import abc
import logging
import math
import pathlib
import shutil
import time
import typing

import torch

from fast_llm.config import Configurable
from fast_llm.core.distributed import allreduce_scalar, safe_barrier
from fast_llm.data.data.abstract import Data
from fast_llm.engine.config_utils.run import Run, is_main_rank, log_main_rank, log_pipeline_parallel_main_rank
from fast_llm.engine.distributed.config import PhaseType
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.multi_stage.config import StageMode
from fast_llm.engine.optimizer.config import ParamGroup
from fast_llm.engine.optimizer.optimizer import Optimizer
from fast_llm.engine.schedule.runner import ScheduleRunner
from fast_llm.engine.schedule.schedule import Schedule
from fast_llm.engine.training.config import (
    TrainerConfig,
    TrainingCheckpointBaseConfig,
    TrainingCheckpointConfig,
)
from fast_llm.engine.training.wandb import Wandb
from fast_llm.logging import format_metrics, log_memory_usage
from fast_llm.utils import Assert, Interrupter, get_and_reset_memory_usage_mib

logger = logging.getLogger(__name__)


class Trainer[ConfigType: TrainerConfig](Configurable[ConfigType], abc.ABC):
    # TODO: Generalize data, schedule, logging, etc.
    _is_setup: bool = False
    _distributed: Distributed
    _run: Run
    _wandb: Wandb
    _optimizer: Optimizer | None
    _completed_steps: int
    _schedule: Schedule

    def __init__(self, config: TrainerConfig):
        super().__init__(config)

        self._do_train = config.training.train_iters > 0

        self._data = self._get_data()
        log_main_rank("Creating model...")
        self._multi_stage = self._config.model.get_model_class()(
            self._config.model,
            optimizer_state_names=self._config.optimizer.state_names() if self._do_train else (),
        )
        self._reference_models = {}
        for name, reference_config in self._config.reference_models.items():
            log_main_rank(f"Creating `{name} reference model...")
            self._reference_models[name] = reference_config.model.get_inference_runner_class()(
                reference_config.model.get_model_class()(reference_config.model)
            )
            self._multi_stage.base_model.add_reference_model(name, self._reference_models[name])

        self._runner = ScheduleRunner(
            config=self._config.schedule,
            multi_stage=self._multi_stage,
            distributed_config=self._config.model.distributed,
        )
        self._loss_definitions = self._multi_stage.base_model.get_loss_definitions()
        self._callbacks = {
            name: config.get_callback(self._multi_stage) for name, config in self._config.callbacks.items()
        }

        self._evaluators = {
            name: config.get_evaluator(name, self._config.training.num_workers)
            for name, config in self._config.training.evaluators.items()
            if config.enabled()
        }

    def setup(self, distributed: Distributed, run: Run) -> None:
        assert distributed.config is self._config.model.distributed
        assert not self._is_setup
        self._distributed = distributed
        self._run = run
        self._wandb = Wandb(self._config.training.wandb, self._run, self._config)

        # Setup the model.
        with torch.no_grad():
            log_main_rank("Setting up model...")
            self._multi_stage.setup(distributed, mode=StageMode.training if self._do_train else StageMode.inference)
            for name, reference_model in self._reference_models.items():
                log_main_rank(f"Setting up `{name}` reference model...")
                reference_model.fast_llm_model.setup(distributed, StageMode.inference)
                reference_model.setup()

        # Setup the optimizer.
        if self._do_train:
            param_groups, grads_for_norm = self._multi_stage.get_param_groups(ParamGroup)
            self._optimizer = self._config.optimizer.optimizer_cls(
                self._config.optimizer,
                param_groups=param_groups,
                grads_for_norm=grads_for_norm,
                distributed=self._distributed,
            )
        else:
            self._optimizer = None

        # Setup the schedules.
        with torch.no_grad():
            self._runner.setup(distributed, self._optimizer)
        # Setup the datasets.
        log_main_rank("Preparing datasets...")

        self._data.setup(None if run.experiment_directory is None else run.experiment_directory / "dataset_cache")
        if self._do_train:
            preprocessing_config = self._multi_stage.get_preprocessing_config(
                PhaseType.training, self._config.schedule.micro_batch_splits
            )
            self._schedule = Schedule(
                config=self._config.schedule,
                multi_stage=self._multi_stage,
                batch_meta=preprocessing_config.get_input_meta(self._data.config.micro_batch_size),
                distributed_config=self._config.model.distributed,
                phase=PhaseType.training,
            )
            self._data.sample_dataset(
                str(PhaseType.training),
                preprocessing_config,
                self._config.training.train_iters * self._schedule.samples_per_batch,
            )

        for name, evaluator in self._evaluators.items():
            run_count = self._config.training.evaluators[name].get_count(self._config.training.train_iters)
            # There may be an extra evaluation after the last training step.
            if not self._config.training.evaluators[name].enabled(self._config.training.train_iters):
                run_count += 1
            evaluator.setup(multi_stage=self._multi_stage, runner=self._runner, data=self._data, run_count=run_count)

        # Make sure everyone is done before continuing.
        safe_barrier(distributed.world_group, "data_preparation", self._config.training.timeout)

        self._is_setup = True

    @abc.abstractmethod
    def _get_data(self) -> Data:
        pass

    def _get_completion_metrics(self) -> dict[str, int | float]:
        assert self._is_setup
        return {
            "total_steps": self._config.training.train_iters,
            "completed_steps": self._completed_steps,
            "consumed_tokens": self._completed_steps * self._batch_size,
            "percent_done": 100 * self._completed_steps / self._config.training.train_iters,
        }

    @property
    def _batch_size(self) -> int:
        return self._schedule.samples_per_batch * self._data.config.micro_batch_size

    def run(self) -> None:
        assert self._is_setup
        with self._wandb:
            self._run_training()
        for callback in self._callbacks.values():
            callback.train_end(self._completed_steps)

    def _run_training(self) -> None:
        self._prepare_training_state()

        log_main_rank("done with setup ...")
        log_pipeline_parallel_main_rank(lambda: log_memory_usage(f"After initial setup", str))
        self._run.save_logged_tensors("init")

        if not self._do_train:
            self._run_evaluators(True, {})
        elif self._completed_steps >= self._config.training.train_iters:
            log_main_rank("Training already completed, nothing to do ...")
        else:
            self._train()

    def _train(self) -> tuple[bool, dict[PhaseType, dict[str, typing.Any]]]:
        # Tracking loss.
        advanced_iters = 0
        skipped_iters = 0
        nan_iters = 0
        total_losses = {loss_def.name: 0.0 for loss_def in self._loss_definitions}

        # Profiling
        profiler = self._config.profiling.get_profiler(
            distributed_config=self._config.model.distributed, start_step=self._completed_steps
        )

        interrupter = Interrupter(self._config.training.checkpoint.enabled())
        train_iterator = self._get_data_iterator(
            PhaseType.training,
            self._completed_steps,
            self._config.training.prefetch_factor,
        )

        log_main_rank("Training ...")

        # TODO: Synchronization is probably unnecessary.
        safe_barrier(self._distributed.world_group, "train begin")

        for callback in self._callbacks.values():
            callback.run_begin(self._completed_steps)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        last_time = start_time
        start_iteration = self._completed_steps
        last_iteration = start_iteration
        stop = False
        with profiler, interrupter:
            while not stop:
                # Iteration starts at 1, so we increment at the beginning.
                self._completed_steps += 1
                is_logging = self._config.training.logs.enabled(self._completed_steps)

                # TODO: Data loader hates getting all micro-batches at once.
                #   (Also preprocessing adds overhead)
                reduced_losses, update_successful, train_metrics = self._runner.run_step(
                    train_iterator,
                    self._schedule,
                    iteration=self._completed_steps,
                    return_metrics=is_logging,
                )

                # Advanced, skipped, and Nan iterations.
                if update_successful:
                    advanced_iters += 1
                    for name, value in reduced_losses.items():
                        total_losses[name] += value
                else:
                    skipped_iters += 1
                    nan_iters += not all(math.isfinite(loss) for loss in reduced_losses.values())

                for callback in self._callbacks.values():
                    callback.step_end(self._completed_steps, reduced_losses, update_successful, train_metrics)
                # Logging.
                metrics = {}
                if is_logging:
                    # TODO: Synchronization is probably unnecessary.
                    safe_barrier(self._distributed.world_group, f"logging {self._completed_steps}")
                    if self._run.is_main_rank:
                        new_time = time.perf_counter()
                        time_per_iteration = (new_time - last_time) / (self._completed_steps - last_iteration)
                        average_time_per_iteration = (new_time - start_time) / (
                            self._completed_steps - start_iteration
                        )
                        last_time = new_time
                        last_iteration = self._completed_steps
                        remaining_time = average_time_per_iteration * (
                            self._config.training.train_iters - self._completed_steps
                        )
                        metrics_key = PhaseType.training
                        metrics[metrics_key] = {
                            "batch_size": self._batch_size,
                            **{
                                name: (value / advanced_iters if advanced_iters > 0 else float("nan"))
                                for name, value in total_losses.items()
                            },
                            **self._get_completion_metrics(),
                            "step_time_ms": time_per_iteration * 1000,
                            "step_time_average_ms": average_time_per_iteration * 1000,
                            "remaining_time": remaining_time,
                            "completion_time": time.time() + remaining_time,
                            "skipped_iters": skipped_iters,
                            "nan_iters": nan_iters,
                            **self._schedule.get_compute_metrics(time_per_iteration),
                            "tokens_per_sec_per_gpu": (
                                self._batch_size / self._config.model.distributed.world_size / time_per_iteration
                            ),
                            "run": self._run.index,
                            **train_metrics,
                            **get_and_reset_memory_usage_mib(),
                        }

                        formatted_metrics = format_metrics(
                            metrics[metrics_key], self._loss_definitions, PhaseType.training
                        )
                        logger.info(formatted_metrics)
                        if self._config.training.wandb.alert.enabled(self._completed_steps):
                            self._wandb.alert("Training results", formatted_metrics, "INFO")

                    advanced_iters = 0
                    skipped_iters = 0
                    nan_iters = 0
                    total_losses = {loss_def.name: 0.0 for loss_def in self._loss_definitions}

                self._run.save_logged_tensors(f"train_{self._completed_steps}")

                profiler.step()

                done = self._completed_steps >= self._config.training.train_iters
                # TODO: Signal-based stop.
                stop = done or self._config.training.shutdown.enabled(self._completed_steps)

                # Evaluation
                self._run_evaluators(done, metrics)

                if is_main_rank() and metrics:
                    self._wandb.log_metrics(self._completed_steps, metrics, commit=True)

                if self._config.training.export.enabled(None if done else self._completed_steps):
                    self._save_checkpoint(self._config.training.export, metrics)

                if interrupter.enabled:
                    stop = stop or allreduce_scalar(
                        interrupter.interrupted, torch.int32, self._distributed.world_group
                    )

                if self._config.training.checkpoint.enabled(None if stop else self._completed_steps):
                    self._save_checkpoint(self._config.training.checkpoint, metrics)

            # The profiler calls the trace_fn at the end and this could lead to
            profiler.step()
        return done, metrics

    def _get_data_iterator(
        self, dataset_name, completed_steps: int = 0, prefetch_factor: int | None = None
    ) -> typing.Iterator[typing.Any]:
        return self._data.get_iterator(
            dataset_name,
            consumed_samples=completed_steps * self._schedule.samples_per_batch,
            num_workers=self._config.training.num_workers,
            prefetch_factor=prefetch_factor,
            timeout=self._config.training.timeout,
        )

    def _prepare_training_state(self) -> None:
        # Setup the training state.
        if (last_iteration := self._get_last_checkpoint()) is None:
            if (path := self._config.pretrained.path) is not None and self._config.pretrained.model_weights:
                log_main_rank(
                    f"Initializing training state from pretrained checkpoint at {path}"
                    f" ({'loading' if self._config.pretrained.optimizer_state else 'resetting'}"
                    f" optimizer state)..."
                )
                self._multi_stage.load_checkpoint(self._config.pretrained)
            else:
                if not self._do_train:
                    raise ValueError(
                        "Evaluation mode, model need to be trained first or pretrained checkpoint is provided for loading"
                    )
                log_main_rank(f"Initializing training state from scratch...")
                self._multi_stage.initialize_weights()

            if self._do_train:
                self._optimizer.reset_state()
            self._completed_steps = 0
        else:
            log_main_rank(lambda: f"Loading checkpoint from iteration {last_iteration}...")
            self._load_checkpoint(self._config.training.checkpoint, last_iteration)

        for name, reference_model in self._reference_models.items():
            pretrained = self._config.reference_models[name].pretrained
            if pretrained.path is not None and pretrained.model_weights:
                log_main_rank(f"Loading weights for `{name}` reference model from {pretrained.path}")
                reference_model.fast_llm_model.load_checkpoint(pretrained)
            else:
                log_main_rank(
                    f"No pretrained checkpoint specified for `{name}` reference model,"
                    f" using a freshly initialized model...",
                    log_fn=logger.warning,
                )
                reference_model.fast_llm_model.initialize_weights()

        Assert.eq(self._completed_steps, last_iteration or 0)
        assert self._multi_stage._is_loaded  # noqa

    def _save_checkpoint(
        self, config: TrainingCheckpointBaseConfig, metrics: dict[str, dict[str, float | int]] | None
    ) -> None:
        # TODO: Move barrier, ok file to FastLLMModel
        checkpoint_base_directory = config.get_save_directory(self._run.experiment_directory)
        checkpoint_directory = checkpoint_base_directory / str(self._completed_steps)

        # Create the checkpoint
        if self._run.is_main_rank:
            logger.info(f"Saving {config.save_name} at iteration {self._completed_steps}")
            checkpoint_directory.mkdir(exist_ok=False, parents=True)
        # Barrier to ensure the directory is created correctly (and didn't exist before).
        safe_barrier(self._distributed.world_group, f"{config.save_name} {self._completed_steps} enter")

        metadata = {
            "optimizer": self._optimizer.save(),
            "completed_steps": self._completed_steps,
        }
        if metrics is not None:
            metadata["metrics"] = metrics
        self._multi_stage.save_checkpoint(
            config.get_save_config(checkpoint_directory, timeout=self._config.training.timeout), metadata
        )

        # Barrier to ensure everyone is done.
        safe_barrier(
            self._distributed.world_group,
            f"{config.save_name} {self._completed_steps} exit",
            timeout=self._config.training.timeout,
        )
        # Mark the checkpoint as complete.
        if self._run.is_main_rank:
            (checkpoint_directory / "ok").touch()
            logger.info(f"Saved {config.save_name} to {checkpoint_directory}")

            to_delete = config.to_delete(sorted(int(path.name) for path in checkpoint_base_directory.iterdir()))

            for iteration in to_delete:
                path = checkpoint_base_directory / str(iteration)
                logger.info(f"Deleting {config.save_name} at {path}")
                try:
                    shutil.rmtree(path, ignore_errors=True)
                except OSError as e:
                    logger.warning(f"Could not remove {config.save_name} directory: {e.args}")

    def _load_checkpoint(self, config: TrainingCheckpointConfig, iteration: int) -> None:
        checkpoint_directory = config.get_save_directory(self._run.experiment_directory) / str(iteration)
        Assert.custom(pathlib.Path.is_file, checkpoint_directory / "ok")

        metadata = self._multi_stage.load_checkpoint(
            config.get_load_config(checkpoint_directory, timeout=self._config.training.timeout)
        )
        assert metadata is not None
        if self._do_train:
            self._optimizer.load(metadata["optimizer"])
        if "schedules" in metadata:
            # Backward compatibility.
            self._completed_steps = metadata["schedules"][PhaseType.training]["completed_steps"]
        else:
            self._completed_steps = metadata["completed_steps"]
        # TODO: Move barrier, ok file to FastLLMModel
        safe_barrier(
            self._distributed.world_group,
            f"load {config.save_name} {iteration} exit",
            timeout=self._config.training.timeout,
        )

    def _get_last_checkpoint(self) -> int | None:
        if self._run.experiment_directory is None:
            return None
        checkpoint_base_directory = (
            self._run.experiment_directory
            / self._config.training.checkpoint.get_save_directory(self._run.experiment_directory)
        )
        if self._run.is_main_rank and checkpoint_base_directory.is_dir():
            checkpoints = [int(path.name) for path in checkpoint_base_directory.iterdir()]
            iteration = max(checkpoints) if checkpoints else -1
        else:
            iteration = -1
        iteration = self._run.broadcast_int(iteration)
        return iteration if iteration >= 0 else None

    def _run_evaluators(self, done: bool, metrics: dict[str, typing.Any] | None = None) -> None:
        for name, evaluator in self._evaluators.items():
            config = self._config.training.evaluators[name]
            if config.enabled(None if done else self._completed_steps):
                evaluator.run(
                    run_index=config.get_run_count(self._completed_steps - 1),
                    metrics=(evaluator_metrics := self._get_completion_metrics()),
                )
                if metrics is not None:
                    if "evaluations" not in metrics:
                        metrics["evaluations"] = {}
                    metrics["evaluations"][name] = evaluator_metrics
