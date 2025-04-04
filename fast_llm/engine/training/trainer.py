import abc
import logging
import math
import pathlib
import shutil
import time
import typing

import torch

from fast_llm.config import Configurable
from fast_llm.core.distributed import safe_barrier
from fast_llm.data.data.abstract import Data
from fast_llm.engine.base_model.config import Preprocessor
from fast_llm.engine.config_utils.run import Run, is_main_rank, log_main_rank, log_pipeline_parallel_main_rank
from fast_llm.engine.distributed.config import PhaseType
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.inference.runner import InferenceRunner
from fast_llm.engine.multi_stage.config import StageMode
from fast_llm.engine.optimizer.config import ParamGroup
from fast_llm.engine.optimizer.optimizer import Optimizer
from fast_llm.engine.schedule.runner import ScheduleRunner
from fast_llm.engine.schedule.schedule import Schedule
from fast_llm.engine.training.config import TrainerConfig, TrainingCheckpointBaseConfig, TrainingCheckpointConfig
from fast_llm.engine.training.wandb import Wandb
from fast_llm.logging import format_metrics, get_memory_usage_mib, log_memory_usage
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)


class Trainer[ConfigType: TrainerConfig](Configurable[ConfigType], abc.ABC):
    config_class: typing.ClassVar[type[TrainerConfig]] = TrainerConfig
    # TODO: Generalize data, schedule, logging, etc.
    _is_setup: bool = False
    _distributed: Distributed
    _run: Run
    _wandb: Wandb
    _optimizer: Optimizer

    _completed_steps: int

    def __init__(self, config: TrainerConfig):
        super().__init__(config)
        self._data = self._get_data()
        log_main_rank("Creating model...")
        self._multi_stage = self._config.model.get_model_class()(
            self._config.model,
            optimizer_state_names=self._config.optimizer.state_names(),
        )
        self._reference_models = {}
        for name, reference_config in self._config.reference_models.items():
            log_main_rank(f"Creating `{name} reference model...")
            self._reference_models[name] = self._config.get_inference_runner_class()(
                reference_config.model.get_model_class()(reference_config.model)
            )
            self._multi_stage.base_model.add_preprocessor(
                self._get_reference_model_preprocessor(name, self._reference_models[name])
            )

        phase: PhaseType
        self._runner = ScheduleRunner(
            config=self._config.schedule,
            multi_stage=self._multi_stage,
            distributed_config=self._config.model.distributed,
        )
        steps_per_split = {
            PhaseType.training: {PhaseType.training.value.lower(): self._config.training.train_iters},
            PhaseType.validation: {
                dataset_name: self._config.training.evaluations[dataset_name].get_iteration_count(
                    self._config.training.train_iters,
                    # There may be an extra evaluation after the last training step.
                    not self._config.training.evaluations[dataset_name].enabled(self._config.training.train_iters),
                )
                for dataset_name in self._config.training.evaluations.keys()
            },
            PhaseType.test: {PhaseType.test.value.lower(): self._config.training.test_iters},
        }
        self._samples_per_split = {
            phase: {
                dataset_name: self._config.batch.batch_size * steps
                for dataset_name, steps in datasets.items()
                if steps > 0
            }
            for phase, datasets in steps_per_split.items()
        }
        # Prune empty phases.
        self._samples_per_split = {k: v for k, v in self._samples_per_split.items() if len(v) > 0}

        self._loss_defs = self._multi_stage.base_model.loss_defs

        # Setup the schedules
        self._schedule = {
            phase: {
                dataset_name: Schedule(
                    multi_stage=self._multi_stage,
                    batch_config=self._config.batch,
                    schedule_config=self._config.schedule,
                    distributed_config=self._config.model.distributed,
                    phase=phase,
                )
                for dataset_name in datasets
            }
            for phase, datasets in self._samples_per_split.items()
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
            self._multi_stage.setup(distributed)
            for name, reference_model in self._reference_models.items():
                log_main_rank(f"Setting up `{name}` reference model...")
                reference_model.fast_llm_model.setup(distributed, StageMode.inference)
                reference_model.setup()

        # Setup the optimizer.
        param_groups, grads_for_norm = self._multi_stage.get_param_groups(ParamGroup)
        self._optimizer = self._config.optimizer.optimizer_cls(
            self._config.optimizer,
            param_groups=param_groups,
            grads_for_norm=grads_for_norm,
            distributed=self._distributed,
        )

        # Setup the schedules.
        with torch.no_grad():
            self._runner.setup(distributed, self._optimizer)
        # Setup the datasets.
        log_main_rank("Preparing datasets...")
        self._data.setup(
            distributed,
            {
                dataset_name: steps
                for datasets in self._samples_per_split.values()
                for dataset_name, steps in datasets.items()
            },
            None if run.experiment_directory is None else run.experiment_directory / "dataset_cache",
            timeout=self._config.training.timeout,
        )
        self._is_setup = True

    @abc.abstractmethod
    def _get_data(self) -> Data:
        pass

    @property
    def _consumed_samples(self) -> int:
        assert self._is_setup
        return self._completed_steps * self._config.batch.batch_size

    @property
    def _consumed_tokens(self) -> int:
        assert self._is_setup
        return self._consumed_samples * self._config.batch.sequence_length

    def _get_completed_evaluation_steps(self, dataset_name) -> int:
        # Number of evaluations steps performed before the current step
        return self._config.training.evaluations[dataset_name].get_iteration_count(self._completed_steps - 1)

    def run(self) -> None:
        assert self._is_setup
        with self._wandb:
            self._run_training()

    def _run_training(self) -> None:
        self._prepare_training_state()
        log_main_rank("done with setup ...")
        log_pipeline_parallel_main_rank(lambda: log_memory_usage(f"After initial setup", str))
        self._run.save_logged_tensors("init")

        if PhaseType.training in self._samples_per_split:
            done = self._completed_steps >= self._config.training.train_iters
            if done:
                metrics = {}
                log_main_rank("Training already completed, nothing to do ...")
            else:
                done, metrics = self._train()
        else:
            done, metrics = True, {}

        if done and PhaseType.test in self._samples_per_split:
            log_main_rank(lambda: f"Running test phase ...")
            test_iterator = self._get_data_iterator(PhaseType.test.value.lower())
            metrics_key = PhaseType.test.value
            metrics[metrics_key] = self._evaluate(
                data_iterator=test_iterator,
                phase=PhaseType.test,
                num_iters=self._config.training.test_iters,
            )
            formatted_metrics = format_metrics(metrics[metrics_key], self._loss_defs, PhaseType.test)
            log_main_rank(formatted_metrics)
            self._wandb.alert("Testing results", formatted_metrics, "WARN")
            # TODO: This may erase some metrics.
            self._wandb.log_metrics(self._completed_steps, metrics)

    def _train(self) -> tuple[bool, dict[PhaseType, dict[str, typing.Any]]]:
        # Tracking loss.
        advanced_iters = 0
        skipped_iters = 0
        nan_iters = 0
        total_losses = {loss_def.name: 0.0 for loss_def in self._loss_defs}

        # Profiling
        profiler = self._config.profiling.get_profiler(
            distributed_config=self._config.model.distributed, start_step=self._completed_steps
        )

        train_iterator = self._get_data_iterator(
            PhaseType.training.value,
            self._completed_steps,
            self._config.training.prefetch_factor,
        )
        evaluation_iterators = {name: None for name in self._config.training.evaluations.keys()}

        log_main_rank("Training ...")

        # TODO: Synchronization is probably unnecessary.
        safe_barrier(self._distributed.world_group, "train begin")
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        last_time = start_time
        start_iteration = self._completed_steps
        last_iteration = start_iteration
        stop = False
        with profiler:
            while not stop:
                # Iteration starts at 1, so we increment at the beginning.
                self._completed_steps += 1
                is_logging = self._config.training.logs.enabled(self._completed_steps)

                # TODO: Data loader hates getting all micro-batches at once.
                #   (Also preprocessing adds overhead)
                reduced_losses, update_successful, train_metrics = self._runner.run_step(
                    train_iterator,
                    self._schedule[PhaseType.training][PhaseType.training.value.lower()],
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
                        model_tflops, hardware_tflops = self.get_tflops(PhaseType.training, time_per_iteration)
                        metrics_key = PhaseType.training.value
                        metrics[metrics_key] = {
                            "train_iters": self._config.training.train_iters,
                            "batch_size": self._config.batch.batch_size,
                            "iteration": self._completed_steps,
                            **{
                                name: (value / advanced_iters if advanced_iters > 0 else float("nan"))
                                for name, value in total_losses.items()
                            },
                            "consumed_samples": self._consumed_samples,
                            "consumed_tokens": self._consumed_tokens,
                            "step_time_ms": time_per_iteration * 1000,
                            "step_time_average_ms": average_time_per_iteration * 1000,
                            "remaining_time": remaining_time,
                            "completion_time": time.time() + remaining_time,
                            "percent_done": 100 * self._completed_steps / self._config.training.train_iters,
                            "skipped_iters": skipped_iters,
                            "nan_iters": nan_iters,
                            "model_tflops": model_tflops,
                            "hardware_tflops": hardware_tflops,
                            "tokens_per_sec_per_gpu": (
                                (self._config.batch.sequence_length * self._config.batch.batch_size)
                                / self._config.model.distributed.world_size
                                / time_per_iteration
                            ),
                            "run": self._run.index,
                            **train_metrics,
                            **get_memory_usage_mib(),
                        }

                        formatted_metrics = format_metrics(metrics[metrics_key], self._loss_defs, PhaseType.training)
                        logger.info(formatted_metrics)
                        if self._config.training.wandb.alert.enabled(self._completed_steps):
                            self._wandb.alert("Training results", formatted_metrics, "INFO")

                    advanced_iters = 0
                    skipped_iters = 0
                    nan_iters = 0
                    total_losses = {loss_def.name: 0.0 for loss_def in self._loss_defs}

                self._run.save_logged_tensors(f"train_{self._completed_steps}")

                profiler.step()

                done = self._completed_steps >= self._config.training.train_iters
                # TODO: Signal-based stop.
                stop = done or self._config.training.shutdown.enabled(self._completed_steps)
                # Evaluation
                # TODO: Adjust valid iterator length.
                if PhaseType.validation in self._samples_per_split and (
                    done
                    or any(
                        evaluation_conf.enabled(self._completed_steps)
                        for evaluation_conf in self._config.training.evaluations.values()
                    )
                ):
                    formatted_metrics = []
                    for dataset_name, evaluation_conf in self._config.training.evaluations.items():
                        if not evaluation_conf.enabled(self._completed_steps):
                            continue
                        if evaluation_iterators[dataset_name] is None:
                            evaluation_iterators[dataset_name] = self._get_data_iterator(
                                dataset_name, self._get_completed_evaluation_steps(dataset_name)
                            )
                        # TODO: formatting metric category as Validation.evaluation_dataset_name
                        #       maybe format each metric with evaluation_dataset_name prefix instead?
                        # TODO: setting performance metrics per evaluation dataset
                        #       maybe to set aggregate performance metrics for all evaluations datasets?
                        metric_key = f"{PhaseType.validation.value}.{dataset_name}"
                        metrics[metric_key] = self._evaluate(
                            data_iterator=evaluation_iterators[dataset_name],
                            phase=PhaseType.validation,
                            num_iters=evaluation_conf.iterations,
                            begin_iter=self._get_completed_evaluation_steps(dataset_name),
                            dataset_name=dataset_name,
                        )
                        formatted_metrics.append(
                            format_metrics(
                                metrics[metric_key],
                                self._loss_defs,
                                PhaseType.validation,
                                dataset_name=dataset_name,
                            )
                        )

                    if len(formatted_metrics) > 0:
                        formatted_metrics = "\n".join(formatted_metrics)
                        log_main_rank(formatted_metrics)
                        if self._config.training.wandb.alert.enabled(self._completed_steps):
                            self._wandb.alert("Validation results", formatted_metrics, "INFO")

                if is_main_rank() and metrics:
                    self._wandb.log_metrics(self._completed_steps, metrics)

                if self._config.training.checkpoint.enabled(None if stop else self._completed_steps):
                    self._save_checkpoint(self._config.training.checkpoint, metrics)

                if self._config.training.export.enabled(None if done else self._completed_steps):
                    self._save_checkpoint(self._config.training.export, metrics)
            # The profiler calls the trace_fn at the end and this could lead to
            profiler.step()
        return done, metrics

    def _evaluate(
        self,
        *,
        data_iterator: typing.Iterator,
        phase: PhaseType,
        num_iters: int,
        begin_iter: int = 0,
        dataset_name: str | None = None,
    ) -> dict[str, float | int]:
        full_phase_name = phase.value if dataset_name is None else f"{phase.value}_{dataset_name}"
        safe_barrier(self._distributed.world_group, f"{full_phase_name} begin")
        begin_time = time.perf_counter()
        total_losses = {loss_def.name: 0.0 for loss_def in self._loss_defs}
        for iter_ in range(num_iters):
            iter_losses, _, _ = self._runner.run_step(
                data_iterator, self._schedule[phase][dataset_name], iteration=begin_iter + iter_
            )
            for name, value in iter_losses.items():
                total_losses[name] += value
            self._run.save_logged_tensors(f"{full_phase_name}_{self._completed_steps}_{iter_}")

        safe_barrier(
            self._distributed.world_group,
            f"{full_phase_name} end",
        )
        end_time = time.perf_counter()
        time_per_iteration = (end_time - begin_time) / num_iters
        model_tflops, hardware_tflops = self.get_tflops(phase, time_per_iteration)
        # TODO add other relevant eval metrics
        metrics = {
            "train_iters": self._config.training.train_iters,
            "batch_size": self._config.batch.batch_size,
            "iteration": self._completed_steps,
            **{name: (value / num_iters) for name, value in total_losses.items()},
            "consumed_samples": self._consumed_samples,
            "consumed_tokens": self._consumed_tokens,
            "step_time_ms": time_per_iteration * 1000,
            "model_tflops": model_tflops,
            "hardware_tflops": hardware_tflops,
            "tokens_per_sec_per_gpu": (
                (self._config.batch.sequence_length * self._config.batch.batch_size)
                / self._config.model.distributed.world_size
                / time_per_iteration
            ),
            **get_memory_usage_mib(),
        }

        return metrics

    def _get_data_iterator(
        self, dataset_name, completed_steps: int = 0, prefetch_factor: int | None = None
    ) -> typing.Iterator[typing.Any]:
        return self._data.get_iterator(
            self._config.batch,
            dataset_name,
            consumed_samples=completed_steps * self._config.batch.batch_size,
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
                log_main_rank(f"Initializing training state from scratch...")
                self._multi_stage.initialize_weights()
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
        # TODO v0.3: Move barrier, ok file to FastLLMModel
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
            (checkpoint_directory / "ok").open("w")
            logger.info(f"Saved {config.save_name} to {checkpoint_directory}")

            to_delete = config.to_delete(sorted(int(path.name) for path in checkpoint_base_directory.iterdir()))

            for iteration in to_delete:
                path = checkpoint_base_directory / str(iteration)
                logger.info(f"Deleting {config.save_name} at {path}")
                try:
                    shutil.rmtree(path, ignore_errors=True)
                except OSError as e:
                    logger.warning(f"Could not remove {config.save_name} directory: {e.args}")

            config.callback.run()

    def _load_checkpoint(self, config: TrainingCheckpointConfig, iteration: int) -> None:
        checkpoint_directory = config.get_save_directory(self._run.experiment_directory) / str(iteration)
        Assert.custom(pathlib.Path.is_file, checkpoint_directory / "ok")

        metadata = self._multi_stage.load_checkpoint(
            config.get_load_config(checkpoint_directory, timeout=self._config.training.timeout)
        )
        self._optimizer.load(metadata["optimizer"])
        if "schedules" in metadata:
            # Backward compatibility.
            self._completed_steps = metadata["schedules"][PhaseType.training.value]["completed_steps"]
        else:
            self._completed_steps = metadata["completed_steps"]
        # TODO v0.3: Move barrier, ok file to FastLLMModel
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

    @abc.abstractmethod
    def get_tflops(self, phase: PhaseType, elapsed_time_per_iteration) -> tuple[int, int]:
        # TODO: Do in model, automate/generalize, get other stats
        pass

    def _get_reference_model_preprocessor(self, name: str, inference_runner: InferenceRunner) -> Preprocessor:
        raise NotImplementedError()
