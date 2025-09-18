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
from fast_llm.data.dataset.config import SamplingParameters
from fast_llm.engine.config_utils.run import Run, is_main_rank, log_main_rank, log_pipeline_parallel_main_rank
from fast_llm.engine.distributed.config import PhaseType
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.evaluation.evaluator import (
    EvaluationMetrics,
    Evaluator,
    EvaluatorRunner,
    EvaluatorSamplingParameters,
    TrainingProgress,
)
from fast_llm.engine.multi_stage.config import StageMode
from fast_llm.engine.multi_stage.fast_llm_model import FastLLMModel
from fast_llm.engine.optimizer.config import ParamGroup
from fast_llm.engine.optimizer.optimizer import Optimizer
from fast_llm.engine.schedule.config import BatchConfig
from fast_llm.engine.schedule.runner import ScheduleRunner
from fast_llm.engine.schedule.schedule import Schedule
from fast_llm.engine.training.config import (
    TrainerConfig,
    TrainingCheckpointBaseConfig,
    TrainingCheckpointConfig,
    TrainingEvaluatorConfig,
)
from fast_llm.engine.training.wandb import Wandb
from fast_llm.logging import format_metrics, log_memory_usage
from fast_llm.utils import Assert, Interrupter, get_and_reset_memory_usage_mib

logger = logging.getLogger(__name__)


class TrainingEvaluator[ConfigType: TrainingEvaluatorConfig](Evaluator[ConfigType]):
    evaluator: Evaluator

    def __init__(
        self,
        name: str,
        eval_config: TrainingEvaluatorConfig,
        batch_config: BatchConfig,
        data_load_num_proc: int,
        train_iters: int | None = None,
    ):
        super().__init__(name, eval_config, batch_config, data_load_num_proc, train_iters)

        self._train_iters = 0 if self._train_iters is None else self._train_iters

        self.evaluator = eval_config.evaluator.get_evaluator(name, batch_config, data_load_num_proc, train_iters)

    def setup(
        self,
        distributed: Distributed,
        run: Run,
        multi_stage: FastLLMModel,
        runner: ScheduleRunner,
        data: Data,
        phase: PhaseType,
    ) -> None:
        self.evaluator.setup(
            distributed,
            run,
            multi_stage,
            runner,
            data,
            phase,
        )

    def run(
        self,
        training_progress: TrainingProgress | None = None,
        run_index: int | None = None,
    ) -> EvaluationMetrics:
        # Run index must be None because it is defined here to be passed to actual evaluator
        assert run_index is None

        # Training progress can be None as it can be run in a training
        #  run without training, just evaluation
        if training_progress is None:
            done = True
            completed_steps = 0
        else:
            done = training_progress.done
            completed_steps = training_progress.completed_steps

        if (done and self.config.enabled()) or self.config.enabled(completed_steps):
            return self.evaluator.run(training_progress, run_index=self._config.get_run_count(completed_steps - 1))
        else:
            return EvaluationMetrics()

    def get_sampling_parameters(self) -> EvaluatorSamplingParameters | None:
        name_samples = self.evaluator.get_sampling_parameters()
        if name_samples is None:
            return None
        run_count = self._config.get_run_count(
            self._train_iters,
            # There may be an extra evaluation after the last training step.s
            not self._config.enabled(self._train_iters),
        )
        return EvaluatorSamplingParameters(name_samples.dataset_name, name_samples.num_samples * run_count)


class Trainer[ConfigType: TrainerConfig](Configurable[ConfigType], abc.ABC):
    # TODO: Generalize data, schedule, logging, etc.
    _is_setup: bool = False
    _distributed: Distributed
    _run: Run
    _wandb: Wandb
    _optimizer: Optimizer

    _completed_steps: int

    _is_evaluation_only: bool

    _evaluator_runner: EvaluatorRunner

    def __init__(self, config: TrainerConfig):
        super().__init__(config)

        self._is_evaluation_only = config.training.train_iters == 0

        self._data = self._get_data()
        log_main_rank("Creating model...")
        self._multi_stage = self._config.model.get_model_class()(
            self._config.model,
            optimizer_state_names=self._config.optimizer.state_names() if not self._is_evaluation_only else (),
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
        self._loss_defs = self._multi_stage.base_model.loss_defs

        if not self._is_evaluation_only:
            steps_per_split = {
                PhaseType.training: {PhaseType.training.value.lower(): self._config.training.train_iters},
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
        else:
            self._samples_per_split = {}

        self._evaluator_runner = EvaluatorRunner(
            evaluator_configs=self._config.training.evaluators,
            batch_config=self._config.batch,
            data_load_num_proc=self._config.training.num_workers,
            train_iters=self._config.training.train_iters,
            wandb_config=self._config.training.wandb,
        )

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
        if self._is_evaluation_only:
            self._optimizer = None
        else:
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
                dataset_name: self._get_sampling_parameters({"num_samples": samples})
                for datasets in self._samples_per_split.values()
                for dataset_name, samples in datasets.items()
            }
            | {
                eval_sampling_params.dataset_name: self._get_sampling_parameters(
                    {"num_samples": eval_sampling_params.num_samples}
                )
                for eval_sampling_params in self._evaluator_runner.get_sampling_parameters()
            },
            None if run.experiment_directory is None else run.experiment_directory / "dataset_cache",
            timeout=self._config.training.timeout,
        )

        # Must be called with all arguments set up
        self._evaluator_runner.setup(
            distributed=self._distributed,
            run=self._run,
            multi_stage=self._multi_stage,
            runner=self._runner,
            data=self._data,
            wandb=self._wandb,
            phase=PhaseType.inference if self._is_evaluation_only else PhaseType.validation,
        )

        self._is_setup = True

    @abc.abstractmethod
    def _get_data(self) -> Data:
        pass

    def _get_sampling_parameters(
        self, parameters: dict[str, typing.Any], _return_dict: bool = False
    ) -> SamplingParameters | dict[str, typing.Any]:
        return parameters if _return_dict else SamplingParameters(**parameters)

    @property
    def _consumed_samples(self) -> int:
        assert self._is_setup
        return self._completed_steps * self._config.batch.batch_size

    @property
    def _consumed_tokens(self) -> int:
        assert self._is_setup
        return self._consumed_samples * self._config.batch.sequence_length

    def run(self) -> None:
        assert self._is_setup
        with self._wandb:
            self._run_training()

    def _run_training(self) -> None:
        self._prepare_training_state()

        log_main_rank("done with setup ...")
        log_pipeline_parallel_main_rank(lambda: log_memory_usage(f"After initial setup", str))
        self._run.save_logged_tensors("init")

        if self._is_evaluation_only:
            assert len(self._samples_per_split) == 0

        if PhaseType.training in self._samples_per_split:
            done = self._completed_steps >= self._config.training.train_iters
            if done:
                metrics = {}
                log_main_rank("Training already completed, nothing to do ...")
            else:
                done, metrics = self._train()
        else:
            metrics = {}
            done = True
            self._evaluator_runner.run(
                metrics=metrics,
                # This is set to ensure that evaluators like lm_eval log results at the correct step if a checkpoint was loaded.
                training_progress=TrainingProgress(
                    done=done,
                    completed_steps=self._completed_steps,
                    consumed_samples=self._consumed_samples,
                    consumed_tokens=self._consumed_tokens,
                ),
            )

        if done and PhaseType.test in self._samples_per_split:
            log_main_rank(lambda: f"Running test phase ...")
            test_iterator = self._get_data_iterator(PhaseType.test.value.lower())
            metrics_key = PhaseType.test.value
            metrics[metrics_key] = self._evaluate_loss(
                data_iterator=test_iterator,
                phase=PhaseType.test,
                num_iters=self._config.training.test_iters,
            )
            formatted_metrics = format_metrics(metrics[metrics_key], self._loss_defs, PhaseType.test)
            log_main_rank(formatted_metrics)
            self._wandb.alert("Testing results", formatted_metrics, "WARN")
            # TODO: This may erase some metrics.
            self._wandb.log_metrics(self._completed_steps, metrics, commit=True)

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

        interrupter = Interrupter(self._config.training.checkpoint.enabled())
        train_iterator = self._get_data_iterator(
            PhaseType.training.value,
            self._completed_steps,
            self._config.training.prefetch_factor,
        )

        has_test_phase = PhaseType.test in self._samples_per_split

        log_main_rank("Training ...")

        # TODO: Synchronization is probably unnecessary.
        safe_barrier(self._distributed.world_group, "train begin")
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
                        model_compute, hardware_compute = self._schedule[PhaseType.training][
                            PhaseType.training.value.lower()
                        ].compute_usage
                        model_tflops = math.nan if model_compute is None else model_compute / time_per_iteration
                        hardware_tflops = (
                            math.nan if hardware_compute is None else hardware_compute / time_per_iteration
                        )

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
                            **get_and_reset_memory_usage_mib(),
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
                self._evaluator_runner.run(
                    metrics=metrics,
                    training_progress=TrainingProgress(
                        done=done,
                        completed_steps=self._completed_steps,
                        consumed_samples=self._consumed_samples,
                        consumed_tokens=self._consumed_tokens,
                    ),
                )

                if is_main_rank() and metrics:
                    self._wandb.log_metrics(self._completed_steps, metrics, commit=not (done and has_test_phase))

                stop = done or self._config.training.shutdown.enabled(self._completed_steps)

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
                if self._is_evaluation_only:
                    raise ValueError(
                        "Evaluation mode, model need to be trained first or pretrained checkpoint is provided for loading"
                    )
                log_main_rank(f"Initializing training state from scratch...")
                self._multi_stage.initialize_weights()

            if not self._is_evaluation_only:
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
        assert metadata is not None
        if not self._is_evaluation_only:
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
