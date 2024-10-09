import abc
import json
import logging
import math
import os
import shlex
import subprocess
import time
import typing

import torch

from fast_llm.core.distributed import safe_barrier
from fast_llm.data.data import Data
from fast_llm.engine.config_utils.run import Run, is_main_rank, log_main_rank, log_pipeline_parallel_main_rank
from fast_llm.engine.distributed.config import PhaseType
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.multi_stage.config import CheckpointConfig, CheckpointType
from fast_llm.engine.multi_stage.fast_llm_model import FastLLMModel
from fast_llm.engine.optimizer.config import ParamGroup
from fast_llm.engine.optimizer.optimizer import Optimizer
from fast_llm.engine.schedule.runner import ScheduleRunner
from fast_llm.engine.schedule.schedule import Schedule
from fast_llm.engine.training.config import TrainerConfig
from fast_llm.logging import format_metrics, get_memory_usage_mib, log_memory_usage
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)


class Trainer:
    config_class: typing.ClassVar[type[TrainerConfig]] = TrainerConfig
    model_class: typing.ClassVar[type[FastLLMModel]] = FastLLMModel
    # TODO: Generalize data, schedule, logging, etc.
    _is_setup: bool = False
    _distributed: Distributed
    _run: Run
    _optimizer: Optimizer
    _completed_steps: int

    def __init__(self, config: TrainerConfig):
        Assert.custom(isinstance, config, self.config_class)
        config.validate()
        self._config = config
        self._data = Data(
            config=self._config.data,
            distributed_config=self._config.distributed,
            # TODO: `vocab_size` is not generic.
            vocab_size=self._config.base_model.vocab_size,  # Noqa
            max_sequence_length=self._config.batch.sequence_length,
        )
        log_main_rank("Creating model...")
        self._multi_stage = self.model_class(
            self._config.model,
            optimizer_state_names=self._config.optimizer.state_names(),
        )
        phase: PhaseType
        self._runner = ScheduleRunner(
            multi_stage=self._multi_stage,
            config=self._config.schedule,
            distributed_config=self._config.distributed,
        )
        steps_per_split = {
            PhaseType.training: self._config.training.train_iters,
            PhaseType.validation: (self._config.training.train_iters // self._config.training.validation_interval + 1)
            * self._config.training.validation_iters,
            PhaseType.test: self._config.training.test_iters,
        }
        self._samples_per_split = {
            phase: self._config.batch.batch_size * steps for phase, steps in steps_per_split.items() if steps > 0
        }
        self._loss_defs = self._multi_stage.base_model.loss_defs

        # Setup the schedules
        self._schedule = {
            phase: Schedule(
                multi_stage=self._multi_stage,
                batch_config=self._config.batch,
                schedule_config=self._config.schedule,
                distributed_config=self._config.distributed,
                phase=phase,
            )
            for phase in self._samples_per_split
        }

    def setup(self, distributed: Distributed, run: Run):
        assert distributed.config is self._config.distributed
        assert not self._is_setup
        self._is_setup = True
        self._distributed = distributed
        self._run = run

        # Setup the model.
        with torch.no_grad():
            self._multi_stage.setup(distributed)

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
        self._data.setup(distributed, self._samples_per_split)

    @property
    def _consumed_samples(self):
        assert self._is_setup
        return self._completed_steps * self._config.batch.batch_size

    @property
    def _consumed_tokens(self):
        assert self._is_setup
        return self._consumed_samples * self._config.batch.sequence_length

    @property
    def _completed_validation_steps(self) -> int:
        # Number of validation steps performed before the current step
        return (
            (self._completed_steps - 1)
            // self._config.training.validation_interval
            * self._config.training.validation_iters
        )

    def run(self):
        assert self._is_setup
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
            test_iterator = self._get_data_iterator(PhaseType.test)
            metrics[PhaseType.test] = self._evaluate(
                data_iterator=test_iterator,
                phase=PhaseType.test,
                num_iters=self._config.training.test_iters,
            )
            formatted_metrics = format_metrics(metrics[PhaseType.test], self._loss_defs, PhaseType.test)
            log_main_rank(formatted_metrics)
            self._run.post_wandb_alert("Testing results", formatted_metrics, "WARN")
            # TODO: This may erase some metrics.
            self._run.log_wandb_metrics(self._completed_steps, metrics)

    def _train(self):
        # Tracking loss.
        advanced_iters = 0
        skipped_iters = 0
        nan_iters = 0
        total_losses = {loss_def.name: 0.0 for loss_def in self._loss_defs}

        # Profiling
        profiler = self._config.profiling.get_profiler(
            distributed_config=self._config.distributed, start_step=self._completed_steps
        )

        # The triton compilation during the first iteration breaks parallel data loading
        # https://github.com/ServiceNow/Fast-LLM/issues/101,
        # so we run the first iteration without it.
        train_iterator = self._get_data_iterator(
            PhaseType.training,
            self._completed_steps,
            self._config.training.prefetch_factor,
        )
        valid_iterator = None

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
                is_logging = (
                    self._config.run.log_interval
                    and (self._completed_steps - self._config.run.log_offset) % self._config.run.log_interval == 0
                )

                # TODO: Data loader hates getting all micro-batches at once.
                #   (Also preprocessing adds overhead)
                reduced_losses, update_successful, train_metrics = self._runner.run_step(
                    train_iterator,
                    self._schedule[PhaseType.training],
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
                        metrics[PhaseType.training] = {
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
                                / self._config.distributed.world_size
                                / time_per_iteration
                            ),
                            "run": self._run.index,
                            **train_metrics,
                            **get_memory_usage_mib(),
                        }

                        formatted_metrics = format_metrics(
                            metrics[PhaseType.training], self._loss_defs, PhaseType.training
                        )
                        logger.info(formatted_metrics)
                        if (
                            self._config.run.wandb_status_interval
                            and (self._completed_steps - self._config.run.log_offset)
                            % self._config.run.wandb_status_interval
                            == 0
                        ):
                            self._run.post_wandb_alert("Training results", formatted_metrics, "INFO")

                    advanced_iters = 0
                    skipped_iters = 0
                    nan_iters = 0
                    total_losses = {loss_def.name: 0.0 for loss_def in self._loss_defs}

                self._run.save_logged_tensors(f"train_{self._completed_steps}")

                profiler.step()

                done = self._completed_steps >= self._config.training.train_iters
                # TODO: Signal-based stop.
                stop = done or (
                    self._config.run.stop_interval
                    and (self._completed_steps - self._config.run.stop_offset) % self._config.run.stop_interval == 0
                )
                # Evaluation
                # TODO: Adjust valid iterator length.
                if PhaseType.validation in self._samples_per_split and (
                    done
                    or (
                        self._config.training.validation_interval
                        and self._completed_steps % self._config.training.validation_interval == 0
                    )
                ):
                    if valid_iterator is None:
                        valid_iterator = self._get_data_iterator(
                            PhaseType.validation, self._completed_validation_steps
                        )
                    metrics[PhaseType.validation] = self._evaluate(
                        data_iterator=valid_iterator,
                        phase=PhaseType.validation,
                        num_iters=self._config.training.validation_iters,
                        begin_iter=self._completed_validation_steps,
                    )
                    formatted_metrics = format_metrics(
                        metrics[PhaseType.validation], self._loss_defs, PhaseType.validation
                    )
                    log_main_rank(formatted_metrics)
                    if (
                        self._config.run.wandb_status_interval
                        and (self._completed_steps - self._config.run.log_offset)
                        % self._config.run.wandb_status_interval
                        == 0
                    ):
                        self._run.post_wandb_alert("Validation results", formatted_metrics, "INFO")

                if is_main_rank() and metrics:
                    self._run.log_wandb_metrics(self._completed_steps, metrics)

                if self._config.run.checkpoint_interval and (
                    stop
                    or (
                        self._config.run.checkpoint_interval
                        and (self._completed_steps - self._config.run.checkpoint_offset)
                        % self._config.run.checkpoint_interval
                        == 0
                    )
                ):
                    self._save_checkpoint(
                        metrics,
                        export=self._config.run.export_interval
                        and (
                            done
                            or (self._completed_steps - self._config.run.checkpoint_offset)
                            % self._config.run.export_interval
                            == 0
                        ),
                    )

        return done, metrics

    def _evaluate(
        self,
        *,
        data_iterator: typing.Iterator,
        phase: PhaseType,
        num_iters: int,
        begin_iter: int = 0,
    ) -> dict[str, float | int]:
        safe_barrier(self._distributed.world_group, f"{phase.value} begin")
        begin_time = time.perf_counter()
        total_losses = {loss_def.name: 0.0 for loss_def in self._loss_defs}
        for iter_ in range(num_iters):
            iter_losses, _, _ = self._runner.run_step(
                data_iterator, self._schedule[phase], iteration=begin_iter + iter_
            )
            for name, value in iter_losses.items():
                total_losses[name] += value
            self._run.save_logged_tensors(f"{phase}_{self._completed_steps}_{iter_}")

        safe_barrier(self._distributed.world_group, f"{phase.value} end")
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
                / self._config.distributed.world_size
                / time_per_iteration
            ),
            **get_memory_usage_mib(),
        }

        return metrics

    def _prepare_training_state(self):
        # Setup the training state.
        if (last_iteration := self._run.get_last_checkpoint()) is None:
            if (
                path := self._config.pretrained.pretrained_checkpoint_path
            ) is not None and self._config.pretrained.load_pretrained_weights:
                log_main_rank(
                    f"Initializing training state from pretrained checkpoint at {path}"
                    f" ({'loading' if self._config.pretrained.load_pretrained_optimizer else 'resetting'}"
                    f" optimizer state)..."
                )
                self._multi_stage.load_pretrained_checkpoint(self._config.pretrained)
            else:
                log_main_rank(f"Initializing training state from scratch...")
                self._multi_stage.initialize_weights()
            self._optimizer.reset_state()
            self._completed_steps = 0
        else:
            log_main_rank(lambda: f"Loading checkpoint from iteration {last_iteration}...")
            with self._run.get_load_checkpoint_context(last_iteration) as context:
                metadata = self._multi_stage.load_distributed_checkpoint_same_format(context.directory)
            self._optimizer.load(metadata["optimizer"])
            if "schedules" in metadata:
                # Backward compatibility.
                self._completed_steps = metadata["schedules"][PhaseType.training.value]["completed_steps"]
            else:
                self._completed_steps = metadata["completed_steps"]

        Assert.eq(self._completed_steps, last_iteration or 0)
        assert self._multi_stage._is_loaded  # noqa

    def _get_data_iterator(self, phase, completed_steps: int = 0, prefetch_factor: int | None = None):
        return self._data.get_iterator(
            self._config.batch,
            phase,
            consumed_samples=completed_steps * self._config.batch.batch_size,
            num_workers=self._config.training.num_workers,
            prefetch_factor=prefetch_factor,
        )

    def _save_checkpoint(self, metrics: dict[PhaseType, dict[str, float | int]] | None, export: bool = False):
        assert self._is_setup
        with self._run.get_save_checkpoint_context(self._completed_steps, export) as checkpoint:
            metadata = {
                "optimizer": self._optimizer.save(),
                "completed_steps": self._completed_steps,
            }
            if metrics is not None:
                metadata["metrics"] = {key.value: value for key, value in metrics.items()}
            self._multi_stage.save_checkpoint(
                CheckpointConfig(checkpoint_type=CheckpointType.distributed, checkpoint_path=checkpoint.directory),
                metadata,
            )
        if export and self._run.is_main_rank and self._config.training.export_callback_script:  # noqa
            custom_env = os.environ.copy()
            if self._config.training.export_callback_env:
                custom_env.update(json.loads(self._config.training.export_callback_env))
            subprocess.Popen(shlex.split(self._config.training.export_callback_script), env=custom_env)

    @abc.abstractmethod
    def get_tflops(self, phase: PhaseType, elapsed_time_per_iteration) -> tuple[int, int]:
        # TODO: Do in model, automate/generalize, get other stats
        pass
