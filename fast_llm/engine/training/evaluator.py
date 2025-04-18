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
from fast_llm.engine.config_utils.run import Run, is_main_rank, log_main_rank, log_pipeline_parallel_main_rank
from fast_llm.engine.distributed.config import PhaseType
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.multi_stage.fast_llm_model import FastLLMModel

from fast_llm.engine.schedule.runner import ScheduleRunner
from fast_llm.engine.schedule.schedule import Schedule
from fast_llm.engine.training.config import (
    TrainerConfig,
    EvaluationConfig,
    EvaluationLossConfig,
    # EvaluationHarnessConfig,
)
from fast_llm.engine.training.wandb import Wandb
from fast_llm.logging import format_metrics, get_memory_usage_mib, log_memory_usage
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)


class Evaluation[ConfigType: EvaluationConfig](Configurable[ConfigType], abc.ABC):
    config_class: typing.ClassVar[type[EvaluationConfig]] = EvaluationConfig

    @classmethod
    def build(
        cls,
        name: str,
        eval_config: EvaluationLossConfig,
        trainer_config: TrainerConfig,
        distributed: Distributed,
        run: Run,
        multi_stage: FastLLMModel,
        runner: ScheduleRunner,
        data: Data,
        get_tflops_func: callable,
    ) -> "Evaluation":
        return cls(
            name=name,
            eval_config=eval_config,
            trainer_config=trainer_config,
            distributed=distributed,
            run=run,
            multi_stage=multi_stage,
            runner=runner,
            data=data,
            get_tflops_func=get_tflops_func,
        )

    @abc.abstractmethod
    def run(
        self,
        done: bool,
        completed_steps: int,
        consumed_samples: int,
        consumed_tokens: int,
    ) -> tuple[dict[str, any], str | None]: ...

    @abc.abstractmethod
    def get_dataset_samples(self) -> tuple[str, int] | None:
        """
        Returns the name and number of required samples in a dataset,
        or None if the evaluation does not rely on Fast-LLM data or
        if the evaluation is skipped for this run.
        """


class EvaluationLoss[ConfigType: EvaluationLossConfig](Evaluation[ConfigType]):
    config_class: typing.ClassVar[type[EvaluationLossConfig]] = EvaluationLossConfig

    def __init__(
        self,
        name: str,
        eval_config: EvaluationLossConfig,
        trainer_config: TrainerConfig,
        distributed: Distributed,
        run: Run,
        multi_stage: FastLLMModel,
        runner: ScheduleRunner,
        data: Data,
        get_tflops_func: callable,
    ):
        self._name = name
        self._eval_config = eval_config
        self._trainer_config = trainer_config
        self._distributed = distributed
        self._run = run
        self._runner = runner
        self._multi_stage = multi_stage
        self._data = data
        self._get_tflops_func = get_tflops_func

        self._loss_defs = self._multi_stage.base_model.loss_defs

        steps = self._eval_config.get_iteration_count(
            self._trainer_config.training.train_iters,
            # There may be an extra evaluation after the last training step.
            not self._eval_config.enabled(self._trainer_config.training.train_iters),
        )

        self._samples = self._trainer_config.batch.batch_size * steps if steps > 0 else None

        # Setup the schedule
        self._schedule = Schedule(
            multi_stage=self._multi_stage,
            batch_config=self._trainer_config.batch,
            schedule_config=self._trainer_config.schedule,
            distributed_config=self._trainer_config.model.distributed,
            phase=PhaseType.validation,
        )

        self._evaluation_iterator = None

    def get_dataset_samples(self) -> tuple[str, int] | None:
        if self._samples is None:
            return None
        return self._name, self._samples

    def run(
        self,
        done: bool,
        completed_steps: int,
        consumed_samples: int,
        consumed_tokens: int,
    ) -> tuple[dict[str, any], str | None]:
        metrics = {}
        formatted_metrics = None
        if self._samples is not None and (done or self._eval_config.enabled(completed_steps)):

            if self._evaluation_iterator is None:
                self._evaluation_iterator = self._get_data_iterator(
                    self._get_completed_evaluation_steps(completed_steps)
                )
            # TODO: formatting metric category as Validation.evaluation_dataset_name
            #       maybe format each metric with evaluation_dataset_name prefix instead?
            # TODO: setting performance metrics per evaluation dataset
            #       maybe to set aggregate performance metrics for all evaluations datasets?
            metric_key = f"{PhaseType.validation.value}.{self._name}"
            metrics[metric_key] = self._evaluate_loss(
                data_iterator=self._evaluation_iterator,
                phase=PhaseType.validation,
                num_iters=self._eval_config.iterations,
                begin_iter=self._get_completed_evaluation_steps(completed_steps),
                completed_steps=completed_steps,
                consumed_samples=consumed_samples,
                consumed_tokens=consumed_tokens,
            )
            formatted_metrics = format_metrics(
                metrics[metric_key],
                self._loss_defs,
                PhaseType.validation,
                dataset_name=self._name,
            )

        return metrics, formatted_metrics

    def _evaluate_loss(
        self,
        *,
        data_iterator: typing.Iterator,
        phase: PhaseType,
        num_iters: int,
        completed_steps: int,
        consumed_samples: int,
        consumed_tokens: int,
        begin_iter: int = 0,
    ) -> dict[str, float | int]:
        full_phase_name = f"{phase.value}_{self._name}"
        safe_barrier(self._distributed.world_group, f"{full_phase_name} begin")
        begin_time = time.perf_counter()
        total_losses = {loss_def.name: 0.0 for loss_def in self._loss_defs}
        for iter_ in range(num_iters):
            iter_losses, _, _ = self._runner.run_step(data_iterator, self._schedule, iteration=begin_iter + iter_)
            for name, value in iter_losses.items():
                total_losses[name] += value
            self._run.save_logged_tensors(f"{full_phase_name}_{completed_steps}_{iter_}")

        safe_barrier(
            self._distributed.world_group,
            f"{full_phase_name} end",
        )
        end_time = time.perf_counter()
        time_per_iteration = (end_time - begin_time) / num_iters
        model_tflops, hardware_tflops = self._get_tflops_func(phase, time_per_iteration)
        # TODO add other relevant eval metrics
        metrics = {
            "train_iters": self._trainer_config.training.train_iters,
            "batch_size": self._trainer_config.batch.batch_size,
            "iteration": completed_steps,
            **{name: (value / num_iters) for name, value in total_losses.items()},
            "consumed_samples": consumed_samples,
            "consumed_tokens": consumed_tokens,
            "step_time_ms": time_per_iteration * 1000,
            "model_tflops": model_tflops,
            "hardware_tflops": hardware_tflops,
            "tokens_per_sec_per_gpu": (
                (self._trainer_config.batch.sequence_length * self._trainer_config.batch.batch_size)
                / self._trainer_config.model.distributed.world_size
                / time_per_iteration
            ),
            **get_memory_usage_mib(),
        }

        return metrics

    def _get_completed_evaluation_steps(self, completed_steps: int) -> int:
        # Number of evaluations steps performed before the current step
        return self._eval_config.get_iteration_count(completed_steps - 1)

    def _get_data_iterator(
        self, completed_steps: int = 0, prefetch_factor: int | None = None
    ) -> typing.Iterator[typing.Any]:
        return self._data.get_iterator(
            self._trainer_config.batch,
            self._name,
            consumed_samples=completed_steps * self._trainer_config.batch.batch_size,
            num_workers=self._trainer_config.training.num_workers,
            prefetch_factor=prefetch_factor,
        )


# class EvaluationHarness[ConfigType: EvaluationHarnessConfig](Evaluation[ConfigType]):
#     config_class: typing.ClassVar[type[EvaluationHarnessConfig]] = EvaluationHarnessConfig

#     @abc.abstractmethod
#     def run(
#         self,
#     ) -> None: ...


# NOTE: This is not a standalone runnable; it's a submodule of Trainer used for code encapsulation.
class Evaluator:
    def __init__(
        self,
        config: TrainerConfig,
        distributed: Distributed,
        run: Run,
        multi_stage: FastLLMModel,
        runner: ScheduleRunner,
        data: Data,
        get_tflops_func: callable,
        wandb: Wandb,
    ):
        self._config = config
        self._wandb = wandb

        self._evaluations = [
            eval_config.get_evaluation_class().build(
                name=name,
                eval_config=eval_config,
                trainer_config=config,
                distributed=distributed,
                run=run,
                multi_stage=multi_stage,
                runner=runner,
                data=data,
                get_tflops_func=get_tflops_func,
            )
            for name, eval_config in config.training.evaluations.items()
        ]

    def get_datasets_samples(self) -> dict[str:int]:
        return {
            el[0]: el[1]
            for el in (evaluation.get_dataset_samples() for evaluation in self._evaluations)
            if el is not None
        }

    def run(
        self,
        metrics: dict[str:any],
        done: bool,
        completed_steps: int,
        consumed_samples: int,
        consumed_tokens: int,
    ):
        formatted_metrics = []
        for evaluation in self._evaluations:
            this_metrics, this_formatted_metrics = evaluation.run(
                done, completed_steps, consumed_samples, consumed_tokens
            )
            if len(this_metrics) == 0:
                continue
            for k, v in this_metrics.items():
                metrics[k] = v
            if this_formatted_metrics is not None:
                formatted_metrics.append(this_formatted_metrics)

        if len(formatted_metrics) > 0:
            formatted_metrics = "\n".join(formatted_metrics)
            log_main_rank(formatted_metrics)
            if self._config.training.wandb.alert.enabled(completed_steps):
                self._wandb.alert("Validation results", formatted_metrics, "INFO")
