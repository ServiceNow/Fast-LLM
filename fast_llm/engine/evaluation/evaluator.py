import abc
import dataclasses
import logging
import time
import typing

from fast_llm.config import Configurable
from fast_llm.core.distributed import safe_barrier
from fast_llm.data.data.abstract import Data
from fast_llm.engine.config_utils.run import Run, log_main_rank
from fast_llm.engine.distributed.config import PhaseType
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.evaluation.config import EvaluatorConfig, EvaluatorConfigBase, LossEvaluatorConfig
from fast_llm.engine.multi_stage.fast_llm_model import FastLLMModel
from fast_llm.engine.schedule.config import BatchConfig
from fast_llm.engine.schedule.runner import ScheduleRunner
from fast_llm.engine.schedule.schedule import Schedule
from fast_llm.engine.training.config import WandbConfig
from fast_llm.engine.training.wandb import Wandb
from fast_llm.logging import format_metrics
from fast_llm.utils import get_and_reset_memory_usage_mib

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TrainingProgress:
    done: bool
    completed_steps: int
    consumed_samples: int
    consumed_tokens: int


@dataclasses.dataclass
class EvaluationMetrics:
    metrics: dict[str, any] = dataclasses.field(default_factory=dict)
    formatted_metrics: str | None = None


@dataclasses.dataclass
class EvaluatorSamplingParameters:
    dataset_name: str
    num_samples: int


class Evaluator[ConfigType: EvaluatorConfig](Configurable[ConfigType], abc.ABC):
    _is_setup: bool = False

    def __init__(
        self,
        name: str,
        eval_config: LossEvaluatorConfig,
        batch_config: BatchConfig,
        data_load_num_proc: int,
        train_iters: int | None = None,
    ):
        super().__init__(eval_config)
        self._name = name
        self._batch_config = batch_config
        self._data_load_num_proc = data_load_num_proc
        self._train_iters = train_iters

    def setup(
        self,
        distributed: Distributed,
        run: Run,
        multi_stage: FastLLMModel,
        runner: ScheduleRunner,
        data: Data,
        phase: PhaseType,
    ) -> None:
        # TODO: check if objects passed are actually set up themselves, if appropriate
        self._distributed = distributed
        self._run = run
        self._runner = runner
        self._multi_stage = multi_stage
        self._data = data
        self._phase = phase

    @abc.abstractmethod
    def run(
        self,
        training_progress: TrainingProgress | None = None,
        run_index: int | None = None,
    ) -> EvaluationMetrics: ...

    @abc.abstractmethod
    def get_sampling_parameters(self) -> EvaluatorSamplingParameters | None:
        """
        Returns the name and number of required samples in a dataset,
        or None if the evaluation does not rely on Fast-LLM data or
        if the evaluation is skipped for this run.
        """


class LossEvaluator[ConfigType: LossEvaluatorConfig](Evaluator[ConfigType]):
    def setup(
        self,
        distributed: Distributed,
        run: Run,
        multi_stage: FastLLMModel,
        runner: ScheduleRunner,
        data: Data,
        phase: PhaseType,
    ) -> None:
        super().setup(distributed, run, multi_stage, runner, data, phase)

        # Setup the schedule
        self._schedule = Schedule(
            multi_stage=self._multi_stage,
            batch_config=self._batch_config,
            schedule_config=runner.config,
            distributed_config=distributed.config,
            phase=PhaseType.validation,
        )

        self._loss_defs = self._multi_stage.base_model.loss_defs
        self._evaluation_iterator = None
        self._is_setup = True

    def get_sampling_parameters(self) -> EvaluatorSamplingParameters | None:
        return (
            None
            if self._config.iterations is None
            else EvaluatorSamplingParameters(
                (self._name if self._config.dataset_name is None else self._config.dataset_name),
                self._config.iterations * self._batch_config.batch_size,
            )
        )

    def run(
        self,
        training_progress: TrainingProgress | None = None,
        run_index: int | None = None,
    ) -> EvaluationMetrics:
        assert self._is_setup
        if run_index is None:
            run_index = 0

        metrics = {}

        if self._evaluation_iterator is None:
            self._evaluation_iterator = self._get_data_iterator(self._get_completed_evaluation_steps(run_index))
        # TODO: formatting metric category as Validation.evaluation_dataset_name
        #       maybe format each metric with evaluation_dataset_name prefix instead?
        # TODO: setting performance metrics per evaluation dataset
        #       maybe to set aggregate performance metrics for all evaluations datasets?
        phase = PhaseType.validation
        metric_key = f"{phase.value}.{self._name}"
        metrics[metric_key] = self._evaluate_loss(
            data_iterator=self._evaluation_iterator,
            phase=phase,
            num_iters=self._config.iterations,
            begin_iter=self._get_completed_evaluation_steps(run_index),
            completed_steps=None if training_progress is None else training_progress.completed_steps,
        )

        if self._train_iters is not None:
            metrics[metric_key]["train_iters"] = self._train_iters

        if training_progress is not None:
            metrics[metric_key]["iteration"] = training_progress.completed_steps
            metrics[metric_key]["consumed_samples"] = training_progress.consumed_samples
            metrics[metric_key]["consumed_tokens"] = training_progress.consumed_tokens

        formatted_metrics = format_metrics(
            metrics[metric_key],
            self._loss_defs,
            phase,
            dataset_name=self._name,
        )

        return EvaluationMetrics(metrics, formatted_metrics)

    def _evaluate_loss(
        self,
        *,
        data_iterator: typing.Iterator,
        phase: PhaseType,
        num_iters: int,
        completed_steps: int | None,
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

            tensor_save_name = (
                f"{full_phase_name}_{iter_}"
                if completed_steps is None
                else f"{full_phase_name}_{completed_steps}_{iter_}"
            )
            self._run.save_logged_tensors(tensor_save_name)

        safe_barrier(
            self._distributed.world_group,
            f"{full_phase_name} end",
        )
        end_time = time.perf_counter()
        time_per_iteration = (end_time - begin_time) / num_iters
        model_tflops, hardware_tflops = self._multi_stage.get_tflops(
            phase,
            time_per_iteration,
            self._batch_config.batch_size,
            self._batch_config.sequence_length,
        )
        # TODO add other relevant eval metrics
        metrics = {
            "batch_size": self._batch_config.batch_size,
            **{name: (value / num_iters) for name, value in total_losses.items()},
            "step_time_ms": time_per_iteration * 1000,
            "model_tflops": model_tflops,
            "hardware_tflops": hardware_tflops,
            "tokens_per_sec_per_gpu": (
                (self._batch_config.sequence_length * self._batch_config.batch_size)
                / self._schedule._distributed.world_size
                / time_per_iteration
            ),
            **get_and_reset_memory_usage_mib(),
        }
        return metrics

    def _get_completed_evaluation_steps(self, run_index: int) -> int:
        # Number of evaluations steps performed before the current step
        return max(0, run_index - 1) * self.config.iterations

    def _get_data_iterator(
        self, completed_steps: int = 0, prefetch_factor: int | None = None
    ) -> typing.Iterator[typing.Any]:
        return self._data.get_iterator(
            self._batch_config,
            self._name,
            consumed_samples=completed_steps * self._batch_config.batch_size,
            num_workers=self._data_load_num_proc,
            prefetch_factor=prefetch_factor,
        )


# NOTE: This is not a standalone runnable; it's a submodule of Trainer used for code encapsulation.
class EvaluatorRunner:
    _is_setup: bool = False

    def __init__(
        self,
        evaluator_configs: dict[str, EvaluatorConfigBase],
        batch_config: BatchConfig,
        data_load_num_proc: int,
        train_iters: int | None = None,
        wandb_config: WandbConfig | None = None,
    ):
        self._wandb_config = wandb_config
        self._evaluators = [
            eval_config.get_evaluator(name, batch_config, data_load_num_proc, train_iters)
            for name, eval_config in evaluator_configs.items()
        ]

    def setup(
        self,
        distributed: Distributed,
        run: Run,
        multi_stage: FastLLMModel,
        runner: ScheduleRunner,
        data: Data,
        wandb: Wandb,
        phase: PhaseType,
    ) -> None:
        self._wandb = wandb
        for evaluator in self._evaluators:
            evaluator.setup(distributed, run, multi_stage, runner, data, phase)
        self._is_setup = True

    def get_sampling_parameters(self) -> list[EvaluatorSamplingParameters]:
        return [
            sampling_params
            for sampling_params in (evaluator.get_sampling_parameters() for evaluator in self._evaluators)
            if sampling_params is not None
        ]

    def run(
        self,
        metrics: dict[str:any],
        training_progress: TrainingProgress | None = None,
    ):
        assert self._is_setup
        formatted_metrics = []
        for evaluator in self._evaluators:
            evaluation_metrics = evaluator.run(training_progress)
            if len(evaluation_metrics.metrics) == 0:
                continue
            for k, v in evaluation_metrics.metrics.items():
                metrics[k] = v
            if evaluation_metrics.formatted_metrics is not None:
                formatted_metrics.append(evaluation_metrics.formatted_metrics)

        if len(formatted_metrics) > 0:
            formatted_metrics = "\n".join(formatted_metrics)
            log_main_rank(formatted_metrics)
            if self._wandb_config is not None and self._wandb_config.alert.enabled(
                0 if training_progress is None else training_progress.completed_steps
            ):
                self._wandb.alert("Validation results", formatted_metrics, "INFO")
