import abc
import dataclasses
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
    EvaluatorConfig,
    EvaluatorLossConfig,
    EvaluatorLmEvalConfig,
)
from fast_llm.engine.training.wandb import Wandb
from fast_llm.logging import format_metrics, get_memory_usage_mib, log_memory_usage
from fast_llm.utils import Assert
from fast_llm.engine.training.lm_eval.fast_llm_wrapper import FastLLMLmEvalWrapper
from fast_llm.engine.training.lm_eval.utils import prepare_lm_eval_simple_eval_params, process_lm_eval_results

# from fast_llm.engine.training.lm_eval.evaluator import simple_evaluate as lm_eval_simple_evaluate
from lm_eval.evaluator import simple_evaluate as lm_eval_simple_evaluate

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TrainingProgressInfo:
    done: bool
    completed_steps: int
    consumed_samples: int
    consumed_tokens: int


@dataclasses.dataclass
class EvaluationMetrics:
    metrics: dict[str, any] = dataclasses.field(default_factory=dict)
    formatted_metrics: str | None = None


class Evaluator[ConfigType: EvaluatorConfig](Configurable[ConfigType], abc.ABC):
    config_class: typing.ClassVar[type[EvaluatorConfig]] = EvaluatorConfig

    _is_setup: bool = False

    @classmethod
    def build(
        cls,
        name: str,
        eval_config: EvaluatorConfig,
        trainer_config: TrainerConfig,
    ) -> "Evaluator":
        return cls(name=name, eval_config=eval_config, trainer_config=trainer_config)

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
        training_progress_info: TrainingProgressInfo | None = None,
    ) -> EvaluationMetrics: ...

    @abc.abstractmethod
    def get_dataset_samples(self) -> tuple[str, int] | None:
        """
        Returns the name and number of required samples in a dataset,
        or None if the evaluation does not rely on Fast-LLM data or
        if the evaluation is skipped for this run.
        """


class EvaluatorLoss[ConfigType: EvaluatorLossConfig](Evaluator[ConfigType]):
    config_class: typing.ClassVar[type[EvaluatorLossConfig]] = EvaluatorLossConfig

    def __init__(
        self,
        name: str,
        eval_config: EvaluatorLossConfig,
        trainer_config: TrainerConfig,
    ):
        self._name = name
        self._eval_config = eval_config
        self._trainer_config = trainer_config

        steps = self._eval_config.get_iteration_count(
            self._trainer_config.training.train_iters,
            # There may be an extra evaluation after the last training step.
            not self._eval_config.enabled(self._trainer_config.training.train_iters),
        )

        self._samples = self._trainer_config.batch.batch_size * steps if steps > 0 else None

        self._evaluation_iterator = None

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
        self._loss_defs = self._multi_stage.base_model.loss_defs
        # Setup the schedule
        self._schedule = Schedule(
            multi_stage=self._multi_stage,
            batch_config=self._trainer_config.batch,
            schedule_config=self._trainer_config.schedule,
            distributed_config=self._trainer_config.model.distributed,
            phase=PhaseType.inference if self._phase == PhaseType.inference else PhaseType.validation,
        )

        self._is_setup = True

    def get_dataset_samples(self) -> tuple[str, int] | None:
        if self._samples is None:
            return None
        return self._name, self._samples

    def run(
        self,
        training_progress_info: TrainingProgressInfo | None = None,
    ) -> EvaluationMetrics:
        assert self._is_setup

        if training_progress_info is None:
            done = True
            completed_steps = 0
            consumed_samples = 0
            consumed_tokens = 0
        else:
            done = training_progress_info.done
            completed_steps = training_progress_info.completed_steps
            consumed_samples = training_progress_info.consumed_samples
            consumed_tokens = training_progress_info.consumed_tokens

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
            phase = PhaseType.inference if self._phase == PhaseType.inference else PhaseType.validation
            metric_key = f"{phase.value}.{self._name}"
            metrics[metric_key] = self._evaluate_loss(
                data_iterator=self._evaluation_iterator,
                phase=phase,
                num_iters=self._eval_config.iterations,
                begin_iter=self._get_completed_evaluation_steps(completed_steps),
                completed_steps=completed_steps,
                consumed_samples=consumed_samples,
                consumed_tokens=consumed_tokens,
            )
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
        model_tflops, hardware_tflops = self._multi_stage.get_tflops(
            phase,
            time_per_iteration,
            self._trainer_config.batch.batch_size,
            self._trainer_config.batch.sequence_length,
        )
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


class EvaluatorLmEval[ConfigType: EvaluatorLmEvalConfig](Evaluator[ConfigType]):
    config_class: typing.ClassVar[type[EvaluatorLmEvalConfig]] = EvaluatorLmEvalConfig

    def __init__(
        self,
        name: str,
        eval_config: EvaluatorLmEvalConfig,
        trainer_config: TrainerConfig,
    ):
        self._name = name
        self._eval_config = eval_config
        self._trainer_config = trainer_config

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

        # TODO: pass mini and batch size of the same length for lm_eval not to crash during training
        #       or implement min batch sequential awareness in fas_llm_wrapper for lm_eval
        self._hf_model = (
            self._multi_stage.config_class.get_huggingface_model_for_causal_lm_class().from_fast_llm_model_in_training(
                self._multi_stage, self._trainer_config, self._runner
            )
        )

        # For reporting purposes, just to indicate it is from Fast-LLM
        # as lm_eval.simple_evaluate will take it for results['config']['model']
        self._hf_model.config.name_or_path = type(self._hf_model).__name__

        self._flm_wrapper = FastLLMLmEvalWrapper(
            model=self._hf_model,
            tokenizer=self._data.tokenizer.tokenizer,
            truncation=self._eval_config.truncation,
            logits_cache=self._eval_config.logits_cache,
            add_bos_token=self._eval_config.add_bos_token,
            prefix_token_id=self._eval_config.prefix_token_id,
        )
        self._is_setup = True

    def run(
        self,
        training_progress_info: TrainingProgressInfo | None = None,
    ) -> EvaluationMetrics:
        assert self._is_setup

        if training_progress_info is None:
            done = True
            completed_steps = 0
            consumed_samples = 0
            consumed_tokens = 0
        else:
            done = training_progress_info.done
            completed_steps = training_progress_info.completed_steps
            consumed_samples = training_progress_info.consumed_samples
            consumed_tokens = training_progress_info.consumed_tokens

        if not (done or self._eval_config.enabled(completed_steps)):
            return EvaluationMetrics()

        # completed_steps is added to output_path like output_path/runs/run_index/completed_steps/

        if self._run.is_main_rank:
            args, simple_eval_kwargs = prepare_lm_eval_simple_eval_params(
                self._eval_config.cli_args, completed_steps, self._run.index
            )
            simple_eval_kwargs["model"] = self._flm_wrapper

            # Needed for reporting as batch_size is set from args not lm for reporting in evaluate
            simple_eval_kwargs["batch_size"] = self._flm_wrapper.batch_size
            simple_eval_kwargs["max_batch_size"] = self._flm_wrapper.max_batch_size

            # As of lm_eval commit 758c5ed891b1ca48acd8d3a0d309a827215796b7
            # Expected to be a string even if empty and not None in simple_evaluate
            simple_eval_kwargs["model_args"] = ""

            results = lm_eval_simple_evaluate(**simple_eval_kwargs)
            self._flm_wrapper.stop_workers()

            # Evaluation_tracker save expects model to be either string, but if model is passed
            # LM wrapper needs to be deep copyable and json serializable
            simple_eval_kwargs["evaluation_tracker"].general_config_tracker.model_source = (
                self._hf_model.config.name_or_path
            )

            if results is not None:
                process_lm_eval_results(
                    args,
                    results,
                    simple_eval_kwargs["evaluation_tracker"],
                    completed_steps,
                    consumed_samples,
                    consumed_tokens,
                )
        else:
            self._flm_wrapper.worker_model_invoke()

        # TODO: do we need it here as self._flm_wrapper.stop_workers() and self._flm_wrapper.worker_model_invoke()
        #       already have barrier
        safe_barrier(self._distributed.world_group, f"Evaluation Harness Run end")

        # lm_eval logs to disc, wandb and prints to screen itself
        return EvaluationMetrics()

    def get_dataset_samples(self) -> tuple[str, int] | None:
        return None


# NOTE: This is not a standalone runnable; it's a submodule of Trainer used for code encapsulation.
class EvaluatorRunner:
    _is_setup: bool = False

    def __init__(
        self,
        config: TrainerConfig,
    ):
        self._config = config
        self._evaluations = [
            eval_config.get_evaluator_class().build(
                name=name,
                eval_config=eval_config,
                trainer_config=config,
            )
            for name, eval_config in config.training.evaluations.items()
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
        for evaluation in self._evaluations:
            evaluation.setup(distributed, run, multi_stage, runner, data, phase)
        self._is_setup = True

    def get_datasets_samples(self) -> dict[str:int]:
        return {
            el[0]: el[1]
            for el in (evaluation.get_dataset_samples() for evaluation in self._evaluations)
            if el is not None
        }

    def run(
        self,
        metrics: dict[str:any],
        training_progress_info: TrainingProgressInfo | None = None,
    ):
        assert self._is_setup
        formatted_metrics = []
        for evaluation in self._evaluations:
            evaluation_metrics = evaluation.run(training_progress_info)
            if len(evaluation_metrics.metrics) == 0:
                continue
            for k, v in evaluation_metrics.metrics.items():
                metrics[k] = v
            if evaluation_metrics.formatted_metrics is not None:
                formatted_metrics.append(evaluation_metrics.formatted_metrics)

        if len(formatted_metrics) > 0:
            formatted_metrics = "\n".join(formatted_metrics)
            log_main_rank(formatted_metrics)
            if self._config.training.wandb.alert.enabled(
                0 if training_progress_info is None else training_progress_info.completed_steps
            ):
                self._wandb.alert("Validation results", formatted_metrics, "INFO")
