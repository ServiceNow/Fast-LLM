import abc
import dataclasses
import logging
import time
import typing

from fast_llm.config import Configurable
from fast_llm.core.distributed import safe_barrier
from fast_llm.data.batch.config import PreprocessedBatch
from fast_llm.data.data.abstract import Data
from fast_llm.engine.base_model.config import LossDef
from fast_llm.engine.config_utils.run import get_run, log_main_rank, run_exists
from fast_llm.engine.distributed.config import PhaseType
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.evaluation.config import EvaluatorConfig, LossEvaluatorConfig
from fast_llm.engine.multi_stage.fast_llm_model import FastLLMModel
from fast_llm.engine.schedule.config import BatchConfig
from fast_llm.engine.schedule.runner import ScheduleRunner
from fast_llm.engine.schedule.schedule import Schedule
from fast_llm.logging import format_metrics
from fast_llm.utils import get_and_reset_memory_usage_mib

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TrainingProgress:
    completed_steps: int
    consumed_samples: int
    consumed_tokens: int


class Evaluator[ConfigType: EvaluatorConfig](Configurable[ConfigType], abc.ABC):
    _is_setup: bool = False
    _multi_stage: FastLLMModel
    _runner: ScheduleRunner
    _data: Data
    _distributed: Distributed

    def __init__(
        self,
        name: str,
        eval_config: LossEvaluatorConfig,
        batch_config: BatchConfig,
        num_workers: int,
    ):
        super().__init__(eval_config)
        self._name = name
        self._batch_config = batch_config
        self._num_workers = num_workers

    @abc.abstractmethod
    def setup(
        self,
        multi_stage: FastLLMModel,
        runner: ScheduleRunner,
        data: Data,
        run_count: int,
    ) -> None:
        self._runner = runner
        self._multi_stage = multi_stage
        self._distributed = multi_stage.distributed
        self._data = data
        self._is_setup = True

    @abc.abstractmethod
    def run(
        self,
        run_index: int | None,
        metrics: dict[str, typing.Any],
    ) -> None:
        pass


class LossEvaluator[ConfigType: LossEvaluatorConfig](Evaluator[ConfigType]):
    _data_iterator: typing.Iterator[PreprocessedBatch] | None = None
    _loss_definitions: list[LossDef]
    _schedule: Schedule
    _data: Data

    def setup(
        self,
        multi_stage: FastLLMModel,
        runner: ScheduleRunner,
        data: Data,
        run_count: int,
    ) -> None:
        super().setup(multi_stage, runner, data, run_count)

        preprocessing_config = self._multi_stage.get_preprocessing_config(self._batch_config, PhaseType.validation)
        self._data.sample_dataset(
            self._name, preprocessing_config, run_count * self._config.iterations * self._batch_config.batch_size
        )
        # Setup the schedule
        self._schedule = Schedule(
            config=runner.config,
            multi_stage=self._multi_stage,
            batch_meta=preprocessing_config.get_batch_meta(),
            distributed_config=self._distributed.config,
            phase=PhaseType.validation,
        )
        self._loss_definitions = self._multi_stage.base_model.get_loss_definitions()
        self._data_iterator = None

    def run(
        self,
        run_index: int,
        metrics: dict[str, typing.Any],
    ) -> None:
        assert self._is_setup
        completed_evaluation_steps = max(0, run_index - 1) * self.config.iterations

        if self._data_iterator is None:
            self._data.get_iterator(
                self._batch_config,
                self._name,
                consumed_samples=completed_evaluation_steps * self._batch_config.batch_size,
                num_workers=self._num_workers,
            )
        safe_barrier(self._distributed.world_group, f"{PhaseType.validation} {self._name} begin")
        begin_time = time.perf_counter()
        total_losses = {loss_def.name: 0.0 for loss_def in self._loss_definitions}
        for iter_ in range(self._config.iterations):
            iter_losses, _, _ = self._runner.run_step(
                self._data_iterator, self._schedule, iteration=completed_evaluation_steps + iter_
            )
            for name, value in iter_losses.items():
                total_losses[name] += value

            if run_exists():
                get_run().save_logged_tensors(
                    f"{PhaseType.validation}_{self._name}_{metrics.get("completed_steps",run_index)}"
                )

        safe_barrier(
            self._distributed.world_group,
            f"{PhaseType.validation} {self._name}  end",
        )
        time_per_iteration = (time.perf_counter() - begin_time) / self._config.iterations

        metrics.update(
            {
                "batch_size": self._batch_config.batch_size,
                **{name: (value / self._config.iterations) for name, value in total_losses.items()},
                "step_time_ms": time_per_iteration * 1000,
                **self._schedule.get_compute_metrics(time_per_iteration),
                "tokens_per_sec_per_gpu": (
                    (self._batch_config.sequence_length * self._batch_config.batch_size)
                    / self._distributed.config.world_size
                    / time_per_iteration
                ),
                **get_and_reset_memory_usage_mib(),
            }
        )

        log_main_rank(
            "\n".join(
                format_metrics(
                    metrics,
                    self._loss_definitions,
                    PhaseType.validation,
                    dataset_name=self._name,
                )
            )
        )
