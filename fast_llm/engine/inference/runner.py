import abc
import typing

from fast_llm.config import NoAutoValidate
from fast_llm.engine.distributed.config import PhaseType
from fast_llm.engine.multi_stage.fast_llm_model import FastLLMModel
from fast_llm.engine.schedule.config import BatchConfig, ScheduleConfig
from fast_llm.engine.schedule.runner import ScheduleRunner
from fast_llm.engine.schedule.schedule import Schedule
from fast_llm.utils import Assert


class InferenceRunner(abc.ABC):
    model_class: typing.ClassVar[type[FastLLMModel]] = FastLLMModel
    batch_config_class: typing.ClassVar[type[BatchConfig]] = BatchConfig

    def __init__(
        self,
        fast_llm_model: FastLLMModel,
        runner: ScheduleRunner | None = None,
    ):
        assert isinstance(fast_llm_model, self.model_class)
        self._fast_llm_model = fast_llm_model

        with NoAutoValidate():
            self._batch_config = self.batch_config_class()
        self._batch_config.setup(self._fast_llm_model.config.distributed)
        self._batch_config.validate()

        if runner is None:
            # We only need a basic schedule and don't care about dimensions.
            self._schedule_config = ScheduleConfig()
            # TODO: Sort things out.

            self._runner = ScheduleRunner(
                config=self._schedule_config,
                multi_stage=self._fast_llm_model,
                distributed_config=self._fast_llm_model.config.distributed,
            )
        else:
            self._schedule_config = runner.config
            self._runner = runner
            # External runner from training loop must be already setup
            assert runner._is_setup

        # TODO: Random state? (Distributed.set_step)
        self._schedule = Schedule(
            multi_stage=self._fast_llm_model,
            batch_config=self._batch_config,
            schedule_config=self._schedule_config,
            distributed_config=self._fast_llm_model.config.distributed,
            phase=PhaseType.inference,
        )

    @property
    def fast_llm_model(self) -> FastLLMModel:
        return self._fast_llm_model

    def setup(self):
        if not self._runner._is_setup:
            self._runner.setup(self._fast_llm_model.distributed)
        else:
            # Means external runner was passed, check it has the same distributed class as the model
            Assert.is_(self._runner._distributed, self._fast_llm_model.distributed)

    def forward(
        self, input_, kwargs: dict, *, iteration: int = 1, return_metrics: bool = False
    ) -> tuple[dict[str, float | int], dict[str, typing.Any] | None]:
        # TODO: Return an actual model output.
        reduced_losses, update_successful, metrics = self._runner.run_step(
            iter((((input_, kwargs),),)),
            self._schedule,
            iteration=iteration,
            return_metrics=return_metrics,
            preprocessed=True,
        )
        assert update_successful
        return reduced_losses, metrics
