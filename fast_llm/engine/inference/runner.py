import abc
import typing

from fast_llm.config import NoAutoValidate
from fast_llm.engine.distributed.config import PhaseType
from fast_llm.engine.multi_stage.fast_llm_model import FastLLMModel
from fast_llm.engine.schedule.config import BatchConfig, ScheduleConfig
from fast_llm.engine.schedule.runner import ScheduleRunner
from fast_llm.engine.schedule.schedule import Schedule


class InferenceRunner(abc.ABC):
    model_class: typing.ClassVar[type[FastLLMModel]] = FastLLMModel

    def __init__(self, fast_llm_model: FastLLMModel):
        assert isinstance(fast_llm_model, self.model_class)
        self._fast_llm_model = fast_llm_model
        # We only need a basic schedule and don't care about dimensions.
        self._schedule_config = ScheduleConfig()
        # TODO: Sort things out.
        with NoAutoValidate():
            self._batch_config = BatchConfig()
        self._batch_config.setup(self._fast_llm_model.config.distributed)
        self._batch_config.validate()
        self._runner = ScheduleRunner(
            config=self._schedule_config,
            multi_stage=self._fast_llm_model,
            distributed_config=self._fast_llm_model.config.distributed,
        )
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
        self._runner.setup(self._fast_llm_model.distributed)

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
