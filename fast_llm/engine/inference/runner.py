import abc
import typing

from fast_llm.config import NoAutoValidate
from fast_llm.engine.distributed.config import PhaseType
from fast_llm.engine.multi_stage.fast_llm_model import FastLLMModel
from fast_llm.engine.schedule.config import BatchConfig, ScheduleConfig
from fast_llm.engine.schedule.runner import ScheduleRunner
from fast_llm.engine.schedule.schedule import Schedule
from fast_llm.engine.training.config import TrainerConfig


class InferenceRunner(abc.ABC):
    model_class: typing.ClassVar[type[FastLLMModel]] = FastLLMModel
    batch_config_class: typing.ClassVar[type[BatchConfig]] = BatchConfig

    def __init__(
        self,
        fast_llm_model: FastLLMModel,
        trainer_config: TrainerConfig | None = None,
        runner: ScheduleRunner | None = None,
    ):
        has_training_args = trainer_config is not None and runner is not None
        has_partial_args = (trainer_config is None) != (runner is None)
        if has_partial_args:
            raise ValueError("Both trainer_config and runner must be provided together or not at all.")
        
        assert isinstance(fast_llm_model, self.model_class)
        self._fast_llm_model = fast_llm_model
        if False:
        #if has_training_args:
            self._trainer_config = trainer_config
            self._schedule_config = self._trainer_config.schedule
            self._batch_config = self._trainer_config.batch
            self._runner = runner
            # External runner from training loop must be already setup
            assert runner._is_setup
        else:
            # We only need a basic schedule and don't care about dimensions.
            self._schedule_config = ScheduleConfig()
            # TODO: Sort things out.
            with NoAutoValidate():
                self._batch_config = self.batch_config_class()
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
        if not self._runner._is_setup:
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
