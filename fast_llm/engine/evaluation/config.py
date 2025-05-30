import abc
import typing

from fast_llm.config import Config, Field, FieldHint, check_field, config_class, skip_valid_if_none
from fast_llm.engine.schedule.config import BatchConfig
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    from fast_llm.engine.evaluation.evaluator import Evaluator, EvaluatorLoss


@config_class()
class EvaluatorConfigBase(Config):
    @abc.abstractmethod
    def get_evaluator(
        self,
        name: str,
        batch_config: BatchConfig,
        data_load_num_proc: int,
        train_iters: int | None = None,
    ) -> "Evaluator":
        pass


@config_class(registry=True)
class EvaluatorConfig(EvaluatorConfigBase):
    _abstract: typing.ClassVar[bool] = True

    @classmethod
    def _from_dict(
        cls,
        default: dict[str, typing.Any],
        strict: bool = True,
        flat: bool = False,
    ) -> typing.Self:
        # TODO v0.x: Remove backward compatibility.
        if not "type" in default:
            default["type"] = "loss"
        return super()._from_dict(default, strict, flat)


@config_class(dynamic_type={EvaluatorConfig: "loss"})
class EvaluatorLossConfig(EvaluatorConfig):
    _abstract: typing.ClassVar[bool] = False

    iterations: int | None = Field(
        default=None,
        desc="Number of iterations for each evaluation phase. Setting to None will disable.",
        hint=FieldHint.feature,
        valid=skip_valid_if_none(check_field(Assert.gt, 0)),
    )

    dataset_name: str | None = Field(default=None)

    def get_evaluator(
        self,
        name: str,
        batch_config: BatchConfig,
        data_load_num_proc: int,
        train_iters: int | None = None,
    ) -> "EvaluatorLoss":
        from fast_llm.engine.evaluation.evaluator import EvaluatorLoss

        return EvaluatorLoss(name, self, batch_config, data_load_num_proc, train_iters)
