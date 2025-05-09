import abc
import typing

from fast_llm.config import (
    Config,
    Field,
    FieldHint,
    check_field,
    config_class,
    skip_valid_if_none,
)
from fast_llm.engine.schedule.config import BatchConfig
from fast_llm.utils import Assert, Registry

if typing.TYPE_CHECKING:
    from fast_llm.engine.evaluation.evaluator import Evaluator, EvaluatorLoss, EvaluatorLmEval, TrainingEvaluator

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


@config_class()
class EvaluatorConfig(EvaluatorConfigBase):
    _abstract: typing.ClassVar[bool] = True
    # TODO: Generalize dynamic types?
    _registry: typing.ClassVar[Registry[str, type["EvaluatorConfig"]]] = Registry[str, type["EvaluationConfig"]](
        "evaluation_class", {}
    )
    type_: typing.ClassVar[str | None] = None
    type: str | None = Field(
        default=None,
        desc="The type of evaluation.",
        hint=FieldHint.core,
    )

    def _validate(self) -> None:
        if self.type is None:
            self.type = self.type_
        # Should be handled in `from_dict`, but can fail if instantiating directly.
        Assert.eq(self.type, self.__class__.type_)
        super()._validate()

    @classmethod
    def _from_dict(
        cls,
        default: dict[str, typing.Any],
        strict: bool = True,
        flat: bool = False,
    ) -> typing.Self:
        type_ = default.get("type")
        if type_ is None:
            # TODO: Remove in version 0.* — this is for backward compatibility.
            #       If 'type' is not provided, it falls back to 'loss'.
            type_ = "loss"
            default["type"] = type_
            actual_cls = EvaluatorLossConfig
            # actual_cls = cls
        else:
            if type_ not in cls._registry:
                raise ValueError(
                    f"Unknown {cls._registry.name} type {type_}." f" Available types: {list(cls._registry.keys())}"
                )
            actual_cls = cls._registry[type_]
            Assert.custom(issubclass, actual_cls, cls)
        if actual_cls == cls:
            return super()._from_dict(default, strict=strict, flat=flat)
        else:
            return actual_cls._from_dict(default, strict=strict, flat=flat)

    def __init_subclass__(cls) -> None:
        if cls._abstract and cls.type_ is not None:
            # Abstract classes should not have a `type_`
            raise ValueError(f"Abstract class {cls.__name__} has type = {cls.type_}, expected None.")
        if cls.type_ is not None:
            if cls.type_ in cls._registry:
                raise ValueError(
                    f"Registry {cls._registry.name} already contains type {cls.type_}."
                    f" Make sure all classes either have a unique or `None` type."
                )
            EvaluatorConfig._registry[cls.type_] = cls
        super().__init_subclass__()



@config_class()
class EvaluatorLossConfig(EvaluatorConfig):
    _abstract: typing.ClassVar[bool] = False
    type_: typing.ClassVar[str | None] = "loss"

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


@config_class()
class EvaluatorLmEvalConfig(EvaluatorConfig):
    _abstract: typing.ClassVar[bool] = False
    type_: typing.ClassVar[str | None] = "lm_eval"

    cli_args: list[str] = Field(
        default_factory=lambda: [],
        desc="lm_eval CLI arguments, excluding those related to model, wandb, batch sizes, and device.",
    )

    truncation: bool = Field(
        default=False,
        desc="Whether to use truncation during tokenization (useful when inputs exceed model's max length);"
        " passed to the Fast-LLM lm_eval model wrapper.",
    )

    logits_cache: bool = Field(
        default=True,
        desc="Whether to enable logits caching for speedup and avoiding recomputation during repeated evaluations;"
        " passed to the Fast-LLM lm_eval model wrapper.",
    )

    add_bos_token: bool = Field(
        default=False,
        desc="Whether to prepend a beginning-of-sequence (BOS) token, required for some models like LLaMA;"
        " passed to the Fast-LLM lm_eval model wrapper.",
    )

    prefix_token_id: int | None = Field(
        default=None,
        desc="Token ID to use as a prefix to the input (e.g., for control codes or prompts);"
        " passed to the Fast-LLM lm_eval model wrapper.",
    )

    def get_evaluator(
        self,
        name: str,
        batch_config: BatchConfig,
        data_load_num_proc: int,
        train_iters: int | None = None,
    ) -> "EvaluatorLmEval":
        from fast_llm.engine.evaluation.evaluator import EvaluatorLmEval

        return EvaluatorLmEval(name, self, batch_config, data_load_num_proc, train_iters)
