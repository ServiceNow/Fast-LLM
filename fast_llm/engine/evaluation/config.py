import abc
import typing

from fast_llm.config import Config, Field, FieldHint, check_field, config_class, skip_valid_if_none
from fast_llm.engine.schedule.config import BatchConfig
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    from fast_llm.engine.evaluation.evaluator import Evaluator, EvaluatorLmEval, LossEvaluator


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
    def _from_dict(cls, default: dict[str, typing.Any], strict: bool = True) -> typing.Self:
        if cls is EvaluatorConfig and cls.get_subclass(default.get("type")) is None:
            # Default subclass.
            return LossEvaluatorConfig._from_dict(default, strict)
        return super()._from_dict(default, strict=strict)


@config_class(dynamic_type={EvaluatorConfig: "loss"})
class LossEvaluatorConfig(EvaluatorConfig):
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
    ) -> "LossEvaluator":
        from fast_llm.engine.evaluation.evaluator import LossEvaluator

        return LossEvaluator(name, self, batch_config, data_load_num_proc, train_iters)


@config_class(dynamic_type={EvaluatorConfig: "lm_eval"})
class LmEvalEvaluatorConfig(EvaluatorConfig):
    _abstract: typing.ClassVar[bool] = False

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

    max_length: int | None = Field(
        default=None,
        desc="Maximum sequence length including both prompt and newly generated tokens."
        " If not set, it is inferred from the Fast-LLM model config or tokenizer.",
    )

    def get_evaluator(
        self,
        name: str,
        batch_config: BatchConfig,
        data_load_num_proc: int,
        train_iters: int | None = None,
    ) -> "EvaluatorLmEval":
        from fast_llm.engine.evaluation.lm_eval.evaluator import LmEvalEvaluator

        return LmEvalEvaluator(name, self, batch_config, data_load_num_proc, train_iters)
