import typing

from fast_llm.config import Field, FieldHint, check_field, config_class, skip_valid_if_none
from fast_llm.data.preprocessing.tokenizer import TokenizerConfig
from fast_llm.engine.config_utils.interval import IntervalConfig
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    from fast_llm.engine.evaluation.evaluator import Evaluator, LossEvaluator
    from fast_llm.engine.evaluation.lm_eval.evaluator import LmEvalEvaluator


@config_class(registry=True)
class EvaluatorConfig(IntervalConfig):
    _abstract: typing.ClassVar[bool] = True

    @classmethod
    def _from_dict(cls, default: dict[str, typing.Any], strict: bool = True) -> typing.Self:
        if cls is EvaluatorConfig and cls.get_subclass(default.get("type")) is None:
            # Default subclass.
            return LossEvaluatorConfig._from_dict(default, strict)
        return super()._from_dict(default, strict=strict)

    def get_run_count(self, training_iterations: int, extra_evaluations: int = 0):
        # Number of completed evaluation runs
        return (self.get_count(training_iterations) + extra_evaluations) if self.enabled() else 0

    def get_evaluator(self, name: str, num_workers: int) -> "Evaluator":
        raise NotImplementedError()


@config_class(dynamic_type={EvaluatorConfig: "loss"})
class LossEvaluatorConfig(EvaluatorConfig):
    _abstract: typing.ClassVar[bool] = False

    iterations: int | None = Field(
        default=None,
        desc="Number of iterations for each evaluation phase. Setting to None will disable.",
        hint=FieldHint.feature,
        valid=skip_valid_if_none(check_field(Assert.gt, 0)),
    )

    def get_evaluator(self, name: str, num_workers: int) -> "LossEvaluator":
        from fast_llm.engine.evaluation.evaluator import LossEvaluator

        return LossEvaluator(self, name, num_workers)


@config_class(dynamic_type={EvaluatorConfig: "lm_eval"})
class LmEvalEvaluatorConfig(EvaluatorConfig):
    _abstract: typing.ClassVar[bool] = False

    tokenizer: TokenizerConfig = Field(
        desc="Configuration for the tokenizer.",
    )
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

    communication_timeout_sec: float = Field(
        default=600.0,
        desc="Maximum wait time (in seconds) for tensor-parallel or data-parallel model "
        "operations such as forward, generate, or gathering data. Needed because some "
        "ranks may have no data or post-processing can be slow, exceeding the default 60s timeout.",
    )

    def get_evaluator(self, name: str, num_workers: int) -> "LmEvalEvaluator":
        from fast_llm.engine.evaluation.lm_eval.evaluator import LmEvalEvaluator

        return LmEvalEvaluator(self, name, num_workers)
