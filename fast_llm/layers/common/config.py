import abc
import typing

from fast_llm.config import Field, FieldHint, config_class
from fast_llm.engine.base_model.config import BaseModelConfig

if typing.TYPE_CHECKING:
    from fast_llm.layers.common.linear import LinearBase, LinearLike


@config_class()
class PeftConfig(BaseModelConfig):
    @abc.abstractmethod
    def apply_linear(self, linear: "LinearBase", **kwargs) -> "LinearLike":
        pass


@config_class()
class NoPeftConfig(PeftConfig):
    _abstract = False

    def apply_linear(self, linear: "LinearBase", **kwargs) -> "LinearLike":
        return linear


@config_class()
class LoRAConfig(PeftConfig):
    _abstract = False

    rank: int = Field(
        default=8,
        desc="The LoRA rank, i.e. the size of the intermediate dimension.",
        hint=FieldHint.stability,
    )
    alpha: float = Field(
        default=8.0,
        desc="The LoRA scaling parameter.",
        hint=FieldHint.stability,
    )
    dropout: float = Field(
        default=0.0,
        desc="Dropout rate for LoRA.",
        hint=FieldHint.stability,
    )

    def apply_linear(self, linear: "LinearBase", **kwargs) -> "LinearLike":
        from fast_llm.layers.common.peft import lora_linear

        # TODO: Init method?
        return lora_linear(
            linear,
            linear.weight.param_init_method,
            linear.weight.param_init_method,
            self.rank,
            self.alpha,
            self.dropout,
            **kwargs,
        )
