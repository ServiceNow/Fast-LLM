import typing

from fast_llm.config import Field, FieldHint, config_class
from fast_llm.engine.base_model.config import BaseModelConfig

if typing.TYPE_CHECKING:
    import torch

    from fast_llm.layers.common.linear.linear import LinearBase, LinearLike
    from fast_llm.layers.common.normalization.normalization import Normalization
    from fast_llm.tensor import ParameterMeta


@config_class()
class PeftConfig(BaseModelConfig):
    def apply_linear(
        self,
        module: "LinearBase",
        enabled: bool,
        out_channel_begin: int | None = None,
        out_channel_end: int | None = None,
    ) -> "LinearLike":
        return self.apply_other(module)

    def apply_normalization(self, module: "Normalization") -> "Normalization":
        return self.apply_other(module)

    def apply_other(self, module: "torch.nn.Module") -> "torch.nn.Module":
        for parameter in module.parameters():
            self.apply_weight(parameter)
        return module

    def apply_weight(self, parameter: "ParameterMeta") -> "ParameterMeta":
        return parameter


@config_class()
class NoPeftConfig(PeftConfig):
    _abstract = False


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
    freeze_others: bool = Field(
        default=True,
        desc="Whether to freeze other layers during training.",
    )

    def apply_linear(
        self,
        module: "LinearBase",
        enabled: bool,
        out_channel_begin: int | None = None,
        out_channel_end: int | None = None,
    ) -> "LinearLike":
        if not enabled:
            return self.apply_other(module)

        from fast_llm.layers.common.linear.linear import InputParallelLinear, OutputParallelLinear
        from fast_llm.layers.common.peft.lora import lora_linear

        if isinstance(module, InputParallelLinear):
            # TODO: Support InputParallelLinear (different output format).
            raise NotImplementedError("LoRA not supported for InputParallelLinear.")
        elif isinstance(module, OutputParallelLinear):
            assert out_channel_begin is None and out_channel_end is None

        # TODO: Init method?
        return lora_linear(module, self.rank, self.alpha, self.dropout, out_channel_begin, out_channel_end)

    def apply_weight(self, parameter: "ParameterMeta") -> "ParameterMeta":
        if self.freeze_others:
            parameter.requires_grad = False
        return parameter
