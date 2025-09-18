"""
TODO: Generalize beyond transformers.
"""

import enum
import typing

from fast_llm.config import Field, FieldHint, config_class
from fast_llm.layers.common.peft.config import LoRAConfig, NoPeftConfig, PeftConfig
from fast_llm.utils import div

if typing.TYPE_CHECKING:
    from fast_llm.layers.common.linear.linear import LinearBase, LinearLike


class TransformerSubLayerName(str, enum.Enum):
    query = "query"
    key = "key"
    value_ = "value"
    key_value = "key_value"
    dense = "dense"
    mlp_1 = "mlp_1"
    mlp_2 = "mlp_2"


@config_class(registry=True)
class TransformerPeftConfig(PeftConfig):
    @classmethod
    def _from_dict(
        cls,
        default: dict[str, typing.Any],
        strict: bool = True,
        flat: bool = False,
    ) -> typing.Self:
        if cls is TransformerPeftConfig and cls.get_subclass(default.get("type")) is None:
            # Default subclass.
            return TransformerNoPeftConfig._from_dict(default, strict, flat)
        return super()._from_dict(default, strict=strict, flat=flat)


@config_class(dynamic_type={TransformerPeftConfig: "none"})
class TransformerNoPeftConfig(NoPeftConfig, TransformerPeftConfig):
    pass


@config_class(dynamic_type={TransformerPeftConfig: "lora"})
class TransformerLoRAConfig(LoRAConfig, TransformerPeftConfig):
    layers: list[TransformerSubLayerName] = Field(
        default=(TransformerSubLayerName.query, TransformerSubLayerName.value_),
        desc="The layers on which to apply LoRA.",
        hint=FieldHint.feature,
    )

    def apply_linear(self, linear: "LinearBase", layer_type: TransformerSubLayerName | None = None) -> "LinearLike":
        out_channel_begin, out_channel_end = None, None
        if layer_type is None or self.layers is None or layer_type in self.layers:
            enabled = True
            if layer_type == TransformerSubLayerName.key:
                out_channel_end = div(linear._out_dim.global_size, 2)
            elif layer_type == TransformerSubLayerName.value_:
                out_channel_begin = div(linear._out_dim.global_size, 2)
        else:
            enabled = False
        return super().apply_linear(linear, enabled, out_channel_begin, out_channel_end)

    def _validate(self) -> None:
        super()._validate()
        if TransformerSubLayerName.mlp_1 in self.layers or TransformerSubLayerName.mlp_2 in self.layers:
            # TODO: Add MLP support.
            raise NotImplementedError("LoRA not supported for MLP.")
        if TransformerSubLayerName.dense in self.layers:
            # TODO: Support InputParallelLinear (different output format).
            raise NotImplementedError("LoRA not supported for attention dense layer.")
        if (
            sum(
                name in self.layers
                for name in (
                    TransformerSubLayerName.key_value,
                    TransformerSubLayerName.key,
                    TransformerSubLayerName.value_,
                )
            )
            > 1
        ):
            raise ValueError(
                f"{TransformerSubLayerName.key_value.value}, {TransformerSubLayerName.key.value} and {TransformerSubLayerName.value_.value} are mutually exclusive."
            )
