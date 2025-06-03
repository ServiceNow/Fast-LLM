import enum
import typing

from fast_llm.config import Field, FieldHint, check_field, config_class
from fast_llm.engine.base_model.config import BaseModelConfig
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    from fast_llm.engine.config_utils.tensor_space import TensorDim
    from fast_llm.layers.common.linear import LinearBase, LinearLike
    from fast_llm.layers.common.normalization import LayerNorm, RMSNorm


class NormalizationImplementation(str, enum.Enum):
    """
    An enum for the available implementations of layer norm.
    """

    auto = "auto"
    torch = "torch"
    fused = "fused"
    fast = "fast"
    triton = "triton"


class NormalizationType(str, enum.Enum):
    """
    An enum for the available normalization layers.
    TODO: Add no_norm type?
    """

    layer_norm = "layer_norm"
    rms_norm = "rms_norm"


@config_class(registry=True)
class NormalizationConfig(BaseModelConfig):
    _abstract = False

    # Normalization type
    type: NormalizationType = Field(
        default=NormalizationType.layer_norm,
        desc="The type of normalization to use, for example Layer Norm or RMS Norm.",
        hint=FieldHint.architecture,
    )
    # TODO: Rename to normalization_epsilon
    epsilon: float = Field(
        default=1e-5,
        desc="Regularizer for the division.",
        hint=FieldHint.architecture,
        valid=check_field(Assert.gt, 0),
    )
    zero_centered: bool = Field(
        default=False,
        desc="Write the normalization weight as `w = 1 + w'`, to improve numerical accuracy when close to one.",
        hint=FieldHint.architecture,
    )
    implementation: NormalizationImplementation = Field(
        default=NormalizationImplementation.auto,
        desc="The implementation to use for the normalization layer.",
        hint=FieldHint.performance,
    )
    # TODO: Rename to normalization_init_range
    initialization_range: float = Field(
        default=0.0,
        desc="Randomize the initialization with a uniform noise. Used to test for issues that may not be visible with the default initialization.",
        hint=FieldHint.testing,
        valid=check_field(Assert.geq, 0),
    )

    def get_layer(self, hidden_dim: "TensorDim") -> "LayerNorm | RMSNorm":
        from fast_llm.layers.common.normalization import LayerNorm, RMSNorm
        from fast_llm.tensor import init_uniform_

        kwargs = {
            "hidden_dim": hidden_dim,
            "eps": self.epsilon,
            "implementation": self.implementation,
            "zero_centered": self.zero_centered,
        }
        if self.initialization_range:
            mean = 0 if self.zero_centered else 1
            kwargs["weight_init_method"] = init_uniform_(
                mean - self.initialization_range, mean + self.initialization_range
            )
        if self.type == NormalizationType.layer_norm:
            if self.initialization_range:
                kwargs["bias_init_method"] = init_uniform_(-self.initialization_range, self.initialization_range)
            return LayerNorm(**kwargs)
        elif self.type == NormalizationType.rms_norm:
            return RMSNorm(**kwargs)
        else:
            raise ValueError(self.type)

    @classmethod
    def _from_dict(
        cls,
        default: dict[str, typing.Any],
        strict: bool = True,
        flat: bool = False,
    ) -> typing.Self:
        cls._handle_renamed_field(default, "normalization_type", "type")
        cls._handle_renamed_field(default, "layer_norm_eps", "epsilon")
        cls._handle_renamed_field(default, "zero_centered_normalization", "zero_centered")
        cls._handle_renamed_field(default, "normalization_implementation", "implementation")
        cls._handle_renamed_field(default, "layer_norm_init_range", "initialization_range")
        return super()._from_dict(default, strict, flat)


for name in NormalizationType:
    # We need this because we are using the reserved field name `type`.
    # TODO: Implement proper dynamic typing.
    NormalizationConfig.register_subclass(name.value, NormalizationConfig)


class PeftType(str, enum.Enum):
    # TODO : Use a dynamic config type instead.
    none = "none"
    lora = "lora"


@config_class()
class PeftConfig(BaseModelConfig):
    _abstract = False

    type: PeftType = Field(
        default=PeftType.none,
        desc="The type of parameter-efficient fine tuning to use Only LoRA is supported at the moment.",
        hint=FieldHint.core,
    )
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
        if self.type == PeftType.none:
            return linear
        elif self.type == PeftType.lora:
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
        else:
            raise NotImplementedError(self.type)
