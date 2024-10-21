import enum
import typing

from fast_llm.config import Field, FieldHint, check_field, config_class
from fast_llm.engine.base_model.config import BaseModelArchitectureConfig, BaseModelConfig
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    from fast_llm.engine.config_utils.tensor_space import TensorDim


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


@config_class()
class NormalizationArchitectureConfig(BaseModelArchitectureConfig):
    _abstract = False
    # TODO: Remove "normalization" from names once we have fully nested configs?
    # Normalization type
    type: NormalizationType = Field(
        default=NormalizationType.layer_norm,
        desc="The type of normalization to use, for example Layer Norm or RMS Norm.",
        hint=FieldHint.core,
    )
    # TODO: Rename to normalization_epsilon
    epsilon: float = Field(
        default=1e-5, desc="Regularizer for the division.", hint=FieldHint.stability, valid=check_field(Assert.gt, 0)
    )
    zero_centered: bool = Field(
        default=False,
        desc="Write the normalization weight as `w = 1 + w'`, to improve numerical accuracy when close to one.",
        hint=FieldHint.stability,
    )

    @classmethod
    def _from_dict(
        cls,
        default: dict[str, typing.Any],
        strict: bool = True,
        flat: bool = False,
    ):
        # TODO v0.2: Remove.
        cls._handle_renamed_field(default, "normalization_type", "type")
        cls._handle_renamed_field(default, "layer_norm_eps", "epsilon")
        cls._handle_renamed_field(default, "zero_centered_normalization", "zero_centered")
        return super()._from_dict(default, strict, flat)


@config_class()
class NormalizationConfig(NormalizationArchitectureConfig, BaseModelConfig):
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

    def get_layer(self, hidden_dim: "TensorDim"):
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
    ):
        cls._handle_renamed_field(default, "normalization_implementation", "implementation")
        cls._handle_renamed_field(default, "layer_norm_init_range", "initialization_range")
        return super()._from_dict(default, strict, flat)
