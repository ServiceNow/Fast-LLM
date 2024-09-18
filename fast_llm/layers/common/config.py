import enum

from fast_llm.config import Field, FieldHint, check_field, config_class
from fast_llm.engine.base_model.config import BaseModelArchitectureConfig, BaseModelConfig
from fast_llm.tensor import TensorDim, init_uniform_
from fast_llm.utils import Assert


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
    normalization_type: NormalizationType = Field(
        default=NormalizationType.layer_norm,
        desc="The type of normalization to use, for example Layer Norm or RMS Norm.",
        hint=FieldHint.core,
    )
    # TODO: Rename to normalization_epsilon
    layer_norm_eps: float = Field(
        default=1e-5, desc="Regularizer for the division.", hint=FieldHint.stability, valid=check_field(Assert.gt, 0)
    )
    zero_centered_normalization: bool = Field(
        default=False,
        desc="Write the normalization weight as `w = 1 + w'`, to improve numerical accuracy when close to one.",
        hint=FieldHint.stability,
    )


@config_class()
class NormalizationConfig(NormalizationArchitectureConfig, BaseModelConfig):
    normalization_implementation: NormalizationImplementation = Field(
        default=NormalizationImplementation.auto,
        desc="The implementation to use for the normalization layer.",
        hint=FieldHint.performance,
    )
    # TODO: Rename to normalization_init_range
    layer_norm_init_range: float = Field(
        default=0.0,
        desc="Randomize the initialization with a uniform noise. Used to test for issues that may not be visible with the default initialization.",
        hint=FieldHint.testing,
        valid=check_field(Assert.geq, 0),
    )

    def get_layer(self, hidden_dim: TensorDim):
        from fast_llm.layers.common.normalization import LayerNorm, RMSNorm

        kwargs = {
            "hidden_dim": hidden_dim,
            "eps": self.layer_norm_eps,
            "implementation": self.normalization_implementation,
            "zero_centered": self.zero_centered_normalization,
        }
        if self.layer_norm_init_range:
            mean = 0 if self.zero_centered_normalization else 1
            kwargs["weight_init_method"] = init_uniform_(
                mean - self.layer_norm_init_range, mean + self.layer_norm_init_range
            )
        if self.normalization_type == NormalizationType.layer_norm:
            if self.layer_norm_init_range:
                kwargs["bias_init_method"] = init_uniform_(-self.layer_norm_init_range, self.layer_norm_init_range)
            return LayerNorm(**kwargs)
        elif self.normalization_type == NormalizationType.rms_norm:
            return RMSNorm(**kwargs)
        else:
            raise ValueError(self.normalization_type)
