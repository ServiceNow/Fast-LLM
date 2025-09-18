import abc
import enum
import typing

from fast_llm.config import Field, FieldHint, check_field, config_class
from fast_llm.engine.base_model.config import BaseModelConfig
from fast_llm.engine.config_utils.parameter import ParameterConfig, combine_lr_scales
from fast_llm.layers.common.peft.config import PeftConfig
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    from fast_llm.engine.config_utils.tensor_dim import TensorDim
    from fast_llm.layers.common.normalization.normalization import Normalization


class NormalizationImplementation(str, enum.Enum):
    """
    An enum for the available implementations of layer norm.
    """

    auto = "auto"
    torch = "torch"
    fused = "fused"
    fast = "fast"
    triton = "triton"


@config_class(registry=True)
class NormalizationConfig(BaseModelConfig):
    lr_scale: float | None = Field(
        default=None,
        desc="Scaling factor for the layer learning rate."
        " Combines multiplicatively with the scale set by the parent layer and individual parameters, if applicable.",
        hint=FieldHint.feature,
    )

    @property
    @abc.abstractmethod
    def module_class(self) -> type["Normalization"]:
        pass

    def get_layer(
        self,
        hidden_dim: "TensorDim",
        *,
        lr_scale: float | None = None,
        peft: PeftConfig | None,
    ) -> "Normalization":
        out = self.module_class(self, hidden_dim, combine_lr_scales(lr_scale, self.lr_scale))
        if peft is not None:
            out = peft.apply_normalization(out)
        return out

    @classmethod
    def _from_dict(cls, default: dict[str, typing.Any], strict: bool = True) -> typing.Self:
        if cls is NormalizationConfig and cls.get_subclass(default.get("type")) is None:
            # Default subclass.
            return LayerNormalizationConfig._from_dict(default, strict)
        return super()._from_dict(default, strict=strict)


@config_class(dynamic_type={NormalizationConfig: "none"})
class NoNormalizationConfig(NormalizationConfig):
    _abstract = False

    @property
    def module_class(self) -> type["Normalization"]:
        from fast_llm.layers.common.normalization.normalization import NoNormalization

        return NoNormalization


@config_class()
class LayerNormalizationBaseConfig(NormalizationConfig):
    """
    Common configuration for layer norm and rms norm
    """

    weight: ParameterConfig = Field(
        desc="Configuration for the weight.",
        hint=FieldHint.architecture,
    )
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

    @property
    @abc.abstractmethod
    def module_class(self):
        pass


@config_class(dynamic_type={NormalizationConfig: "layer_norm"})
class LayerNormalizationConfig(LayerNormalizationBaseConfig):
    bias: ParameterConfig = Field(
        desc="Configuration for the weight.",
        hint=FieldHint.architecture,
    )
    _abstract = False

    @property
    def module_class(self):
        from fast_llm.layers.common.normalization.normalization import LayerNormalization

        return LayerNormalization


@config_class(dynamic_type={NormalizationConfig: "rms_norm"})
class RMSNormalizationConfig(LayerNormalizationBaseConfig):
    _abstract = False

    @property
    def module_class(self):
        from fast_llm.layers.common.normalization.normalization import RMSNormalization

        return RMSNormalization
