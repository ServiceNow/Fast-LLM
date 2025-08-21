import abc
import enum
import typing

from fast_llm.config import Field, FieldHint, check_field, config_class
from fast_llm.engine.base_model.config import BaseModelConfig
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    import torch

    from fast_llm.engine.config_utils.tensor_space import TensorDim
    from fast_llm.layers.common.linear import LinearBase, LinearLike
    from fast_llm.layers.common.normalization import LayerNorm, RMSNorm


@config_class()
class LLMBlockConfig(BaseModelConfig):
    _abstract = False

    per_layer_lr_scale: list[float] | None = Field(
        default=None,
        desc="Custom learning rate scale for each layer.",
        doc="May be used to freeze some layers by setting their scale to zero.",
        hint=FieldHint.feature,
    )


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
    pass

    @abc.abstractmethod
    def get_layer(self, hidden_dim: "TensorDim", lr_scale: float | None) -> "torch.nn.Module":
        pass

    @classmethod
    def _from_dict(
        cls,
        default: dict[str, typing.Any],
        strict: bool = True,
        flat: bool = False,
    ) -> typing.Self:
        if cls is NormalizationConfig and cls.get_subclass(default.get("type")) is None:
            # Default subclass.
            return LayerNormalizationConfig._from_dict(default, strict, flat)
        return super()._from_dict(default, strict=strict, flat=flat)


@config_class(dynamic_type={NormalizationConfig: "none"})
class NoNormalizationConfig(NormalizationConfig):
    _abstract = False

    def get_layer(self, hidden_dim: "TensorDim", lr_scale: float | None) -> "torch.nn.Module":
        return torch.nn.Identity()


@config_class()
class LayerNormalizationBaseConfig(NormalizationConfig):
    """
    Common configuration for layer norm and rms norm
    """

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

    def get_layer(self, hidden_dim: "TensorDim", lr_scale: float | None = None) -> "LayerNorm | RMSNorm":
        from fast_llm.tensor import init_uniform_centered_

        kwargs = {
            "hidden_dim": hidden_dim,
            "eps": self.epsilon,
            "implementation": self.implementation,
            "zero_centered": self.zero_centered,
            "lr_scale": lr_scale,
        }
        if self.initialization_range:
            mean = 0 if self.zero_centered else 1
            kwargs["weight_init_method"] = init_uniform_centered_(self.initialization_range, mean=mean)
        return self.module_class(**kwargs)

    @property
    @abc.abstractmethod
    def module_class(self):
        pass

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


@config_class(dynamic_type={NormalizationConfig: "layer_norm"})
class LayerNormalizationConfig(LayerNormalizationBaseConfig):
    _abstract = False

    @property
    def module_class(self):
        from fast_llm.layers.common.normalization import LayerNorm

        return LayerNorm


@config_class(dynamic_type={NormalizationConfig: "rms_norm"})
class RMSNormalizationConfig(LayerNormalizationBaseConfig):
    _abstract = False

    @property
    def module_class(self):
        from fast_llm.layers.common.normalization import RMSNorm

        return RMSNorm


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
