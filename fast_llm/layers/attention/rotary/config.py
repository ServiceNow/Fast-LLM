import abc
import math
import typing
import warnings

from fast_llm.config import Field, FieldHint, config_class
from fast_llm.engine.base_model.config import BaseModelConfig
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.functional.config import TritonConfig
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    from fast_llm.layers.attention.rotary.rotary import DefaultRotary, Llama3Rotary, NoRotary, Rotary, YarnRotary


@config_class(registry=True)
class RotaryConfig(BaseModelConfig):
    # TODO: Move rotary to its own submodule.

    @classmethod
    def _from_dict(
        cls,
        default: dict[str, typing.Any],
        strict: bool = True,
        flat: bool = False,
    ) -> typing.Self:
        if cls is RotaryConfig and cls.get_subclass(default.get("type")) is None:
            # Default subclass.
            return NoRotaryConfig._from_dict(default, strict, flat)
        return super()._from_dict(default, strict=strict, flat=flat)

    def get_layer(self, head_size_dim: TensorDim) -> "Rotary":
        return self._get_configurable_class()(self, head_size_dim)

    @classmethod
    @abc.abstractmethod
    def _get_configurable_class(self) -> "type[Rotary]":
        pass


@config_class(dynamic_type={RotaryConfig: "none"})
class NoRotaryConfig(RotaryConfig):
    _abstract = False

    @classmethod
    def _get_configurable_class(self) -> "type[NoRotary]":
        from fast_llm.layers.attention.rotary.rotary import NoRotary

        return NoRotary


@config_class(dynamic_type={RotaryConfig: "default"})
class DefaultRotaryConfig(RotaryConfig):
    _abstract = False
    theta: float = Field(
        default=10000,
        desc="Scale for the rotary positional embeddings",
        hint=FieldHint.architecture,
    )
    # TODO: Make a backup implementation that doesn't affect the layout.
    triton: bool = Field(
        default=True,
        desc="Enable the triton implementation of the rotary embeddings. Affects the model layout.",
        hint=FieldHint.architecture,
    )

    @property
    def complex_format(self) -> bool:
        # TODO: Make a backup implementation that doesn't affect the layout.
        return not self.triton

    def _validate(self) -> None:
        super()._validate()
        if self.triton and not TritonConfig.TRITON_ENABLED:
            warnings.warn("Triton is disabled, but the triton rotary kernel will be used anyway.")

    def _get_configurable_class(self) -> "type[DefaultRotary]":
        from fast_llm.layers.attention.rotary.rotary import DefaultRotary

        return DefaultRotary


@config_class(dynamic_type={RotaryConfig: "llama3"})
class Llama3RotaryConfig(DefaultRotaryConfig):
    """
    Llama3 scaling: https://github.com/meta-llama/llama-models/blob/baf7b01b6e62bc7126c7b558d2b67d4533142680/models/llama3/reference_impl/model.py#L45-L67
    """

    # TODO: Add descriptions.
    scale_factor: float = Field(default=8.0, hint=FieldHint.feature)
    low_frequency_factor: float = Field(default=1.0, hint=FieldHint.feature)
    high_frequency_factor: float = Field(default=4.0, hint=FieldHint.feature)
    original_context_length: int = Field(default=8192, hint=FieldHint.feature)

    def _validate(self) -> None:
        super()._validate()
        Assert.gt(self.high_frequency_factor, self.low_frequency_factor)

    def _get_configurable_class(self) -> "type[Llama3Rotary]":
        from fast_llm.layers.attention.rotary.rotary import Llama3Rotary

        return Llama3Rotary


@config_class(dynamic_type={RotaryConfig: "yarn"})
class YarnRotaryConfig(DefaultRotaryConfig):
    """
    Yarn scaling:
    https://github.com/huggingface/transformers/blob/006d9249ec0270ff6c4d3840979d23fe94bdc763/src/transformers/modeling_rope_utils.py#L163
    [original paper](https://arxiv.org/abs/2309.00071)
    """

    # TODO: Add descriptions.
    scale_factor: float = Field(default=8.0, hint=FieldHint.feature)
    attention_factor: None | float = Field(
        default=None,
        hint=FieldHint.feature,
    )
    beta_fast: float = Field(
        default=32.0,
        hint=FieldHint.feature,
    )
    beta_slow: float = Field(
        default=1.0,
        hint=FieldHint.feature,
    )
    original_context_length: int = Field(default=8192, hint=FieldHint.feature)

    def _validate(self) -> None:
        if self.attention_factor is None:
            if "attention_factor" in self._explicit_fields:
                # TODO: hack to be able to load models with attention_factor set to None/null in the config (e.g. https://huggingface.co/ServiceNow-AI/Apriel-5B-Instruct/blob/main/config.json)
                self._explicit_fields.remove("attention_factor")
                delattr(self, "attention_factor")
            with self._set_implicit_default():
                self.attention_factor = 0.1 * math.log(self.scale_factor) + 1.0
        super()._validate()

    def _get_configurable_class(self) -> "type[YarnRotary]":
        from fast_llm.layers.attention.rotary.rotary import YarnRotary

        return YarnRotary
