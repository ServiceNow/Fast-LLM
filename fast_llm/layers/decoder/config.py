import enum
import typing

from fast_llm.config import Field, FieldHint, check_field, config_class
from fast_llm.engine.config_utils.parameter import combine_lr_scales
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import _BIG_PRIMES, DistributedConfig
from fast_llm.layers.block.config import BlockConfig, BlockKwargs
from fast_llm.layers.common.normalization.config import NormalizationConfig
from fast_llm.layers.common.peft.config import PeftConfig
from fast_llm.utils import Assert, normalize_probabilities

if typing.TYPE_CHECKING:
    from fast_llm.layers.decoder.block import BlockWithBias, DecoderBlock
    from fast_llm.layers.decoder.stochastic_mixer import StochasticMixer


class StochasticMixerKwargs(BlockKwargs):
    """Kwargs keys for stochastic mixer."""

    mixer_name = "stochastic_mixer_name"
    generator = "stochastic_mixer_generator"


@config_class()
class BlockWithBiasConfig(BlockConfig):
    """
    A common interface for various blocks and block layers.
    """

    @property
    def layer_class(self) -> "type[BlockWithBias]":
        raise NotImplementedError()

    def get_layer(
        self,
        distributed_config: DistributedConfig,
        hidden_dim: TensorDim,
        *,
        lr_scale: float | None,
        peft: PeftConfig | None,
        return_bias: bool = False,
    ) -> "BlockWithBias":
        return self.layer_class(
            self,
            distributed_config,
            hidden_dim=hidden_dim,
            lr_scale=combine_lr_scales(lr_scale, self.lr_scale),
            peft=peft,
            return_bias=return_bias,
        )


@config_class(registry=True)
class MLPBaseConfig(BlockWithBiasConfig):
    _abstract = True

    def get_layer(
        self,
        distributed_config: DistributedConfig,
        hidden_dim: TensorDim,
        *,
        output_dim: TensorDim | None = None,
        lr_scale: float | None,
        peft: PeftConfig | None,
        return_bias: bool = False,
    ) -> "BlockWithBias":
        return self.layer_class(
            self,
            distributed_config,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            lr_scale=combine_lr_scales(lr_scale, self.lr_scale),
            peft=peft,
            return_bias=return_bias,
        )

    @classmethod
    def _from_dict(cls, default: dict[str, typing.Any], strict: bool = True) -> typing.Self:
        if cls is MLPBaseConfig and cls.get_subclass(default.get("type")) is None:
            from fast_llm.layers.decoder.mlp.config import MLPConfig

            # Default subclass.
            return MLPConfig._from_dict(default, strict)
        return super()._from_dict(default, strict=strict)


class StochasticMixerSamplingStrategy(enum.StrEnum):
    """Strategy for sampling mixers in a stochastic mixer."""

    uniform = "uniform"
    weighted = "weighted"


@config_class(registry=True)
class MixerConfig(BlockWithBiasConfig):
    """
    Base config class for mixers.
    """

    @classmethod
    def _from_dict(cls, default: dict[str, typing.Any], strict: bool = True) -> typing.Self:
        if cls is MixerConfig and cls.get_subclass(default.get("type")) is None:
            from fast_llm.layers.attention.config import AttentionConfig

            # Default subclass.
            return AttentionConfig._from_dict(default, strict)
        return super()._from_dict(default, strict=strict)


@config_class(dynamic_type={MixerConfig: "stochastic"})
class StochasticMixerConfig(MixerConfig):
    """
    Stochastic mixer that uniformly samples from multiple mixer options during training.

    For supernet training, each forward pass randomly selects one mixer to execute,
    training all mixers with different subsets of data.
    """

    _abstract = False

    mixers: dict[str, MixerConfig] = Field(
        desc="Dict of mixer options to sample from (must contain at least 1). "
        "Keys are mixer names used for debugging and namespacing.",
        hint=FieldHint.architecture,
    )

    sampling_strategy: StochasticMixerSamplingStrategy = Field(
        default=StochasticMixerSamplingStrategy.uniform,
        desc="Strategy for sampling mixers during training.",
        hint=FieldHint.feature,
    )

    sampling_weights: dict[str, float] | None = Field(
        default=None,
        desc="Sampling probability for each mixer by name (will be normalized to sum to 1.0). "
        "Only used when sampling_strategy='weighted'. "
        "If None with uniform strategy, all mixers have equal probability.",
        hint=FieldHint.feature,
    )

    main_mixer_name: str | None = Field(
        default=None,
        desc="Name of the main mixer. "
        "Used for inference/eval, checkpoint loading (receives pretrained weights), "
        "and checkpoint saving (only this mixer is exported). "
        "If None, uses the first mixer in the dict.",
        hint=FieldHint.feature,
    )

    seed_shift: int = Field(
        default=_BIG_PRIMES[11],
        desc="Seed shift for mixer sampling reproducibility.",
        hint=FieldHint.optional,
    )

    def _validate(self) -> None:
        super()._validate()

        # Validate mixers dict is not empty
        Assert.gt(len(self.mixers), 0)

        # Set main_mixer_name to first mixer if not specified
        if self.main_mixer_name is None:
            with self._set_implicit_default():
                self.main_mixer_name = next(iter(self.mixers.keys()))

        # Validate main mixer name exists
        if self.main_mixer_name not in self.mixers:
            raise ValueError(f"main_mixer_name '{self.main_mixer_name}' not found in mixers")

        # Validate and normalize sampling weights
        if self.sampling_weights is not None:
            Assert.eq(set(self.sampling_weights.keys()), set(self.mixers.keys()))
            # Normalize weights to sum to 1.0 (also validates non-negative and positive sum)
            normalized_values = normalize_probabilities(list(self.sampling_weights.values()))
            self.sampling_weights = dict(zip(self.sampling_weights.keys(), normalized_values))

    @property
    def layer_class(self) -> "type[StochasticMixer]":
        from fast_llm.layers.decoder.stochastic_mixer import StochasticMixer

        return StochasticMixer


@config_class(dynamic_type={BlockConfig: "decoder"})
class DecoderBlockConfig(BlockConfig):
    _abstract = False
    mixer: MixerConfig = Field()
    mlp: MLPBaseConfig = Field()
    # TODO: Review names
    normalization: NormalizationConfig = Field(
        desc="Configuration for the block normalization layers.",
        hint=FieldHint.architecture,
    )
    # TODO: Review names
    dropout: float = Field(
        default=0.0,
        desc="Dropout applied to the residual connections.",
        hint=FieldHint.feature,
        valid=check_field(Assert.geq, 0),
    )
    distillation_model: str | None = Field(
        default=None,
        desc="Name of the reference model to use for activation-level distillation.",
        hint=FieldHint.feature,
    )
    activation_distillation_factor: float = Field(
        default=0.0,
        desc="Factor to scale the activation-level distillation loss by.",
        hint=FieldHint.feature,
        valid=check_field(Assert.geq, 0),
    )

    def _validate(self) -> None:
        super()._validate()
        if self.activation_distillation_factor > 0.0 and self.distillation_model is None:
            raise ValueError("Activation distillation requires a distillation_model.")

    @property
    def layer_class(self) -> "type[DecoderBlock]":
        from fast_llm.layers.decoder.block import DecoderBlock

        return DecoderBlock

    def get_layer(
        self,
        distributed_config: DistributedConfig,
        hidden_dim: TensorDim,
        lr_scale: float | None,
        peft: PeftConfig | None,
        return_input: bool = False,
    ) -> "DecoderBlock":
        return self.layer_class(
            self,
            distributed_config,
            hidden_dim=hidden_dim,
            lr_scale=combine_lr_scales(lr_scale, self.lr_scale),
            peft=peft,
            return_input=return_input,
        )

    def get_distillation_models(self) -> set[str]:
        if self.distillation_model is not None and self.activation_distillation_factor > 0.0:
            return {self.distillation_model}
        return set()
