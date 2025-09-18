import typing

from fast_llm.config import Field, FieldHint, check_field, config_class
from fast_llm.engine.base_model.config import LossDef, Preprocessor
from fast_llm.engine.config_utils.parameter import combine_lr_scales
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.layers.block.config import BaseBlockConfig, BlockConfig
from fast_llm.layers.common.normalization.config import NormalizationConfig
from fast_llm.layers.common.peft.config import PeftConfig
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    from fast_llm.layers.decoder.block import BlockWithBias, DecoderBlock


@config_class()
class BlockWithBiasConfig(BaseBlockConfig):
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
        lr_scale: float | None,
        peft: PeftConfig | None,
    ) -> "BlockWithBias":
        return self.layer_class(
            self,
            distributed_config,
            hidden_dim=hidden_dim,
            lr_scale=combine_lr_scales(lr_scale, self.lr_scale),
            peft=peft,
        )


@config_class(registry=True)
class MLPBaseConfig(BlockWithBiasConfig):
    _abstract = True

    @classmethod
    def _from_dict(
        cls,
        default: dict[str, typing.Any],
        strict: bool = True,
        flat: bool = False,
    ) -> typing.Self:
        if cls is MLPBaseConfig and cls.get_subclass(default.get("type")) is None:
            from fast_llm.layers.decoder.mlp.config import MLPConfig

            # Default subclass.
            return MLPConfig._from_dict(default, strict, flat)
        return super()._from_dict(default, strict=strict, flat=flat)


@config_class(registry=True)
class MixerConfig(BlockWithBiasConfig):
    """
    Base config class for mixers.
    """

    @classmethod
    def _from_dict(
        cls,
        default: dict[str, typing.Any],
        strict: bool = True,
        flat: bool = False,
    ) -> typing.Self:
        if cls is MixerConfig and cls.get_subclass(default.get("type")) is None:
            from fast_llm.layers.attention.config import AttentionConfig

            # Default subclass.
            return AttentionConfig._from_dict(default, strict, flat)
        return super()._from_dict(default, strict=strict, flat=flat)


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

    @property
    def layer_class(self) -> "type[DecoderBlock]":
        from fast_llm.layers.decoder.block import DecoderBlock

        return DecoderBlock

    def get_preprocessors(self, distributed_config: DistributedConfig) -> list[Preprocessor]:
        return self.mixer.get_preprocessors(distributed_config) + self.mlp.get_preprocessors(distributed_config)

    def get_loss_definitions(self, count: int = 1) -> list[LossDef]:
        return self.mixer.get_loss_definitions(count=count) + self.mlp.get_loss_definitions(count=count)
