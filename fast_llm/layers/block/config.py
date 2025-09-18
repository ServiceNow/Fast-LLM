import typing

from fast_llm.config import Field, FieldHint, check_field, config_class
from fast_llm.engine.base_model.config import BaseModelConfig, Preprocessor
from fast_llm.engine.config_utils.parameter import combine_lr_scales
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.layers.common.normalization.config import NormalizationConfig
from fast_llm.layers.common.peft.config import PeftConfig
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    from fast_llm.layers.block.block import BlockLayer


# TODO: Generalize these beyond language models? (Ex. vision)


class BlockDimNames:
    # A set of common tensor dim names packed into a namespace.
    # Input dimensions (variable)
    # TODO: Does batch belong here?
    batch = "batch"
    # TODO: Distinguish micro-sequence?
    sequence_q = "sequence_q"
    sequence_q_tp = "sequence_q_tp"
    sequence_k = "sequence_k"
    hidden = "hidden"


class BlockKwargs:
    sequence_first = "sequence_first"
    hidden_dims = "hidden_dims"
    sequence_q_dim = "sequence_q_dim"
    sequence_k_dim = "sequence_k_dim"
    # TODO: These are confusing
    sequence_length = "sequence_length"
    sequence_lengths = "sequence_lengths"
    # TODO: Belongs elsewhere?
    grad_output = "grad_output"


@config_class()
class BlockLayerConfig(BaseModelConfig):
    """
    A common class for mixers and mlps, which have the same interface.
    """

    _abstract = True
    block: "BlockConfig" = Field(init=False)

    lr_scale: float | None = Field(
        default=None,
        desc="Scaling factor for the layer learning rate."
        " Combines multiplicatively with the scale set by the parent and child layers, if applicable.",
        hint=FieldHint.feature,
    )

    @property
    def layer_class(self) -> "type[BlockLayer]":
        raise NotImplementedError()

    def get_layer(
        self,
        distributed_config: DistributedConfig,
        hidden_dim: TensorDim,
        lr_scale: float | None,
        peft: PeftConfig | None,
    ) -> "BlockLayer":
        return self.layer_class(
            self,
            distributed_config,
            hidden_dim=hidden_dim,
            lr_scale=combine_lr_scales(lr_scale, self.lr_scale),
            peft=peft,
        )

    def get_preprocessors(self, distributed_config: DistributedConfig) -> list[Preprocessor]:
        # TODO: Move to actual layers?
        return []


@config_class(registry=True)
class MLPBaseConfig(BlockLayerConfig):
    _abstract = True

    @classmethod
    def _from_dict(
        cls,
        default: dict[str, typing.Any],
        strict: bool = True,
        flat: bool = False,
    ) -> typing.Self:
        if cls is MLPBaseConfig and cls.get_subclass(default.get("type")) is None:
            from fast_llm.layers.block.mlp.config import MLPConfig

            # Default subclass.
            return MLPConfig._from_dict(default, strict, flat)
        return super()._from_dict(default, strict=strict, flat=flat)


@config_class(registry=True)
class MixerConfig(BlockLayerConfig):
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


@config_class()
class BlockConfig(BaseModelConfig):
    _abstract = False
    mixer: MixerConfig = Field()
    mlp: MLPBaseConfig = Field()
    # TODO: Review names
    normalization: NormalizationConfig = Field(
        desc="Configuration for the block normalization layers.",
        hint=FieldHint.architecture,
    )
    lr_scale: float | None = Field(
        default=None,
        desc="Scaling factor for the layer learning rate."
        " Combines multiplicatively with the scale set by the parent and child layers, if applicable.",
        hint=FieldHint.feature,
    )
    # TODO: Review names
    dropout: float = Field(
        default=0.0,
        desc="Dropout applied to the residual connections.",
        hint=FieldHint.feature,
        valid=check_field(Assert.geq, 0),
    )
    # TODO: Move these, not specific to a single block.
    num_layers: int = Field(
        default=12,
        desc="Number of blocks in the model.",
        hint=FieldHint.architecture,
        valid=check_field(Assert.geq, 0),
    )
    hidden_size: int = Field(
        default=1024,
        desc="Size of the transformer's main hidden dimension, e.g., for its input and output layers.",
        hint=FieldHint.architecture,
        valid=check_field(Assert.gt, 0),
    )

    def get_layer(
        self,
        distributed_config: DistributedConfig,
        hidden_dim: TensorDim,
        lr_scale: float | None,
        peft: PeftConfig | None = None,
        return_input: bool = False,
    ):
        from fast_llm.layers.block.block import Block

        return Block(
            self,
            distributed_config,
            hidden_dim=hidden_dim,
            lr_scale=combine_lr_scales(lr_scale, self.lr_scale),
            peft=peft,
            return_input=return_input,
        )

    def get_preprocessors(self, distributed_config: DistributedConfig) -> list[Preprocessor]:
        # TODO: Move to actual layers?
        return self.mixer.get_preprocessors(distributed_config) + self.mlp.get_preprocessors(distributed_config)
