import typing

from fast_llm.config import Field, FieldHint, check_field, config_class
from fast_llm.engine.base_model.config import BaseModelConfig
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
    A common class for mixers and mlps, which have the exact same interface.
    """

    _abstract = True
    block: "BlockConfig" = Field(init=False)

    @property
    def layer_class(self) -> "type[BlockLayer]":
        raise NotImplementedError()

    def get_layer(
        self,
        block_config: "BlockConfig",
        distributed_config: DistributedConfig,
        hidden_dim: TensorDim,
        block_index: int,
        name: str,
    ) -> "BlockLayer":
        return self.layer_class(
            self,
            block_config,
            distributed_config,
            hidden_dim,
            block_index,
            name,
        )


@config_class(registry=True)
class MixerConfig(BlockLayerConfig):
    _abstract = True

    # Needed for backward compatibility. TODO: Standardize to `mixer`
    module_name: typing.ClassVar[str] = "mixer"

    def _validate(self) -> None:
        assert hasattr(self, "block")
        Assert.is_(self.block.mixer, self)
        super()._validate()

    @classmethod
    def _from_dict(
        cls,
        default: dict[str, typing.Any],
        strict: bool = True,
        flat: bool = False,
    ) -> typing.Self:
        if cls is MixerConfig and cls.get_subclass(default.get("type")) is None:
            from fast_llm.layers.transformer.config import AttentionConfig

            # Default subclass.
            return AttentionConfig._from_dict(default, strict, flat)
        return super()._from_dict(default, strict=strict, flat=flat)


@config_class(registry=True)
class MLPBaseConfig(BlockLayerConfig):
    _abstract = True

    def _validate(self) -> None:
        assert hasattr(self, "block")
        Assert.is_(self.block.mlp, self)
        super()._validate()

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


@config_class()
class BlockConfig(BaseModelConfig):
    _abstract = False
    mixer: MixerConfig = Field(
        desc="Configuration for the mixer.",
        hint=FieldHint.architecture,
    )
    mlp: MLPBaseConfig = Field(
        desc="Configuration for the MLP.",
        hint=FieldHint.architecture,
    )
    # TODO: Allow separate initializations?
    normalization: NormalizationConfig = Field(
        desc="Configuration for the normalization layers architecture.",
        hint=FieldHint.architecture,
    )
    peft: PeftConfig = Field(
        desc="Configuration for the parameter-efficient fine tuning.",
        hint=FieldHint.architecture,
    )
    # TODO: Review names
    hidden_dropout: float = Field(
        default=0.0,
        desc="Dropout applied to the residual connections.",
        hint=FieldHint.feature,
        valid=check_field(Assert.geq, 0),
    )
    debug_transformer: int = Field(
        default=0,
        desc="Log the output of each operation in a transformer layer.",
        hint=FieldHint.logging,
        valid=check_field(Assert.geq, 0),
    )
    debug_transformer_memory: bool = Field(
        default=False,
        desc="Log the memory usage after each operation in a transformer layer..",
        hint=FieldHint.logging,
    )
    add_linear_biases: bool = Field(
        default=True,
        desc="Whether to add biases to linear layers. May be overridden in individual layer configs.",
        hint=FieldHint.architecture,
    )

    # TODO: Move these, not specific to a single block.
    num_blocks: int = Field(
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
    full_precision_residual: bool = Field(
        default=False,
        desc="Store the residuals for the transformer in full precision (`optimization_dtype`).",
        hint=FieldHint.stability,
    )
    per_layer_lr_scale: list[float] | None = Field(
        default=None,
        desc="Custom learning rate scale for each layer.",
        doc="May be used to freeze some layers by setting their scale to zero.",
        hint=FieldHint.feature,
    )
