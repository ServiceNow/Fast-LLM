from fast_llm.config import Field, FieldHint, check_field, config_class
from fast_llm.engine.base_model.config import BaseModelConfig
from fast_llm.layers.block.mlp.config import MLPConfig
from fast_llm.layers.block.peft import TransformerPeftConfig
from fast_llm.layers.common.normalization.config import NormalizationConfig
from fast_llm.utils import Assert

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
# TODO: Use composition instead
class BlockConfig(MLPConfig, BaseModelConfig):

    # TODO: Review names
    normalization: NormalizationConfig = Field(
        desc="Configuration for the normalization layers architecture.",
        hint=FieldHint.architecture,
    )
    peft: TransformerPeftConfig = Field(
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
    full_precision_residual: bool = Field(
        default=False,
        desc="Store the residuals for the transformer in full precision (`optimization_dtype`).",
        hint=FieldHint.stability,
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
        desc="Add biases to linear layers. May be overridden for individual layers.",
        hint=FieldHint.architecture,
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
    per_layer_lr_scale: list[float | None] | None = Field(
        default=None,
        desc="Custom learning rate scale for each layer.",
        doc="May be used to freeze some layers by setting their scale to zero.",
        hint=FieldHint.feature,
    )

    # TODO: Review initialization
    init_method_std: float = Field(
        default=None,
        desc="Default scale for weight initialization. Default: hidden_size**-0.5",
        hint=FieldHint.optional,
        valid=check_field(Assert.geq, 0),
    )
    init_method_max: float | None = Field(
        default=None,
        desc="Max value for clamping initialized weights. Default: float('inf')",
        hint=FieldHint.optional,
    )
    init_method_min: float | None = Field(
        default=None,
        desc="Min value for clamping initialized weights. Default: -float('inf')",
        hint=FieldHint.optional,
    )
