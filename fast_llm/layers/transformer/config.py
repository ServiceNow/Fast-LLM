import functools
import logging
import typing
import warnings

from fast_llm.config import Config, Field, FieldHint, check_field, config_class, skip_valid_if_none
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.engine.config_utils.tensor_space import CompositeTensorDim, TensorDim, TensorSpace
from fast_llm.engine.distributed.config import DistributedConfig, DistributedDimNames
from fast_llm.functional.config import TritonConfig
from fast_llm.layers.block.config import AddLinearBiasChoices, BlockConfig, BlockDimNames, BlockKwargs
from fast_llm.layers.transformer.rotary.config import RotaryConfig
from fast_llm.utils import Assert, div

if typing.TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class TransformerDimNames(BlockDimNames):
    # A set of common tensor dim names packed into a namespace.
    # Self-attention dimensions
    head_groups = "head_groups"
    group_heads = "group_heads"
    key_and_value = "key_value"
    kv_channels = "kv_channels"
    composite_heads = "composite_heads"
    composite_query = "composite_query"
    composite_key_value = "composite_key_value"
    composite_dense = "composite_dense"


class TransformerKwargs(BlockKwargs):
    rotary_freq_q = "rotary_freq_q"
    rotary_freq_k = "rotary_freq_k"
    attention_mask = "attention_mask"
    attention_mask_value = "attention_mask_value"
    sequence_lengths = "sequence_lengths"
    cu_seqlens_q = "cu_seqlens_q"
    cu_seqlens_k = "cu_seqlens_k"
    max_seqlen_q = "max_seqlen_q"
    max_seqlen_k = "max_seqlen_k"
    # TODO: Review these
    presents = "presents"
    past_key_values = "past_key_values"


class AttentionConfig(Config):
    # TODO: Make mixer class dynamic.
    _abstract = False

    # TODO: Review names
    rotary: RotaryConfig = Field(
        desc="Configuration for the rotary positional embeddings.",
        hint=FieldHint.architecture,
    )
    num_attention_heads: int = Field(default=8, desc="Number of attention heads.", hint=FieldHint.architecture)
    head_groups: int = Field(
        default=1,
        desc="Number of head group for grouped query attention.",
        doc="Set to 1 for multi-query attention, `num_attention_heads` for multi-head.",
        hint=FieldHint.architecture,
        valid=check_field(Assert.gt, 0),
    )
    kv_channels: int = Field(
        default=None,
        desc="Number of key and value channels, i.e., hidden dimension of each attention head. Default: hidden_size // num_attention_heads",
        hint=FieldHint.architecture,
        valid=check_field(Assert.gt, 0),
    )
    attention_dropout: float = Field(
        default=0.0,
        desc="Dropout applied to the attention intermediate states.",
        hint=FieldHint.feature,
        valid=check_field(Assert.geq, 0),
    )
    # Use flash attention if possible (fp16 or bf16)
    use_flash_attention: bool = Field(
        default=True, desc="Enable Flash Attention if possible.", hint=FieldHint.optional
    )
    window_size: int | None = Field(
        default=None,
        desc="Size of the attention sliding window. Warning: this parameter is not part of the architecture and must be redefined when loading a pretrained model.",
        hint=FieldHint.feature,
        valid=skip_valid_if_none(check_field(Assert.geq, 0)),
    )
    max_window_layers: int | None = Field(
        default=None,
        desc="The number of layers that use SWA (Sliding Window Attention). The bottom layers use SWA while the top use full attention.",
        hint=FieldHint.optional,
        valid=skip_valid_if_none(check_field(Assert.geq, 0)),
    )
    attention_lr_scale: float | None = Field(
        default=None,
        desc="Custom learning rate scale for the Attention projection weights.",
        doc="Can be used in muP to scale the Attention learning rate by 1/width_factor",
        hint=FieldHint.feature,
        valid=skip_valid_if_none(check_field(Assert.geq, 0)),
    )
    attention_softmax_scale_power: float = Field(
        default=0.5,
        desc="The scaling power to apply to kv_channel in the attention calculation. "
        " Under Standard Parameterization (SP): default to 0.5. "
        " Under muP (if scaling kv_channels size): use 1. "
        " Under muP (if scaling number of heads instead of kv_channels): use 0.5.",
        valid=skip_valid_if_none(check_field(Assert.geq, 0)),
    )

    def _validate(self) -> None:
        super()._validate()

        if not TritonConfig.TRITON_ENABLED:
            warnings.warn("Triton is disabled, but triton rotary kernel will be used anyway.")

        Assert.multiple(self.num_attention_heads, self.head_groups)

    @functools.cached_property
    def projection_size(self):
        assert self._validated
        return self.num_attention_heads * self.kv_channels

    def do_use_flash_attention(self, distributed_config: DistributedConfig) -> bool:
        return self.use_flash_attention and distributed_config.training_dtype in (DataType.float16, DataType.bfloat16)

    def setup_tensor_space(self, tensor_space: TensorSpace) -> None:
        tensor = tensor_space.distributed_config.get_distributed_dim(DistributedDimNames.tensor)
        # Needed for multiple inheritance.

        tensor_space.add_tensor_dim(
            head_groups := TensorDim(
                TransformerDimNames.head_groups, self.head_groups, tensor if self.head_groups > 1 else None
            )
        )
        tensor_space.add_tensor_dim(
            group_heads := TensorDim(
                TransformerDimNames.group_heads,
                div(self.num_attention_heads, self.head_groups),
                None if self.head_groups > 1 else tensor,
            )
        )
        tensor_space.add_tensor_dim(key_and_value := TensorDim(TransformerDimNames.key_and_value, 2))
        tensor_space.add_tensor_dim(kv_channels := TensorDim(TransformerDimNames.kv_channels, self.kv_channels))
        tensor_space.add_tensor_dim(
            CompositeTensorDim(TransformerDimNames.composite_heads, (head_groups, group_heads))
        )
        tensor_space.add_tensor_dim(
            CompositeTensorDim(TransformerDimNames.composite_query, (head_groups, group_heads, kv_channels))
        )
        tensor_space.add_tensor_dim(
            CompositeTensorDim(TransformerDimNames.composite_key_value, (key_and_value, head_groups, kv_channels))
        )
        tensor_space.add_tensor_dim(
            CompositeTensorDim(TransformerDimNames.composite_dense, (head_groups, group_heads, kv_channels))
        )


@config_class()
# TODO: Use composition for attention config
class TransformerConfig(AttentionConfig, BlockConfig):
    _abstract = False

    # TODO: Review names
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
    init_method_std_qkv: float = Field(
        default=None,
        desc="Scale for the query, key and value weight initialization. Default: init_method_std",
        hint=FieldHint.optional,
        valid=check_field(Assert.geq, 0),
    )
    init_method_max_qkv: float | None = Field(
        default=None,
        desc="Max value for clamping initialized weights for query, key and value matrices. Default: float('inf')",
        hint=FieldHint.optional,
    )
    init_method_min_qkv: float | None = Field(
        default=None,
        desc="Min value for clamping initialized weights for query, key and value matrices. Default: -float('inf')",
        hint=FieldHint.optional,
    )
    init_method_std_attn_proj: float = Field(
        default=None,
        desc="Scale for the attention projection weight initialization. Default: init_method_std",
        hint=FieldHint.optional,
        valid=check_field(Assert.geq, 0),
    )
    init_method_max_attn_proj: float | None = Field(
        default=None,
        desc="Max value for clamping initialized weights for attention projection. Default: float('inf')",
        hint=FieldHint.optional,
    )
    init_method_min_attn_proj: float | None = Field(
        default=None,
        desc="Min value for clamping initialized weights for attention projection. Default: -float('inf')",
        hint=FieldHint.optional,
    )
    init_method_std_mlp_1: float = Field(
        default=None,
        desc="Scale for the MLP first layer weight initialization. Default: init_method_std",
        hint=FieldHint.optional,
        valid=check_field(Assert.geq, 0),
    )
    init_method_max_mlp_1: float | None = Field(
        default=None,
        desc="Max value for clamping initialized weights for MLP first layer. Default: float('inf')",
        hint=FieldHint.optional,
    )
    init_method_min_mlp_1: float | None = Field(
        default=None,
        desc="Min value for clamping initialized weights for MLP first layer. Default: -float('inf')",
        hint=FieldHint.optional,
    )
    init_method_std_mlp_2: float = Field(
        default=None,
        desc="Scale for the MLP second layer weight initialization. Default: init_method_std",
        hint=FieldHint.optional,
        valid=check_field(Assert.geq, 0),
    )
    init_method_max_mlp_2: float | None = Field(
        default=None,
        desc="Max value for clamping initialized weights for MLP second layer. Default: float('inf')",
        hint=FieldHint.optional,
    )
    init_method_min_mlp_2: float | None = Field(
        default=None,
        desc="Min value for clamping initialized weights for MLP second layer. Default: -float('inf')",
        hint=FieldHint.optional,
    )
    # Use random inits instead of constant values, useful for debugging.
    random_bias_init: bool = Field(
        default=False,
        desc="Initialize the biases using the initialization method of their respective weights instead of setting them to zero. Used to test for issues that may not be visible when the biases are zero.",
        hint=FieldHint.testing,
    )

    def _validate(self) -> None:
        with self._set_implicit_default():
            if self.kv_channels is None:
                self.kv_channels = div(self.hidden_size, self.num_attention_heads)
            if self.init_method_std is None:
                self.init_method_std = self.hidden_size**-0.5
            if self.init_method_std_qkv is None:
                self.init_method_std_qkv = self.init_method_std
            if self.init_method_std_attn_proj is None:
                self.init_method_std_attn_proj = self.init_method_std / max(2 * self.num_layers, 1) ** 0.5
            if self.init_method_std_mlp_1 is None:
                self.init_method_std_mlp_1 = self.init_method_std
            if self.init_method_std_mlp_2 is None:
                self.init_method_std_mlp_2 = self.init_method_std / max(2 * self.num_layers, 1) ** 0.5
            if self.init_method_max_qkv is None:
                self.init_method_max_qkv = self.init_method_max
            if self.init_method_min_qkv is None:
                self.init_method_min_qkv = self.init_method_min
            if self.init_method_max_attn_proj is None:
                self.init_method_max_attn_proj = self.init_method_max
            if self.init_method_min_attn_proj is None:
                self.init_method_min_attn_proj = self.init_method_min
            if self.init_method_max_mlp_1 is None:
                self.init_method_max_mlp_1 = self.init_method_max
            if self.init_method_min_mlp_1 is None:
                self.init_method_min_mlp_1 = self.init_method_min
            if self.init_method_max_mlp_2 is None:
                self.init_method_max_mlp_2 = self.init_method_max
            if self.init_method_min_mlp_2 is None:
                self.init_method_min_mlp_2 = self.init_method_min
            if self.init_method_min is not None and self.init_method_max is not None:
                Assert.leq(self.init_method_min, self.init_method_max)
            if self.init_method_min_qkv is not None and self.init_method_max_qkv is not None:
                Assert.leq(self.init_method_min, self.init_method_max)
            if self.init_method_min_qkv is not None and self.init_method_max_qkv is not None:
                Assert.leq(self.init_method_min_qkv, self.init_method_max_qkv)
            if self.init_method_min_attn_proj is not None and self.init_method_max_attn_proj is not None:
                Assert.leq(self.init_method_min_attn_proj, self.init_method_max_attn_proj)
            if self.init_method_min_mlp_1 is not None and self.init_method_max_mlp_1 is not None:
                Assert.leq(self.init_method_min_mlp_1, self.init_method_max_mlp_1)
            if self.init_method_min_mlp_2 is not None and self.init_method_max_mlp_2 is not None:
                Assert.leq(self.init_method_min_mlp_2, self.init_method_max_mlp_2)

        super()._validate()

    @property
    def add_attn_qkv_bias(self) -> bool:
        if isinstance(self.add_linear_biases, bool):
            return self.add_linear_biases
        if self.add_linear_biases == AddLinearBiasChoices.nowhere:
            return False
        return True

    @property
    def add_attn_dense_bias(self) -> bool:
        if isinstance(self.add_linear_biases, bool):
            return self.add_linear_biases
        if self.add_linear_biases == AddLinearBiasChoices.everywhere:
            return True
        return False
