import functools
import logging
import typing
import warnings

from fast_llm.config import Field, FieldHint, check_field, config_class, skip_valid_if_none
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.engine.config_utils.initialization import InitializationConfig, Initializer, init_normal_, init_zeros_
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.functional.config import TritonConfig
from fast_llm.layers.block.config import BlockConfig, BlockDimNames, BlockKwargs, MixerConfig
from fast_llm.layers.transformer.rotary.config import RotaryConfig
from fast_llm.utils import Assert, div

logger = logging.getLogger(__name__)


class AttentionDimNames(BlockDimNames):
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


class AttentionKwargs(BlockKwargs):
    rotary_freq_q = "rotary_freq_q"
    rotary_freq_k = "rotary_freq_k"
    attention_mask = "attention_mask"
    attention_mask_value = "attention_mask_value"
    cu_seqlens_q = "cu_seqlens_q"
    cu_seqlens_k = "cu_seqlens_k"
    max_seqlen_q = "max_seqlen_q"
    max_seqlen_k = "max_seqlen_k"
    # TODO: Review these
    presents = "presents"
    past_key_values = "past_key_values"


@config_class(dynamic_type={MixerConfig: "attention"})
class AttentionConfig(MixerConfig):
    _abstract = False

    # Needed for backward compatibility. TODO: remove
    module_name: typing.ClassVar[str] = "attn"

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
    qkv_weight_initialization: InitializationConfig = Field(
        desc="Initialization configuration for the query, key and value layer weights."
        " Default: normal(std=hidden_size**-0.5)",
        hint=FieldHint.feature,
    )
    qkv_bias_initialization: InitializationConfig = Field(
        desc="Initialization configuration for the query, key and value layer biases. Default: fill with zeros.",
        hint=FieldHint.feature,
    )
    dense_weight_initialization: InitializationConfig = Field(
        desc="Initialization configuration for the dense layer weight."
        " Default: normal(std=(2 * num_blocks * hidden_size)**-0.5)",
        hint=FieldHint.feature,
    )
    dense_bias_initialization: InitializationConfig = Field(
        desc="Initialization configuration for the dense layer biases. Default: fill with zeros.",
        hint=FieldHint.feature,
    )

    def _validate(self) -> None:
        with self._set_implicit_default():
            # TODO: hidden_size not yet validated.
            if self.kv_channels is None:
                self.kv_channels = div(self.block.hidden_size, self.num_attention_heads)

        super()._validate()

        if not TritonConfig.TRITON_ENABLED:
            warnings.warn("Triton is disabled, but triton rotary kernel will be used anyway.")

        Assert.multiple(self.num_attention_heads, self.head_groups)
        if not self.qkv_bias_initialization.is_default:
            assert self.add_qkv_bias
        if not self.dense_bias_initialization.is_default:
            assert self.add_dense_bias

    @functools.cached_property
    def projection_size(self):
        assert self._validated
        return self.num_attention_heads * self.kv_channels

    def do_use_flash_attention(self, distributed_config: DistributedConfig) -> bool:
        return self.use_flash_attention and distributed_config.training_dtype in (DataType.float16, DataType.bfloat16)

    @functools.cached_property
    def add_qkv_bias(self) -> bool:
        return self.block.add_linear_biases

    @functools.cached_property
    def add_dense_bias(self) -> bool:
        return self.block.add_linear_biases

    @functools.cached_property
    def qkv_weight_initialization_method(self) -> Initializer:
        if self.qkv_weight_initialization.is_default:
            return init_normal_(0, self.block.hidden_size**-0.5)
        else:
            return self.qkv_weight_initialization.get_initializer()

    @functools.cached_property
    def qkv_bias_initialization_method(self) -> Initializer:
        if self.qkv_bias_initialization.is_default:
            return init_zeros_
        else:
            return self.qkv_bias_initialization.get_initializer()

    @functools.cached_property
    def dense_weight_initialization_method(self) -> Initializer:
        if self.dense_weight_initialization.is_default:
            return init_normal_(0, self.block.hidden_size**-0.5 / max(2 * self.block.num_blocks, 1))
        else:
            return self.dense_weight_initialization.get_initializer()

    @functools.cached_property
    def dense_bias_initialization_method(self) -> Initializer:
        if self.dense_bias_initialization.is_default:
            return init_zeros_
        else:
            return self.dense_bias_initialization.get_initializer()


@config_class()
# TODO: Remove
class TransformerConfig(BlockConfig):
    pass
