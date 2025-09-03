import functools
import logging
import typing
import warnings

from fast_llm.config import Field, FieldHint, check_field, config_class, skip_valid_if_none
from fast_llm.engine.base_model.config import Preprocessor
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.functional.config import TritonConfig
from fast_llm.layers.attention.rotary.config import RotaryConfig
from fast_llm.layers.block.config import BlockKwargs, MixerConfig
from fast_llm.layers.common.linear.config import AffineLinearConfig
from fast_llm.utils import Assert, div

if typing.TYPE_CHECKING:
    from fast_llm.layers.attention.attention import Attention

logger = logging.getLogger(__name__)


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
    # TODO: Make mixer class dynamic.
    _abstract = False

    query_layer: AffineLinearConfig = Field(
        desc="Configuration for the query layer.",
        hint=FieldHint.architecture,
    )
    key_layer: AffineLinearConfig = Field(
        desc="Configuration for the key layer.",
        hint=FieldHint.architecture,
    )
    # TODO: Use
    value_layer: AffineLinearConfig = Field(
        desc="Configuration for the value layer.",
        hint=FieldHint.architecture,
    )
    dense_layer: AffineLinearConfig = Field(
        desc="Initialization configuration for the dense layer.",
        hint=FieldHint.feature,
    )
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
    attention_softmax_scale_power: float = Field(
        default=0.5,
        desc="The scaling power to apply to kv_channel in the attention calculation. "
        " Under Standard Parameterization (SP): default to 0.5. "
        " Under muP (if scaling kv_channels size): use 1. "
        " Under muP (if scaling number of heads instead of kv_channels): use 0.5.",
        valid=skip_valid_if_none(check_field(Assert.geq, 0)),
    )

    def set_defaults(self, hidden_size: int):
        if self.kv_channels is None:
            with self._set_implicit_default():
                self.kv_channels = div(hidden_size, self.num_attention_heads)

    def _validate(self) -> None:
        super()._validate()

        if not TritonConfig.TRITON_ENABLED:
            warnings.warn("Triton is disabled, but triton rotary kernel will be used anyway.")

        Assert.multiple(self.num_attention_heads, self.head_groups)

    @property
    def layer_class(self) -> "type[Attention]":
        from fast_llm.layers.attention.attention import Attention

        return Attention

    @functools.cached_property
    def projection_size(self):
        assert self._validated
        return self.num_attention_heads * self.kv_channels

    def do_use_flash_attention(self, distributed_config: DistributedConfig) -> bool:
        return self.use_flash_attention and distributed_config.training_dtype in (DataType.float16, DataType.bfloat16)

    def get_preprocessors(self, distributed_config: DistributedConfig) -> list[Preprocessor]:
        # We have multiple identical rotary modules/preprocessors, so it's simpler to make a new one here.
        # TODO: Find a better solution.
        preprocessors: list[Preprocessor] = [self.rotary.get_layer(TensorDim("kv_channels", self.kv_channels))]
        if self.do_use_flash_attention(distributed_config):
            from fast_llm.layers.attention.preprocessing import FlashAttnVarlenPreprocessor

            preprocessors.append(FlashAttnVarlenPreprocessor(self, distributed_config))
        else:
            from fast_llm.layers.attention.preprocessing import BackupAttentionPreprocessor

            preprocessors.append(BackupAttentionPreprocessor(self, distributed_config))
        return preprocessors
