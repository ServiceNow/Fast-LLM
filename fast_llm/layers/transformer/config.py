import enum
import functools
import logging
import math
import typing
import warnings

from fast_llm.config import Field, FieldHint, check_field, config_class, skip_valid_if_none
from fast_llm.engine.base_model.config import BaseModelConfig
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.engine.config_utils.tensor_space import CompositeTensorDim, TensorDim, TensorSpace
from fast_llm.engine.distributed.config import DistributedConfig, DistributedDimNames
from fast_llm.functional.config import TritonConfig
from fast_llm.layers.common.config import (
    BaseBlockConfig,
    BaseBlockLoRAConfig,
    BaseBlockSubLayerName,
    LLMDimNames,
    PeftConfig,
)
from fast_llm.utils import Assert, div

if typing.TYPE_CHECKING:
    pass

    from fast_llm.layers.common.linear import LinearBase, LinearLike

logger = logging.getLogger(__name__)


class TransformerDimNames(LLMDimNames):
    # Self-attention dimensions
    head_groups = "head_groups"
    group_heads = "group_heads"
    key_and_value = "key_value"
    kv_channels = "kv_channels"
    composite_heads = "composite_heads"
    composite_query = "composite_query"
    composite_key_value = "composite_key_value"
    composite_dense = "composite_dense"


class TransformerKwargs:
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
    sequence_first = "sequence_first"
    hidden_dims = "hidden_dims"
    sequence_q_dim = "sequence_q_dim"
    sequence_k_dim = "sequence_k_dim"
    sequence_length = "sequence_length"
    # TODO: Move
    grad_output = "grad_output"


class TransformerLossNames:
    load_balancing_loss = "load_balancing_loss"
    router_z_loss = "router_z_loss"


class RotaryEmbeddingType(str, enum.Enum):
    none = "none"
    default = "default"
    llama3 = "llama3"
    yarn = "yarn"


@config_class(registry=True)
class RotaryConfig(BaseModelConfig):
    _abstract = False
    type: RotaryEmbeddingType = Field(
        default=RotaryEmbeddingType.none,
        desc="The type of rotary embedding to use. Choices: none, default, llama3.",
        hint=FieldHint.architecture,
    )
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
    # TODO: These are not really architecture parameters, but we want to import them from huggingface.
    scale_factor: float = Field(
        default=8.0, desc="Scaling factor for llama3-type scaling.", hint=FieldHint.architecture
    )
    low_frequency_factor: float = Field(
        default=1.0, desc="Low frequency factor for llama3-type scaling.", hint=FieldHint.feature
    )
    high_frequency_factor: float = Field(
        default=4.0, desc="High frequency factor for llama3-type scaling.", hint=FieldHint.feature
    )
    original_context_length: int = Field(
        default=8192, desc="Original context length for llama3/yarn-type scaling.", hint=FieldHint.feature
    )
    attention_factor: None | float = Field(
        default=None,
        desc="Attention factor for yarn-type scaling.",
        hint=FieldHint.feature,
    )
    beta_fast: float = Field(
        default=32.0,
        desc="Beta-fast for yarn-type scaling.",
        hint=FieldHint.feature,
    )
    beta_slow: float = Field(
        default=1.0,
        desc="Beta-slow for yarn-type scaling.",
        hint=FieldHint.feature,
    )

    @property
    def enabled(self) -> bool:
        return self.type != RotaryEmbeddingType.none

    @property
    def complex_format(self) -> bool:
        # TODO: Make a backup implementation that doesn't affect the layout.
        return self.enabled and not self.triton

    def _validate(self) -> None:
        super()._validate()
        if self.triton and not TritonConfig.TRITON_ENABLED:
            warnings.warn("Triton is disabled, but the triton rotary kernel will be used anyway.")


for name in RotaryEmbeddingType:
    # We need this because we are using the reserved field name `type`.
    # TODO: Implement proper dynamic typing.
    RotaryConfig.register_subclass(name.value, RotaryConfig)


class TransformerSubLayerName(BaseBlockSubLayerName):
    # TODO: Use this to replace AddLinearBiasChoices.
    query = "query"
    key = "key"
    value_ = "value"
    key_value = "key_value"
    dense = "dense"


@config_class(dynamic_type={PeftConfig: "transformer_lora"})
class TransformerLoRaConfig(BaseBlockLoRAConfig):
    """
    LoRa config that applies to transformer layer. If this is used with GPTBaseModel it is reused for all transformer layers.
    Note, this does not freeze layers.
    If you want to freeze weights, you need to do so explicitly by setting the corresponding layer's lr_scales (embeddings/mlp etc.) to 0.
    """

    layers: list[TransformerSubLayerName] = Field(
        default=None,
        desc="The layers on which to apply LoRA.",
        hint=FieldHint.feature,
    )

    def apply_linear(self, linear: "LinearBase", layer_type: TransformerSubLayerName | None = None) -> "LinearLike":
        if layer_type is None or self.layers is None or layer_type in self.layers:
            if layer_type == TransformerSubLayerName.key:
                return super().apply_linear(linear, out_channel_end=div(linear._out_dim.global_size, 2))
            elif layer_type == TransformerSubLayerName.value_:
                return super().apply_linear(linear, out_channel_begin=div(linear._out_dim.global_size, 2))
            else:
                return super().apply_linear(linear)
            # elif self.freeze_others:
            #     linear.weight.requires_grad = False
        return linear

    def _validate(self) -> None:
        if self.layers is None:
            with self._set_implicit_default():
                # Setting the default layers only whee PeFT is enabled
                # so they don't appear when serializing the default transformer config.
                self.layers = [TransformerSubLayerName.query, TransformerSubLayerName.value_]
        super()._validate()
        if TransformerSubLayerName.dense in self.layers:
            # TODO: Support InputParallelLinear (different output format).
            raise NotImplementedError("LoRA not supported for attention dense layer.")
        if (
            sum(
                name in self.layers
                for name in (
                    TransformerSubLayerName.key_value,
                    TransformerSubLayerName.key,
                    TransformerSubLayerName.value_,
                )
            )
            > 1
        ):
            raise ValueError(
                f"{TransformerSubLayerName.key_value.value}, {TransformerSubLayerName.key.value} and {TransformerSubLayerName.value_.value} are mutually exclusive."
            )


# for name in PeftType:
#     # We need this because we are using the reserved field name `type`.
#     # TODO: Implement proper dynamic typing.
#     TransformerPeftConfig.register_subclass(name.value, TransformerPeftConfig)


@config_class()
class TransformerConfig(BaseBlockConfig):
    _abstract = False
    # normalization: NormalizationConfig = Field(
    #     desc="Configuration for the normalization layers architecture.",
    #     hint=FieldHint.architecture,
    # )
    rotary: RotaryConfig = Field(
        desc="Configuration for the rotary positional embeddings.",
        hint=FieldHint.architecture,
    )
    # peft: PeftConfig = FieldUpdate(
    #     desc="Configuration for the parameter-efficient fine tuning.",
    #     hint=FieldHint.architecture,
    # )
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
    attention_dropout: float = Field(
        default=0.0,
        desc="Dropout applied to the attention intermediate states.",
        hint=FieldHint.feature,
        valid=check_field(Assert.geq, 0),
    )
    kv_channels: int = Field(
        default=None,
        desc="Number of key and value channels, i.e., hidden dimension of each attention head. Default: hidden_size // num_attention_heads",
        hint=FieldHint.architecture,
        valid=check_field(Assert.gt, 0),
    )
    num_attention_heads: int = Field(default=8, desc="Number of attention heads.", hint=FieldHint.architecture)
    head_groups: int = Field(
        default=1,
        desc="Number of head group for grouped query attention.",
        doc="Set to 1 for multi-query attention, `num_attention_heads` for multi-head.",
        hint=FieldHint.architecture,
        valid=check_field(Assert.gt, 0),
    )

    def _validate(self) -> None:
        with self._set_implicit_default():
            if self.kv_channels is None:
                self.kv_channels = div(self.hidden_size, self.num_attention_heads)
        Assert.multiple(self.num_attention_heads, self.head_groups)
        Assert.geq(self.attention_dropout, 0)
        super()._validate()

    #     with self._set_implicit_default():
    #         if self.ffn_hidden_size is None:
    #             self.ffn_hidden_size = 4 * self.hidden_size
    #         if self.kv_channels is None:
    #             self.kv_channels = div(self.hidden_size, self.num_attention_heads)
    #         if self.activation_type is None:
    #             self.activation_type = ActivationType.silu if self.gated else ActivationType.gelu
    #         if self.init_method_std is None:
    #             self.init_method_std = self.hidden_size**-0.5
    #         if self.init_method_std_qkv is None:
    #             self.init_method_std_qkv = self.init_method_std
    #         if self.init_method_std_attn_proj is None:
    #             self.init_method_std_attn_proj = self.init_method_std / max(2 * self.num_layers, 1) ** 0.5
    #         if self.init_method_std_mlp_1 is None:
    #             self.init_method_std_mlp_1 = self.init_method_std
    #         if self.init_method_std_mlp_2 is None:
    #             self.init_method_std_mlp_2 = self.init_method_std / max(2 * self.num_layers, 1) ** 0.5
    #         if self.init_method_max_qkv is None:
    #             self.init_method_max_qkv = self.init_method_max
    #         if self.init_method_min_qkv is None:
    #             self.init_method_min_qkv = self.init_method_min
    #         if self.init_method_max_attn_proj is None:
    #             self.init_method_max_attn_proj = self.init_method_max
    #         if self.init_method_min_attn_proj is None:
    #             self.init_method_min_attn_proj = self.init_method_min
    #         if self.init_method_max_mlp_1 is None:
    #             self.init_method_max_mlp_1 = self.init_method_max
    #         if self.init_method_min_mlp_1 is None:
    #             self.init_method_min_mlp_1 = self.init_method_min
    #         if self.init_method_max_mlp_2 is None:
    #             self.init_method_max_mlp_2 = self.init_method_max
    #         if self.init_method_min_mlp_2 is None:
    #             self.init_method_min_mlp_2 = self.init_method_min
    #         if self.init_method_min is not None and self.init_method_max is not None:
    #             Assert.leq(self.init_method_min, self.init_method_max)
    #         if self.init_method_min_qkv is not None and self.init_method_max_qkv is not None:
    #             Assert.leq(self.init_method_min, self.init_method_max)
    #         if self.init_method_min_qkv is not None and self.init_method_max_qkv is not None:
    #             Assert.leq(self.init_method_min_qkv, self.init_method_max_qkv)
    #         if self.init_method_min_attn_proj is not None and self.init_method_max_attn_proj is not None:
    #             Assert.leq(self.init_method_min_attn_proj, self.init_method_max_attn_proj)
    #         if self.init_method_min_mlp_1 is not None and self.init_method_max_mlp_1 is not None:
    #             Assert.leq(self.init_method_min_mlp_1, self.init_method_max_mlp_1)
    #         if self.init_method_min_mlp_2 is not None and self.init_method_max_mlp_2 is not None:
    #             Assert.leq(self.init_method_min_mlp_2, self.init_method_max_mlp_2)
    #     self.num_unshared_experts = self.num_experts - self.num_shared_experts

    #     super()._validate()

    #     # if not TritonConfig.TRITON_ENABLED:
    #     #     warnings.warn("Triton is disabled, but triton rotary kernel will be used anyway.")

    #     Assert.leq(self.num_shared_experts, self.num_experts)
    #     Assert.leq(self.num_shared_experts + self.num_experts_per_token, self.num_experts)
    #     Assert.multiple(self.num_attention_heads, self.head_groups)
    #     Assert.geq(self.attention_dropout, 0)

    @functools.cached_property
    def projection_size(self):
        assert self._validated
        return self.num_attention_heads * self.kv_channels

    # # @property
    # def add_mlp_bias(self) -> bool:
    #     if isinstance(self.add_linear_biases, bool):
    #         return self.add_linear_biases
    #     if self.add_linear_biases == AddLinearBiasChoices.everywhere:
    #         return True
    #     return False

    # @property
    # def add_attn_qkv_bias(self) -> bool:
    #     if isinstance(self.add_linear_biases, bool):
    #         return self.add_linear_biases
    #     if self.add_linear_biases == AddLinearBiasChoices.nowhere:
    #         return False
    #     return True

    # @property
    # def add_attn_dense_bias(self) -> bool:
    #     if isinstance(self.add_linear_biases, bool):
    #         return self.add_linear_biases
    #     if self.add_linear_biases == AddLinearBiasChoices.everywhere:
    #         return True
    #     return False

    @classmethod
    def _from_dict(
        cls,
        default: dict[str, typing.Any],
        strict: bool = True,
        flat: bool = False,
    ) -> typing.Self:
        # TODO v0.x: Remove backward compatibility.
        cls._handle_renamed_field(
            default,
            "use_rotary_embeddings",
            ("rotary", "type"),
            lambda x: RotaryEmbeddingType.default if x else RotaryEmbeddingType.none,
        )
        cls._handle_renamed_field(default, "rotary_embedding_scale", ("rotary", "theta"), lambda x: math.exp(-x))
        cls._handle_renamed_field(default, "triton_rotary", ("rotary", "triton"))
        return super()._from_dict(default, strict, flat)

    def setup_tensor_space(self, tensor_space: TensorSpace, block_name: str = "") -> None:
        super().setup_tensor_space(tensor_space, block_name)
        tensor = tensor_space.distributed_config.get_distributed_dim(DistributedDimNames.tensor)

        # Self-attention dimensions
        tensor_space.add_tensor_dim(
            head_groups := TensorDim(
                f"{TransformerDimNames.head_groups}_{block_name}",
                self.head_groups,
                tensor if self.head_groups > 1 else None,
            )
        )
        tensor_space.add_tensor_dim(
            group_heads := TensorDim(
                f"{TransformerDimNames.group_heads}_{block_name}",
                div(self.num_attention_heads, self.head_groups),
                None if self.head_groups > 1 else tensor,
            )
        )
        tensor_space.add_tensor_dim(key_and_value := TensorDim(f"{TransformerDimNames.key_and_value}_{block_name}", 2))
        tensor_space.add_tensor_dim(
            kv_channels := TensorDim(f"{TransformerDimNames.kv_channels}_{block_name}", self.kv_channels)
        )
        tensor_space.add_tensor_dim(
            CompositeTensorDim(f"{TransformerDimNames.composite_heads}_{block_name}", (head_groups, group_heads))
        )
        tensor_space.add_tensor_dim(
            CompositeTensorDim(
                f"{TransformerDimNames.composite_query}_{block_name}", (head_groups, group_heads, kv_channels)
            )
        )
        tensor_space.add_tensor_dim(
            CompositeTensorDim(
                f"{TransformerDimNames.composite_key_value}_{block_name}", (key_and_value, head_groups, kv_channels)
            )
        )
        tensor_space.add_tensor_dim(
            CompositeTensorDim(
                f"{TransformerDimNames.composite_dense}_{block_name}", (head_groups, group_heads, kv_channels)
            )
        )

    def do_use_flash_attention(self, distributed_config: DistributedConfig) -> bool:
        use_flash_attention = self.use_flash_attention and distributed_config.training_dtype in (
            DataType.float16,
            DataType.bfloat16,
        )

        # Config parameter `window_size` only can be used with flash attention
        if not use_flash_attention:
            Assert.is_(self.window_size, None)

        return use_flash_attention
