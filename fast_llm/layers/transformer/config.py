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
from fast_llm.functional.config import ActivationType, MLPRecomputeLevel, TritonConfig
from fast_llm.layers.common.config import NormalizationConfig, PeftConfig, PeftType
from fast_llm.utils import Assert, div

if typing.TYPE_CHECKING:
    import torch

    from fast_llm.layers.common.linear import LinearBase, LinearLike
    from fast_llm.tensor import ParameterMeta

logger = logging.getLogger(__name__)


class RoutingType(str, enum.Enum):
    topk = "aux_loss"
    sinkhorn = "sinkhorn"


class TransformerDimNames:
    # A set of common tensor dim names packed into a namespace.
    # Input dimensions (variable)
    # TODO: Does batch belong here?
    batch = "batch"
    # TODO: Distinguish micro-sequence?
    sequence_q = "sequence_q"
    sequence_q_tp = "sequence_q_tp"
    sequence_k = "sequence_k"
    hidden = "hidden"
    # Self-attention dimensions
    head_groups = "head_groups"
    group_heads = "group_heads"
    key_and_value = "key_value"
    kv_channels = "kv_channels"
    composite_heads = "composite_heads"
    composite_query = "composite_query"
    composite_key_value = "composite_key_value"
    composite_dense = "composite_dense"
    # MLP dimensions
    mlp = "mlp"
    gate_and_up = "gate_and_up"
    composite_gated_mlp = "composite_gated_mlp"
    experts = "experts"
    top_experts = "top_experts"
    shared_experts = "shared_experts"
    unshared_experts = "unshared_experts"
    composite_expert_mlp = "composite_expert_mlp"
    composite_gated_expert_mlp = "composite_gated_expert_mlp"
    composite_shared_expert_mlp = "composite_shared_expert_mlp"
    composite_gated_shared_expert_mlp = "composite_gated_shared_expert_mlp"


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


@config_class()
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


class AddLinearBiasChoices(str, enum.Enum):
    nowhere = "nowhere"
    everywhere = "everywhere"
    only_attn_qkv = "only_attn_qkv"


class TransformerSubLayerName(str, enum.Enum):
    # TODO: Use this to replace AddLinearBiasChoices.
    query = "query"
    key = "key"
    value_ = "value"
    key_value = "key_value"
    dense = "dense"
    mlp_1 = "mlp_1"
    mlp_2 = "mlp_2"


@config_class()
class TransformerPeftConfig(PeftConfig):
    layers: list[TransformerSubLayerName] = Field(
        default=None,
        desc="The layers on which to apply LoRA.",
        hint=FieldHint.feature,
    )
    freeze_others: bool = Field(
        default=True,
        desc="Whether to freeze other layers during training.",
    )

    def apply_linear(self, linear: "LinearBase", layer_type: TransformerSubLayerName | None = None) -> "LinearLike":
        if self.type != PeftType.none:
            if layer_type is None or self.layers is None or layer_type in self.layers:
                if layer_type == TransformerSubLayerName.key:
                    return super().apply_linear(linear, out_channel_end=div(linear._out_dim.global_size, 2))
                elif layer_type == TransformerSubLayerName.value_:
                    return super().apply_linear(linear, out_channel_begin=div(linear._out_dim.global_size, 2))
                else:
                    return super().apply_linear(linear)
            elif self.freeze_others:
                linear.weight.requires_grad = False
        return linear

    def apply_other(self, module: "torch.nn.Module") -> "torch.nn.Module":
        if self.type != PeftType.none and self.freeze_others:
            for parameter in module.parameters():
                parameter.requires_grad = False
        return module

    def apply_weight(self, parameter: "ParameterMeta") -> "ParameterMeta":
        if self.type != PeftType.none and self.freeze_others:
            parameter.requires_grad = False
        return parameter

    def _validate(self) -> None:
        if self.layers is None:
            with self._set_implicit_default():
                # Setting the default layers only whee PeFT is enabled
                # so they don't appear when serializing the default transformer config.
                self.layers = (
                    [TransformerSubLayerName.query, TransformerSubLayerName.value_]
                    if self.type == PeftType.lora
                    else []
                )
        if self.type != PeftType.none:
            if TransformerSubLayerName.mlp_1 in self.layers or TransformerSubLayerName.mlp_2 in self.layers:
                # TODO: Add MLP support.
                raise NotImplementedError("LoRA not supported for MLP.")
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


@config_class()
class TransformerConfig(BaseModelConfig):
    _abstract = False
    normalization: NormalizationConfig = Field(
        default_factory=NormalizationConfig,
        desc="Configuration for the normalization layers architecture.",
        hint=FieldHint.architecture,
    )
    rotary: RotaryConfig = Field(
        default_factory=RotaryConfig,
        desc="Configuration for the rotary positional embeddings.",
        hint=FieldHint.architecture,
    )
    peft: TransformerPeftConfig = Field(
        default_factory=TransformerPeftConfig,
        desc="Configuration for the parameter-efficient fine tuning.",
        hint=FieldHint.architecture,
    )
    num_layers: int = Field(
        default=12,
        desc="Number of layers in the transformer.",
        hint=FieldHint.architecture,
        valid=check_field(Assert.geq, 0),
    )
    hidden_size: int = Field(
        default=1024,
        desc="Size of the transformer's main hidden dimension, e.g., for its input and output layers.",
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
    add_linear_biases: bool | AddLinearBiasChoices = Field(
        default=True,
        desc="Add biases to all, none or Q, K, V layers. Accepted values: True, False, or AddLinearBiasChoices.",
        hint=FieldHint.architecture,
    )
    ffn_hidden_size: int = Field(
        default=None,
        desc="Hidden dimension of the MLP intermediate state. Default: 4 * hidden_size.",
        hint=FieldHint.architecture,
        valid=check_field(Assert.gt, 0),
    )
    kv_channels: int = Field(
        default=None,
        desc="Number of key and value channels, i.e., hidden dimension of each attention head. Default: hidden_size // num_attention_heads",
        hint=FieldHint.architecture,
        valid=check_field(Assert.gt, 0),
    )
    gated: bool = Field(default=False, desc="Enable gated MLP.", hint=FieldHint.architecture)
    num_experts: int = Field(
        default=1,
        desc="Number of MLP experts in a Mixture of Expert (MoE) model",
        hint=FieldHint.architecture,
        valid=check_field(Assert.gt, 0),
    )
    num_shared_experts: int = Field(
        default=0,
        desc="Number of MLP experts that are shared between all tokens, i.e., always enabled.",
        hint=FieldHint.architecture,
        valid=check_field(Assert.geq, 0),
    )
    num_unshared_experts: int = Field(
        init=False,
        desc="Number of MLP experts excluding shared ones",
        hint=FieldHint.architecture,
        valid=check_field(Assert.geq, 0),
    )
    num_experts_per_token: int = Field(
        default=1,
        desc="Active experts for each token in a MoE model.",
        hint=FieldHint.architecture,
        valid=check_field(Assert.gt, 0),
    )
    expert_routing_type: RoutingType = Field(
        default=RoutingType.topk,
        desc="The routing method, i.e., the method used to assign experts to tokens.",
        hint=FieldHint.architecture,
    )
    activation_type: ActivationType = Field(
        default=None,
        desc="The MLP intermediate activation type. Default: SiLU for gated MLP, GeLU otherwise.",
        hint=FieldHint.core,
    )
    # Default: hidden_size**-0.5
    # TODO: Allow custom initialization (InitializationConfig?)
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
    attention_dropout: float = Field(
        default=0.0,
        desc="Dropout applied to the attention intermediate states.",
        hint=FieldHint.feature,
        valid=check_field(Assert.geq, 0),
    )
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
    # normalization_implementation: NormalizationImplementation = NormalizationImplementation.auto
    mlp_recompute_level: MLPRecomputeLevel = Field(
        default=MLPRecomputeLevel.none,
        desc="Set which of the MLP intermediate activations will be recomputed during the backward passes. This provides a trade-off between memory and speed.",
        hint=FieldHint.performance,
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
    # Use random inits instead of constant values, useful for debugging.
    random_bias_init: bool = Field(
        default=False,
        desc="Initialize the biases using the initialization method of their respective weights instead of setting them to zero. Used to test for issues that may not be visible when the biases are zero.",
        hint=FieldHint.testing,
    )
    expert_auxiliary_loss_coefficient: float = Field(
        default=0.01,
        desc="Scale of the load balancing auxiliary loss for topk routing.",
        hint=FieldHint.feature,
        valid=check_field(Assert.geq, 0),
    )
    expert_z_loss_coefficient: float = Field(
        default=0.0,
        desc="Regularize the router during training by applying Z-loss to the logits.",
        hint=FieldHint.feature,
        valid=check_field(Assert.geq, 0),
    )
    moe_jitter_eps: float = Field(
        default=0.0,
        desc="Regularize the router during training by applying a random multiplicative noise `uniform(1-eps, 1+eps)` to the logits.",
        hint=FieldHint.feature,
        valid=check_field(Assert.geq, 0),
    )
    mlp_lr_scale: float | None | list[float | None] = Field(
        default=None,
        desc="Custom learning rate scale for each expert.",
        doc="May be used to freeze some experts by setting their scale to zero.",
        hint=FieldHint.feature,
    )
    router_lr_scale: float | None = Field(
        default=None,
        desc="Custom learning rate for the MoE router weight.",
        hint=FieldHint.feature,
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
    dropless_moe: bool = Field(
        default=True, desc="Evaluate all the experts at once using dropless MoE.", hint=FieldHint.expert
    )
    dropless_dynamic_shape: bool = Field(
        default=False,
        desc="Use a dynamic shape for dropless MLP instead of the worst-case value."
        " Reduces memory usage, but increases fragmentation and requires CPU synchronisation. Not recommended.",
        hint=FieldHint.expert,
    )

    def _validate(self) -> None:
        with self._set_implicit_default():
            if self.ffn_hidden_size is None:
                self.ffn_hidden_size = 4 * self.hidden_size
            if self.kv_channels is None:
                self.kv_channels = div(self.hidden_size, self.num_attention_heads)
            if self.activation_type is None:
                self.activation_type = ActivationType.silu if self.gated else ActivationType.gelu
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
        self.num_unshared_experts = self.num_experts - self.num_shared_experts

        super()._validate()

        if not TritonConfig.TRITON_ENABLED:
            warnings.warn("Triton is disabled, but triton rotary kernel will be used anyway.")

        Assert.leq(self.num_shared_experts, self.num_experts)
        Assert.leq(self.num_shared_experts + self.num_experts_per_token, self.num_experts)
        Assert.multiple(self.num_attention_heads, self.head_groups)
        Assert.geq(self.attention_dropout, 0)
        Assert.geq(self.hidden_dropout, 0)

        if isinstance(self.mlp_lr_scale, list):
            Assert.eq(len(self.mlp_lr_scale), self.num_experts)
            for scale in self.mlp_lr_scale:
                if scale is not None:
                    Assert.geq(scale, 0)
        elif self.mlp_lr_scale is not None:
            Assert.geq(self.mlp_lr_scale, 0)

    @functools.cached_property
    def projection_size(self):
        assert self._validated
        return self.num_attention_heads * self.kv_channels

    @property
    def add_mlp_bias(self) -> bool:
        if isinstance(self.add_linear_biases, bool):
            return self.add_linear_biases
        if self.add_linear_biases == AddLinearBiasChoices.everywhere:
            return True
        return False

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

    def setup_tensor_space(self, tensor_space: TensorSpace) -> None:
        tensor = tensor_space.distributed_config.get_distributed_dim(DistributedDimNames.tensor)

        # Hidden dimension
        tensor_space.add_tensor_dim(TensorDim(TransformerDimNames.hidden, self.hidden_size))

        # Self-attention dimensions
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

        # MLP dimensions
        tensor_space.add_tensor_dim(mlp := TensorDim(TransformerDimNames.mlp, self.ffn_hidden_size, tensor))
        tensor_space.add_tensor_dim(gate_and_up := TensorDim(TransformerDimNames.gate_and_up, 2 if self.gated else 1))
        tensor_space.add_tensor_dim(CompositeTensorDim(TransformerDimNames.composite_gated_mlp, (gate_and_up, mlp)))
        tensor_space.add_tensor_dim(experts := TensorDim(TransformerDimNames.experts, self.num_experts))
        tensor_space.add_tensor_dim(CompositeTensorDim(TransformerDimNames.composite_expert_mlp, (experts, mlp)))
        tensor_space.add_tensor_dim(
            CompositeTensorDim(TransformerDimNames.composite_gated_expert_mlp, (experts, gate_and_up, mlp))
        )
        tensor_space.add_tensor_dim(TensorDim(TransformerDimNames.top_experts, self.num_experts_per_token))
        tensor_space.add_tensor_dim(TensorDim(TransformerDimNames.unshared_experts, self.num_unshared_experts))

        # shared_experts
        if self.num_shared_experts:
            tensor_space.add_tensor_dim(
                shared_experts := TensorDim(TransformerDimNames.shared_experts, self.num_shared_experts)
            )
            tensor_space.add_tensor_dim(
                CompositeTensorDim(TransformerDimNames.composite_shared_expert_mlp, (shared_experts, mlp))
            )
            tensor_space.add_tensor_dim(
                CompositeTensorDim(
                    TransformerDimNames.composite_gated_shared_expert_mlp, (shared_experts, gate_and_up, mlp)
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
