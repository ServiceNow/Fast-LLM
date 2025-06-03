import abc
import enum
import typing

from fast_llm.config import Field, FieldHint, check_field, config_class, skip_valid_if_none
from fast_llm.engine.base_model.config import BaseModelConfig
from fast_llm.engine.config_utils.tensor_space import CompositeTensorDim, TensorDim, TensorSpace
from fast_llm.engine.distributed.config import DistributedDimNames
from fast_llm.functional.config import ActivationType, MLPRecomputeLevel
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    from fast_llm.engine.config_utils.tensor_space import TensorDim
    from fast_llm.layers.common.linear import LinearBase, LinearLike
    from fast_llm.layers.common.normalization import LayerNorm, RMSNorm


class RotaryEmbeddingType(str, enum.Enum):
    none = "none"
    default = "default"
    llama3 = "llama3"
    yarn = "yarn"


class LLMDimNames:
    input_hidden = "input_hidden"
    output_hidden = "output_hidden"
    # A set of common tensor dim names packed into a namespace.
    # Input dimensions (variable)
    # TODO: Does batch belong here?
    batch = "batch"
    # TODO: Distinguish micro-sequence?
    sequence_q = "sequence_q"
    sequence_q_tp = "sequence_q_tp"
    sequence_k = "sequence_k"
    hidden = "hidden"
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


class NormalizationImplementation(str, enum.Enum):
    """
    An enum for the available implementations of layer norm.
    """

    auto = "auto"
    torch = "torch"
    fused = "fused"
    fast = "fast"
    triton = "triton"


class NormalizationType(str, enum.Enum):
    """
    An enum for the available normalization layers.
    TODO: Add no_norm type?
    """

    layer_norm = "layer_norm"
    rms_norm = "rms_norm"


@config_class(registry=True)
class NormalizationConfig(BaseModelConfig):
    _abstract = False

    # Normalization type
    type: NormalizationType = Field(
        default=NormalizationType.layer_norm,
        desc="The type of normalization to use, for example Layer Norm or RMS Norm.",
        hint=FieldHint.architecture,
    )
    # TODO: Rename to normalization_epsilon
    epsilon: float = Field(
        default=1e-5,
        desc="Regularizer for the division.",
        hint=FieldHint.architecture,
        valid=check_field(Assert.gt, 0),
    )
    zero_centered: bool = Field(
        default=False,
        desc="Write the normalization weight as `w = 1 + w'`, to improve numerical accuracy when close to one.",
        hint=FieldHint.architecture,
    )
    implementation: NormalizationImplementation = Field(
        default=NormalizationImplementation.auto,
        desc="The implementation to use for the normalization layer.",
        hint=FieldHint.performance,
    )
    # TODO: Rename to normalization_init_range
    initialization_range: float = Field(
        default=0.0,
        desc="Randomize the initialization with a uniform noise. Used to test for issues that may not be visible with the default initialization.",
        hint=FieldHint.testing,
        valid=check_field(Assert.geq, 0),
    )

    def get_layer(self, hidden_dim: "TensorDim", lr_scale: float | None = None) -> "LayerNorm | RMSNorm":
        from fast_llm.layers.common.normalization import LayerNorm, RMSNorm
        from fast_llm.tensor import init_uniform_

        kwargs = {
            "hidden_dim": hidden_dim,
            "eps": self.epsilon,
            "implementation": self.implementation,
            "zero_centered": self.zero_centered,
            "lr_scale": lr_scale,
        }
        if self.initialization_range:
            mean = 0 if self.zero_centered else 1
            kwargs["weight_init_method"] = init_uniform_(
                mean - self.initialization_range, mean + self.initialization_range
            )
        if self.type == NormalizationType.layer_norm:
            if self.initialization_range:
                kwargs["bias_init_method"] = init_uniform_(-self.initialization_range, self.initialization_range)
            return LayerNorm(**kwargs)
        elif self.type == NormalizationType.rms_norm:
            return RMSNorm(**kwargs)
        else:
            raise ValueError(self.type)

    @classmethod
    def _from_dict(
        cls,
        default: dict[str, typing.Any],
        strict: bool = True,
        flat: bool = False,
    ) -> typing.Self:
        cls._handle_renamed_field(default, "normalization_type", "type")
        cls._handle_renamed_field(default, "layer_norm_eps", "epsilon")
        cls._handle_renamed_field(default, "zero_centered_normalization", "zero_centered")
        cls._handle_renamed_field(default, "normalization_implementation", "implementation")
        cls._handle_renamed_field(default, "layer_norm_init_range", "initialization_range")
        return super()._from_dict(default, strict, flat)


for name in NormalizationType:
    # We need this because we are using the reserved field name `type`.
    # TODO: Implement proper dynamic typing.
    NormalizationConfig.register_subclass(name.value, NormalizationConfig)


class PeftType(str, enum.Enum):
    # TODO : Use a dynamic config type instead.
    none = "none"
    lora = "lora"


@config_class(registry=True)
class PeftConfig(BaseModelConfig):

    type: PeftType = Field(
        default=PeftType.none,
        desc="The type of parameter-efficient fine tuning to use Only LoRA is supported at the moment.",
        hint=FieldHint.core,
    )

    @abc.abstractmethod
    def apply_linear(self, linear: "LinearBase", **kwargs) -> "LinearLike":
        pass

    @classmethod
    def _from_dict(
        cls,
        default: dict[str, typing.Any],
        strict: bool = True,
        flat: bool = False,
    ) -> typing.Self:
        if cls is PeftConfig and cls.get_subclass(default.get("type")) is None:
            # Default subclass.
            return EmptyPeftConfig._from_dict(default, strict, flat)
        return super()._from_dict(default, strict=strict, flat=flat)


@config_class(dynamic_type={PeftConfig: "none"})
class EmptyPeftConfig(PeftConfig):
    """
    A dummy PeftConfig that does nothing.
    """

    _abstract = False

    def apply_linear(self, *args, **kwargs) -> "LinearLike":
        return args[0]


@config_class(dynamic_type={PeftConfig: "lora"})
class LoRAConfig(PeftConfig):
    """
    LoRA configuration.
    """

    _abstract = False
    rank: int = Field(
        default=8,
        desc="The LoRA rank, i.e. the size of the intermediate dimension.",
        hint=FieldHint.stability,
    )
    alpha: float = Field(
        default=8.0,
        desc="The LoRA scaling parameter.",
        hint=FieldHint.stability,
    )
    dropout: float = Field(
        default=0.0,
        desc="Dropout rate for LoRA.",
        hint=FieldHint.stability,
    )

    def apply_linear(self, linear: "LinearBase", **kwargs) -> "LinearLike":
        from fast_llm.layers.common.peft import lora_linear

        # TODO: Init method?
        return lora_linear(
            linear,
            linear.weight.param_init_method,
            linear.weight.param_init_method,
            self.rank,
            self.alpha,
            self.dropout,
            **kwargs,
        )


class RoutingType(str, enum.Enum):
    topk = "aux_loss"
    sinkhorn = "sinkhorn"


class AddLinearBiasChoices(str, enum.Enum):
    nowhere = "nowhere"
    everywhere = "everywhere"
    only_attn_qkv = "only_attn_qkv"


class BaseBlockSubLayerName:
    mlp_1 = "mlp_1"
    mlp_2 = "mlp_2"


@config_class(dynamic_type={PeftConfig: "base_lora"})
class BaseBlockLoRAConfig(LoRAConfig):
    """
    TODO: Add support for MLP.
    """

    _abstract = False

    layers: list[BaseBlockSubLayerName] = Field(
        default=None,
        desc="The layers on which to apply LoRA.",
        hint=FieldHint.feature,
    )

    def apply_linear(self, linear: "LinearBase", layer_type: BaseBlockSubLayerName | None = None) -> "LinearLike":
        if layer_type is None or self.layers is None or layer_type in self.layers:
            return super().apply_linear(linear)
        return linear

    def _validate(self) -> None:
        if self.layers is None:
            with self._set_implicit_default():
                self.layers = []
        if BaseBlockSubLayerName.mlp_1 in self.layers or BaseBlockSubLayerName.mlp_2 in self.layers:
            # TODO: Add MLP support.
            raise NotImplementedError("LoRA not supported for MLP.")


# for name in PeftType:
#     # We need this because we are using the reserved field name `type`.
#     # TODO: Implement proper dynamic typing.
#     BaseBlockPeftConfig.register_subclass(name.value, BaseBlockPeftConfig)


@config_class()
class BaseBlockConfig(BaseModelConfig):

    _abstract = False
    peft: PeftConfig = Field(
        # default_factory=lambda: PeftConfig(type=PeftType.none),
        desc="Configuration for the parameter-efficient fine tuning.",
        hint=FieldHint.architecture,
    )
    normalization: NormalizationConfig = Field(
        desc="Configuration for the normalization layers architecture.",
        hint=FieldHint.architecture,
    )
    hidden_dropout: float = Field(
        default=0.0,
        desc="Dropout applied to the residual connections.",
        hint=FieldHint.feature,
        valid=check_field(Assert.geq, 0),
    )
    debug_block: int = Field(
        default=0,
        desc="Log the output of each operation in each layer.",
        hint=FieldHint.logging,
        valid=check_field(Assert.geq, 0),
    )
    debug_block_memory: bool = Field(
        default=False,
        desc="Log the memory usage after each operation in each layer.",
        hint=FieldHint.logging,
    )
    num_experts: int = Field(
        default=1,
        desc="Number of MLP experts in a Mixture of Expert (MoE) model",
        hint=FieldHint.architecture,
        valid=check_field(Assert.gt, 0),
    )

    lr_scale: float = Field(
        default=1.0,
        desc="Custom learning rate scale for full block. note, ",
        doc="May be used to freeze some layers by setting their scale to zero. Note, in non-hybrid models (GPT model) all layers share same config and setting lr_scale to 0 will freeze all layers. Consider using norm_lr_scale, mlp_lr_scale etc. instead.",
        hint=FieldHint.feature,
    )

    norm_lr_scale: float | None | list[float | None] = Field(
        default=None,
        desc="Custom learning rate scale for each normalization layer.",
        doc="May be used to freeze some normalization layers by setting their scale to zero.",
        hint=FieldHint.feature,
    )
    mlp_lr_scale: float | None | list[float | None] = Field(
        default=None,
        desc="Custom learning rate scale for each expert.",
        doc="May be used to freeze some experts by setting their scale to zero.",
        hint=FieldHint.feature,
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
    ffn_hidden_size: int = Field(
        default=None,
        desc="Hidden dimension of the MLP intermediate state. Default: 4 * hidden_size.",
        hint=FieldHint.architecture,
        valid=check_field(Assert.gt, 0),
    )
    gated: bool = Field(default=False, desc="Enable gated MLP.", hint=FieldHint.architecture)
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
    # normalization_implementation: NormalizationImplementation = NormalizationImplementation.auto
    mlp_recompute_level: MLPRecomputeLevel = Field(
        default=MLPRecomputeLevel.none,
        desc="Set which of the MLP intermediate activations will be recomputed during the backward passes. This provides a trade-off between memory and speed.",
        hint=FieldHint.performance,
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
    router_lr_scale: float | None = Field(
        default=None,
        desc="Custom learning rate for the MoE router weight.",
        hint=FieldHint.feature,
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
    add_linear_biases: bool | AddLinearBiasChoices = Field(
        default=True,
        desc="Add biases to all, none or Q, K, V layers. Accepted values: True, False, or AddLinearBiasChoices.",
        hint=FieldHint.architecture,
    )

    def _validate(self) -> None:
        with self._set_implicit_default():
            if self.ffn_hidden_size is None:
                self.ffn_hidden_size = 4 * self.hidden_size
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
        Assert.geq(
            self.hidden_dropout, 0
        )  # Do we need to check it here again given that its is already asserted in the config field?
        if self.norm_lr_scale is not None:
            Assert.geq(self.norm_lr_scale, 0)

        if isinstance(self.mlp_lr_scale, list):
            Assert.eq(len(self.mlp_lr_scale), self.num_experts)
            for scale in self.mlp_lr_scale:
                if scale is not None:
                    Assert.geq(scale, 0)
        elif self.mlp_lr_scale is not None:
            Assert.geq(self.mlp_lr_scale, 0)
        super()._validate()
        Assert.leq(self.num_shared_experts, self.num_experts)
        Assert.leq(self.num_shared_experts + self.num_experts_per_token, self.num_experts)

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

    def setup_tensor_space(self, tensor_space: TensorSpace, block_name: str = "") -> None:
        tensor = tensor_space.distributed_config.get_distributed_dim(DistributedDimNames.tensor)

        # Hidden dimension
        tensor_space.add_tensor_dim(TensorDim(f"{LLMDimNames.hidden}_{block_name}", self.hidden_size))

        # MLP dimensions
        tensor_space.add_tensor_dim(mlp := TensorDim(f"{LLMDimNames.mlp}_{block_name}", self.ffn_hidden_size, tensor))
        tensor_space.add_tensor_dim(
            gate_and_up := TensorDim(f"{LLMDimNames.gate_and_up}_{block_name}", 2 if self.gated else 1)
        )
        tensor_space.add_tensor_dim(
            CompositeTensorDim(f"{LLMDimNames.composite_gated_mlp}_{block_name}", (gate_and_up, mlp))
        )
        tensor_space.add_tensor_dim(experts := TensorDim(f"{LLMDimNames.experts}_{block_name}", self.num_experts))
        tensor_space.add_tensor_dim(
            CompositeTensorDim(f"{LLMDimNames.composite_expert_mlp}_{block_name}", (experts, mlp))
        )
        tensor_space.add_tensor_dim(
            CompositeTensorDim(f"{LLMDimNames.composite_gated_expert_mlp}_{block_name}", (experts, gate_and_up, mlp))
        )
        tensor_space.add_tensor_dim(TensorDim(f"{LLMDimNames.top_experts}_{block_name}", self.num_experts_per_token))
        tensor_space.add_tensor_dim(
            TensorDim(f"{LLMDimNames.unshared_experts}_{block_name}", self.num_unshared_experts)
        )

        # shared_experts
        if self.num_shared_experts:
            tensor_space.add_tensor_dim(
                shared_experts := TensorDim(f"{LLMDimNames.shared_experts}_{block_name}", self.num_shared_experts)
            )
            tensor_space.add_tensor_dim(
                CompositeTensorDim(f"{LLMDimNames.composite_shared_expert_mlp}_{block_name}", (shared_experts, mlp))
            )
            tensor_space.add_tensor_dim(
                CompositeTensorDim(
                    f"{LLMDimNames.composite_gated_shared_expert_mlp}_{block_name}",
                    (shared_experts, gate_and_up, mlp),
                )
            )
