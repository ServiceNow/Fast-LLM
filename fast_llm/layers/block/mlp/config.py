import enum
import typing

from fast_llm.config import Config, Field, FieldHint, check_field, config_class, skip_valid_if_none
from fast_llm.functional.config import ActivationType, MLPRecomputeLevel
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    pass


class MLPLossNames:
    load_balancing_loss = "load_balancing_loss"
    router_z_loss = "router_z_loss"


class RoutingType(str, enum.Enum):
    topk = "aux_loss"
    sinkhorn = "sinkhorn"


@config_class()
class MLPConfig(Config):
    # TODO: Review names    # TODO: Separate MoE?
    _abstract = False
    ffn_hidden_size: int = Field(
        default=None,
        desc="Hidden dimension of the MLP intermediate state. Default: 4 * hidden_size.",
        hint=FieldHint.architecture,
        valid=check_field(Assert.gt, 0),
    )
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
    gated: bool = Field(default=False, desc="Enable gated MLP.", hint=FieldHint.architecture)
    activation_type: ActivationType = Field(
        default=None,
        desc="The MLP intermediate activation type. Default: SiLU for gated MLP, GeLU otherwise.",
        hint=FieldHint.core,
    )
    # normalization_implementation: NormalizationImplementation = NormalizationImplementation.auto
    mlp_recompute_level: MLPRecomputeLevel = Field(
        default=MLPRecomputeLevel.none,
        desc="Set which of the MLP intermediate activations will be recomputed during the backward passes. This provides a trade-off between memory and speed.",
        hint=FieldHint.performance,
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
    mlp_lr_scale: float | None | tuple[float | None, ...] = Field(
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
    dropless_moe: bool = Field(
        default=True, desc="Evaluate all the experts at once using dropless MoE.", hint=FieldHint.expert
    )
    dropless_dynamic_shape: bool = Field(
        default=False,
        desc="Use a dynamic shape for dropless MLP instead of the worst-case value."
        " Reduces memory usage, but increases fragmentation and requires CPU synchronisation. Not recommended.",
        hint=FieldHint.expert,
    )
    # TODO: Review initialization
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

    @property
    def add_mlp_bias(self) -> bool:
        from fast_llm.layers.block.config import AddLinearBiasChoices

        # TODO: Make this work without inheritance.
        if isinstance(self.add_linear_biases, bool):
            return self.add_linear_biases
        if self.add_linear_biases == AddLinearBiasChoices.everywhere:
            return True
        return False

    def _validate(self) -> None:
        with self._set_implicit_default():
            if self.activation_type is None:
                self.activation_type = ActivationType.silu if self.gated else ActivationType.gelu
            # TODO: Make this work without inheritance.
            if self.ffn_hidden_size is None:
                self.ffn_hidden_size = 4 * self.hidden_size
            # TODO: Review initialization
            if self.init_method_std_mlp_1 is None:
                self.init_method_std_mlp_1 = self.init_method_std
            if self.init_method_std_mlp_2 is None:
                self.init_method_std_mlp_2 = self.init_method_std / max(2 * self.num_layers, 1) ** 0.5
            if self.init_method_max_mlp_1 is None:
                self.init_method_max_mlp_1 = self.init_method_max
            if self.init_method_min_mlp_1 is None:
                self.init_method_min_mlp_1 = self.init_method_min
            if self.init_method_max_mlp_2 is None:
                self.init_method_max_mlp_2 = self.init_method_max
            if self.init_method_min_mlp_2 is None:
                self.init_method_min_mlp_2 = self.init_method_min
            if self.init_method_min_mlp_1 is not None and self.init_method_max_mlp_1 is not None:
                Assert.leq(self.init_method_min_mlp_1, self.init_method_max_mlp_1)
            if self.init_method_min_mlp_2 is not None and self.init_method_max_mlp_2 is not None:
                Assert.leq(self.init_method_min_mlp_2, self.init_method_max_mlp_2)

        self.num_unshared_experts = self.num_experts - self.num_shared_experts

        super()._validate()

        Assert.leq(self.num_shared_experts, self.num_experts)
        Assert.leq(self.num_shared_experts + self.num_experts_per_token, self.num_experts)

        if isinstance(self.mlp_lr_scale, tuple):
            Assert.eq(len(self.mlp_lr_scale), self.num_experts)
            for scale in self.mlp_lr_scale:
                if scale is not None:
                    Assert.geq(scale, 0)
        elif self.mlp_lr_scale is not None:
            Assert.geq(self.mlp_lr_scale, 0)
