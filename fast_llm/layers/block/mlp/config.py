import enum
import typing

from fast_llm.config import Config, Field, FieldHint, check_field, config_class
from fast_llm.engine.config_utils.initialization import FillInitializationConfig, NormalInitializationConfig
from fast_llm.functional.config import ActivationType, MLPRecomputeLevel
from fast_llm.layers.common.linear.config import AffineLinearConfig, LinearConfig
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
    # TODO: Review names
    # TODO: Separate MoE?
    _abstract = False
    # TODO: Configure experts separately?
    layer_1: AffineLinearConfig = Field(
        desc="Configuration for the first MLP layer.",
        hint=FieldHint.architecture,
    )
    # TODO: Separate gate and up
    layer_2: AffineLinearConfig = Field(
        desc="Configuration for the second MLP layer.",
        hint=FieldHint.architecture,
    )
    router: LinearConfig = Field(
        # TODO: Improve default?
        desc="Configuration for the MoE router.",
        hint=FieldHint.feature,
    )
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
        # TODO: Make this work without inheritance.
        for layer, bias, scale in zip(
            (self.layer_1, self.layer_2, self.router),
            (self.add_linear_biases, self.add_linear_biases, False),
            (1, max(self.num_layers, 1), 1),
        ):
            layer.default = AffineLinearConfig(
                bias=bias,
                weight_initialization=NormalInitializationConfig(std=(self.hidden_size * scale) ** -0.5),
                bias_initialization=FillInitializationConfig(value=0),
                apply_peft=False,
            )

        with self._set_implicit_default():
            if self.activation_type is None:
                self.activation_type = ActivationType.silu if self.gated else ActivationType.gelu
            if self.ffn_hidden_size is None:
                self.ffn_hidden_size = 4 * self.hidden_size

        self.num_unshared_experts = self.num_experts - self.num_shared_experts

        super()._validate()

        Assert.leq(self.num_shared_experts, self.num_experts)
        Assert.leq(self.num_shared_experts + self.num_experts_per_token, self.num_experts)
