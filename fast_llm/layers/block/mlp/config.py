import enum
import typing

from fast_llm.config import Field, FieldHint, check_field, config_class, skip_valid_if_none
from fast_llm.engine.config_utils.initialization import init_normal_, init_zeros_
from fast_llm.engine.config_utils.tensor_space import CompositeTensorDim, TensorDim, TensorSpace
from fast_llm.engine.distributed.config import DistributedDimNames
from fast_llm.functional.config import ActivationType, MLPRecomputeLevel
from fast_llm.layers.block.config import BlockLayerConfig
from fast_llm.layers.common.linear.config import LinearConfig
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    from fast_llm.layers.block.mlp.mlp import MLPBase


class MLPDimNames:
    pass
    # MLP dimensions
    # mlp = "mlp"
    # gate_and_up = "gate_and_up"
    # composite_gated_mlp = "composite_gated_mlp"
    # experts = "experts"
    # top_experts = "top_experts"
    # shared_experts = "shared_experts"
    # unshared_experts = "unshared_experts"
    # composite_expert_mlp = "composite_expert_mlp"
    # composite_gated_expert_mlp = "composite_gated_expert_mlp"
    # composite_shared_expert_mlp = "composite_shared_expert_mlp"
    # composite_gated_shared_expert_mlp = "composite_gated_shared_expert_mlp"


class MLPLossNames:
    load_balancing_loss = "load_balancing_loss"
    router_z_loss = "router_z_loss"


class RoutingType(str, enum.Enum):
    topk = "aux_loss"
    sinkhorn = "sinkhorn"


@config_class(dynamic_type={BlockLayerConfig: "mlp"})
class MLPConfig(BlockLayerConfig):
    # TODO: Review names
    # TODO: Separate MoE?
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
    dropless_moe: bool = Field(
        default=True, desc="Evaluate all the experts at once using dropless MoE.", hint=FieldHint.expert
    )
    dropless_dynamic_shape: bool = Field(
        default=False,
        desc="Use a dynamic shape for dropless MLP instead of the worst-case value."
        " Reduces memory usage, but increases fragmentation and requires CPU synchronisation. Not recommended.",
        hint=FieldHint.expert,
    )
    layer_1: LinearConfig = Field(
        desc="Configuration for the first MLP layer.",
        hint=FieldHint.architecture,
    )
    layer_2: LinearConfig = Field(
        desc="Configuration for the second MLP layer.",
        hint=FieldHint.architecture,
    )
    router: LinearConfig = Field(
        # TODO: Improve default?
        desc="Configuration for the MoE router.",
        hint=FieldHint.feature,
    )

    @property
    def layer_class(self) -> "type[MLPBase]":
        if self.num_experts > 1:
            from fast_llm.layers.block.mlp.mixture_of_experts import MixtureOfExpertMLP

            return MixtureOfExpertMLP
        else:
            from fast_llm.layers.block.mlp.mlp import MLP

            return MLP

    def _validate(self) -> None:
        assert hasattr(self, "block")
        for layer, bias, scale in zip(
            (self.layer_1, self.layer_2, self.router),
            (self.block.add_linear_biases, self.block.add_linear_biases, False),
            (1, max(self.block.num_blocks, 1), 1),
        ):
            layer.default = LinearConfig(
                bias=bias,
                weight_initialization=init_normal_(0, (self.block.hidden_size * scale) ** -0.5),
                bias_initialization=init_zeros_,
                apply_peft=False,
            )

        with self._set_implicit_default():
            if self.activation_type is None:
                self.activation_type = ActivationType.silu if self.gated else ActivationType.gelu
            # TODO: `hidden_size` not yet validated.
            if self.ffn_hidden_size is None:
                self.ffn_hidden_size = 4 * self.block.hidden_size

        self.num_unshared_experts = self.num_experts - self.num_shared_experts

        super()._validate()

        Assert.leq(self.num_shared_experts, self.num_experts)
        Assert.leq(self.num_shared_experts + self.num_experts_per_token, self.num_experts)

        if isinstance(self.mlp_lr_scale, list):
            Assert.eq(len(self.mlp_lr_scale), self.num_experts)
            for scale in self.mlp_lr_scale:
                if scale is not None:
                    Assert.geq(scale, 0)
        elif self.mlp_lr_scale is not None:
            Assert.geq(self.mlp_lr_scale, 0)

    def setup_tensor_space(self, tensor_space: TensorSpace) -> None:
        tensor = tensor_space.distributed_config.get_distributed_dim(DistributedDimNames.tensor)

        # MLP dimensions
        tensor_space.add_tensor_dim(mlp := TensorDim(MLPDimNames.mlp, self.ffn_hidden_size, tensor))
        tensor_space.add_tensor_dim(gate_and_up := TensorDim(MLPDimNames.gate_and_up, 2 if self.gated else 1))
        tensor_space.add_tensor_dim(CompositeTensorDim(MLPDimNames.composite_gated_mlp, (gate_and_up, mlp)))
        tensor_space.add_tensor_dim(experts := TensorDim(MLPDimNames.experts, self.num_experts))
        tensor_space.add_tensor_dim(CompositeTensorDim(MLPDimNames.composite_expert_mlp, (experts, mlp)))
        tensor_space.add_tensor_dim(
            CompositeTensorDim(MLPDimNames.composite_gated_expert_mlp, (experts, gate_and_up, mlp))
        )
        tensor_space.add_tensor_dim(TensorDim(MLPDimNames.top_experts, self.num_experts_per_token))
        # composite_gated_expert_mlp
        # composite_expert_mlp
