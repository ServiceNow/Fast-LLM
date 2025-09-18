import logging
import typing
import warnings

import torch

from fast_llm.core.distributed import ProcessGroup, set_generator
from fast_llm.engine.base_model.config import ResourceUsageConfig
from fast_llm.engine.config_utils.initialization import init_normal_
from fast_llm.engine.config_utils.tensor_dim import CompositeTensorDim, TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.functional.triton.mlp import mlp_autograd, mlp_autograd_looped
from fast_llm.functional.triton.sparse_copy import get_sparse_map
from fast_llm.layers.attention.config import AttentionKwargs
from fast_llm.layers.block.config import BlockKwargs
from fast_llm.layers.common.auxiliary_loss import AuxiliaryLoss, z_loss
from fast_llm.layers.common.peft.config import PeftConfig
from fast_llm.layers.decoder.mlp.config import MLPLossNames, MoEMLPConfig, RoutingType
from fast_llm.layers.decoder.mlp.mlp import MLPBase
from fast_llm.tensor import TensorMeta
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)


class MixtureOfExpertMLP[ConfigType: MoEMLPConfig](MLPBase[ConfigType]):
    """
    MoeLayer following implementation from
    https://github.com/NVIDIA/Megatron-LM/blob/46ebc0e4202c980d98900000d455f754a7ff9d4b/megatron/model/transformer.py#L346
    With custom routing implementation supporting both topk and sinkhorn routing

    TODO: Bias
    TODO: Sequence-tensor-parallel
    TODO: Expert parallel
    """

    _config: ConfigType
    _group: ProcessGroup

    def __init__(
        self,
        config: ConfigType,
        distributed_config: DistributedConfig,
        *,
        # TODO: Review `hidden_dim` and `block_index`
        hidden_dim: TensorDim,
        lr_scale: float | None,
        peft: PeftConfig | None,
    ):
        Assert.gt(config.experts, 1)
        # TODO: Implement?
        assert not config.add_linear_biases, "Biases not supported for MoE."
        super().__init__(
            config,
            distributed_config,
            hidden_dim=hidden_dim,
            lr_scale=lr_scale,
            peft=peft,
        )
        self.router = self._config.router.get_layer(
            self._hidden_dim,
            TensorDim("router_experts", self._config.unshared_experts),
            default_weight_initialization=init_normal_(std=self._hidden_size**-0.5),
            lr_scale=self._lr_scale,
            peft=self._peft,
        )
        dropless_moe = self._config.dropless
        if dropless_moe and self._sequence_parallel:
            warnings.warn(
                "Dropless MoE not supported for sequence-tensor-parallel, falling back to looped implementation."
            )
            dropless_moe = False
        self._mlp_forward = self._forward_dropless if dropless_moe else self._forward_looped

        if self._debug.enabled:
            self._top_expert_dim = TensorDim("top_experts", self._config.experts_per_token)

    def _get_intermediate_dims(self) -> tuple[TensorDim, TensorDim]:
        intermediate_1_dim, intermediate_2_dim = super()._get_intermediate_dims()
        experts_dim = TensorDim("experts", self._config.experts)
        return (
            CompositeTensorDim("moe_intermediate_1", (experts_dim, intermediate_1_dim)),
            CompositeTensorDim("moe_intermediate_2", (experts_dim, intermediate_2_dim)),
        )

    def forward(
        self, input_: torch.Tensor, kwargs: dict, losses: dict | None = None, metrics: dict | None = None
    ) -> torch.Tensor:
        hidden_states = input_.flatten(0, -2)
        logits = self.router(hidden_states)
        if self._debug.enabled:
            self._debug(
                logits, "Router logits", kwargs[BlockKwargs.hidden_dims][:-1] + (self._top_expert_dim,), kwargs
            )

        # Apply z_loss if applicable
        if self._config.z_loss_coefficient > 0.0:
            logits = z_loss(
                logits,
                self._config.z_loss_coefficient,
                self.training,
                grad_scale=kwargs.get("grad_output"),
                losses=losses,
                loss_name=MLPLossNames.router_z_loss,
            )

        # Apply input_jitter if applicable:
        if self.training and self._config.jitter_eps > 0.0:
            with set_generator(self._distributed.pp_generator):
                logits = self._apply_input_jitter(logits)

        # Routing
        if self._config.routing == RoutingType.topk:
            scores, top_experts = self._topk_routing(logits, kwargs.get(BlockKwargs.grad_output), losses)
            if self._config.shared_experts > 0:
                scores, top_experts = self._add_shared_experts(top_experts, scores)
        elif self._config.routing == RoutingType.sinkhorn:
            scores, top_experts = self._sinkhorn_routing(logits)
        else:
            raise NotImplementedError(self._config.routing)

        if self._debug.enabled:
            # To log all ranks set `global_=False`
            self._debug(
                scores, "Router scores", kwargs[BlockKwargs.hidden_dims][:-1] + (self._top_expert_dim,), kwargs
            )
            self._debug(
                top_experts,
                "Router top experts",
                kwargs[BlockKwargs.hidden_dims][:-1] + (self._top_expert_dim,),
                kwargs,
            )

        return self._mlp_forward(hidden_states, scores, top_experts).view_as(input_), None  # noqa

    def _forward_dropless(
        self, hidden_states: torch.Tensor, scores: torch.Tensor, top_experts: torch.Tensor
    ) -> torch.Tensor:
        # Compute token counts and the sparse mapping (dense_row, top_index) -> sparse_row.
        sparse_map = get_sparse_map(
            top_experts, self._config.experts, dynamic_shape=self._config.dropless_dynamic_shape
        )

        # Sparse MLP
        return mlp_autograd(
            hidden_states,
            scores,
            self.layer_1.weight,
            None,
            self.layer_2.weight,
            None,
            gated=self._config.gated,
            activation_type=self._config.activation,
            group=self._parallel_dim.group,
            sequence_parallel=self._sequence_parallel,
            training=self.training,
            recompute_level=self._config.recompute_level,
            transposed_layer_2_weight=True,
            sparse_map=sparse_map,
        )

    def _forward_looped(
        self, hidden_states: torch.Tensor, scores: torch.Tensor, top_experts: torch.Tensor
    ) -> torch.Tensor:
        return mlp_autograd_looped(
            hidden_states,
            scores,
            top_experts,
            self.layer_1.weight,
            self.layer_2.weight,
            self._config.experts,
            self._config.gated,
            self._config.activation,
            self._parallel_dim.group,
            self._sequence_parallel,
            self.training,
            self._config.recompute_level,
        )

    @torch.compile
    def _apply_input_jitter(self, logits: torch.Tensor) -> torch.Tensor:
        return logits * torch.empty_like(logits).uniform_(1.0 - self._config.jitter_eps, 1.0 + self._config.jitter_eps)

    def _topk_routing(
        self,
        logits: torch.Tensor,
        grad_scale: float | None = None,
        losses: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        top_logits, top_experts = torch.topk(logits, k=self._config.experts_per_token, dim=-1)
        scores = torch.softmax(top_logits, dim=-1, dtype=torch.float32)
        if losses is not None or (self.training and grad_scale is not None):
            probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
            mask = torch.nn.functional.one_hot(top_experts, num_classes=self._config.unshared_experts).sum(dim=1)
            # Auxiliary loss, corresponding to the sum of probabilities for the top experts.
            # In the optimal case (uniform distribution), loss = experts_per_token / num_experts.
            # In the worst case (whole distribution in the top experts), loss = 1.
            aux_loss = torch.sum(
                probs.flatten(0, -2).mean(dim=0) * mask.flatten(0, -2).mean(dim=0, dtype=torch.float32)
            )
            if losses is not None:
                losses[MLPLossNames.load_balancing_loss].append(aux_loss.detach())
            if self.training and grad_scale is not None:
                scores = AuxiliaryLoss.apply(
                    scores,
                    aux_loss,
                    self._config.unshared_experts * self._config.auxiliary_loss_coefficient * grad_scale,
                )
        return scores, top_experts

    def _add_shared_experts(
        self, scores: torch.Tensor, top_experts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Add the shared experts (last ones) to the top experts.
        shared_experts = torch.arange(
            self._config.unshared_experts,
            self._config.experts,
            device=top_experts.device,
            dtype=top_experts.dtype,
        )[None].repeat(top_experts.size(0), 1)
        top_experts = torch.cat((shared_experts, top_experts), dim=1)
        # Add scores of 1 to scores for shared experts.
        scores = torch.cat((scores.new_ones(scores.size(0), self._config.shared_experts), scores), dim=1)
        return scores, top_experts

    def _sinkhorn_routing(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training:
            _, top_experts = torch.topk(sinkhorn(logits), k=self._config.experts_per_token, dim=-1)
            logits = self._sinkhorn_activation(logits)
            scores = torch.gather(logits, -1, top_experts)
        else:
            logits = self._sinkhorn_activation(logits)
            scores, top_experts = torch.topk(logits, k=self._config.experts_per_token, dim=-1)
        return scores, top_experts

    def _sinkhorn_activation(self, logits: torch.Tensor) -> torch.Tensor:
        return (
            torch.sigmoid(logits)
            if self._config.experts_per_token == 1
            else torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
        )

    def get_compute_usage(self, input_: TensorMeta, kwargs: dict[str, typing.Any], config: ResourceUsageConfig) -> int:
        if kwargs[AttentionKwargs.sequence_first]:
            sequence_dim, batch_dim, hidden_dim = input_.dims
        else:
            batch_dim, sequence_dim, hidden_dim = input_.dims

        # Applying the tokens per expert on the batch dim so the super() call works as intended.
        moe_batch_dim = TensorDim(
            f"moe_{batch_dim.name}", batch_dim.global_size * self._config.experts_per_token, batch_dim.parallel_dim
        )

        if kwargs[AttentionKwargs.sequence_first]:
            dims = sequence_dim, moe_batch_dim, hidden_dim
        else:
            dims = moe_batch_dim, sequence_dim, hidden_dim

        # Also adjust the dtype in case of full-precision residual
        moe_input = TensorMeta.from_dims(dims, tensor_name=f"moe_{input_.tensor_name}", dtype=input_.dtype)

        return super().get_compute_usage(moe_input, kwargs, config) + self.router.get_compute_usage(input_, config)


def sinkhorn(cost: torch.Tensor, tolerance: float = 1e-5, eps=1e-9) -> torch.Tensor:
    """Sinkhorn based MoE routing function"""
    with torch.no_grad():
        cost = torch.exp(cost.to(torch.float32))
        d0 = torch.ones(cost.shape[:-1], device=cost.device, dtype=cost.dtype)
        d1 = torch.ones(cost.shape[-1:], device=cost.device, dtype=cost.dtype)

        error = eps**-1
        d1_old = d1
        while error > tolerance:
            d0 = d0.numel() ** -1 / (torch.sum(d1 * cost, -1) + eps)
            d1 = d1.numel() ** -1 / (torch.sum((d0.unsqueeze(-1) * cost).flatten(0, -2), 0) + eps)
            error = torch.mean(torch.abs(d1_old - d1))
            d1_old = d1
        return d1 * cost * d0.unsqueeze(-1)
