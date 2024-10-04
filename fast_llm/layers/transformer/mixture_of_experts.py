import logging
import warnings

import torch

from fast_llm.core.distributed import ProcessGroup, set_generator
from fast_llm.engine.config_utils.run import log_pipeline_parallel_main_rank
from fast_llm.engine.config_utils.tensor_space import TensorSpace
from fast_llm.functional.triton.mlp import mlp_autograd, mlp_autograd_looped
from fast_llm.functional.triton.sparse_copy import get_sparse_map
from fast_llm.layers.common.auxiliary_loss import AuxiliaryLoss, z_loss
from fast_llm.layers.common.linear import Linear
from fast_llm.layers.transformer.config import (
    RoutingType,
    TransformerConfig,
    TransformerDimNames,
    TransformerKwargs,
    TransformerLossNames,
)
from fast_llm.layers.transformer.mlp import MLPBase
from fast_llm.logging import log_distributed_grad, log_distributed_tensor, log_memory_usage
from fast_llm.tensor import TensorMeta, init_normal_
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)


class MixtureOfExpertMLP(MLPBase):
    """
    MoeLayer following implementation from
    https://github.com/NVIDIA/Megatron-LM/blob/46ebc0e4202c980d98900000d455f754a7ff9d4b/megatron/model/transformer.py#L346
    With custom routing implementation supporting both topk and sinkhorn routing

    TODO: Merge with MLP?
    TODO: Bias
    TODO: Sequence-tensor-parallel
    TODO: Expert parallel
    """

    _group: ProcessGroup

    def __init__(self, config: TransformerConfig, tensor_space: TensorSpace, name: str = "mlp"):
        Assert.gt(config.num_experts, 1)
        # TODO: Implement?
        assert not config.add_linear_biases, "Biases not supported for MoE."
        super().__init__(config, tensor_space, name)
        self._config = config
        self._tensor_space = tensor_space
        self._debug_mode = self._config.debug_transformer or self._config.debug_transformer_memory

        self._num_experts = config.num_experts
        self._experts_per_token = config.num_experts_per_token
        self._num_shared_experts = config.num_shared_experts
        self._num_unshared_experts = config.num_unshared_experts

        self._routing_type = config.expert_routing_type
        self._load_balancing_factor = config.expert_auxiliary_loss_coefficient
        self._z_loss_factor = config.expert_z_loss_coefficient
        self._moe_jitter_eps = config.moe_jitter_eps

        self.router = Linear(
            tensor_space.get_tensor_dim(TransformerDimNames.hidden),
            tensor_space.get_tensor_dim(TransformerDimNames.unshared_experts),
            bias=False,
            weight_init_method=init_normal_(std=config.init_method_std),
            lr_scale=config.router_lr_scale,
        )
        dropless_moe = config.dropless_moe
        if dropless_moe and tensor_space.distributed_config.sequence_tensor_parallel:
            warnings.warn(
                "Dropless MoE not supported for sequence-tensor-parallel, falling back to looped implementation."
            )
            dropless_moe = False
        self._mlp_forward = self._forward_dropless if dropless_moe else self._forward_looped
        self._dynamic_shape = config.dropless_dynamic_shape

    def forward(self, input_: torch.Tensor, kwargs: dict, losses: dict | None = None, metrics: dict | None = None):
        hidden_states = input_.flatten(0, -2)
        logits = self.router(hidden_states)
        if self._debug_mode:
            self._debug_log(logits, "Router logits", TransformerDimNames.experts, kwargs)

        # Apply z_loss if applicable
        if self._z_loss_factor > 0.0:
            logits = z_loss(
                logits,
                self._z_loss_factor,
                self.training,
                grad_scale=kwargs.get("grad_output"),
                losses=losses,
                loss_name=TransformerLossNames.router_z_loss,
            )

        # Apply input_jitter if applicable:
        if self.training and self._moe_jitter_eps > 0.0:
            with set_generator(self._tensor_space.distributed.pp_generator):
                logits = self._apply_input_jitter(logits)

        # Routing
        if self._routing_type == RoutingType.topk:
            scores, top_experts = self._topk_routing(logits, kwargs.get(TransformerKwargs.grad_output), losses)
            if self._num_shared_experts > 0:
                scores, top_experts = self._add_shared_experts(top_experts, scores)
        elif self._routing_type == RoutingType.sinkhorn:
            scores, top_experts = self._sinkhorn_routing(logits)
        else:
            raise NotImplementedError(self._routing_type)

        if self._debug_mode:
            # To log all ranks set `global_=False`
            self._debug_log(scores, "Router scores", TransformerDimNames.top_experts, kwargs)
            self._debug_log(top_experts, "Router top experts", TransformerDimNames.top_experts, kwargs)

        return self._mlp_forward(hidden_states, scores, top_experts).view_as(input_), None  # noqa

    def _forward_dropless(self, hidden_states: torch.Tensor, scores: torch.Tensor, top_experts: torch.Tensor):
        # Compute token counts and the sparse mapping (dense_row, top_index) -> sparse_row.
        sparse_map = get_sparse_map(top_experts, self._num_experts, dynamic_shape=self._dynamic_shape)

        # Sparse MLP
        return mlp_autograd(
            hidden_states,
            scores,
            self.layer_1.weight,
            None,
            self.layer_2.weight,
            None,
            gated=self._gated,
            activation_type=self._activation_type,
            group=self._intermediate_dim.parallel_group,
            sequence_parallel=self._sequence_parallel,
            training=self.training,
            recompute_level=self._recompute_level,
            transposed_layer_2_weight=True,
            sparse_map=sparse_map,
        )

    def _forward_looped(self, hidden_states: torch.Tensor, scores: torch.Tensor, top_experts: torch.Tensor):
        return mlp_autograd_looped(
            hidden_states,
            scores,
            top_experts,
            self.layer_1.weight,
            self.layer_2.weight,
            self._num_experts,
            self._gated,
            self._activation_type,
            self._intermediate_dim.parallel_group,
            self._sequence_parallel,
            self.training,
            self._recompute_level,
        )

    @torch.compile
    def _apply_input_jitter(self, logits: torch.Tensor) -> torch.Tensor:
        return logits * torch.empty_like(logits).uniform_(1.0 - self._moe_jitter_eps, 1.0 + self._moe_jitter_eps)

    def _topk_routing(
        self,
        logits: torch.Tensor,
        grad_scale: float | None = None,
        losses: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        top_logits, top_experts = torch.topk(logits, k=self._experts_per_token, dim=-1)
        scores = torch.softmax(top_logits, dim=-1, dtype=torch.float32)
        if losses is not None or (self.training and grad_scale is not None):
            probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
            mask = torch.nn.functional.one_hot(top_experts, num_classes=self._num_unshared_experts).sum(dim=1)
            # Auxiliary loss, corresponding to the sum of probabilities for the top experts.
            # In the optimal case (uniform distribution), loss = experts_per_token / num_experts.
            # In the worst case (whole distribution in the top experts), loss = 1.
            aux_loss = torch.sum(
                probs.flatten(0, -2).mean(dim=0) * mask.flatten(0, -2).mean(dim=0, dtype=torch.float32)
            )
            if losses is not None:
                losses[TransformerLossNames.load_balancing_loss].append(aux_loss.detach())
            if self.training and grad_scale is not None:
                scores = AuxiliaryLoss.apply(
                    scores, aux_loss, self._num_unshared_experts * self._load_balancing_factor * grad_scale
                )
        return scores, top_experts

    def _add_shared_experts(
        self, scores: torch.Tensor, top_experts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Add the shared experts (last ones) to the top experts.
        shared_experts = torch.arange(
            self._num_unshared_experts, self._num_experts, device=top_experts.device, dtype=top_experts.dtype
        )[None].repeat(top_experts.size(0), 1)
        top_experts = torch.cat((shared_experts, top_experts), dim=1)
        # Add scores of 1 to scores for shared experts.
        scores = torch.cat((scores.new_ones(scores.size(0), self._num_shared_experts), scores), dim=1)
        return scores, top_experts

    def _sinkhorn_routing(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training:
            _, top_experts = torch.topk(sinkhorn(logits), k=self._experts_per_token, dim=-1)
            logits = self._sinkhorn_activation(logits)
            scores = torch.gather(logits, -1, top_experts)
        else:
            logits = self._sinkhorn_activation(logits)
            scores, top_experts = torch.topk(logits, k=self._experts_per_token, dim=-1)
        return scores, top_experts

    def _sinkhorn_activation(self, logits: torch.Tensor) -> torch.Tensor:
        return (
            torch.sigmoid(logits)
            if self._experts_per_token == 1
            else torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
        )

    def _debug_log(
        self,
        tensor: torch.Tensor | None,
        name: str,
        dim_name: str,
        kwargs: dict,
        global_: bool = True,
    ):
        if self._config.debug_transformer_memory:
            log_pipeline_parallel_main_rank(lambda: log_memory_usage(f"{self._name} {name}", str))
        if self._config.debug_transformer and tensor is not None:
            # TODO: Local vs global
            meta = self._get_meta(tensor, name, dim_name, kwargs)
            log_distributed_tensor(
                "",
                tensor.view_as(meta),
                level=self._config.debug_transformer,
                meta=meta,
                distributed=self._tensor_space.distributed,
                global_=global_,
            )
            if tensor.requires_grad:
                log_distributed_grad(
                    "",
                    tensor,
                    level=self._config.debug_transformer,
                    meta=self._get_meta(tensor, name + " grad", dim_name, kwargs),
                    distributed=self._tensor_space.distributed,
                    grad_fn=lambda tensor_: tensor_.view_as(meta),
                    global_=global_,
                )

    def _get_meta(self, tensor: torch.Tensor, name: str, dim_name: str, kwargs: dict):
        return TensorMeta.from_dims(
            kwargs[TransformerKwargs.hidden_dims][:-1] + (self._tensor_space.get_tensor_dim(dim_name),),
            tensor_name=f"{self._name} {name}",
            dtype=tensor.dtype,
        )


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
