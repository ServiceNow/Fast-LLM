import typing
import warnings

import torch

from fast_llm.core.distributed import ProcessGroup
from fast_llm.core.ops import gather_op
from fast_llm.engine.base_model.config import ResourceUsageConfig
from fast_llm.engine.config_utils.initialization import init_normal_, init_ones_
from fast_llm.engine.config_utils.tensor_dim import CompositeTensorDim, ConcatenatedTensorDim, TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.functional.triton import triton_available
from fast_llm.functional.triton.mlp import mlp_autograd, mlp_autograd_looped
from fast_llm.functional.triton.sparse_copy import get_sparse_map
from fast_llm.layers.block.config import BlockKwargs
from fast_llm.layers.common.peft.config import PeftConfig
from fast_llm.layers.decoder.mlp.config import Gemma4MoEMLPConfig, MoEImplementation
from fast_llm.layers.decoder.mlp.mlp import MLP
from fast_llm.tensor import TensorMeta
from fast_llm.utils import Assert


class Gemma4MoEMLP[ConfigType: Gemma4MoEMLPConfig](MLP[ConfigType]):
    """
    Gemma4 feedforward block with a dense MLP branch in parallel with routed experts.

    The dense branch uses the inherited `layer_1` and `layer_2`. The sparse branch
    uses Gemma4-specific router math and separate expert weights.
    """

    _config: ConfigType
    _group: ProcessGroup

    def __init__(
        self,
        config: ConfigType,
        distributed_config: DistributedConfig,
        *,
        hidden_dim: TensorDim,
        output_dim: TensorDim | None = None,
        lr_scale: float | None,
        peft: PeftConfig | None,
        return_bias: bool = True,
    ):
        Assert.gt(config.experts, 1)
        Assert.eq(config.shared_experts, 0)
        assert not config.add_linear_biases, "Biases not supported for Gemma4 MoE."
        super().__init__(
            config,
            distributed_config,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            lr_scale=lr_scale,
            peft=peft,
            return_bias=return_bias,
        )
        Assert.is_(self._output_dim, self._hidden_dim)

        self.post_feedforward_norm_1 = self._config.post_feedforward_norm_1.get_layer(
            self._hidden_dim, lr_scale=self._lr_scale, peft=self._peft
        )
        self.pre_feedforward_norm_2 = self._config.pre_feedforward_norm_2.get_layer(
            self._hidden_dim, lr_scale=self._lr_scale, peft=self._peft
        )
        self.post_feedforward_norm_2 = self._config.post_feedforward_norm_2.get_layer(
            self._hidden_dim, lr_scale=self._lr_scale, peft=self._peft
        )

        experts_dim = TensorDim("experts", self._config.experts)
        self._expert_intermediate_2_dim = TensorDim(
            "expert_intermediate", self._config.moe_intermediate_size, self._parallel_dim
        )
        if self._config.gated:
            expert_intermediate_1_dim = ConcatenatedTensorDim(
                "expert_gate_and_up", (self._expert_intermediate_2_dim, self._expert_intermediate_2_dim)
            )
        else:
            expert_intermediate_1_dim = self._expert_intermediate_2_dim
        expert_layer_1_dim = CompositeTensorDim("expert_layer_1", (experts_dim, expert_intermediate_1_dim))
        expert_layer_2_dim = CompositeTensorDim("expert_layer_2", (experts_dim, self._expert_intermediate_2_dim))

        self.router = self._config.router.get_layer(
            self._hidden_dim,
            experts_dim,
            default_weight_initialization=init_normal_(std=self._hidden_size**-0.5),
            lr_scale=self._lr_scale,
            peft=self._peft,
        )
        self.router_scale = self._config.router_scale.get_parameter(
            (self._hidden_dim,),
            default_initialization=init_ones_,
            lr_scale=self._lr_scale,
            peft=None,
        )
        self.per_expert_scale = self._config.per_expert_scale.get_parameter(
            (experts_dim,),
            default_initialization=init_ones_,
            lr_scale=self._lr_scale,
            peft=None,
        )

        self.expert_layer_1 = self._config.expert_layer_1.get_layer(
            self._hidden_dim,
            expert_layer_1_dim,
            default_weight_initialization=init_normal_(std=self._hidden_size**-0.5),
            default_add_bias=False,
            sequence_parallel=self._sequence_parallel,
            lr_scale=self._lr_scale,
            peft=self._peft,
        )
        self.expert_layer_2 = self._config.expert_layer_2.get_layer(
            expert_layer_2_dim,
            self._hidden_dim,
            default_weight_initialization=init_normal_(std=self._hidden_size**-0.5),
            default_add_bias=False,
            sequence_parallel=self._sequence_parallel,
            transposed_weight=True,
            lr_scale=self._lr_scale,
            peft=self._peft,
        )

        implementation = self._config.implementation
        if implementation == MoEImplementation.auto:
            implementation = MoEImplementation.dropless if triton_available else MoEImplementation.looped
        if implementation == MoEImplementation.dropless and not triton_available:
            warnings.warn("Dropless MoE not available without Triton, falling back to looped implementation.")
            implementation = MoEImplementation.looped
        self._expert_forward = (
            self._forward_dropless if implementation == MoEImplementation.dropless else self._forward_looped
        )
        self._top_expert_dim = TensorDim("top_experts", self._config.experts_per_token)

    def _forward(
        self,
        input_: torch.Tensor,
        kwargs: dict[str, typing.Any],
        losses: dict[str, typing.Any] | None = None,
        metrics: dict[str, typing.Any] | None = None,
    ) -> tuple[torch.Tensor, None]:
        if isinstance(input_, TensorMeta):
            return TensorMeta.from_dims(input_.dims[:-1] + (self._output_dim,), "MLP output"), None

        dense_out, dense_bias = super()._forward(input_, kwargs, losses, metrics)
        if dense_bias is not None:
            dense_out = dense_out + dense_bias
        dense_out = self.post_feedforward_norm_1(dense_out)

        residual = kwargs[BlockKwargs.pre_mlp_residual]
        hidden_states = residual.flatten(0, -2)

        router_in = torch.rms_norm(
            hidden_states.to(self.router_scale.dtype),
            (self._hidden_size,),
            None,
            self._config.router_norm_eps,
        )
        router_in = (router_in * self.router_scale * (self._hidden_size**-0.5)).type_as(hidden_states)
        logits = self.router(router_in)

        scores, top_experts = self._topk_routing(logits)
        sparse_in = self.pre_feedforward_norm_2(hidden_states)
        sparse_out = self._expert_forward(sparse_in, scores, top_experts).view_as(residual)
        sparse_out = self.post_feedforward_norm_2(sparse_out)

        out = dense_out + sparse_out
        self._debug(out, None, (kwargs.get(BlockKwargs.hidden_token_dim), self._hidden_dim), kwargs)
        return out, None

    def _topk_routing(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
        scores, top_experts = torch.topk(probs, k=self._config.experts_per_token, dim=-1)
        scores = scores / scores.sum(dim=-1, keepdim=True)
        scores = scores * self.per_expert_scale[top_experts]
        return scores.type_as(logits), top_experts

    def _forward_dropless(
        self, hidden_states: torch.Tensor, scores: torch.Tensor, top_experts: torch.Tensor
    ) -> torch.Tensor:
        if self._sequence_parallel:
            top_experts_for_map = gather_op(top_experts, self._parallel_dim.group, dim=0)
        else:
            top_experts_for_map = top_experts
        sparse_map = get_sparse_map(top_experts_for_map, self._config.experts, dynamic_shape=False)
        return mlp_autograd(
            hidden_states,
            scores,
            self.expert_layer_1.weight,
            None,
            self.expert_layer_2.weight,
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
            self.expert_layer_1.weight,
            self.expert_layer_2.weight,
            self._config.experts,
            self._config.gated,
            self._config.activation,
            self._parallel_dim.group,
            self._sequence_parallel,
            self.training,
            self._config.recompute_level,
        )

    def get_compute_usage(self, input_: TensorMeta, kwargs: dict[str, typing.Any], config: ResourceUsageConfig) -> int:
        token_dim, hidden_dim = input_.dims
        token_count = token_dim.global_size if config.global_ else token_dim.size
        hidden_size = hidden_dim.global_size if config.global_ else hidden_dim.size
        expert_intermediate_size = (
            self._expert_intermediate_2_dim.global_size if config.global_ else self._expert_intermediate_2_dim.size
        )
        routed_token_count = token_count * self._config.experts_per_token
        expert_layer_1_output_size = expert_intermediate_size * (2 if self._config.gated else 1)
        linear_compute_factor = 2 * (config.forward + 2 * config.backward)
        expert_layer_1_compute = (
            linear_compute_factor * routed_token_count * hidden_size * expert_layer_1_output_size
        )
        expert_layer_2_compute = linear_compute_factor * routed_token_count * expert_intermediate_size * hidden_size
        return (
            super().get_compute_usage(input_, kwargs, config)
            + expert_layer_1_compute
            + expert_layer_2_compute
            + self.router.get_compute_usage(input_, config)
        )
