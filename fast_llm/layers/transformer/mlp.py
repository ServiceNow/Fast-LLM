import typing
from abc import ABC

import torch

from fast_llm.engine.base_model.base_model import Layer
from fast_llm.engine.config_utils.tensor_space import TensorSpace
from fast_llm.functional.config import TritonConfig
from fast_llm.functional.triton.mlp import mlp_autograd, torch_mlp_activation, triton_mlp_activation_autograd
from fast_llm.layers.common.linear import LinearBase
from fast_llm.layers.transformer.config import TransformerConfig, TransformerDimNames, TransformerSubLayerName
from fast_llm.tensor import init_normal_, init_zeros_
from fast_llm.utils import Assert


class MLPBase(Layer, ABC):
    def __init__(self, config: TransformerConfig, tensor_space: TensorSpace, name: str = "mlp"):
        super().__init__()
        self._name = name

        init_method_1 = init_normal_(
            std=config.init_method_std_mlp_1,
            min_val=config.init_method_min_mlp_1,
            max_val=config.init_method_max_mlp_1,
        )
        init_method_2 = init_normal_(
            std=config.init_method_std_mlp_2,
            min_val=config.init_method_min_mlp_2,
            max_val=config.init_method_max_mlp_2,
        )

        hidden_dim = tensor_space.get_tensor_dim(TransformerDimNames.hidden)
        self._intermediate_dim = tensor_space.get_tensor_dim(TransformerDimNames.composite_expert_mlp)
        self._sequence_parallel = tensor_space.distributed_config.sequence_tensor_parallel
        self._recompute_level = config.mlp_recompute_level

        self._gated = config.gated
        self._activation_type = config.activation_type
        self._activation_fn = triton_mlp_activation_autograd if TritonConfig.TRITON_ENABLED else torch_mlp_activation

        # So both layers' weights have shape (num_experts [* gate_up] * ffn, hidden_size)
        self.layer_1 = LinearBase(
            hidden_dim,
            tensor_space.get_tensor_dim(TransformerDimNames.composite_gated_expert_mlp),
            bias=config.add_mlp_bias,
            weight_init_method=init_method_1,
            bias_init_method=init_method_1 if config.random_bias_init else init_zeros_,
            lr_scale=tuple(config.mlp_lr_scale) if isinstance(config.mlp_lr_scale, list) else config.mlp_lr_scale,
        )
        self.layer_2 = LinearBase(
            self._intermediate_dim,
            hidden_dim,
            bias=config.add_mlp_bias,
            weight_init_method=init_method_2,
            bias_init_method=init_method_2 if config.random_bias_init else init_zeros_,
            auto_bias_grad_accumulation=tensor_space.distributed_config.tensor_parallel > 1,
            transposed_weight=True,
            lr_scale=tuple(config.mlp_lr_scale) if isinstance(config.mlp_lr_scale, list) else config.mlp_lr_scale,
        )

        # PEFT.
        self.layer_1 = config.peft.apply_linear(self.layer_1, TransformerSubLayerName.mlp_1)
        self.layer_2 = config.peft.apply_linear(self.layer_2, TransformerSubLayerName.mlp_2)


class MLP(MLPBase):
    def __init__(self, config: TransformerConfig, tensor_space: TensorSpace, name: str = "mlp"):
        Assert.eq(config.num_experts, 1)
        super().__init__(config, tensor_space, name)

    def forward(
        self,
        input_: torch.Tensor,
        kwargs: dict[str, typing.Any],
        losses: dict[str, typing.Any] | None = None,
        metrics: dict[str, typing.Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        parallel_group = self._intermediate_dim.parallel_group
        return (
            mlp_autograd(
                input_,
                None,
                self.layer_1.weight,
                self.layer_1.bias,
                self.layer_2.weight,
                None if parallel_group else self.layer_2.bias,
                gated=self._gated,
                activation_type=self._activation_type,
                group=parallel_group,
                sequence_parallel=self._sequence_parallel,
                training=self.training,
                recompute_level=self._recompute_level,
                transposed_layer_2_weight=self.layer_2.transposed_weight,
            ),
            self.layer_2.bias if parallel_group else None,
        )
