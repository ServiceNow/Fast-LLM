import typing
from abc import ABC

import torch

from fast_llm.engine.base_model.base_model import Layer
from fast_llm.engine.config_utils.tensor_space import TensorSpace
from fast_llm.functional.config import TritonConfig
from fast_llm.functional.triton.mlp import (
    mlp_autograd,
    torch_mlp_activation,
    triton_mlp_activation_autograd,
    triton_mlp_activation_backward,
    triton_mlp_activation_forward,
)
from fast_llm.functional.triton.sparse_copy import (
    SparseMap,
    copy_dense_to_sparse_forward,
    copy_sparse_to_dense_backward,
    copy_sparse_to_dense_forward,
)
from fast_llm.layers.common.linear import LinearBase
from fast_llm.layers.transformer.config import TransformerConfig, TransformerDimNames
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
        self.layer_1 = self._config.peft.apply_linear(
            LinearBase(
                hidden_dim,
                tensor_space.get_tensor_dim(TransformerDimNames.composite_gated_expert_mlp),
                bias=config.add_mlp_bias,
                weight_init_method=init_method_1,
                bias_init_method=init_method_1 if config.random_bias_init else init_zeros_,
                lr_scale=tuple(config.mlp_lr_scale),
            )
        )
        self.layer_2 = self._config.peft.apply_linear(
            LinearBase(
                self._intermediate_dim,
                hidden_dim,
                bias=config.add_mlp_bias,
                weight_init_method=init_method_2,
                bias_init_method=init_method_2 if config.random_bias_init else init_zeros_,
                auto_bias_grad_accumulation=tensor_space.distributed_config.tensor_parallel > 1,
                transposed_weight=True,
                lr_scale=tuple(config.mlp_lr_scale),
            )
        )

    def forward_only(
        self,
        input_: torch.Tensor,
        scores: torch.Tensor | None,
        sparse_map: SparseMap | None = None,
    ) -> tuple[torch.Tensor, list[typing.Any] | None]:
        # Sparse copy
        input_shape = input_.shape
        intermediate_0 = input_ if sparse_map is None else copy_dense_to_sparse_forward(input_, sparse_map)[0]

        # Layer 1
        intermediate_1, _ = self.layer_1.forward_only(intermediate_0, sparse_map)

        if self._recompute_level.recompute_sparse_input:
            intermediate_0 = None
        else:
            input_ = None

        # Activation
        if TritonConfig.TRITON_ENABLED:
            intermediate_2, _ = triton_mlp_activation_forward(intermediate_1, self._gated, self._activation_type)
        else:
            do_grad = self.training and not self._recompute_level.recompute_activation
            with torch.set_grad_enabled(do_grad):
                intermediate_2 = torch_mlp_activation(
                    intermediate_1.detach().requires_grad_(do_grad), self._gated, self._activation_type
                )
        if self._recompute_level.recompute_layer_1:
            intermediate_1 = None

        # Layer 2
        intermediate_3, _ = self.layer_2.forward_only(intermediate_2, sparse_map)

        # Context
        if self._recompute_level.recompute_activation or not self.training:
            intermediate_2 = None

        # Sparse copy
        if sparse_map is None:
            output = intermediate_3
            intermediate_3 = None
        else:
            output, _ = copy_sparse_to_dense_forward(intermediate_3, scores, sparse_map)

        context = (
            [
                input_,
                scores,
                intermediate_0,
                intermediate_1,
                intermediate_2,
                intermediate_3,
                sparse_map,
                input_shape,
            ]
            if self.training
            else None
        )
        return output, context

    def backward(self, grad_output: torch.Tensor, context: typing.Any) -> torch.Tensor:
        (
            input_,
            scores,
            intermediate_0,
            intermediate_1,
            intermediate_2,
            intermediate_3,
            sparse_map,
            input_shape,
        ) = context
        context.clear()

        # Sparse copy
        if sparse_map is None:
            grad_scores = None
        else:
            grad_output, grad_scores = copy_sparse_to_dense_backward(grad_output, (sparse_map, intermediate_3, scores))

        grad_intermediate_2, handle = self.layer_2.backward_input(grad_output, ())

        # Sparse input recomputation
        if intermediate_0 is None:
            intermediate_0 = input_ if sparse_map is None else copy_dense_to_sparse_forward(input_, sparse_map)[0]

        # Layer 1 recomputation
        if intermediate_1 is None:
            intermediate_1 = self.layer_1.forward_only(intermediate_0, sparse_map)

        # Activation recomputation and/or backward
        if TritonConfig.TRITON_ENABLED:
            grad_intermediate_1, intermediate_2_ = triton_mlp_activation_backward(
                grad_intermediate_2, (intermediate_1, self._gated, self._activation_type), intermediate_2 is None
            )
        else:
            if intermediate_2 is None:
                with torch.set_grad_enabled(True):
                    intermediate_2_ = torch_mlp_activation(
                        intermediate_1.detach().requires_grad_(True), self._gated, self._activation_type
                    )
            else:
                intermediate_2_ = intermediate_2
            intermediate_2_.backward(grad_intermediate_2)
            grad_intermediate_1 = intermediate_1.grad

        # Layer 2 parameter grad
        del grad_intermediate_2, intermediate_1
        update_linear_gradients(
            intermediate_2_ if intermediate_2 is None else intermediate_2,
            weight_2,
            bias_2,
            grad_output,
            transposed_layer_2_weight,
            sparse_map,
        )
        del grad_output, intermediate_2, intermediate_2_

        # Layer 1 backward
        grad_input = output_parallel_linear_backward(
            grad_intermediate_1,
            (intermediate_0, weight_1, bias_1, group, sequence_parallel, False, sparse_map),
        )

        # Sparse copy
        if sparse_map is not None:
            grad_input = copy_dense_to_sparse_backward(grad_input, (sparse_map, input_shape))

        return grad_input, grad_scores


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
