import dataclasses
import logging
import typing

import torch

from fast_llm.core.ops import gather_op, reduce_op, reduce_scatter_op
from fast_llm.engine.base_model.base_model import FastLLMModule
from fast_llm.engine.config_utils.tensor_space import TensorDim
from fast_llm.functional.autograd import wrap_forward_backward
from fast_llm.functional.config import TritonConfig
from fast_llm.functional.triton.sparse_copy import SparseMap
from fast_llm.functional.triton.sparse_linear import (
    dense_matmul,
    input_inner_sparse_matmul,
    input_row_sparse_matmul,
    output_sparse_matmul,
)
from fast_llm.tensor import ParameterMeta, accumulate_gradient, init_zeros_, param_get_and_unset_is_zero

logger = logging.getLogger(__name__)


def maybe_transpose(tensor: torch.Tensor, transpose: bool) -> torch.Tensor:
    return tensor.t() if transpose else tensor


@dataclasses.dataclass
class LinearContext:
    # TODO: Check for memory leak
    input_: torch.Tensor | None
    sparse_map: SparseMap | None


class LinearLike(FastLLMModule):
    def __init__(self):
        super().__init__()
        self._forward = wrap_forward_backward(self.forward_only, self.backward)

    def forward(self, input_: torch.Tensor, sparse_map: SparseMap | None = None) -> torch.Tensor:
        return self._forward(input_, sparse_map)

    def forward_only(
        self, input_: torch.Tensor, sparse_map: SparseMap | None = None
    ) -> tuple[torch.Tensor, LinearContext]:
        raise NotImplementedError()

    def backward(self, grad_output: torch.Tensor, context: LinearContext) -> torch.Tensor:
        context, gather_handle = self.backward_gather_input(context)
        grad_input, reduce_handle = self.backward_activation(grad_output, context)
        if gather_handle is not None:
            gather_handle()
        self.backward_parameters(grad_output, context)
        if reduce_handle is not None:
            gather_handle()
        return grad_input

    def backward_gather_input(self, context: LinearContext) -> tuple[LinearContext, typing.Callable[[], None] | None]:
        return context, None

    def backward_activation(
        self, grad_output: torch.Tensor, context: LinearContext
    ) -> tuple[torch.Tensor, typing.Callable[[], None] | None]:
        raise NotImplementedError()

    def backward_parameters(self, grad_output: torch.Tensor, context: LinearContext) -> None:
        raise NotImplementedError()


class LinearBase(LinearLike):
    """
    A base module for linear layers holding weights and biases.
    """

    def __init__(
        self,
        in_dim: TensorDim,
        out_dim: TensorDim,
        *,
        bias=True,
        weight_init_method,
        bias_init_method=init_zeros_,
        transposed_weight: bool = False,
        auto_bias_grad_accumulation: bool = False,
        lr_scale: float | None | tuple[float | None, ...] = None,
    ):
        super().__init__()
        self._transposed_weight = transposed_weight
        self._in_dim = in_dim
        self._out_dim = out_dim
        self._lr_scale = lr_scale
        self._weight_init_method = weight_init_method
        self.weight = ParameterMeta.from_dims(
            (self._in_dim, self._out_dim) if self._transposed_weight else (self._out_dim, self._in_dim),
            init_method=weight_init_method,
            auto_grad_accumulation=False,
            lr_scale=lr_scale,
        )
        if bias:
            self.bias = ParameterMeta.from_dims(
                (self._out_dim,),
                init_method=bias_init_method,
                weight_decay=False,
                auto_grad_accumulation=auto_bias_grad_accumulation,
                lr_scale=lr_scale,
            )
        else:
            self.bias = None
        self._forward = wrap_forward_backward(self.forward_only, self.backward)

    @property
    def transposed_weight(self) -> bool:
        return self._transposed_weight

    def backward_parameters(self, grad_output: torch.Tensor, context: LinearContext) -> None:
        """
        Calculate the weight and bias gradients for a linear layer.
        TODO: fused_dense_cuda fuses weight gradient with bias gradient, but not with grad accumulation.
          Which one is best? (and can we fuse everything?)
        """

        grad_output = grad_output.flatten(0, -2)
        input_ = context.input_.flatten(0, -2)
        lhs, rhs = (input_.t(), grad_output) if self._transposed_weight else (grad_output.t(), input_)

        if not self.weight.requires_grad:
            pass
        elif TritonConfig.TRITON_LINEAR or context.sparse_map is not None:
            # This assumes the transposed_weight is True for input_sparse, False for output_sparse.
            input_row_sparse_matmul(
                lhs,
                rhs,
                context.sparse_map,
                out=weight.grad_buffer,  # noqa
                accumulate=not param_get_and_unset_is_zero(self.weight),
            )
        elif weight.grad_buffer.dtype == grad_output.dtype:  # noqa
            beta = 1 - param_get_and_unset_is_zero(self.weight)
            torch.addmm(
                weight.grad_buffer,  # noqa
                lhs,
                rhs,
                beta=beta,
                alpha=1,
                out=weight.grad_buffer,  # noqa
            )
        else:
            accumulate_gradient(self.weight, torch.mm(lhs, rhs))
        if self.bias is not None and self.bias.requires_grad:
            accumulate_gradient(self.bias, grad_output.sum(dim=0))


class Linear(LinearBase):
    """
    A basic linear layer without tensor parallelism.
    """

    def __init__(
        self,
        in_dim: TensorDim,
        out_dim: TensorDim,
        *,
        bias=True,
        weight_init_method,
        bias_init_method=init_zeros_,
        transposed_weight: bool = False,
        lr_scale: float | None | tuple[float | None, ...] = None,
    ):
        assert in_dim.parallel_dim is None
        assert out_dim.parallel_dim is None
        super().__init__(
            in_dim,
            out_dim,
            bias=bias,
            weight_init_method=weight_init_method,
            bias_init_method=bias_init_method,
            transposed_weight=transposed_weight,
            lr_scale=lr_scale,
        )

    def forward_only(
        self, input_: torch.Tensor, sparse_map: SparseMap | None = None
    ) -> tuple[torch.Tensor, LinearContext]:
        assert sparse_map is None

        # Matmul
        if TritonConfig.TRITON_LINEAR:
            assert self.bias is None
            output = dense_matmul(
                input_.flatten(0, -2),
                maybe_transpose(self.weight, not self._transposed_weight),
            ).unflatten(0, input_.shape[:-1])
        else:
            output = torch.nn.functional.linear(
                input_, maybe_transpose(self.weight, self._transposed_weight), self.bias
            )
        return output, LinearContext(input_, None)

    def backward_activation(
        self, grad_output: torch.Tensor, context: LinearContext
    ) -> tuple[torch.Tensor, typing.Callable[[], None] | None]:
        weight_t = maybe_transpose(self.weight, self._transposed_weight)

        # Input grad
        if TritonConfig.TRITON_LINEAR:
            grad_input = dense_matmul(grad_output.flatten(0, -2), weight_t).view(
                *grad_output.shape[:-1], weight_t.size(-1)
            )
        else:
            grad_input = grad_output.matmul(weight_t)

        return grad_input, None


class OutputParallelLinear(LinearBase):
    """
    A linear layer with output (column) tensor parallelism.
    """

    def __init__(
        self,
        in_dim: TensorDim,
        out_dim: TensorDim,
        *,
        bias=True,
        weight_init_method,
        bias_init_method=init_zeros_,
        transposed_weight: bool = False,
        sequence_parallel: bool = False,
        lr_scale: float | None | tuple[float | None, ...] = None,
    ):
        assert in_dim.parallel_dim is None
        self._group_size = 1 if out_dim.parallel_dim is None else out_dim.parallel_dim.size
        self._sequence_parallel = sequence_parallel and self._group_size > 1
        super().__init__(
            in_dim,
            out_dim,
            bias=bias,
            weight_init_method=weight_init_method,
            bias_init_method=bias_init_method,
            transposed_weight=transposed_weight,
            lr_scale=lr_scale,
        )

    def forward_only(
        self, input_: torch.Tensor, sparse_map: SparseMap | None = None
    ) -> tuple[torch.Tensor, LinearContext]:

        # Gather sequence-parallel slices (non-overlapped)
        input1 = gather_op(input_, self._out_dim.parallel_group, dim=0) if self._sequence_parallel else input_

        # Matmul
        if TritonConfig.TRITON_LINEAR or sparse_map is not None:
            assert self.bias is None
            if sparse_map is not None:
                assert not self._transposed_weight
            output = output_sparse_matmul(
                input1.flatten(0, -2),
                maybe_transpose(self.weight, not self._transposed_weight),
                sparse_map,
            ).unflatten(0, input_.shape[:-1])
        else:
            output = torch.nn.functional.linear(
                input1, maybe_transpose(self.weight, self._transposed_weight), self.bias
            )

        return output, LinearContext(input_, sparse_map)

    def backward_gather_input(self, context: LinearContext) -> tuple[LinearContext, typing.Callable[[], None] | None]:
        # Gather sequence-parallel slices (overlapped)
        if self._sequence_parallel:
            input_, gather_handle = gather_op(context.input_, self._out_dim.parallel_group, dim=0, async_op=True)
            context = dataclasses.replace(context, input_=input_)
        else:
            gather_handle = None
        return context, gather_handle

    def backward_activation(
        self, grad_output: torch.Tensor, context: LinearContext
    ) -> tuple[torch.Tensor, typing.Callable[[], None] | None]:
        weight_t = maybe_transpose(self.weight, self._transposed_weight)

        # Input grad
        if TritonConfig.TRITON_LINEAR or context.sparse_map is not None:
            grad_input = input_inner_sparse_matmul(grad_output.flatten(0, -2), weight_t, context.sparse_map).view(
                *grad_output.shape[:-1], weight_t.size(-1)
            )
        else:
            grad_input = grad_output.matmul(weight_t)

        # Reduce input grad (overlapped)
        grad_input, reduce_handle = (reduce_scatter_op if self._sequence_parallel else reduce_op)(
            grad_input, group=self._out_dim.parallel_group, async_op=True
        )

        return grad_input, reduce_handle


class InputParallelLinear(LinearBase):
    """
    A linear layer with input (row) tensor parallelism.
    """

    def __init__(
        self,
        in_dim: TensorDim,
        out_dim: TensorDim,
        *,
        bias=True,
        weight_init_method,
        bias_init_method=init_zeros_,
        sequence_parallel: bool = False,
        transposed_weight: bool = False,
        lr_scale: float | None | tuple[float | None, ...] = None,
    ):
        assert out_dim.parallel_dim is None
        self._group_size = 1 if in_dim.parallel_dim is None else in_dim.parallel_dim.size
        self._sequence_parallel = sequence_parallel and self._group_size > 1
        super().__init__(
            in_dim,
            out_dim,
            bias=bias,
            weight_init_method=weight_init_method,
            bias_init_method=bias_init_method,
            transposed_weight=transposed_weight,
            # Tensor-parallel bias is computed in _bias_dropout_grad.
            auto_bias_grad_accumulation=self._group_size > 1,
            lr_scale=lr_scale,
        )

    def forward_only(
        self, input_: torch.Tensor, sparse_map: SparseMap | None = None
    ) -> tuple[tuple[torch.Tensor, torch.Tensor | None], LinearContext]:
        # TODO: Fix signature
        # Matmul
        if TritonConfig.TRITON_LINEAR or sparse_map is not None:
            assert self.bias is None
            if sparse_map is not None:
                assert self._transposed_weight
            output = input_inner_sparse_matmul(
                input_.flatten(0, -2), maybe_transpose(self.weight, not self._transposed_weight), sparse_map
            ).unflatten(0, input_.shape[:-1])
        else:
            output = torch.nn.functional.linear(
                input_, maybe_transpose(self.weight, self._transposed_weight), self.bias
            )

        # Reduce input grad (non-overlapped)
        output = (reduce_scatter_op if self._sequence_parallel else reduce_op)(
            output, group=self._in_dim.parallel_group
        )
        return (output, self.bias if self._in_dim.parallel_group else None), LinearContext(input_, sparse_map)

    def backward_activation(
        self, grad_output: torch.Tensor, context: LinearContext
    ) -> tuple[torch.Tensor, typing.Callable[[], None] | None]:
        weight_t = maybe_transpose(self.weight, self._transposed_weight)

        # Gather sequence-parallel slices (non-overlapped)
        if self._sequence_parallel:
            grad_output = gather_op(grad_output, self._in_dim.parallel_group, dim=0)

        # Input grad
        if TritonConfig.TRITON_LINEAR or context.sparse_map is not None:
            grad_input = output_sparse_matmul(grad_output.flatten(0, -2), weight_t, context.sparse_map).view(
                *grad_output.shape[:-1], weight_t.size(-1)
            )
        else:
            grad_input = grad_output.matmul(weight_t)

        return grad_input, None
