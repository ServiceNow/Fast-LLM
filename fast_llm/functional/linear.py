"""
Forward and backward pass of linear layers.
"""

import torch

from fast_llm.core.distributed import ProcessGroup
from fast_llm.core.ops import gather_op, reduce_op, reduce_scatter_op
from fast_llm.functional.autograd import wrap_forward_backward
from fast_llm.functional.config import TritonConfig
from fast_llm.functional.triton.sparse_copy import SparseMap
from fast_llm.functional.triton.sparse_linear import (
    dense_matmul,
    input_inner_sparse_matmul,
    input_row_sparse_matmul,
    output_sparse_matmul,
)
from fast_llm.tensor import accumulate_gradient, param_get_and_unset_is_zero


def maybe_transpose(tensor: torch.Tensor, transpose: bool):
    return tensor.t() if transpose else tensor


def update_linear_gradients(
    input_: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    grad_output: torch.Tensor,
    transposed_weight: bool,
    sparse_map: SparseMap | None,
):
    """
    Calculate the weight and bias gradients for a linear layer.
    TODO: fused_dense_cuda fuses weight gradient with bias gradient, but not with grad accumulation.
      Which one is best? (and can we fuse everything?)
    """

    grad_output = grad_output.flatten(0, -2)
    input_ = input_.flatten(0, -2)
    lhs, rhs = (input_.t(), grad_output) if transposed_weight else (grad_output.t(), input_)

    if TritonConfig.TRITON_LINEAR or sparse_map is not None:
        # This assumes the transposed_weight is True for input_sparse, False for output_sparse.
        input_row_sparse_matmul(
            lhs,
            rhs,
            sparse_map,
            out=weight.grad_buffer,  # noqa
            accumulate=not param_get_and_unset_is_zero(weight),
        )
    elif weight.grad_buffer.dtype == grad_output.dtype:  # noqa
        beta = 1 - param_get_and_unset_is_zero(weight)
        torch.addmm(
            weight.grad_buffer,  # noqa
            lhs,
            rhs,
            beta=beta,
            alpha=1,
            out=weight.grad_buffer,  # noqa
        )
    else:
        accumulate_gradient(weight, torch.mm(lhs, rhs))
    if bias is not None:
        accumulate_gradient(bias, grad_output.sum(dim=0))


def linear_forward(
    input_: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, transposed_weight: bool = False
):
    # Matmul
    if TritonConfig.TRITON_LINEAR:
        assert bias is None
        output = dense_matmul(
            input_.flatten(0, -2),
            maybe_transpose(weight, not transposed_weight),
        ).unflatten(0, input_.shape[:-1])
    else:
        output = torch.nn.functional.linear(input_, maybe_transpose(weight, transposed_weight), bias)
    return output, (input_, weight, bias, transposed_weight)


def linear_backward(grad_output: torch.Tensor, context: tuple):
    input_, weight, bias, transposed_weight = context
    weight_t = maybe_transpose(weight, transposed_weight)

    # Input grad
    if TritonConfig.TRITON_LINEAR:
        grad_input = dense_matmul(grad_output.flatten(0, -2), weight_t).view_as(input_)
    else:
        grad_input = grad_output.matmul(weight_t)

    # Parameter grad
    update_linear_gradients(input_, weight, bias, grad_output, transposed_weight, None)
    return grad_input


def output_parallel_linear_forward(
    input_: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    group: ProcessGroup | None,
    sequence_parallel: bool,
    transposed_weight: bool = False,
    sparse_map: SparseMap | None = None,
):
    # Gather sequence-parallel slices (non-overlapped)
    input1 = gather_op(input_, group, dim=0) if sequence_parallel else input_

    # Matmul
    if TritonConfig.TRITON_LINEAR or sparse_map is not None:
        assert bias is None
        if sparse_map is not None:
            assert not transposed_weight
        output = output_sparse_matmul(
            input1.flatten(0, -2),
            maybe_transpose(weight, not transposed_weight),
            sparse_map,
        ).unflatten(0, input_.shape[:-1])
    else:
        output = torch.nn.functional.linear(input1, maybe_transpose(weight, transposed_weight), bias)

    return output, (
        input_,
        weight,
        bias,
        group,
        sequence_parallel,
        transposed_weight,
        sparse_map,
    )


def output_parallel_linear_backward(grad_output: torch.Tensor, context: tuple):
    input_, weight, bias, group, sequence_parallel, transposed_weight, sparse_map = context
    weight_t = maybe_transpose(weight, transposed_weight)

    # Gather sequence-parallel slices (overlapped)
    if sequence_parallel:
        input_, gather_handle = gather_op(input_, group, dim=0, async_op=True)
    else:
        gather_handle = None

    # Input grad
    if TritonConfig.TRITON_LINEAR or sparse_map is not None:
        grad_input = input_inner_sparse_matmul(grad_output.flatten(0, -2), weight_t, sparse_map).view_as(input_)
    else:
        grad_input = grad_output.matmul(weight_t)

    # Reduce input grad (overlapped)
    grad_input, reduce_handle = (reduce_scatter_op if sequence_parallel else reduce_op)(
        grad_input, group=group, async_op=True
    )
    if sequence_parallel:
        gather_handle.wait()

    # Parameter grad
    update_linear_gradients(input_, weight, bias, grad_output, transposed_weight, sparse_map)

    if reduce_handle:
        reduce_handle.wait()
    return grad_input


def input_parallel_linear_forward(
    input_: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    group: ProcessGroup | None,
    sequence_parallel: bool,
    transposed_weight: bool = False,
    sparse_map: SparseMap | None = None,
):
    # Matmul
    if TritonConfig.TRITON_LINEAR or sparse_map is not None:
        assert bias is None
        if sparse_map is not None:
            assert transposed_weight
        output = input_inner_sparse_matmul(
            input_.flatten(0, -2), maybe_transpose(weight, not transposed_weight), sparse_map
        ).unflatten(0, input_.shape[:-1])
    else:
        output = torch.nn.functional.linear(input_, maybe_transpose(weight, transposed_weight), bias)

    # Reduce input grad (non-overlapped)
    output = (reduce_scatter_op if sequence_parallel else reduce_op)(output, group=group)
    return output, (
        input_,
        weight,
        bias,
        group,
        sequence_parallel,
        transposed_weight,
        sparse_map,
    )


def input_parallel_linear_backward(grad_output: torch.Tensor, context: tuple):
    input_, weight, bias, group, sequence_parallel, transposed_weight, sparse_map = context
    weight_t = maybe_transpose(weight, transposed_weight)

    # Gather sequence-parallel slices (non-overlapped)
    if sequence_parallel:
        grad_output = gather_op(grad_output, group, dim=0)

    # Input grad
    if TritonConfig.TRITON_LINEAR or sparse_map is not None:
        grad_input = output_sparse_matmul(grad_output.flatten(0, -2), weight_t, sparse_map).view_as(input_)
    else:
        grad_input = grad_output.matmul(weight_t)

    # Parameter grad
    update_linear_gradients(input_, weight, bias, grad_output, transposed_weight, sparse_map)

    return grad_input


linear_autograd = wrap_forward_backward(linear_forward, linear_backward)

output_parallel_linear_autograd = wrap_forward_backward(
    output_parallel_linear_forward, output_parallel_linear_backward
)

_input_parallel_linear = wrap_forward_backward(input_parallel_linear_forward, input_parallel_linear_backward)


def input_parallel_linear_autograd(
    input_: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    group: ProcessGroup | None,
    sequence_parallel: bool,
    transposed_weight: bool = False,
    sparse_map: SparseMap | None = None,
):
    # Autograd goes nuts it this goes in the function.
    return (
        _input_parallel_linear(
            input_,
            weight,
            None if group else bias,
            group,
            sequence_parallel,
            transposed_weight,
            sparse_map,
        ),
        bias if group else None,
    )
