"""
Advanced distributed ops, pytorch functions and associated utils used within fast llm.
Todo: Move all core methods elsewhere (functional?).
"""

import logging

import torch
import torch._dynamo  # noqa
import torch.autograd

from fast_llm.core.distributed import ProcessGroup, ReduceOp, all_gather_into_tensor, all_reduce, reduce_scatter_tensor
from fast_llm.utils import Assert, div

logger = logging.getLogger(__name__)


def reduce_op(input_, group: ProcessGroup | None, *, op: ReduceOp = ReduceOp.SUM, async_op: bool = False):
    if group:
        handle = all_reduce(input_, group=group, async_op=async_op, op=op)
    else:
        handle = None
    return (input_, handle) if async_op else input_


def split_op(input_, group: ProcessGroup | None, dim: int):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""
    if group:
        split_size = div(input_.size(dim), group.size())
        input_ = input_.split(split_size, dim=dim)[group.rank()]
    return input_


def insert_dim(shape, factor: int, dim: int):
    """Insert the specified dimension into a shape"""
    return *shape[:dim], factor, *shape[dim:]


def mult_dim(shape, factor: int, dim: int):
    """Multiply the specified dimension in a shape"""
    return *shape[:dim], shape[dim] * factor, *shape[dim + 1 :]


def div_dim(shape, factor: int, dim: int):
    """Divide the specified dimension in a shape"""
    return *shape[:dim], div(shape[dim], factor), *shape[dim + 1 :]


def swap_mult_dim(tensor: torch.Tensor, factor: int, old_dim: int, new_dim: int):
    """ "Split" a tensor along a specified dimension, then "concatenate" along another dimension."""
    base_shape = div_dim(tensor.shape, factor, old_dim)
    return (
        tensor.view(insert_dim(base_shape, factor, old_dim))
        .movedim(old_dim, new_dim)
        .reshape(mult_dim(base_shape, factor, new_dim))
    )


def gather_op(input_, group: ProcessGroup | None, dim: int, async_op: bool = False, out=None):
    """Gather tensors and concatenate along the last dimension."""
    # Bypass the function if we are using only 1 GPU.
    if not group:
        out = input_ if out is None else out.copy_(input_)
        return (out, None) if async_op else out
    # We can't use the output shape because all_gather expects contiguous tensors.
    if out is None:
        out = torch.empty(mult_dim(input_.shape, group.size(), 0), device=input_.device, dtype=input_.dtype)
    else:
        Assert.eq(dim, 0)
    handle = all_gather_into_tensor(out, input_.contiguous(), group=group, async_op=async_op)
    # Combine in the channel dimension
    if dim != 0:
        assert not async_op
        # TODO: contiguous?
        out = swap_mult_dim(out, group.size(), 0, dim)
    return (out, handle) if async_op else out


def reduce_scatter_op(
    input_, group: ProcessGroup | None, *, op: ReduceOp = ReduceOp.SUM, dim: int = 0, async_op: bool = False
):
    """Reduce-scatter the input tensor across model parallel group."""
    # Bypass the function if we are using only 1 GPU.
    if not group:
        return (input_, None) if async_op else input_
    output = torch.empty(div_dim(input_.shape, group.size(), dim), device=input_.device, dtype=input_.dtype)
    if dim != 0:
        input_ = swap_mult_dim(input_, group.size(), dim, 0)
    # TODO: May give the wrong output without the contiguous call.
    handle = reduce_scatter_tensor(output, input_.contiguous(), group=group, async_op=async_op, op=op)
    return (output, handle) if async_op else output


class _ReduceBackward(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def symbolic(graph, input_, group: ProcessGroup | None):  # noqa
        return input_

    @staticmethod
    def forward(ctx, input_, group: ProcessGroup | None):  # noqa
        ctx.group = group
        return input_

    @staticmethod
    def backward(ctx, grad_output):  # noqa
        return reduce_op(grad_output.contiguous(), ctx.group), None


class _ReduceForward(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_, group: ProcessGroup | None):  # noqa
        return reduce_op(input_, group)

    @staticmethod
    def forward(ctx, input_, group: ProcessGroup | None):  # noqa
        return reduce_op(input_, group)

    @staticmethod
    def backward(ctx, grad_output):  # noqa
        return grad_output, None


class _Split(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_, group: ProcessGroup | None, dim: int):  # noqa
        return split_op(input_, group, dim)

    @staticmethod
    def forward(ctx, input_, group: ProcessGroup | None, dim: int):  # noqa
        ctx.group = group
        ctx.dim = dim
        return split_op(input_, group, dim)

    @staticmethod
    def backward(ctx, grad_output):  # noqa
        return gather_op(grad_output, ctx.group, ctx.dim), None, None


class _Gather(torch.autograd.Function):
    """Gather the input from model parallel region and concatenate."""

    @staticmethod
    def symbolic(graph, input_, group: ProcessGroup | None, dim: int, reduce_grads: bool):  # noqa
        return gather_op(input_, group, dim)

    @staticmethod
    def forward(ctx, input_, group: ProcessGroup | None, dim: int, reduce_grads: bool):  # noqa
        ctx.group = group
        ctx.dim = dim
        ctx.reduce_grads = reduce_grads
        return gather_op(input_, group, dim)

    @staticmethod
    def backward(ctx, grad_output):  # noqa
        return (reduce_scatter_op if ctx.reduce_grads else split_op)(grad_output, ctx.group, ctx.dim), None, None, None


class _ReduceScatter(torch.autograd.Function):
    """Reduce scatter the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_, group: ProcessGroup | None, dim: int):  # noqa
        return reduce_scatter_op(input_, group, dim=dim)

    @staticmethod
    def forward(ctx, input_, group: ProcessGroup | None, dim: int):  # noqa
        ctx.group = group
        ctx.dim = dim
        return reduce_scatter_op(input_, group, dim=dim)

    @staticmethod
    def backward(ctx, grad_output):  # noqa
        return gather_op(grad_output, ctx.group, ctx.dim), None, None


# TODO: Torch compile currently doesn't work for pytorch functions with non-tensor inputs (still true?).
#   (no real optimization opportunity anyway, except maybe for unused swap_mult_dim)


@torch._dynamo.disable  # noqa
def reduce_forward(input_, group: ProcessGroup | None):
    return _ReduceForward.apply(input_, group)


@torch._dynamo.disable  # noqa
def reduce_backward(input_, group: ProcessGroup | None):
    return _ReduceBackward.apply(input_, group)


@torch._dynamo.disable  # noqa
def split(input_, group: ProcessGroup | None, dim: int):
    return _Split.apply(input_, group, dim)


@torch._dynamo.disable  # noqa
def gather(input_, group: ProcessGroup | None, dim: int, reduce_grads: bool = False):
    return _Gather.apply(input_, group, dim, reduce_grads)


@torch._dynamo.disable  # noqa
def reduce_scatter(input_, group: ProcessGroup | None, dim: int):
    return _ReduceScatter.apply(input_, group, dim)
