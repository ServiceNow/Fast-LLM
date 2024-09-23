"""
A triton implementation of some basic pointwise kernels.
These triton kernels tend to be much faster than their pytorch equivalent (observed up to ~2x on A100).
"""

import torch
import triton
from triton import language as tl

from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.functional.config import TritonConfig


@triton.jit
def triton_copy_kernel(
    input_ptr,
    out_ptr,
    numel: tl.constexpr,
    block_size: tl.constexpr,
):
    # TODO: Int64 ptr only if needed?
    block_start = tl.program_id(axis=0).to(tl.int64) * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < numel
    input_ = tl.load(input_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, input_, mask=mask)


def triton_copy(
    input_,
    out,
):
    """
    A triton implementation of tensor copying (`torch.Tensor.copy_()`).
    """
    if not TritonConfig.TRITON_ENABLED:
        return out.copy_(input_)
    # TODO: Improve assumptions.
    assert input_.is_contiguous()
    assert out.is_contiguous()
    numel = input_.numel()
    grid = lambda meta: (triton.cdiv(numel, meta["block_size"]),)
    triton_copy_kernel[grid](input_, out, numel, block_size=TritonConfig.POINTWISE_BLOCK_SIZE)
    return out


@triton.jit
def triton_fill_kernel(
    input_ptr,
    value: tl.constexpr,
    numel: tl.constexpr,
    dtype: tl.constexpr,
    block_size: tl.constexpr,
):
    # TODO: Int64 ptr only if needed?
    block_start = tl.program_id(axis=0).to(tl.int64) * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < numel
    tl.store(input_ptr + offsets, tl.full((block_size,), value, dtype), mask=mask)


def triton_fill(
    input_: torch.Tensor,
    value: float | int,
):
    """
    A faster triton implementation of tensor copying (`torch.Tensor.fill_()`).
    """
    if not TritonConfig.TRITON_ENABLED:
        return input_.fill_(value)
    # TODO: Improve assumptions.
    assert input_.is_contiguous()
    numel = input_.numel()
    grid = lambda meta: (triton.cdiv(numel, meta["block_size"]),)
    triton_fill_kernel[grid](
        input_,
        value,  # noqa
        numel,  # noqa
        DataType.from_torch(input_.dtype).triton,
        block_size=TritonConfig.POINTWISE_BLOCK_SIZE,  # noqa
    )
    return input_


@triton.jit
def triton_add_kernel(
    input_ptr,
    other_ptr,
    out_ptr,
    numel: tl.constexpr,
    block_size: tl.constexpr,
):
    # TODO: Int64 ptr only if needed?
    block_start = tl.program_id(axis=0).to(tl.int64) * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < numel
    input_ = tl.load(input_ptr + offsets, mask=mask)
    other = tl.load(other_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, input_ + other, mask=mask)


def triton_add(
    input_,
    other,
    out: torch.Tensor | None = None,
):
    """
    A faster triton implementation of tensor addition (`torch.Tensor.add()`).
    """
    if not TritonConfig.TRITON_ENABLED:
        return torch.add(input_, other, out=out)
    # TODO: Improve assumptions.
    assert input_.is_contiguous()
    assert other.is_contiguous()
    numel = input_.numel()
    if out is None:
        out = torch.empty_like(input_)
    grid = lambda meta: (triton.cdiv(numel, meta["block_size"]),)
    triton_add_kernel[grid](input_, other, out, numel, block_size=TritonConfig.POINTWISE_BLOCK_SIZE)
    return out
