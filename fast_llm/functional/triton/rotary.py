import torch
import triton
from triton import language as tl

from fast_llm.functional.autograd import wrap_forward_backward
from fast_llm.functional.config import TritonConfig
from fast_llm.utils import div


@triton.jit
def triton_rotary_kernel(
    input_ptr,
    frequencies_ptr,
    stride_0,
    stride_1,
    stride_2,
    rotary_dim: tl.constexpr,
    num_heads: tl.constexpr,
    rotary_block_size: tl.constexpr,
    head_block_size: tl.constexpr,
    backward: tl.constexpr,
):
    # TODO: Int64 ptr if needed?
    pid_0 = tl.program_id(axis=0)
    pid_1 = tl.program_id(axis=1)
    pid_2 = tl.program_id(axis=2)

    offsets = tl.arange(0, rotary_block_size)
    head_offsets = pid_2 * head_block_size + tl.arange(0, head_block_size)[:, None]
    input_offsets = stride_0 * pid_0 + stride_1 * pid_1 + stride_2 * head_offsets + offsets[None, :]
    input_re_ptr = input_ptr + input_offsets
    input_im_ptr = input_re_ptr + rotary_dim

    if rotary_block_size % rotary_dim == 0 and num_heads % head_block_size == 0:
        input_re = tl.load(input_re_ptr).to(tl.float32)
        input_im = tl.load(input_im_ptr).to(tl.float32)
    else:
        mask = (offsets[None, :] < rotary_dim) & (head_offsets < num_heads)
        input_re = tl.load(input_re_ptr, mask=mask).to(tl.float32)
        input_im = tl.load(input_im_ptr, mask=mask).to(tl.float32)

    # Computing frequencies here is faster but hurts precision, so we load pre-computed ones instead.
    frequencies_offsets = 2 * rotary_dim * pid_1 + offsets
    frequencies_re_ptr = frequencies_ptr + frequencies_offsets
    frequencies_im_ptr = frequencies_re_ptr + rotary_dim
    frequencies_re = tl.load(frequencies_re_ptr)
    frequencies_im = tl.load(frequencies_im_ptr)

    if backward:
        out_re = input_re * frequencies_re + input_im * frequencies_im
        out_im = input_im * frequencies_re - input_re * frequencies_im
    else:
        out_re = input_re * frequencies_re - input_im * frequencies_im
        out_im = input_im * frequencies_re + input_re * frequencies_im

    if rotary_block_size % rotary_dim == 0 and num_heads % head_block_size == 0:
        tl.store(input_re_ptr, out_re)
        tl.store(input_im_ptr, out_im)
    else:
        tl.store(input_re_ptr, out_re, mask=mask)  # noqa
        tl.store(input_im_ptr, out_im, mask=mask)


def triton_rotary_(
    input_: torch.Tensor,
    frequencies: torch.Tensor,
    backward: bool = False,
):
    # TODO: Improve assumptions.
    # TODO: Make a transposed version to avoid contiguous call in key backward.
    # TODO: Improve block size heuristics.
    assert input_.stride(-1) == 1, f"{input_.shape} {input_.stride()}"
    batch_size, seq_len, num_heads, kv_channels = input_.shape
    rotary_dim = div(kv_channels, 2)
    rotary_block_size = triton.next_power_of_2(rotary_dim)
    head_block_size = triton.cdiv(TritonConfig.POINTWISE_BLOCK_SIZE, rotary_block_size)
    if head_block_size > num_heads:
        head_block_size = triton.next_power_of_2(num_heads)

    triton_rotary_kernel[(batch_size, seq_len, triton.cdiv(num_heads, head_block_size))](
        input_,
        frequencies,
        input_.stride(0),
        input_.stride(1),
        input_.stride(2),
        rotary_dim,
        num_heads,
        rotary_block_size,
        head_block_size,
        backward,  # noqa
    )
    return input_


def triton_rotary_forward_(input_: torch.Tensor, frequencies: torch.Tensor):
    return triton_rotary_(input_, frequencies), frequencies


def triton_rotary_backward_(grad_output: torch.Tensor, context: torch.Tensor):
    # TODO: Make a transposed version to avoid contiguous call in key backward.
    if grad_output.stride(-1) != 1:
        grad_output = grad_output.contiguous()
    return triton_rotary_(grad_output, context, True)


triton_rotary_autograd_ = wrap_forward_backward(triton_rotary_forward_, triton_rotary_backward_)
