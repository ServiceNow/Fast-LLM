import torch

from fast_llm.functional.config import TritonConfig
from fast_llm.functional.triton import tl, tl_arange, tl_constexpr, triton, triton_jit
from fast_llm.functional.utils import wrap_forward_backward
from fast_llm.utils import div


@triton_jit()
def triton_rotary_kernel(
    input_ptr,
    frequencies_ptr,
    stride_0,
    stride_1,
    stride_2,
    rotary_dim: tl_constexpr,
    num_heads: tl_constexpr,
    rotary_block_size: tl_constexpr,
    head_block_size: tl_constexpr,
    seq_len: tl_constexpr,
    backward: tl_constexpr,
):
    # TODO: Int64 ptr if needed?
    pid_0 = tl.program_id(axis=0)  # Folded (batch * seq) index
    pid_1 = tl.program_id(axis=1)  # Head block index
    position_id = pid_0 % seq_len

    col_offsets = tl_arange(0, rotary_block_size)
    head_row = pid_1 * head_block_size + tl_arange(0, head_block_size)
    base = stride_0 * (pid_0 // seq_len) + stride_1 * position_id

    # Load re and im halves separately so both are in registers simultaneously,
    # avoiding a partner load (reading the same cache lines again with shuffled indices).
    re_offsets = base + stride_2 * head_row[:, None] + col_offsets[None, :]
    im_offsets = re_offsets + rotary_dim

    if rotary_block_size % rotary_dim == 0 and num_heads % head_block_size == 0:
        x_re = tl.load(input_ptr + re_offsets).to(tl.float32)
        x_im = tl.load(input_ptr + im_offsets).to(tl.float32)
    else:
        mask = (col_offsets[None, :] < rotary_dim) & (head_row[:, None] < num_heads)
        x_re = tl.load(input_ptr + re_offsets, mask=mask).to(tl.float32)
        x_im = tl.load(input_ptr + im_offsets, mask=mask).to(tl.float32)

    freq_base = frequencies_ptr + 2 * rotary_dim * position_id
    freq_re = tl.load(freq_base + col_offsets)
    freq_im = tl.load(freq_base + rotary_dim + col_offsets)

    # fwd: out_re = cos*re - sin*im,  out_im = cos*im + sin*re
    # bwd: conjugate rotation, sin signs flipped
    if backward:
        out_re = x_re * freq_re[None, :] + x_im * freq_im[None, :]
        out_im = x_im * freq_re[None, :] - x_re * freq_im[None, :]
    else:
        out_re = x_re * freq_re[None, :] - x_im * freq_im[None, :]
        out_im = x_im * freq_re[None, :] + x_re * freq_im[None, :]

    if rotary_block_size % rotary_dim == 0 and num_heads % head_block_size == 0:
        tl.store(input_ptr + re_offsets, out_re)
        tl.store(input_ptr + im_offsets, out_im)
    else:
        tl.store(input_ptr + re_offsets, out_re, mask=mask)
        tl.store(input_ptr + im_offsets, out_im, mask=mask)


def triton_rotary_(
    input_: torch.Tensor,
    frequencies: torch.Tensor,
    is_key_value: bool = False,
    backward: bool = False,
) -> torch.Tensor:
    # TODO: Make a transposed version to avoid contiguous call in key backward.
    out = input_
    if input_.stride(-1) != 1:
        input_ = input_.contiguous()
    if input_.ndim == 3:
        input_ = input_.unsqueeze(0)
        frequencies = frequencies.unsqueeze(0)
    if is_key_value:
        input_ = input_.chunk(2, dim=-2)[0]
    batch_size, seq_len, num_heads, head_size = input_.shape
    rotary_dim = div(head_size, 2)
    rotary_block_size = triton.next_power_of_2(rotary_dim)
    head_block_size = triton.cdiv(TritonConfig.POINTWISE_BLOCK_SIZE, rotary_block_size)
    if head_block_size > num_heads:
        head_block_size = triton.next_power_of_2(num_heads)

    # Folded the large y dim into the x dim as gridDim.x is 32 bit while gridDim.y and gridDim.z are 16 bit registers
    triton_rotary_kernel[(batch_size * seq_len, triton.cdiv(num_heads, head_block_size))](
        input_,
        frequencies,
        input_.stride(0),
        input_.stride(1),
        input_.stride(2),
        rotary_dim,
        num_heads,
        rotary_block_size,
        head_block_size,
        seq_len,
        backward,  # noqa
    )
    return out


def triton_rotary_forward_(
    input_: torch.Tensor, frequencies: torch.Tensor, is_key_value: bool = False
) -> tuple[torch.Tensor, tuple[torch.Tensor, bool]]:
    return triton_rotary_(input_, frequencies, is_key_value), (frequencies, is_key_value)


def triton_rotary_backward_(grad_output: torch.Tensor, context: tuple[torch.Tensor, bool]) -> torch.Tensor:
    frequencies, is_key_value = context
    if grad_output.stride(-1) != 1:
        grad_output = grad_output.contiguous()
    return triton_rotary_(grad_output, frequencies, is_key_value, True)


triton_rotary_autograd_ = wrap_forward_backward(triton_rotary_forward_, triton_rotary_backward_)
