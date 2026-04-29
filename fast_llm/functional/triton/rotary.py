import os

import torch

from fast_llm.functional.triton import TritonConfig, tl, tl_arange, tl_constexpr, triton, triton_autotune, triton_jit
from fast_llm.functional.utils import wrap_forward_backward
from fast_llm.utils import div

autotune_configs = (
    TritonConfig({"head_block_size": 16}, num_warps=4),
    TritonConfig({"head_block_size": 8}, num_warps=4),
    TritonConfig({"head_block_size": 4}, num_warps=4),
    TritonConfig({"head_block_size": 2}, num_warps=4),
    TritonConfig({"head_block_size": 1}, num_warps=4),
    TritonConfig({"head_block_size": 16}, num_warps=8),
    TritonConfig({"head_block_size": 8}, num_warps=8),
    TritonConfig({"head_block_size": 4}, num_warps=8),
)

if os.environ.get("FAST_LLM_SKIP_TRITON_AUTOTUNE"):
    autotune_configs = (autotune_configs[0],)


@triton_autotune(
    configs=autotune_configs,
    key=["rotary_dim", "num_heads", "seq_len"],
)
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
    seq_len: tl_constexpr,
    backward: tl_constexpr,
    head_block_size: tl_constexpr,  # injected by autotune
):
    # TODO: Int64 ptr if needed?
    pid_0 = tl.program_id(axis=0)  # Folded (batch * seq) index
    pid_1 = tl.program_id(axis=1)  # Head block index
    position_id = pid_0 % seq_len

    # Full-head column offsets: [0, 1, …, 2*rotary_dim-1]
    col_offsets = tl_arange(0, 2 * rotary_block_size)
    head_row = pid_1 * head_block_size + tl_arange(0, head_block_size)
    base = stride_0 * (pid_0 // seq_len) + stride_1 * position_id
    input_offsets = base + stride_2 * head_row[:, None] + col_offsets[None, :]

    # Load full head as one contiguous block per head (re and im halves together)
    if rotary_block_size % rotary_dim == 0 and num_heads % head_block_size == 0:
        x = tl.load(input_ptr + input_offsets).to(tl.float32)
    else:
        mask = (col_offsets[None, :] < 2 * rotary_dim) & (head_row[:, None] < num_heads)
        x = tl.load(input_ptr + input_offsets, mask=mask).to(tl.float32)

    # Partner: x[e + rotary_dim] for re-columns, x[e - rotary_dim] for im-columns.
    # These are the same cache lines as x, so expect L2 hits after the x load above.
    partner_col = tl.where(col_offsets < rotary_dim, col_offsets + rotary_dim, col_offsets - rotary_dim)
    partner_offsets = base + stride_2 * head_row[:, None] + partner_col[None, :]
    if rotary_block_size % rotary_dim == 0 and num_heads % head_block_size == 0:
        x_partner = tl.load(input_ptr + partner_offsets).to(tl.float32)
    else:
        x_partner = tl.load(input_ptr + partner_offsets, mask=mask).to(tl.float32)

    # Frequencies: same index for both halves (cos/sin repeat for re and im columns)
    freq_col = tl.where(col_offsets < rotary_dim, col_offsets, col_offsets - rotary_dim)
    freq_base = frequencies_ptr + 2 * rotary_dim * position_id
    freq_re = tl.load(freq_base + freq_col)
    freq_im = tl.load(freq_base + rotary_dim + freq_col)

    # out[e] = x[e]*cos ± x_partner[e]*sin
    # fwd: sign=-1 for re columns (cos*re - sin*im), +1 for im (cos*im + sin*re)
    # bwd: conjugate rotation flips the sign
    if backward:
        sign = tl.where(col_offsets < rotary_dim, 1.0, -1.0)
    else:
        sign = tl.where(col_offsets < rotary_dim, -1.0, 1.0)

    out = x * freq_re[None, :] + sign[None, :] * x_partner * freq_im[None, :]

    if rotary_block_size % rotary_dim == 0 and num_heads % head_block_size == 0:
        tl.store(input_ptr + input_offsets, out)
    else:
        tl.store(input_ptr + input_offsets, out, mask=mask)


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

    # Folded the large y dim into the x dim as gridDim.x is 32 bit while gridDim.y and gridDim.z are 16 bit registers
    grid = lambda meta: (batch_size * seq_len, triton.cdiv(num_heads, meta["head_block_size"]))
    triton_rotary_kernel[grid](
        input_,
        frequencies,
        input_.stride(0),
        input_.stride(1),
        input_.stride(2),
        rotary_dim,
        num_heads,
        rotary_block_size,
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
