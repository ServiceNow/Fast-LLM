import torch

import triton
import triton.language as tl
from fast_llm.functional.autograd import wrap_forward_backward
from fast_llm.functional.config import TritonConfig
from fast_llm.tensor import param_get_and_unset_is_zero


@triton.jit
def triton_normalization_forward_kernel(
    input_ptr,
    output_ptr,
    weight_ptr,
    bias_ptr,
    inv_var_ptr,
    n_cols,
    eps,
    has_bias: tl.constexpr,
    zero_centered: tl.constexpr,
    block_size: tl.constexpr,
):
    # Program dimensions
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, block_size)
    mask = cols < n_cols
    offsets = row * n_cols + cols

    # Input
    input_ = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Mean
    if has_bias:
        mean = tl.sum(input_, axis=0) / n_cols
        input_ = tl.where(mask, input_ - mean, 0.0)

    # Standard deviation
    inv_var = 1 / tl.sqrt(tl.sum(input_ * input_, axis=0) / n_cols + eps)
    tl.store(inv_var_ptr + row, inv_var)

    # Weight
    weight = tl.load(weight_ptr + cols, mask=mask)
    if zero_centered:
        weight += 1

    output = input_ * inv_var * weight

    # Bias
    if has_bias:
        bias = tl.load(bias_ptr + cols, mask=mask)
        output = output + bias

    # Output
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def triton_normalization_backward_kernel_1(
    grad_input_ptr,
    grad_output_ptr,
    grad_weight_partial_ptr,
    grad_bias_partial_ptr,
    output_ptr,
    weight_ptr,
    bias_ptr,
    inv_var_ptr,
    n_cols,
    n_rows,
    has_bias: tl.constexpr,
    zero_centered: tl.constexpr,
    block_size: tl.constexpr,
    block_size_row: tl.constexpr,
):
    # row_start = tl.program_id(0)*block_size_row
    rows = tl.program_id(0) * block_size_row + tl.arange(0, block_size_row)[:, None]
    row_mask = rows < n_rows

    cols = tl.arange(0, block_size)[None, :]

    col_mask = cols < n_cols
    mask = col_mask & row_mask
    offsets = rows * n_cols + cols

    # Load data
    output = tl.load(output_ptr + offsets, mask=mask, other=0).to(tl.float32)
    grad_output = tl.load(grad_output_ptr + offsets, mask=mask, other=0).to(tl.float32)
    weight = tl.load(weight_ptr + cols, mask=col_mask).to(tl.float32)
    if zero_centered:
        weight += 1

    inv_var = tl.load(inv_var_ptr + rows, mask=row_mask)

    # Bias
    if has_bias:
        bias = tl.load(bias_ptr + cols, mask=col_mask).to(tl.float32)
        output = output - bias

    # Input grad
    input_normalized = tl.where(mask, output / weight, 0.0)
    weight_grad_output = tl.where(mask, weight * grad_output * inv_var, 0.0)
    grad_input = weight_grad_output - input_normalized * (
        tl.sum(input_normalized * weight_grad_output, axis=1)[:, None] / n_cols
    )

    if has_bias:
        grad_input = grad_input - tl.sum(weight_grad_output, axis=1)[:, None] / n_cols
    tl.store(grad_input_ptr + offsets, grad_input, mask=mask)

    # Parameter grad partial sums
    parameter_offsets = tl.program_id(0) * n_cols + cols
    grad_weight_partial_ptr = grad_weight_partial_ptr + parameter_offsets
    grad_weight_partial = (grad_output * input_normalized).to(weight.dtype)
    grad_weight_partial = tl.sum(grad_weight_partial, axis=0)[None, :]

    if has_bias:
        grad_bias_partial_ptr = grad_bias_partial_ptr + parameter_offsets
        grad_bias_partial = tl.sum(grad_output.to(weight.dtype), axis=0)[None, :]

    tl.store(grad_weight_partial_ptr, grad_weight_partial, mask=col_mask)
    if has_bias:
        tl.store(grad_bias_partial_ptr, grad_bias_partial, mask=col_mask)  # noqa


@triton.jit
def triton_normalization_backward_kernel_2(
    grad_weight_partial_ptr,
    grad_bias_partial_ptr,
    grad_weight_ptr,
    grad_bias_ptr,
    m,  # GROUP_SIZE_M
    n_cols,  # number of columns
    has_bias: tl.constexpr,
    accumulate_grad: tl.constexpr,
    block_size_m: tl.constexpr,
    block_size_n: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * block_size_n + tl.arange(0, block_size_n)
    grad_weight_partial_sum = tl.zeros((block_size_m, block_size_n), dtype=tl.float32)
    if has_bias:
        grad_bias_partial_sum = tl.zeros((block_size_m, block_size_n), dtype=tl.float32)
    col_mask = cols < n_cols

    # Partial sums.
    for i in range(0, m, block_size_m):
        rows = i + tl.arange(0, block_size_m)
        mask = (rows[:, None] < m) & (cols[None, :] < n_cols)
        offsets = rows[:, None] * n_cols + cols[None, :]
        grad_weight_partial_sum += tl.load(grad_weight_partial_ptr + offsets, mask=mask, other=0.0)
        if has_bias:
            grad_bias_partial_sum += tl.load(grad_bias_partial_ptr + offsets, mask=mask, other=0.0)  # noqa

    # Final sum.
    grad_weight = tl.sum(grad_weight_partial_sum, axis=0)
    if accumulate_grad:
        grad_weight = tl.load(grad_weight_ptr + cols, mask=col_mask) + grad_weight
    tl.store(grad_weight_ptr + cols, grad_weight, mask=col_mask)

    if has_bias:
        grad_bias = tl.sum(grad_bias_partial_sum, axis=0)  # noqa
        if accumulate_grad:
            grad_bias = tl.load(grad_bias_ptr + cols, mask=col_mask) + grad_bias
        tl.store(grad_bias_ptr + cols, grad_bias, mask=col_mask)


def triton_normalization_forward(
    input_: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float,
    training: bool,
    zero_centered: bool,
):
    assert weight.shape == input_.shape[-1:]
    if bias is not None:
        assert weight.shape == bias.shape
    assert input_.is_contiguous()
    n_rows = input_.shape[:-1].numel()
    n_cols = weight.numel()

    output = torch.empty_like(input_)
    inv_var = torch.empty(n_rows, dtype=torch.float32, device="cuda")

    block_size = triton.next_power_of_2(n_cols)
    assert block_size * input_.element_size() <= TritonConfig.MAX_BLOCK_SIZE_BYTES
    num_warps = min(max(block_size // 256, 1), 8)

    triton_normalization_forward_kernel[(n_rows,)](
        input_,
        output,
        weight,
        bias,
        inv_var,
        n_cols,
        eps,
        bias is not None,
        zero_centered,
        block_size,
        num_warps=num_warps,
        num_ctas=1,
    )
    # Note: the context must be explicitly cleared to prevent a memory leak.
    context = [output, weight, bias, inv_var, eps, zero_centered] if training else None
    return output, context


def triton_normalization_backward(grad_output: torch.Tensor, context: list):
    output, weight, bias, inv_var, eps, zero_centered = context
    # We delete the context to prevent a memory leak
    context.clear()
    has_bias = bias is not None

    grad_output = grad_output.contiguous()

    n_rows = grad_output.shape[:-1].numel()
    n_cols = weight.numel()
    # TODO: Improve heuristics
    #   The ones from triton tutorial (32, 128) are terrible.
    #   These seem to match torch compile heuristics and were near-optimal for A100 tests with [8192, 4096], bf16.
    block_size_m = 64
    block_size_n = 8

    block_size = triton.next_power_of_2(n_cols)
    max_block_size = TritonConfig.MAX_BLOCK_SIZE_BYTES // 4
    assert block_size <= max_block_size
    block_size_row = max_block_size // block_size

    num_warps = min(max(block_size // 256, 1), 8)

    num_blocks_row = triton.cdiv(n_rows, block_size_row)

    grad_input = torch.empty_like(grad_output)

    grad_is_zero = param_get_and_unset_is_zero(weight)
    grad_weight = weight.grad_buffer
    # TODO: Any point in making it full precision?
    grad_weight_partial = grad_output.new_empty(num_blocks_row, n_cols)

    if has_bias:
        assert param_get_and_unset_is_zero(bias) == grad_is_zero
        grad_bias = bias.grad_buffer
        grad_bias_partial = grad_output.new_empty(num_blocks_row, n_cols)
    else:
        grad_bias_partial, grad_bias = None, None

    triton_normalization_backward_kernel_1[(num_blocks_row,)](
        grad_input,
        grad_output,
        grad_weight_partial,
        grad_bias_partial,
        output,
        weight,
        bias,
        inv_var,
        n_cols,
        n_rows,
        has_bias,
        zero_centered,
        block_size,
        block_size_row,
        num_warps=num_warps,
    )
    triton_normalization_backward_kernel_2[(triton.cdiv(n_cols, block_size_n),)](
        grad_weight_partial,
        grad_bias_partial,
        grad_weight,
        grad_bias,
        num_blocks_row,
        n_cols,
        has_bias,
        not grad_is_zero,
        block_size_m,
        block_size_n,
        num_ctas=1,
    )
    return grad_input


triton_normalization_autograd = wrap_forward_backward(triton_normalization_forward, triton_normalization_backward)


@torch.compile
def rms_norm(input_: torch.Tensor, weight: torch.Tensor, eps: float):
    # TODO: Backward pass is extremely slow.
    input_dtype = input_.dtype
    input_ = input_.to(torch.float32)
    return (weight * input_ * torch.rsqrt(input_.pow(2).mean(dim=-1, keepdim=True) + eps)).to(dtype=input_dtype)
