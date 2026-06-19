import typing

import torch

from fast_llm.functional.config import TritonConfig
from fast_llm.functional.triton import tl, tl_arange, tl_constexpr, tl_full, triton, triton_jit
from fast_llm.functional.utils import wrap_forward_backward
from fast_llm.tensor import param_get_and_unset_is_zero


@triton_jit()
def triton_normalization_forward_kernel(
    input_ptr,
    output_ptr,
    weight_ptr,
    bias_ptr,
    inv_var_ptr,
    n_cols,
    eps,
    has_bias: tl_constexpr,
    has_weight: tl_constexpr,
    zero_centered: tl_constexpr,
    block_size: tl_constexpr,
):
    # Program dimensions
    row = tl.program_id(0).to(tl.int64)
    cols = tl_arange(0, block_size)
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
    if has_weight:
        weight = tl.load(weight_ptr + cols, mask=mask)
        if zero_centered:
            weight += 1
        output = input_ * inv_var * weight
    else:
        output = input_ * inv_var

    # Bias
    if has_bias:
        bias = tl.load(bias_ptr + cols, mask=mask)
        output = output + bias

    # Output
    tl.store(output_ptr + offsets, output, mask=mask)


@triton_jit()
def _normalization_backward_terms(
    output_ptr,
    grad_output_ptr,
    weight_ptr,
    bias_ptr,
    offsets,
    cols,
    mask,
    col_mask,
    inv_var,
    eps,
    has_bias: tl_constexpr,
    has_weight: tl_constexpr,
    zero_centered: tl_constexpr,
):
    # Load one (block_size_row, block_size_col) tile and derive the two per-element terms the
    # backward needs: the normalized input and `weight * grad_output * inv_var`. Also returns
    # the loaded grad_output for the parameter-grad partials.
    output = tl.load(output_ptr + offsets, mask=mask, other=0).to(tl.float32)
    grad_output = tl.load(grad_output_ptr + offsets, mask=mask, other=0).to(tl.float32)
    if has_bias:
        bias = tl.load(bias_ptr + cols, mask=col_mask).to(tl.float32)
        output = output - bias
    if has_weight:
        weight = tl.load(weight_ptr + cols, mask=col_mask).to(tl.float32)
        if zero_centered:
            weight += 1
        weight_regularised = tl.where(weight >= 0, tl.maximum(weight, eps), tl.minimum(weight, -eps))
        input_normalized = tl.where(mask, output / weight_regularised, 0.0)
        weight_grad_output = tl.where(mask, weight * grad_output * inv_var, 0.0)
    else:
        # weight == 1 everywhere: forward output = input * inv_var, so input_normalized = output
        input_normalized = tl.where(mask, output, 0.0)
        weight_grad_output = tl.where(mask, grad_output * inv_var, 0.0)
    return input_normalized, weight_grad_output, grad_output


@triton_jit()
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
    eps,
    has_bias: tl_constexpr,
    has_weight: tl_constexpr,
    parameter_grad: tl_constexpr,
    zero_centered: tl_constexpr,
    block_size_col: tl_constexpr,
    block_size_row: tl_constexpr,
    two_pass: tl_constexpr,
):
    # Each program owns a `block_size_row × block_size_col` register tile (occupancy set by the
    # tile, independent of n_cols) and writes grad_input for every row it visits while
    # accumulating the per-column parameter-grad partials that kernel_2 later reduces.
    #
    # grad_input[r, c] = weight_grad_output[r, c] - input_normalized[r, c] * correction[r],
    # where correction[r] = mean_c(input_normalized * weight_grad_output) (+ mean_c(weight_grad_output)
    # for the bias term). The column mean needs a full row, so a row wider than one chunk uses two
    # passes (pass 1 reduces the per-row corrections, pass 2 re-reads to write grad_input and the
    # partials); a row that fits in one chunk is single pass.
    #
    # Single pass grid-strides the rows: the launch bounds the program count (and thus the
    # partial-buffer height kernel_2 reduces) independently of n_rows, so each program folds many
    # row tiles into one fp32-accumulated partial instead of emitting one partial per tile — the
    # difference between a small reduction for kernel_2 and one whose height grows with n_rows.
    # Two pass keeps one tile per program (its rows already stride the column chunks, and a
    # per-chunk cross-tile accumulator would not fit in registers at wide n_cols).
    row_block = tl.program_id(0)

    if two_pass:
        rows = row_block * block_size_row + tl_arange(0, block_size_row)[:, None]
        row_mask = rows < n_rows
        inv_var = tl.load(inv_var_ptr + rows, mask=row_mask)
        normalization_correction = tl_full((block_size_row, 1), 0.0, dtype=tl.float32)
        if has_bias:
            bias_correction = tl_full((block_size_row, 1), 0.0, dtype=tl.float32)
        for col_start in range(0, n_cols, block_size_col):
            cols = col_start + tl_arange(0, block_size_col)[None, :]
            col_mask = cols < n_cols
            mask = col_mask & row_mask
            input_normalized, weight_grad_output, _ = _normalization_backward_terms(
                output_ptr,
                grad_output_ptr,
                weight_ptr,
                bias_ptr,
                rows * n_cols + cols,
                cols,
                mask,
                col_mask,
                inv_var,
                eps,
                has_bias,
                has_weight,
                zero_centered,
            )
            normalization_correction += tl.sum(input_normalized * weight_grad_output, axis=1)[:, None]
            if has_bias:
                bias_correction += tl.sum(weight_grad_output, axis=1)[:, None]
        normalization_correction = normalization_correction / n_cols
        if has_bias:
            bias_correction = bias_correction / n_cols

        for col_start in range(0, n_cols, block_size_col):
            cols = col_start + tl_arange(0, block_size_col)[None, :]
            col_mask = cols < n_cols
            mask = col_mask & row_mask
            offsets = rows * n_cols + cols
            input_normalized, weight_grad_output, grad_output = _normalization_backward_terms(
                output_ptr,
                grad_output_ptr,
                weight_ptr,
                bias_ptr,
                offsets,
                cols,
                mask,
                col_mask,
                inv_var,
                eps,
                has_bias,
                has_weight,
                zero_centered,
            )
            grad_input = weight_grad_output - input_normalized * normalization_correction
            if has_bias:
                grad_input = grad_input - bias_correction
            tl.store(grad_input_ptr + offsets, grad_input, mask=mask)
            if parameter_grad:
                # Reduce the row partials in fp32 (the product/sum stay fp32; the store casts to
                # the partial buffer's dtype). Casting before the sum would reduce in low
                # precision and degrade the parameter grads.
                parameter_offsets = row_block * n_cols + cols
                grad_weight_partial = tl.sum(grad_output * input_normalized, axis=0)[None, :]
                tl.store(grad_weight_partial_ptr + parameter_offsets, grad_weight_partial, mask=col_mask)
                if has_bias:
                    grad_bias_partial = tl.sum(grad_output, axis=0)[None, :]
                    tl.store(grad_bias_partial_ptr + parameter_offsets, grad_bias_partial, mask=col_mask)  # noqa
    else:
        cols = tl_arange(0, block_size_col)[None, :]
        col_mask = cols < n_cols
        if parameter_grad:
            # fp32 accumulator folded over every row tile this program visits (one store per
            # program, so kernel_2 reduces only `num_programs` rows). fp32 here, store casts.
            grad_weight_partial = tl_full((1, block_size_col), 0.0, dtype=tl.float32)
            if has_bias:
                grad_bias_partial = tl_full((1, block_size_col), 0.0, dtype=tl.float32)
        row_step = tl.num_programs(0) * block_size_row
        for row_start in range(row_block * block_size_row, n_rows, row_step):
            rows = row_start + tl_arange(0, block_size_row)[:, None]
            row_mask = rows < n_rows
            inv_var = tl.load(inv_var_ptr + rows, mask=row_mask)
            mask = col_mask & row_mask
            offsets = rows * n_cols + cols
            input_normalized, weight_grad_output, grad_output = _normalization_backward_terms(
                output_ptr,
                grad_output_ptr,
                weight_ptr,
                bias_ptr,
                offsets,
                cols,
                mask,
                col_mask,
                inv_var,
                eps,
                has_bias,
                has_weight,
                zero_centered,
            )
            grad_input = weight_grad_output - input_normalized * (
                tl.sum(input_normalized * weight_grad_output, axis=1)[:, None] / n_cols
            )
            if has_bias:
                grad_input = grad_input - tl.sum(weight_grad_output, axis=1)[:, None] / n_cols
            tl.store(grad_input_ptr + offsets, grad_input, mask=mask)
            if parameter_grad:
                grad_weight_partial += tl.sum(grad_output * input_normalized, axis=0)[None, :]
                if has_bias:
                    grad_bias_partial += tl.sum(grad_output, axis=0)[None, :]
        if parameter_grad:
            parameter_offsets = row_block * n_cols + cols
            tl.store(grad_weight_partial_ptr + parameter_offsets, grad_weight_partial, mask=col_mask)
            if has_bias:
                tl.store(grad_bias_partial_ptr + parameter_offsets, grad_bias_partial, mask=col_mask)  # noqa


@triton_jit()
def triton_normalization_backward_kernel_2(
    grad_weight_partial_ptr,
    grad_bias_partial_ptr,
    grad_weight_ptr,
    grad_bias_ptr,
    m,  # GROUP_SIZE_M
    n_cols,  # number of columns
    has_bias: tl_constexpr,
    accumulate_grad: tl_constexpr,
    block_size_m: tl_constexpr,
    block_size_n: tl_constexpr,
):
    pid = tl.program_id(0)
    cols = pid * block_size_n + tl_arange(0, block_size_n)
    grad_weight_partial_sum = tl_full((block_size_m, block_size_n), 0, dtype=tl.float32)
    if has_bias:
        grad_bias_partial_sum = tl_full((block_size_m, block_size_n), 0, dtype=tl.float32)
    col_mask = cols < n_cols

    # Partial sums.
    for i in range(0, m, block_size_m):
        rows = i + tl_arange(0, block_size_m)
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
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
    training: bool,
    zero_centered: bool,
) -> tuple[torch.Tensor, list[typing.Any]] | None:
    # Note: Converting input automatically to training dtype to match Apex behaviour,
    #  needed for full precision residual.
    # TODO: Review this?
    if weight is not None:
        assert weight.shape == input_.shape[-1:]
        if bias is not None:
            assert weight.shape == bias.shape
    assert input_.is_contiguous()
    n_rows = input_.shape[:-1].numel()
    n_cols = input_.shape[-1]

    output = torch.empty_like(input_, dtype=weight.dtype if weight is not None else input_.dtype)
    inv_var = torch.empty(n_rows, dtype=torch.float32, device=input_.device)

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
        weight is not None,
        zero_centered,
        block_size,
        num_warps=num_warps,
        num_ctas=1,
    )
    # Note: the context must be explicitly cleared to prevent a memory leak.
    context = [output, weight, bias, inv_var, eps, zero_centered] if training else None
    return output, context


# kernel_1 launch configuration. block_size_row × block_size_col is the register tile that governs
# occupancy. A row is handled in one pass when its tile fits in registers; otherwise the columns are
# chunked and the row is processed in two passes (reduce the corrections, then re-read to write
# grad_input and the partials). The re-read is the cost of two-pass, so single pass is used as wide
# as it fits.
_KERNEL_1_NARROW_MAX_COLS = 4096  # rows this narrow always fit a multi-row single-pass tile
_KERNEL_1_MAX_ROWS = 8  # cap on block_size_row (the register-tile height) for narrow rows
_KERNEL_1_WIDE_COL_CHUNK = 4096  # two-pass column-chunk width
_KERNEL_1_WIDE_ROWS = 4  # two-pass block_size_row
# Single-pass bound on the program count (and thus the partial-buffer height kernel_2 reduces):
# enough programs per SM to saturate memory bandwidth, but small enough that kernel_2's reduction
# stays cheap. Two waves is the knee on H100 — one wave (×1) starves grad_input latency-hiding on
# tall-narrow rows, more only re-inflates kernel_2. Programs beyond `n_rows / block_size_row` are
# pointless, so it is also capped there.
_KERNEL_1_ROW_BLOCKS_PER_SM = 2

_kernel_1_sm_count: int | None = None


def _kernel_1_target_row_blocks(device: torch.device) -> int:
    global _kernel_1_sm_count
    if _kernel_1_sm_count is None:
        _kernel_1_sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    return _kernel_1_sm_count * _KERNEL_1_ROW_BLOCKS_PER_SM


def _kernel_1_wide_single_pass(block_size: int, n_rows: int, has_bias: bool, max_block_size: int) -> bool:
    # A wide row can still go single pass — one warp-saturated tile spanning the whole row — which
    # avoids the two-pass re-read. Bias roughly doubles the live registers per element, so the tile
    # only fits up to half the cap with bias. When it fits, single pass wins once there are enough
    # rows to fill the SMs; with fewer rows the two-pass chunking is faster.
    fits = block_size <= (max_block_size // 2 if has_bias else max_block_size)
    enough_rows = n_rows >= block_size // (2 if has_bias else 8)
    return fits and enough_rows


def _triton_normalization_backward_kernel_1_config(
    n_cols: int, n_rows: int, has_bias: bool
) -> tuple[int, int, bool, int, int]:
    block_size = triton.next_power_of_2(n_cols)
    max_block_size = TritonConfig.MAX_BLOCK_SIZE_BYTES // 4
    if block_size <= _KERNEL_1_NARROW_MAX_COLS:
        # Whole row in one chunk, multiple rows per tile. Cap the row count so a narrow row doesn't
        # blow up the register tile (hurting occupancy); wider rows keep their smaller count.
        block_size_col = block_size
        block_size_row = max(1, min(max_block_size // block_size_col, _KERNEL_1_MAX_ROWS))
        two_pass = False
        num_warps = min(max(block_size_col // 256, 1), 8)
    elif _kernel_1_wide_single_pass(block_size, n_rows, has_bias, max_block_size):
        # Whole row in one chunk, one row per tile, maximum warps to cover the wide reduction.
        block_size_col = block_size
        block_size_row = 1
        two_pass = False
        num_warps = 32
    else:
        # Wide row: chunk the columns and re-read.
        block_size_col = _KERNEL_1_WIDE_COL_CHUNK
        block_size_row = _KERNEL_1_WIDE_ROWS
        two_pass = True
        num_warps = 16
    return block_size_col, block_size_row, two_pass, num_warps, 1


def triton_normalization_backward(grad_output: torch.Tensor, context: list[typing.Any]) -> torch.Tensor:
    output, weight, bias, inv_var, eps, zero_centered = context
    # We delete the context to prevent a memory leak
    context.clear()
    has_bias = bias is not None
    has_weight = weight is not None

    parameter_grad = weight.requires_grad if has_weight else False
    if has_weight:
        assert parameter_grad == hasattr(weight, "grad_buffer")
    if has_bias:
        assert parameter_grad == bias.requires_grad

    grad_output = grad_output.contiguous()

    n_rows = grad_output.shape[:-1].numel()
    n_cols = grad_output.shape[-1]
    # TODO: Improve heuristics
    #   The ones from triton tutorial (32, 128) are terrible.
    #   These seem to match torch compile heuristics and were near-optimal for A100 tests with [8192, 4096], bf16.
    block_size_m = 64
    block_size_n = 8

    block_size_col, block_size_row, two_pass, num_warps, num_stages = _triton_normalization_backward_kernel_1_config(
        n_cols, n_rows, has_bias
    )

    num_tiles = triton.cdiv(n_rows, block_size_row)
    # Single pass grid-strides the rows, so the program count (the partial-buffer height) is bounded
    # below the tile count; two pass keeps one program per tile.
    num_blocks_row = num_tiles if two_pass else min(num_tiles, _kernel_1_target_row_blocks(grad_output.device))

    grad_input = torch.empty_like(grad_output)

    if parameter_grad:
        grad_is_zero = param_get_and_unset_is_zero(weight)
        grad_weight = weight.grad_buffer
        # TODO: Any point in making it full precision?
        grad_weight_partial = grad_output.new_empty(num_blocks_row, n_cols)
    else:
        grad_is_zero = True
        grad_weight = None
        grad_weight_partial = None

    if has_bias and parameter_grad:
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
        eps,
        has_bias,
        has_weight,
        parameter_grad,
        zero_centered,
        block_size_col,
        block_size_row,
        two_pass,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if parameter_grad:
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
def rms_norm(input_: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    # TODO: Backward pass is extremely slow.
    input_dtype = input_.dtype
    input_ = input_.to(torch.float32)
    return (weight * input_ * torch.rsqrt(input_.pow(2).mean(dim=-1, keepdim=True) + eps)).to(dtype=input_dtype)
