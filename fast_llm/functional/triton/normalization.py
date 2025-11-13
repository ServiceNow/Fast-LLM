import math
import typing

import torch

from fast_llm.functional.autograd import wrap_forward_backward
from fast_llm.functional.config import TritonConfig
from fast_llm.functional.triton import tl, tl_constexpr, triton, triton_jit
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
    zero_centered: tl_constexpr,
    block_size: tl_constexpr,
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
    parameter_grad: tl_constexpr,
    zero_centered: tl_constexpr,
    block_size: tl_constexpr,
    block_size_row: tl_constexpr,
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
    weight_regularised = tl.where(weight >= 0, tl.maximum(weight, eps), tl.minimum(weight, -eps))
    input_normalized = tl.where(mask, output / weight_regularised, 0.0)
    weight_grad_output = tl.where(mask, weight * grad_output * inv_var, 0.0)
    grad_input = weight_grad_output - input_normalized * (
        tl.sum(input_normalized * weight_grad_output, axis=1)[:, None] / n_cols
    )

    if has_bias:
        grad_input = grad_input - tl.sum(weight_grad_output, axis=1)[:, None] / n_cols
    tl.store(grad_input_ptr + offsets, grad_input, mask=mask)

    # Parameter grad partial sums
    if parameter_grad:
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
) -> tuple[torch.Tensor, list[typing.Any]] | None:
    # Note: Converting input automatically to training dtype to match Apex behaviour,
    #  needed for full precision residual.
    # TODO: Review this?
    assert weight.shape == input_.shape[-1:]
    if bias is not None:
        assert weight.shape == bias.shape
    assert input_.is_contiguous()
    n_rows = input_.shape[:-1].numel()
    n_cols = weight.numel()

    output = torch.empty_like(input_, dtype=weight.dtype)
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


def triton_normalization_backward(grad_output: torch.Tensor, context: list[typing.Any]) -> torch.Tensor:
    output, weight, bias, inv_var, eps, zero_centered = context
    # We delete the context to prevent a memory leak
    context.clear()
    has_bias = bias is not None

    parameter_grad = weight.requires_grad
    assert parameter_grad == hasattr(weight, "grad_buffer")
    if has_bias:
        assert parameter_grad == bias.requires_grad

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
        parameter_grad,
        zero_centered,
        block_size,
        block_size_row,
        num_warps=num_warps,
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


# from mamba2
@triton.heuristics({"HAS_BIAS": lambda args: args["B"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["Z"] is not None})
@triton.jit
def _layer_norm_fwd_1pass_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Z,  # pointer to the other branch
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    stride_z_row,
    M,  # number of rows in X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    group = tl.program_id(1)
    X += row * stride_x_row + group * N
    Y += row * stride_y_row + group * N
    if HAS_Z:
        Z += row * stride_z_row + group * N
    if not IS_RMS_NORM:
        Mean += group * M
    Rstd += group * M
    W += group * N
    if HAS_BIAS:
        B += group * N
    # Compute mean and variance
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    if HAS_Z and not NORM_BEFORE_GATE:
        z = tl.load(Z + cols, mask=cols < N).to(tl.float32)
        x *= z * tl.sigmoid(z)
    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=0) / N
        tl.store(Mean + row, mean)
        xbar = tl.where(cols < N, x - mean, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    else:
        xbar = tl.where(cols < N, x, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    mask = cols < N
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    if HAS_BIAS:
        b = tl.load(B + cols, mask=mask).to(tl.float32)
    x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
    y = x_hat * w + b if HAS_BIAS else x_hat * w
    if HAS_Z and NORM_BEFORE_GATE:
        z = tl.load(Z + cols, mask=mask).to(tl.float32)
        y *= z * tl.sigmoid(z)
    # Write output
    tl.store(Y + cols, y, mask=mask)


def _layer_norm_fwd(x, weight, bias, eps, z=None, out=None, group_size=None, norm_before_gate=True, is_rms_norm=False):
    M, N = x.shape
    if group_size is None:
        group_size = N
    assert N % group_size == 0
    ngroups = N // group_size
    assert x.stride(-1) == 1
    if z is not None:
        assert z.stride(-1) == 1
        assert z.shape == (M, N)
    assert weight.shape == (N,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)
    # allocate output
    if out is not None:
        assert out.shape == x.shape
    else:
        out = torch.empty_like(x)
    assert out.stride(-1) == 1
    mean = torch.empty((ngroups * M,), dtype=torch.float32, device=x.device) if not is_rms_norm else None
    rstd = torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
    if group_size > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    num_warps = min(max(BLOCK_N // 256, 1), 8)
    grid = (M, ngroups)
    with torch.cuda.device(x.device.index):
        _layer_norm_fwd_1pass_kernel[grid](
            x,
            out,
            weight,
            bias,
            z,
            mean,
            rstd,
            x.stride(0),
            out.stride(0),
            z.stride(0) if z is not None else 0,
            M,
            group_size,
            eps,
            BLOCK_N=BLOCK_N,
            NORM_BEFORE_GATE=norm_before_gate,
            IS_RMS_NORM=is_rms_norm,
            num_warps=num_warps,
        )
    return out, mean, rstd


@triton.heuristics({"HAS_BIAS": lambda args: args["B"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["Z"] is not None})
@triton.heuristics({"RECOMPUTE_OUTPUT": lambda args: args["Y"] is not None})
@triton.jit
def _layer_norm_bwd_kernel(
    X,  # pointer to the input
    W,  # pointer to the weights
    B,  # pointer to the biases
    Z,  # pointer to the other branch
    Y,  # pointer to the output to be recomputed
    DY,  # pointer to the output gradient
    DX,  # pointer to the input gradient
    DW,  # pointer to the partial sum of weights gradient
    DB,  # pointer to the partial sum of biases gradient
    DZ,  # pointer to the other branch
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_z_row,
    stride_y_row,
    stride_dy_row,
    stride_dx_row,
    stride_dz_row,
    stride_dw_row,
    stride_db_row,
    M,  # number of rows in X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    rows_per_program,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Map the program id to the elements of X, DX, and DY it should compute.
    row_block_id = tl.program_id(0)
    group = tl.program_id(1)
    row_start = row_block_id * rows_per_program
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    X += row_start * stride_x_row + group * N
    if HAS_Z:
        Z += row_start * stride_z_row + group * N
        DZ += row_start * stride_dz_row + group * N
    DY += row_start * stride_dy_row + group * N
    DX += row_start * stride_dx_row + group * N
    if RECOMPUTE_OUTPUT:
        Y += row_start * stride_y_row + group * N
    if not IS_RMS_NORM:
        Mean += group * M
    Rstd += group * M
    W += group * N
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    if (RECOMPUTE_OUTPUT or HAS_Z) and HAS_BIAS:
        B += group * N
        b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    dw = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if HAS_BIAS:
        db = tl.zeros((BLOCK_N,), dtype=tl.float32)
    row_end = min((row_block_id + 1) * rows_per_program, M)
    for row in range(row_start, row_end):
        # Load data to SRAM
        x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
        dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
        if not IS_RMS_NORM:
            mean = tl.load(Mean + row)
        if HAS_Z and not NORM_BEFORE_GATE:
            z = tl.load(Z + cols, mask=mask, other=0.0).to(tl.float32)
            x_og = x
            x = x_og * z * tl.sigmoid(z)
        rstd = tl.load(Rstd + row)
        # Compute dx
        xhat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
        xhat = tl.where(mask, xhat, 0.0)
        if HAS_Z and NORM_BEFORE_GATE:
            z = tl.load(Z + cols, mask=mask, other=0.0).to(tl.float32)
            z_sigmoid = tl.sigmoid(z)
            y = xhat * w + b if HAS_BIAS else xhat * w
            if RECOMPUTE_OUTPUT:
                tl.store(Y + cols, y * z * z_sigmoid, mask=mask)
            dz = dy * y * z_sigmoid * (1 + z * (1 - z_sigmoid))
            tl.store(DZ + cols, dz, mask=mask)
            dy *= z * z_sigmoid
        else:
            if RECOMPUTE_OUTPUT:
                y = xhat * w + b if HAS_BIAS else xhat * w
                tl.store(Y + cols, y, mask=mask)
        wdy = w * dy
        c1 = tl.sum(xhat * wdy, axis=0) / N
        if not IS_RMS_NORM:
            c2 = tl.sum(wdy, axis=0) / N
            dx = (wdy - (xhat * c1 + c2)) * rstd
        else:
            dx = (wdy - xhat * c1) * rstd
        dw += dy * xhat
        if HAS_BIAS:
            db += dy
        if HAS_Z and not NORM_BEFORE_GATE:
            z_sigmoid = tl.sigmoid(z)
            dz = dx * x_og * z_sigmoid * (1 + z * (1 - z_sigmoid))
            tl.store(DZ + cols, dz, mask=mask)
            dx *= z * z_sigmoid
        # Write dx
        tl.store(DX + cols, dx, mask=mask)

        X += stride_x_row
        if HAS_Z:
            Z += stride_z_row
            DZ += stride_dz_row
        if RECOMPUTE_OUTPUT:
            Y += stride_y_row
        DY += stride_dy_row
        DX += stride_dx_row
    tl.store(DW + row_block_id * stride_dw_row + group * N + cols, dw, mask=mask)
    if HAS_BIAS:
        tl.store(DB + row_block_id * stride_db_row + group * N + cols, db, mask=mask)


def _layer_norm_bwd(
    dy,
    x,
    weight,
    bias,
    eps,
    mean,
    rstd,
    z=None,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
    recompute_output=False,
    dz=None,
    out=None,
):
    M, N = x.shape
    if group_size is None:
        group_size = N
    assert N % group_size == 0
    ngroups = N // group_size
    assert x.stride(-1) == 1
    assert dy.stride(-1) == 1
    assert dy.shape == (M, N)
    if z is not None:
        assert z.stride(-1) == 1
        assert z.shape == (M, N)
    assert weight.shape == (N,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)
    # allocate output
    dx = torch.empty_like(x)
    if dz is not None:
        assert z is not None
        assert dz.shape == z.shape
        assert dz.stride(-1) == 1
    else:
        dz = torch.empty_like(z) if z is not None else None
    if recompute_output:
        if out is None:
            out = torch.empty_like(x)
        assert out.shape == x.shape

    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
    if group_size > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    num_warps = min(max(BLOCK_N // 256, 1), 8)
    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
    # If group size is small (e.g., 64), we're only using 1 warp. So having just 108 programs
    # would limit the occupancy.
    nrow_groups = math.ceil(sm_count * math.ceil(4 / num_warps) / ngroups)
    _dw = torch.empty((nrow_groups, N), dtype=torch.float32, device=weight.device)
    _db = torch.empty((nrow_groups, N), dtype=torch.float32, device=bias.device) if bias is not None else None
    rows_per_program = math.ceil(M / nrow_groups)
    grid = (nrow_groups, ngroups)
    with torch.cuda.device(x.device.index):
        _layer_norm_bwd_kernel[grid](
            x,
            weight,
            bias,
            z,
            out if recompute_output else None,
            dy,
            dx,
            _dw,
            _db,
            dz,
            mean,
            rstd,
            x.stride(0),
            z.stride(0) if z is not None else 0,
            0 if not recompute_output else out.stride(0),
            dy.stride(0),
            dx.stride(0),
            dz.stride(0) if dz is not None else 0,
            _dw.stride(0),
            _db.stride(0) if _db is not None else 0,
            M,
            group_size,
            eps,
            rows_per_program,
            BLOCK_N=BLOCK_N,
            NORM_BEFORE_GATE=norm_before_gate,
            IS_RMS_NORM=is_rms_norm,
            num_warps=num_warps,
        )
    dw = _dw.sum(0).to(weight.dtype)
    db = _db.sum(0).to(bias.dtype) if bias is not None else None
    return (dx, dw, db, dz) if not recompute_output else (dx, dw, db, dz, out)
