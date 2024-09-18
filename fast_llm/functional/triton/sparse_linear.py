import torch
import triton
import triton.language as tl

from fast_llm.functional.config import SparseMap
from fast_llm.utils import Assert, div

autotune_configs = [
    triton.Config(
        {"block_size_row": 128, "block_size_col": 256, "block_size_inner": 64, "group_size_row": 8},
        num_stages=3,
        num_warps=8,
    ),
    triton.Config(
        {"block_size_row": 64, "block_size_col": 256, "block_size_inner": 32, "group_size_row": 8},
        num_stages=4,
        num_warps=4,
    ),
    triton.Config(
        {"block_size_row": 128, "block_size_col": 128, "block_size_inner": 32, "group_size_row": 8},
        num_stages=4,
        num_warps=4,
    ),
    triton.Config(
        {"block_size_row": 128, "block_size_col": 64, "block_size_inner": 32, "group_size_row": 8},
        num_stages=4,
        num_warps=4,
    ),
    triton.Config(
        {"block_size_row": 64, "block_size_col": 128, "block_size_inner": 32, "group_size_row": 8},
        num_stages=4,
        num_warps=4,
    ),
    triton.Config(
        {"block_size_row": 128, "block_size_col": 32, "block_size_inner": 32, "group_size_row": 8},
        num_stages=4,
        num_warps=4,
    ),
    triton.Config(
        {"block_size_row": 64, "block_size_col": 32, "block_size_inner": 32, "group_size_row": 8},
        num_stages=5,
        num_warps=2,
    ),
    triton.Config(
        {"block_size_row": 32, "block_size_col": 64, "block_size_inner": 32, "group_size_row": 8},
        num_stages=5,
        num_warps=2,
    ),
]


@triton.autotune(
    configs=autotune_configs,
    key=["row_dim", "col_dim", "inner_dim"],
)
@triton.jit
def dense_matmul_kernel(
    lhs_ptr,
    rhs_ptr,
    out_ptr,
    row_dim: tl.constexpr,
    col_dim: tl.constexpr,
    inner_dim: tl.constexpr,
    lhs_stride_row: tl.constexpr,
    lhs_stride_inner: tl.constexpr,
    rhs_stride_inner: tl.constexpr,
    rhs_stride_col: tl.constexpr,
    out_stride_row: tl.constexpr,
    out_stride_col: tl.constexpr,
    accumulate: tl.constexpr,
    masked: tl.constexpr,
    block_size_row: tl.constexpr,
    block_size_col: tl.constexpr,
    block_size_inner: tl.constexpr,
    group_size_row: tl.constexpr,
):
    # Safety checks
    # TODO: Any better way to handle optional masking?
    if not masked:
        tl.static_assert(row_dim % block_size_row == 0)
        tl.static_assert(col_dim % block_size_col == 0)
        tl.static_assert(inner_dim % block_size_inner == 0)

    # Reorganize blocks to maximize cache reuse.
    pid_row, pid_col = tl.swizzle2d(
        tl.program_id(axis=0),
        tl.program_id(axis=1),
        tl.cdiv(row_dim, block_size_row),
        tl.cdiv(col_dim, block_size_col),
        group_size_row,
    )

    # Grid offsets
    row_offset = pid_row * block_size_row
    col_offset = pid_col * block_size_col

    # Pointers
    row_range = tl.arange(0, block_size_row)[:, None] + row_offset
    col_range = tl.arange(0, block_size_col)[None, :] + col_offset
    inner_range = tl.arange(0, block_size_inner)
    lhs_ptr += row_range * lhs_stride_row + inner_range[None, :] * lhs_stride_inner
    rhs_ptr += inner_range[:, None] * rhs_stride_inner + col_range * rhs_stride_col
    out_ptr += row_range * out_stride_row + col_range * out_stride_col

    # Matrix multiplication
    if masked:
        row_mask = row_range < row_dim
        col_mask = col_range < col_dim
        inner_mask = inner_range < inner_dim
        out = tl.dot(
            tl.load(lhs_ptr, mask=row_mask * inner_mask[None, :], other=0),
            tl.load(rhs_ptr, mask=inner_mask[:, None] * col_mask, other=0),
            out_dtype=tl.float32,
        )
    else:
        out = tl.dot(tl.load(lhs_ptr), tl.load(rhs_ptr), out_dtype=tl.float32)

    for k in range(1, inner_dim // block_size_inner):
        lhs_ptr += block_size_inner * lhs_stride_inner
        rhs_ptr += block_size_inner * rhs_stride_inner
        if masked:
            inner_range += block_size_inner
            inner_mask = inner_range < inner_dim
            out += tl.dot(
                tl.load(lhs_ptr, mask=row_mask & inner_mask[None, :], other=0),  # noqa
                tl.load(rhs_ptr, mask=inner_mask[:, None] & col_mask, other=0),  # noqa
                out_dtype=tl.float32,
            )
        else:
            out += tl.dot(tl.load(lhs_ptr), tl.load(rhs_ptr))

    # Output
    if masked:
        out_mask = row_mask & col_mask
        if accumulate:
            out += tl.load(out_ptr, mask=out_mask)
        tl.store(out_ptr, out, mask=out_mask)
    else:
        if accumulate:
            out += tl.load(out_ptr)
        tl.store(out_ptr, out)


def dense_matmul(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    out: torch.Tensor | None = None,
    accumulate: bool = False,
):
    """
    Standard matrix multiplication.
    """
    # Shape
    row_dim, inner_dim = lhs.shape
    inner_dim_1, col_dim = rhs.shape
    Assert.eq(inner_dim, inner_dim_1)
    if out is None:
        assert not accumulate
        out = lhs.new_empty(row_dim, col_dim)

    grid = lambda meta: (triton.cdiv(row_dim, meta["block_size_row"]), triton.cdiv(col_dim, meta["block_size_col"]))
    dense_matmul_kernel[grid](
        lhs,
        rhs,
        out,
        row_dim,
        col_dim,
        inner_dim,
        lhs.stride(0),
        lhs.stride(1),
        rhs.stride(0),
        rhs.stride(1),
        out.stride(0),
        out.stride(1),
        accumulate,
        not (row_dim % 128 == col_dim % 256 == inner_dim % 64 == 0),
    )
    return out


@triton.autotune(
    configs=autotune_configs,
    # Excluding `row_dim` because it causes the compile time to skyrocket.
    key=["col_sparse_dim", "inner_dim", "sparse_dim"],
)
@triton.jit
def output_sparse_matmul_kernel(
    lhs_ptr,
    rhs_ptr,
    out_ptr,
    expert_ends_ptr,
    row_dim: tl.constexpr,
    col_sparse_dim: tl.constexpr,
    inner_dim: tl.constexpr,
    sparse_dim: tl.constexpr,
    padded_sparse_dim: tl.constexpr,
    lhs_stride_row: tl.constexpr,
    lhs_stride_inner: tl.constexpr,
    rhs_stride_inner: tl.constexpr,
    rhs_stride_col: tl.constexpr,
    out_stride_row: tl.constexpr,
    out_stride_col: tl.constexpr,
    accumulate: tl.constexpr,
    block_size_row: tl.constexpr,
    block_size_col: tl.constexpr,
    block_size_inner: tl.constexpr,
    group_size_row: tl.constexpr,
):
    # Safety checks
    tl.static_assert(row_dim % block_size_row == 0)
    tl.static_assert(col_sparse_dim % block_size_col == 0)
    tl.static_assert(inner_dim % block_size_inner == 0)
    tl.static_assert(sparse_dim <= padded_sparse_dim)

    # Reorganize blocks to maximize cache reuse.
    pid_row, pid_col = tl.swizzle2d(
        tl.program_id(axis=0),
        tl.program_id(axis=1),
        row_dim // block_size_row,
        col_sparse_dim // block_size_col,
        group_size_row,
    )

    # Grid offsets
    row_offset = pid_row * block_size_row
    col_sparse_offset = pid_col * block_size_col
    sparse_range = tl.arange(0, padded_sparse_dim)
    expert_ends = tl.load(expert_ends_ptr + sparse_range, mask=sparse_range < sparse_dim, other=row_dim)
    sparse_index = tl.sum((expert_ends <= row_offset).to(tl.int64))  # noqa
    if sparse_index == sparse_dim:
        return
    col_dense_offset = col_sparse_offset + sparse_index * col_sparse_dim

    # Pointers
    row_range = tl.arange(0, block_size_row)[:, None]
    col_range = tl.arange(0, block_size_col)[None, :]
    inner_range = tl.arange(0, block_size_inner)
    lhs_ptr += (row_offset + row_range) * lhs_stride_row + inner_range[None, :] * lhs_stride_inner
    rhs_ptr += inner_range[:, None] * rhs_stride_inner + (col_dense_offset + col_range) * rhs_stride_col
    out_ptr += (row_offset + row_range) * out_stride_row + (col_sparse_offset + col_range) * out_stride_col

    # Matrix multiplication
    out = tl.dot(tl.load(lhs_ptr), tl.load(rhs_ptr), out_dtype=tl.float32)
    for k in range(1, inner_dim // block_size_inner):
        lhs_ptr += block_size_inner * lhs_stride_inner
        rhs_ptr += block_size_inner * rhs_stride_inner
        out += tl.dot(tl.load(lhs_ptr), tl.load(rhs_ptr))

    if accumulate:
        out += tl.load(out_ptr)

    # Output
    tl.store(out_ptr, out)


def output_sparse_matmul(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    sparse_map: SparseMap | None,
    out: torch.Tensor | None = None,
    accumulate: bool = False,
):
    """
    Output-sparse matrix multiplication with a sparse column dimension,
    i.e., with a mapping row_index -> sparse_index (obtained from expert_ends).
    Ex.: MLP layer 1 forward (Y = X x W1^T), MLP layer 2 input grad (gY = gZ x W2).
    Formula: out[i, js] = sum_k(lhs[i, k] * rhs[k, jd]), where jd = js + col_sparse_dim * sparse_index[i]
      sparse_index[i] = sum(expert_ends <= i)
    TODO: Assumes sparse_index is constant within a block.
    """
    if sparse_map is None:
        return dense_matmul(lhs, rhs, out, accumulate)
    # Shape
    row_dim, inner_dim = lhs.shape
    inner_dim_1, col_dense_dim = rhs.shape
    Assert.eq(row_dim, sparse_map.num_rows)
    Assert.eq(inner_dim, inner_dim_1)
    col_sparse_dim = div(col_dense_dim, sparse_map.num_experts)
    if out is None:
        assert not accumulate
        out = lhs.new_empty(row_dim, col_sparse_dim)

    grid = lambda meta: (div(row_dim, meta["block_size_row"]), div(col_sparse_dim, meta["block_size_col"]))
    output_sparse_matmul_kernel[grid](
        lhs,
        rhs,
        out,
        sparse_map.expert_ends,
        row_dim,
        col_sparse_dim,
        inner_dim,
        sparse_map.num_experts,
        triton.next_power_of_2(sparse_map.num_experts),
        lhs.stride(0),
        lhs.stride(1),
        rhs.stride(0),
        rhs.stride(1),
        out.stride(0),
        out.stride(1),
        accumulate,
    )
    return out


@triton.autotune(
    configs=autotune_configs,
    # Excluding `row_dim` because it causes the compile time to skyrocket.
    key=["col_dim", "inner_sparse_dim", "sparse_dim"],
)
@triton.jit
def input_inner_sparse_matmul_kernel(
    lhs_ptr,
    rhs_ptr,
    out_ptr,
    expert_ends_ptr,
    row_dim: tl.constexpr,
    col_dim: tl.constexpr,
    inner_sparse_dim: tl.constexpr,
    sparse_dim: tl.constexpr,
    padded_sparse_dim: tl.constexpr,
    lhs_stride_row: tl.constexpr,
    lhs_stride_inner: tl.constexpr,
    rhs_stride_inner: tl.constexpr,
    rhs_stride_col: tl.constexpr,
    out_stride_row: tl.constexpr,
    out_stride_col: tl.constexpr,
    accumulate: tl.constexpr,
    block_size_row: tl.constexpr,
    block_size_col: tl.constexpr,
    block_size_inner: tl.constexpr,
    group_size_row: tl.constexpr,
):
    # Safety checks
    tl.static_assert(row_dim % block_size_row == 0)
    tl.static_assert(col_dim % block_size_col == 0)
    tl.static_assert(inner_sparse_dim % block_size_inner == 0)

    # Reorganize blocks to maximize cache reuse.
    pid_row, pid_col = tl.swizzle2d(
        tl.program_id(axis=0),
        tl.program_id(axis=1),
        row_dim // block_size_row,
        col_dim // block_size_col,
        group_size_row,
    )

    # Grid offsets
    row_offset = pid_row * block_size_row

    sparse_range = tl.arange(0, padded_sparse_dim)
    expert_ends = tl.load(expert_ends_ptr + sparse_range, mask=sparse_range < sparse_dim, other=row_dim)
    sparse_index = tl.sum((expert_ends <= row_offset).to(tl.int64))  # noqa
    if sparse_index == sparse_dim:
        return
    inner_dense_offset = sparse_index * inner_sparse_dim
    col_offset = pid_col * block_size_col

    # Pointers
    row_range = tl.arange(0, block_size_row)[:, None]
    col_range = tl.arange(0, block_size_col)[None, :]
    inner_range = tl.arange(0, block_size_inner)
    lhs_ptr += (row_offset + row_range) * lhs_stride_row + inner_range[None, :] * lhs_stride_inner
    rhs_ptr += (inner_dense_offset + inner_range[:, None]) * rhs_stride_inner + (
        col_offset + col_range
    ) * rhs_stride_col
    out_ptr += (row_offset + row_range) * out_stride_row + (col_offset + col_range) * out_stride_col

    # Matrix multiplication
    out = tl.dot(tl.load(lhs_ptr), tl.load(rhs_ptr), out_dtype=tl.float32)
    for k in range(1, inner_sparse_dim // block_size_inner):
        lhs_ptr += block_size_inner * lhs_stride_inner
        rhs_ptr += block_size_inner * rhs_stride_inner
        out += tl.dot(tl.load(lhs_ptr), tl.load(rhs_ptr))

    if accumulate:
        out += tl.load(out_ptr)

    # Output
    tl.store(out_ptr, out)


def input_inner_sparse_matmul(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    sparse_map: SparseMap | None,
    out: torch.Tensor | None = None,
    accumulate: bool = False,
):
    """
    Left-input-sparse matrix multiplication with a sparse inner dimension,
    i.e., with a mapping row_index -> sparse_index (obtained from expert_ends).
    Ex.: MLP layer 2 forward (Z = Y x W2^T), MLP layer 1 input grad (gX = gY x W1).
    Formula: out[i, j] = sum_ks(lhs[i, ks] * rhs[kd, j]), where kd = ks + inner_sparse_dim * sparse_index[i]
      sparse_index[i] = sum(expert_ends <= i)
    TODO: Assumes sparse_index is constant within a block.
    """
    if sparse_map is None:
        return dense_matmul(lhs, rhs, out, accumulate)
    # Shape
    row_dim, inner_sparse_dim = lhs.shape
    inner_dense_dim, col_dim = rhs.shape
    Assert.eq(inner_sparse_dim, div(inner_dense_dim, sparse_map.num_experts))
    Assert.eq(row_dim, sparse_map.num_rows)

    if out is None:
        assert not accumulate
        out = lhs.new_empty(row_dim, col_dim)

    grid = lambda meta: (div(row_dim, meta["block_size_row"]), div(col_dim, meta["block_size_col"]))
    input_inner_sparse_matmul_kernel[grid](
        lhs.data,
        rhs,
        out,
        sparse_map.expert_ends,
        row_dim,
        col_dim,
        inner_sparse_dim,
        sparse_map.num_experts,
        triton.next_power_of_2(sparse_map.num_experts),
        lhs.stride(0),
        lhs.stride(1),
        rhs.stride(0),
        rhs.stride(1),
        out.stride(0),
        out.stride(1),
        accumulate,
    )
    return out


@triton.autotune(
    configs=autotune_configs,
    # Excluding `inner_dim` because it causes the compile time to skyrocket.
    key=["row_dense_dim", "row_sparse_dim", "col_dim"],
)
@triton.jit
def input_row_sparse_matmul_kernel(
    lhs_ptr,
    rhs_ptr,
    out_ptr,
    expert_ends_ptr,
    expert_pad_begins_ptr,
    row_dense_dim: tl.constexpr,
    row_sparse_dim: tl.constexpr,
    col_dim: tl.constexpr,
    inner_dim: tl.constexpr,
    lhs_stride_row: tl.constexpr,
    lhs_stride_inner: tl.constexpr,
    rhs_stride_inner: tl.constexpr,
    rhs_stride_col: tl.constexpr,
    out_stride_row: tl.constexpr,
    out_stride_col: tl.constexpr,
    accumulate: tl.constexpr,
    block_size_row: tl.constexpr,
    block_size_col: tl.constexpr,
    block_size_inner: tl.constexpr,
    group_size_row: tl.constexpr,
):
    # Safety checks
    tl.static_assert(row_sparse_dim % block_size_row == 0)
    tl.static_assert(col_dim % block_size_col == 0)
    tl.static_assert(inner_dim % block_size_inner == 0)
    tl.static_assert(row_dense_dim % row_sparse_dim == 0)

    # Reorganize blocks to maximize cache reuse.
    pid_row, pid_col = tl.swizzle2d(
        tl.program_id(axis=0),
        tl.program_id(axis=1),
        row_dense_dim // block_size_row,
        col_dim // block_size_col,
        group_size_row,
    )

    # Grid offsets
    row_dense_offset = pid_row * block_size_row
    sparse_index = row_dense_offset // row_sparse_dim
    row_sparse_offset = row_dense_offset % row_sparse_dim
    col_offset = pid_col * block_size_col
    inner_begin = tl.load(expert_ends_ptr + sparse_index - 1, mask=sparse_index > 0, other=0)
    inner_end = tl.load(expert_pad_begins_ptr + sparse_index)
    inner_offset = (inner_begin // block_size_inner) * block_size_inner

    # Pointers
    row_range = tl.arange(0, block_size_row)[:, None]
    col_range = tl.arange(0, block_size_col)[None, :]
    inner_range = tl.arange(0, block_size_inner) + inner_offset
    lhs_ptr += (row_sparse_offset + row_range) * lhs_stride_row
    rhs_ptr += (col_offset + col_range) * rhs_stride_col
    out_ptr += (row_dense_offset + row_range) * out_stride_row + (col_offset + col_range) * out_stride_col

    # Matrix multiplication
    mask = (inner_begin <= inner_range) & (inner_range < inner_end)
    out = tl.dot(
        tl.load(lhs_ptr + inner_range[None, :] * lhs_stride_inner, mask=mask[None, :], other=0),
        tl.load(rhs_ptr + inner_range[:, None] * rhs_stride_inner, mask=mask[:, None], other=0),
    )
    for i in range(1, tl.cdiv(inner_end - inner_offset, block_size_inner)):
        inner_range += block_size_inner
        mask = (inner_begin <= inner_range) & (inner_range < inner_end)
        out += tl.dot(
            tl.load(lhs_ptr + inner_range[None, :] * lhs_stride_inner, mask=mask[None, :], other=0),
            tl.load(rhs_ptr + inner_range[:, None] * rhs_stride_inner, mask=mask[:, None], other=0),
        )

    if accumulate:
        out += tl.load(out_ptr)

    # Output
    tl.store(out_ptr, out)


def input_row_sparse_matmul(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    sparse_map: SparseMap | None,
    out: torch.Tensor | None = None,
    accumulate: bool = False,
):
    """
    Left-input-sparse matrix multiplication with a sparse row dimension,
    i.e., with a mapping inner_index -> sparse_index.
    Ex.: MLP layer 1 weight grad (gW1 = gY^T x X), MLP layer 2 weight grad (gW2^T = Y^T x gZ).
    Formula: out[id, j] = sum_ks(lhs[is, ks] * rhs[ks, j]), where
      sparse_begin[sparse_index[id]] <= ks < sparse_end[sparse_index[id]],
      id = is + row_sparse_dim * sparse_index[id],
      sparse_index[id] = id // row_sparse_dim
    TODO: Assumes sparse_begin, sparse_end are multiple of BLOCK_SIZE_INNER.
    """
    if sparse_map is None:
        return dense_matmul(lhs, rhs, out, accumulate)
    # Shape
    row_sparse_dim, inner_dim = lhs.shape
    inner_dim_1, col_dim = rhs.shape
    Assert.eq(inner_dim, inner_dim_1)
    Assert.eq(inner_dim, sparse_map.num_rows)

    row_dense_dim = row_sparse_dim * sparse_map.num_experts

    if out is None:
        assert not accumulate
        out = lhs.new_empty(row_dense_dim, col_dim)

    grid = lambda meta: (div(row_dense_dim, meta["block_size_row"]), div(col_dim, meta["block_size_col"]))
    input_row_sparse_matmul_kernel[grid](
        lhs,
        rhs,
        out,
        sparse_map.expert_ends,
        sparse_map.expert_pad_begins,
        row_dense_dim,
        row_sparse_dim,
        col_dim,
        inner_dim,
        lhs.stride(0),
        lhs.stride(1),
        rhs.stride(0),
        rhs.stride(1),
        out.stride(0),
        out.stride(1),
        accumulate,
    )
    return out


class OutputSparseLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lhs, rhs, sparse_map):  # noqa
        ctx.sparse_map = sparse_map
        ctx.save_for_backward(lhs, rhs)
        return output_sparse_matmul(lhs, rhs, sparse_map)

    @staticmethod
    def backward(ctx, grad_out):  # noqa
        grad_out = grad_out.contiguous()
        lhs, rhs = ctx.saved_tensors
        grad_lhs = input_inner_sparse_matmul(grad_out, rhs.t(), ctx.sparse_map)
        grad_rhs = input_row_sparse_matmul(lhs.t(), grad_out, ctx.sparse_map).t()
        return grad_lhs, grad_rhs, None, None


output_sparse_linear_autograd = OutputSparseLinear.apply


class InputSparseLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lhs, rhs, sparse_map):  # noqa
        ctx.sparse_map = sparse_map
        ctx.save_for_backward(lhs, rhs)
        return input_inner_sparse_matmul(lhs, rhs, sparse_map)

    @staticmethod
    def backward(ctx, grad_out):  # noqa
        grad_out = grad_out.contiguous()
        lhs, rhs = ctx.saved_tensors
        grad_lhs = output_sparse_matmul(grad_out, rhs.t(), ctx.sparse_map)
        grad_rhs = input_row_sparse_matmul(grad_out.t(), lhs, ctx.sparse_map)
        return grad_lhs, grad_rhs, None, None


input_sparse_linear_autograd = InputSparseLinear.apply
