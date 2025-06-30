import dataclasses

import torch

from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.functional.autograd import wrap_forward_backward
from fast_llm.functional.config import MAX_DROPLESS_BLOCK_SIZE_ROW, TritonConfig
from fast_llm.functional.triton import tl, tl_constexpr, triton, triton_jit


@dataclasses.dataclass()
class SparseMap:
    sparse_rows: torch.Tensor
    # The end row for each expert, including padding. `expert_ends[i] = expert_begins[i] + padded_tokens_per_expert[i]`
    expert_ends: torch.Tensor
    # The end row for each expert, excluding padding. `expert_pad_begins[i] = expert_begins[i] + unpadded_tokens_per_expert[i]`
    expert_pad_begins: torch.Tensor
    # The number of rows un the dense tensor, i.e., the number of tokens.
    num_rows_dense: int
    # The number of sparse rows, including padding. `num_rows = expert_ends[-1]`
    num_rows: int
    # The number of sparse rows, excluding padding. `num_rows_unpadded = num_rows_dense * num_experts_per_token`
    num_rows_unpadded: int
    num_experts: int
    num_experts_per_token: int


@triton_jit()
def copy_dense_to_sparse_kernel(
    input_ptr,
    output_ptr,
    scores_ptr,
    sparse_rows_ptr,
    num_columns: tl_constexpr,
    num_experts_per_token: tl_constexpr,
    block_size: tl_constexpr,
):
    dense_row = tl.program_id(0)
    offsets = tl.arange(0, block_size) + block_size * tl.program_id(1)
    mask = None if num_columns % block_size == 0 else offsets < num_columns
    out = tl.load(input_ptr + dense_row * num_columns + offsets, mask=mask)
    # Write to each expert.
    for top_index in range(num_experts_per_token):
        sparse_row = tl.load(sparse_rows_ptr + dense_row * num_experts_per_token + top_index)
        out_scaled = (
            out
            if scores_ptr is None
            else out * tl.load(scores_ptr + dense_row * num_experts_per_token + top_index).to(tl.float32)
        )
        tl.store(output_ptr + sparse_row * num_columns + offsets, out_scaled, mask=mask)


def copy_dense_to_sparse(input_: torch.Tensor, scores: torch.Tensor | None, sparse_map: SparseMap) -> torch.Tensor:
    # output[sparse_row[dense_row, :]] = input_[dense_row, None] * (score[dense_row, :] or 1)
    hidden_size = input_.size(1)
    # A worst-case static shape.
    out = input_.new_empty((sparse_map.num_rows, hidden_size))
    copy_dense_to_sparse_kernel[(input_.size(0), triton.cdiv(hidden_size, TritonConfig.POINTWISE_BLOCK_SIZE))](
        input_,
        out,
        scores,
        sparse_map.sparse_rows,
        hidden_size,  # noqa
        sparse_map.num_experts_per_token,  # noqa
        TritonConfig.POINTWISE_BLOCK_SIZE,
    )
    return out


@triton_jit()
def copy_sparse_to_dense_kernel(
    input_ptr,
    output_ptr,
    scores_ptr,
    sparse_rows_ptr,
    num_columns: tl_constexpr,
    num_experts_per_token: tl_constexpr,
    block_size: tl_constexpr,
):
    dense_row = tl.program_id(0)
    offsets = tl.arange(0, block_size) + block_size * tl.program_id(1)
    mask = None if num_columns % block_size == 0 else offsets < num_columns
    out = tl.zeros((block_size,), tl.float32)
    # Sum over experts.
    for top_index in range(num_experts_per_token):
        sparse_row = tl.load(sparse_rows_ptr + dense_row * num_experts_per_token + top_index)
        input_ = tl.load(input_ptr + sparse_row * num_columns + offsets, mask=mask)
        if scores_ptr is not None:
            input_ *= tl.load(scores_ptr + dense_row * num_experts_per_token + top_index).to(tl.float32)
        out += input_
    tl.store(output_ptr + dense_row * num_columns + offsets, out, mask=mask)


def copy_sparse_to_dense(input_: torch.Tensor, scores: torch.Tensor | None, sparse_map: SparseMap) -> torch.Tensor:
    # output[dense_row] = (input_[sparse_row[dense_row, :]] * (score[dense_row, :] or 1).sum(1)
    hidden_size = input_.size(1)
    out = input_.new_empty((sparse_map.num_rows_dense, hidden_size))
    copy_sparse_to_dense_kernel[
        (sparse_map.num_rows_dense, triton.cdiv(hidden_size, TritonConfig.POINTWISE_BLOCK_SIZE))
    ](
        input_,
        out,
        scores,
        sparse_map.sparse_rows,
        hidden_size,  # noqa
        sparse_map.num_experts_per_token,  # noqa
        TritonConfig.POINTWISE_BLOCK_SIZE,
    )
    return out


@triton_jit()
def copy_sparse_to_dense_grad_score_kernel(
    input_ptr,
    grad_output_ptr,
    grad_scores_ptr,
    sparse_rows_ptr,
    num_columns: tl_constexpr,
    num_experts_per_token: tl_constexpr,
    block_size: tl_constexpr,
):
    dense_row = tl.program_id(0)
    top_index = tl.program_id(1)
    sparse_row = tl.load(sparse_rows_ptr + dense_row * num_experts_per_token + top_index)

    grad_output_ptr += dense_row * num_columns
    input_ptr += sparse_row * num_columns
    offsets = tl.arange(0, block_size)

    if num_columns % block_size == 0:
        grad_scores = tl.load(input_ptr + offsets).to(tl.float32) * tl.load(grad_output_ptr + offsets).to(tl.float32)
    else:
        mask = offsets < num_columns
        grad_scores = tl.load(input_ptr + offsets, mask=mask).to(tl.float32) * tl.load(
            grad_output_ptr + offsets, mask=mask
        ).to(tl.float32)
    for i in range(1, tl.cdiv(num_columns, block_size)):
        offsets += block_size
        if num_columns % block_size == 0:
            grad_scores += tl.load(input_ptr + offsets).to(tl.float32) * tl.load(grad_output_ptr + offsets).to(
                tl.float32
            )
        else:
            mask = offsets < num_columns
            grad_scores += tl.load(input_ptr + offsets, mask=mask).to(tl.float32) * tl.load(
                grad_output_ptr + offsets, mask=mask
            ).to(tl.float32)

    tl.store(grad_scores_ptr + dense_row * num_experts_per_token + top_index, tl.sum(grad_scores))


def copy_sparse_to_dense_grad_score(
    input_: torch.Tensor, grad_output: torch.Tensor, sparse_map: SparseMap
) -> torch.Tensor:
    # grad_scores[dense_row, top_index] = (input_[sparse_row[dense_row, top_index]]*grad_output[dense_row]).sum()
    out = input_.new_empty(sparse_map.num_rows_dense, sparse_map.num_experts_per_token)
    copy_sparse_to_dense_grad_score_kernel[(sparse_map.num_rows_dense, sparse_map.num_experts_per_token)](
        input_,
        grad_output,
        out,
        sparse_map.sparse_rows,
        input_.size(1),  # noqa
        sparse_map.num_experts_per_token,  # noqa
        TritonConfig.POINTWISE_BLOCK_SIZE,
    )
    return out


def copy_dense_to_sparse_forward(
    input_: torch.Tensor, sparse_map: SparseMap
) -> tuple[torch.Tensor, tuple[SparseMap, tuple[int, ...]]]:
    return copy_dense_to_sparse(input_.flatten(0, -2), None, sparse_map), (sparse_map, input_.shape)


def copy_dense_to_sparse_backward(
    grad_output: torch.Tensor, context: tuple[SparseMap, tuple[int, ...]]
) -> torch.Tensor:
    sparse_map, input_shape = context
    return copy_sparse_to_dense(grad_output.contiguous(), None, sparse_map).view(input_shape)


copy_dense_to_sparse_autograd = wrap_forward_backward(copy_dense_to_sparse_forward, copy_dense_to_sparse_backward)


def copy_sparse_to_dense_forward(
    input_: torch.Tensor, scores: torch.Tensor, sparse_map: SparseMap
) -> tuple[torch.Tensor, tuple[SparseMap, torch.Tensor, torch.Tensor]]:
    return copy_sparse_to_dense(input_, scores, sparse_map), (sparse_map, input_, scores)


def copy_sparse_to_dense_backward(grad_output: torch.Tensor, context: tuple[SparseMap, torch.Tensor, torch.Tensor]):
    sparse_map, input_, scores = context
    grad_input = copy_dense_to_sparse(grad_output, scores, sparse_map)
    grad_scores = copy_sparse_to_dense_grad_score(input_, grad_output, sparse_map)
    return grad_input, grad_scores


copy_sparse_to_dense_autograd = wrap_forward_backward(copy_sparse_to_dense_forward, copy_sparse_to_dense_backward)


@triton_jit()
def sparse_map_kernel(
    top_experts_ptr,
    expert_ends_ptr,
    expert_pad_begins_ptr,
    sparse_rows_ptr,
    num_sparse_rows: tl_constexpr,
    num_experts: tl_constexpr,
    pad_to_multiple: tl_constexpr,
    block_size: tl_constexpr,
    block_size_expert: tl_constexpr,
    dtype: tl_constexpr,
):
    """
    Since the methods we want (histogram, argsort) are not readily available in triton,
    we use a one-hot representation to get the quantities we want.
    TODO: Next triton release will support tl.histogram, maybe argsort.
    """
    block_range = tl.arange(0, block_size)
    expert_range = tl.arange(0, block_size_expert)
    expert_mask = None if block_size_expert == num_experts else expert_range < num_experts

    if num_sparse_rows >= block_size:
        expert_index = tl.load(top_experts_ptr + block_range)
    else:
        # Mask outside the expert range so the masked values never contribute to the expert counts.
        expert_index = tl.load(top_experts_ptr + block_range, mask=block_range < num_sparse_rows, other=num_experts)

    # Count the number of tokens per expert for each block (tl.histogram), and sum over blocks.
    expert_counts = tl.sum((expert_index[:, None] == expert_range[None, :]).to(dtype), 0)  # noqa
    for i in range(1, tl.cdiv(num_sparse_rows, block_size)):
        block_range += block_size
        if num_sparse_rows % block_size == 0:
            expert_index = tl.load(top_experts_ptr + block_range)
        else:
            expert_index = tl.load(
                top_experts_ptr + block_range, mask=block_range < num_sparse_rows, other=num_experts
            )
        expert_counts += tl.sum((expert_index[:, None] == expert_range[None, :]).to(dtype), 0)  # noqa

    if pad_to_multiple is None:
        expert_counts_padded = expert_counts
    else:
        # Round up to the next multiple.
        expert_counts_padded = (expert_counts + pad_to_multiple - 1) // pad_to_multiple * pad_to_multiple

    expert_ends = tl.cumsum(expert_counts_padded)
    expert_begins = expert_ends - expert_counts_padded

    if expert_ends_ptr is not None:
        tl.store(expert_ends_ptr + expert_range, expert_ends, mask=expert_mask)

    if expert_pad_begins_ptr is not None:
        tl.store(expert_pad_begins_ptr + expert_range, expert_begins + expert_counts, mask=expert_mask)

    if sparse_rows_ptr is not None:
        # Assign a new unique index to each row so that it lies in the range (expert_begin, expert_end)
        # for its assigned expert.
        block_range = tl.arange(0, block_size)
        for i in range(tl.cdiv(num_sparse_rows, block_size)):
            if num_sparse_rows % block_size == 0:
                mask = None
                expert_index = tl.load(top_experts_ptr + block_range)
            else:
                mask = block_range < num_sparse_rows
                expert_index = tl.load(top_experts_ptr + block_range, mask=mask, other=num_experts)
            expert_one_hot = (expert_index[:, None] == expert_range[None, :]).to(dtype)  # noqa
            # Use a cumsum and add block begins so that ones in each row become consecutive integers (starting at 1),
            # then shift these ranges to the expert ranges using the begins as offsets,
            # and filter out the relevant values by multiplying by the one-hot matrix.
            expert_offsets = (tl.cumsum(expert_one_hot, 0) + expert_begins[None, :]) * expert_one_hot
            # At this point each column contains exactly one non-zero value corresponding to the sparse row index +1,
            # which we recover with a sum.
            tl.store(sparse_rows_ptr + block_range, tl.sum(expert_offsets, 1) - 1, mask=mask)
            # Advance the expert begins so that we don't reuse the same indices in the next block.
            expert_begins += tl.sum(expert_one_hot, 0)
            block_range += block_size


def sparse_map_pytorch(
    top_experts: torch.Tensor,
    num_experts: int,
    pad_to_multiple: int = MAX_DROPLESS_BLOCK_SIZE_ROW,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    expert_counts = torch.bincount(top_experts.flatten(), minlength=num_experts)
    assert expert_counts.numel() == num_experts
    padded_expert_counts = (expert_counts + pad_to_multiple - 1) // pad_to_multiple * pad_to_multiple
    expert_ends = padded_expert_counts.cumsum(0)
    expert_begins = expert_ends - padded_expert_counts
    expert_pad_begins = expert_begins + expert_counts
    sparse_rows_unpadded = top_experts.flatten().sort()[1].sort()[1].view_as(top_experts)
    expert_begins_unpadded = expert_counts.cumsum(0) - expert_counts
    sparse_rows = sparse_rows_unpadded - expert_begins_unpadded[top_experts] + expert_begins[top_experts]
    return expert_ends, expert_pad_begins, sparse_rows


def get_sparse_map(
    top_experts: torch.Tensor,
    num_experts: int,
    *,
    dynamic_shape: bool = False,
    pad_to_multiple: int = MAX_DROPLESS_BLOCK_SIZE_ROW,
    block_size=TritonConfig.POINTWISE_BLOCK_SIZE,
    use_triton: bool | None = None,
) -> SparseMap:
    num_rows_dense, num_experts_per_token = top_experts.shape
    num_rows_unpadded = num_rows_dense * num_experts_per_token
    max_rows = (num_rows_unpadded + num_experts * pad_to_multiple) // pad_to_multiple * pad_to_multiple
    dtype = torch.int16 if max_rows < 32768 else torch.int32
    if (use_triton is None and TritonConfig.TRITON_ENABLED) or use_triton:
        expert_ends, expert_pad_begins = top_experts.new_empty((2 * num_experts,), dtype=dtype).chunk(2)
        sparse_rows = expert_ends.new_empty(num_rows_dense, num_experts_per_token)
        sparse_map_kernel[(triton.cdiv(num_rows_dense, block_size),)](
            top_experts,
            expert_ends,
            expert_pad_begins,
            sparse_rows,
            num_rows_unpadded,
            num_experts,  # noqa
            pad_to_multiple,  # noqa
            block_size,
            triton.next_power_of_2(num_experts),
            DataType.from_torch(dtype).triton,
        )
    else:
        expert_ends, expert_pad_begins, sparse_rows = sparse_map_pytorch(top_experts, num_experts, pad_to_multiple)

    return SparseMap(
        sparse_rows=sparse_rows,
        expert_ends=expert_ends,
        expert_pad_begins=expert_pad_begins,
        num_rows_dense=num_rows_dense,
        num_rows=expert_ends[-1].item() if dynamic_shape else max_rows,
        num_rows_unpadded=num_rows_unpadded,
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
    )
