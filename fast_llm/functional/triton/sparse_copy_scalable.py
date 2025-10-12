"""
Scalable sparse map kernel for large numbers of experts (e.g., 128+).

The original sparse_map_kernel uses one-hot encoding which creates large
intermediate tensors (block_size x num_experts) that exceed register limits
for num_experts > 32.

This implementation uses a multi-pass approach with atomic operations to
handle arbitrary numbers of experts efficiently.
"""

import torch

from fast_llm.functional.config import MAX_DROPLESS_BLOCK_SIZE_ROW
from fast_llm.functional.triton import tl, tl_constexpr, triton, triton_jit


@triton_jit()
def sparse_map_histogram_kernel(
    top_experts_ptr,
    expert_counts_ptr,
    num_sparse_rows: tl_constexpr,
    num_experts: tl_constexpr,
    block_size: tl_constexpr,
):
    """
    First pass: Count tokens per expert using atomic operations.
    This avoids materializing large one-hot matrices.
    """
    block_start = tl.program_id(0) * block_size
    offsets = tl.arange(0, block_size) + block_start
    mask = offsets < num_sparse_rows

    # Load expert indices for this block
    expert_indices = tl.load(top_experts_ptr + offsets, mask=mask, other=num_experts)

    # Atomically increment counts for each expert
    # This is much more efficient than one-hot encoding for large num_experts
    for i in range(block_size):
        if block_start + i < num_sparse_rows:
            expert_id = tl.load(top_experts_ptr + block_start + i)
            # Atomic add ensures thread-safety across all blocks
            tl.atomic_add(expert_counts_ptr + expert_id, 1)


@triton_jit()
def sparse_map_assign_kernel(
    top_experts_ptr,
    sparse_rows_ptr,
    expert_begins_ptr,
    expert_atomic_counters_ptr,
    num_sparse_rows: tl_constexpr,
    block_size: tl_constexpr,
):
    """
    Second pass: Assign sparse row indices using atomic counters per expert.
    Each token atomically claims the next available slot for its expert.
    """
    block_start = tl.program_id(0) * block_size
    offsets = tl.arange(0, block_size) + block_start
    mask = offsets < num_sparse_rows

    # Load expert indices
    expert_indices = tl.load(top_experts_ptr + offsets, mask=mask, other=0)

    # For each token, atomically claim a slot in its expert's range
    for i in range(block_size):
        if block_start + i < num_sparse_rows:
            expert_id = tl.load(top_experts_ptr + block_start + i)
            expert_begin = tl.load(expert_begins_ptr + expert_id)

            # Atomically get the next available index for this expert
            local_offset = tl.atomic_add(expert_atomic_counters_ptr + expert_id, 1)
            sparse_row = expert_begin + local_offset

            tl.store(sparse_rows_ptr + block_start + i, sparse_row)


@triton_jit()
def sparse_map_assign_chunked_kernel(
    top_experts_ptr,
    sparse_rows_ptr,
    expert_begins_ptr,
    expert_chunk_start: tl_constexpr,
    expert_chunk_size: tl_constexpr,
    num_sparse_rows: tl_constexpr,
    block_size: tl_constexpr,
    dtype: tl_constexpr,
):
    """
    Alternative second pass: Process experts in chunks to reduce memory pressure.
    This processes only expert_chunk_size experts at a time, scanning through all tokens.

    Better for very large num_experts as it keeps working set small.
    """
    block_start = tl.program_id(0) * block_size
    offsets = tl.arange(0, block_size) + block_start
    mask = offsets < num_sparse_rows

    # Load expert indices for this block
    expert_indices = tl.load(top_experts_ptr + offsets, mask=mask, other=-1)

    # Process experts in the current chunk
    expert_range = tl.arange(0, expert_chunk_size) + expert_chunk_start
    expert_begins = tl.load(expert_begins_ptr + expert_range)

    # For each expert in chunk, find matching tokens and assign indices
    for expert_offset in range(expert_chunk_size):
        expert_id = expert_chunk_start + expert_offset
        expert_begin = tl.load(expert_begins_ptr + expert_id)

        # Find tokens going to this expert
        matches = (expert_indices == expert_id).to(dtype)

        # Compute cumulative sum to get local indices (0, 1, 2, ...)
        # This gives each matching token a unique consecutive index
        cumsum = tl.cumsum(matches)
        local_indices = (cumsum - matches) * matches  # Shift by 1 and mask

        # Compute final sparse row indices
        sparse_rows = (expert_begin + local_indices) * matches

        # Store results (only for matching tokens)
        # Use max to handle non-matching tokens (they get 0 which we ignore)
        current_values = tl.load(sparse_rows_ptr + offsets, mask=mask, other=0)
        new_values = tl.maximum(current_values, sparse_rows)
        tl.store(sparse_rows_ptr + offsets, new_values, mask=mask)


def get_sparse_map_scalable(
    top_experts: torch.Tensor,
    num_experts: int,
    *,
    pad_to_multiple: int = MAX_DROPLESS_BLOCK_SIZE_ROW,
    block_size: int = 1024,
    expert_chunk_size: int = 32,
    use_atomic_assign: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Scalable sparse map computation for large numbers of experts.

    Args:
        top_experts: [num_rows_dense, num_experts_per_token] tensor of expert indices
        num_experts: Total number of experts
        pad_to_multiple: Padding for each expert's allocation
        block_size: Block size for Triton kernels
        expert_chunk_size: Number of experts to process at once (for chunked approach)
        use_atomic_assign: If True, use atomic-based assignment (faster but requires more memory)
                          If False, use chunked approach (slower but more memory efficient)

    Returns:
        expert_ends: Cumulative end index for each expert (including padding)
        expert_pad_begins: Start of padding for each expert
        sparse_rows: Remapped row indices [num_rows_dense, num_experts_per_token]
    """
    device = top_experts.device
    dtype = top_experts.dtype
    num_rows_dense, num_experts_per_token = top_experts.shape
    num_sparse_rows = num_rows_dense * num_experts_per_token

    # Pass 1: Histogram using atomics
    expert_counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
    num_blocks = triton.cdiv(num_sparse_rows, block_size)

    sparse_map_histogram_kernel[(num_blocks,)](
        top_experts.flatten(),
        expert_counts,
        num_sparse_rows,
        num_experts,
        block_size,
    )

    # Compute padded counts and offsets (on CPU is fine, small tensor)
    expert_counts_cpu = expert_counts.cpu()
    padded_counts = ((expert_counts_cpu + pad_to_multiple - 1) // pad_to_multiple * pad_to_multiple)
    expert_ends = padded_counts.cumsum(0).to(device)
    expert_begins = expert_ends - padded_counts.to(device)
    expert_pad_begins = expert_begins + expert_counts

    # Pass 2: Assign sparse indices
    sparse_rows = torch.empty_like(top_experts, dtype=torch.int32)

    if use_atomic_assign:
        # Faster approach: Use atomic counters per expert
        expert_atomic_counters = torch.zeros(num_experts, dtype=torch.int32, device=device)

        sparse_map_assign_kernel[(num_blocks,)](
            top_experts.flatten(),
            sparse_rows.flatten(),
            expert_begins,
            expert_atomic_counters,
            num_sparse_rows,
            block_size,
        )
    else:
        # Memory-efficient approach: Process experts in chunks
        sparse_rows.fill_(0)  # Initialize

        for chunk_start in range(0, num_experts, expert_chunk_size):
            chunk_end = min(chunk_start + expert_chunk_size, num_experts)
            actual_chunk_size = chunk_end - chunk_start

            sparse_map_assign_chunked_kernel[(num_blocks,)](
                top_experts.flatten(),
                sparse_rows.flatten(),
                expert_begins,
                chunk_start,
                actual_chunk_size,
                num_sparse_rows,
                block_size,
                torch.int32,  # dtype for intermediate calculations
            )

    return expert_ends, expert_pad_begins, sparse_rows
