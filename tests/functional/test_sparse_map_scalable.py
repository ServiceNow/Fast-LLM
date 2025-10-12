"""
Tests for scalable sparse map kernel that supports large numbers of experts (128+).

Tests verify:
1. New kernel matches old kernel for small num_experts (<=32)
2. New kernel matches PyTorch fallback for large num_experts (>32)
3. Correctness of histogram and index assignment
4. Both atomic and chunked variants work correctly
"""

import pytest
import torch

from fast_llm.functional.triton.sparse_copy import get_sparse_map, sparse_map_pytorch
from fast_llm.functional.triton.sparse_copy_scalable import get_sparse_map_scalable
from fast_llm.utils import Assert


def generate_test_experts(num_tokens: int, num_experts: int, experts_per_token: int, device: str = "cuda"):
    """Generate random expert assignments for testing."""
    return torch.randint(0, num_experts, (num_tokens, experts_per_token), device=device)


def validate_sparse_map_correctness(
    sparse_rows: torch.Tensor,
    expert_ends: torch.Tensor,
    expert_pad_begins: torch.Tensor,
    top_experts: torch.Tensor,
    num_experts: int,
):
    """
    Validate that a sparse map satisfies all invariants:
    1. All sparse_rows are unique within each expert's range
    2. Expert ranges are non-overlapping and consecutive
    3. Token counts match histogram
    """
    num_tokens, experts_per_token = top_experts.shape

    # Check expert ranges are valid
    expert_begins = torch.cat([torch.tensor([0], device=expert_ends.device), expert_ends[:-1]])
    assert torch.all(expert_begins <= expert_pad_begins), "Pad begins must be >= begins"
    assert torch.all(expert_pad_begins <= expert_ends), "Pad begins must be <= ends"

    # Check each token's assignment
    for token_idx in range(num_tokens):
        for expert_slot in range(experts_per_token):
            expert_id = top_experts[token_idx, expert_slot].item()
            sparse_row = sparse_rows[token_idx, expert_slot].item()

            # Check sparse_row is in valid range for this expert
            expert_begin = expert_begins[expert_id].item()
            expert_end = expert_ends[expert_id].item()
            assert expert_begin <= sparse_row < expert_end, (
                f"Token {token_idx} expert {expert_id}: "
                f"sparse_row {sparse_row} not in range [{expert_begin}, {expert_end})"
            )

    # Check uniqueness: all sparse_rows should be unique (no collisions)
    all_sparse_rows = sparse_rows.flatten()
    unique_sparse_rows = torch.unique(all_sparse_rows)
    assert len(unique_sparse_rows) == len(all_sparse_rows), "Sparse rows must be unique (no collisions)"

    # Check histogram correctness
    flat_experts = top_experts.flatten()
    for expert_id in range(num_experts):
        expected_count = (flat_experts == expert_id).sum().item()
        expert_begin = expert_begins[expert_id].item()
        expert_pad_begin = expert_pad_begins[expert_id].item()
        actual_count = expert_pad_begin - expert_begin

        assert actual_count == expected_count, (
            f"Expert {expert_id}: count mismatch. Expected {expected_count}, got {actual_count}"
        )


@pytest.mark.parametrize("num_experts", [4, 8, 16, 32])
@pytest.mark.parametrize("num_tokens", [64, 256, 1024])
@pytest.mark.parametrize("experts_per_token", [1, 2, 4])
@pytest.mark.parametrize("use_atomic", [True, False])
def test_scalable_kernel_matches_original_small(num_experts, num_tokens, experts_per_token, use_atomic):
    """
    Test that the new scalable kernel produces identical results to the original kernel
    for small numbers of experts where the original works fine.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = "cuda"
    top_experts = generate_test_experts(num_tokens, num_experts, experts_per_token, device)

    # Get results from original kernel
    sparse_map_original = get_sparse_map(top_experts, num_experts, use_triton=True)

    # Get results from new scalable kernel
    expert_ends_new, expert_pad_begins_new, sparse_rows_new = get_sparse_map_scalable(
        top_experts, num_experts, use_atomic_assign=use_atomic
    )

    # Results should be identical
    torch.testing.assert_close(sparse_map_original.expert_ends, expert_ends_new)
    torch.testing.assert_close(sparse_map_original.expert_pad_begins, expert_pad_begins_new)
    torch.testing.assert_close(sparse_map_original.sparse_rows, sparse_rows_new)

    # Validate correctness
    validate_sparse_map_correctness(
        sparse_rows_new, expert_ends_new, expert_pad_begins_new, top_experts, num_experts
    )


@pytest.mark.parametrize("num_experts", [64, 96, 128, 256])
@pytest.mark.parametrize("num_tokens", [128, 512])
@pytest.mark.parametrize("experts_per_token", [2, 4])
@pytest.mark.parametrize("use_atomic", [True, False])
def test_scalable_kernel_matches_pytorch_large(num_experts, num_tokens, experts_per_token, use_atomic):
    """
    Test that the new scalable kernel produces results matching the PyTorch fallback
    for large numbers of experts where the original Triton kernel may fail.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = "cuda"
    top_experts = generate_test_experts(num_tokens, num_experts, experts_per_token, device)

    # Get results from PyTorch fallback (always correct reference)
    expert_ends_pytorch, expert_pad_begins_pytorch, sparse_rows_pytorch = sparse_map_pytorch(
        top_experts, num_experts
    )

    # Get results from new scalable kernel
    expert_ends_new, expert_pad_begins_new, sparse_rows_new = get_sparse_map_scalable(
        top_experts, num_experts, use_atomic_assign=use_atomic
    )

    # Results should match PyTorch reference
    torch.testing.assert_close(expert_ends_pytorch, expert_ends_new)
    torch.testing.assert_close(expert_pad_begins_pytorch, expert_pad_begins_new)
    torch.testing.assert_close(sparse_rows_pytorch, sparse_rows_new)

    # Validate correctness
    validate_sparse_map_correctness(
        sparse_rows_new, expert_ends_new, expert_pad_begins_new, top_experts, num_experts
    )


@pytest.mark.parametrize("num_experts", [128])
@pytest.mark.parametrize("num_tokens", [1024])
@pytest.mark.parametrize("experts_per_token", [4])
def test_gpt_oss_config(num_experts, num_tokens, experts_per_token):
    """
    Test with GPT-OSS specific configuration: 128 experts, 4 active per token.
    This is the primary use case that motivated the scalable kernel.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = "cuda"
    top_experts = generate_test_experts(num_tokens, num_experts, experts_per_token, device)

    # Test both atomic and chunked variants
    for use_atomic in [True, False]:
        expert_ends, expert_pad_begins, sparse_rows = get_sparse_map_scalable(
            top_experts, num_experts, use_atomic_assign=use_atomic
        )

        # Validate correctness
        validate_sparse_map_correctness(sparse_rows, expert_ends, expert_pad_begins, top_experts, num_experts)

        # Verify it matches PyTorch reference
        expert_ends_ref, expert_pad_begins_ref, sparse_rows_ref = sparse_map_pytorch(top_experts, num_experts)
        torch.testing.assert_close(expert_ends, expert_ends_ref)
        torch.testing.assert_close(expert_pad_begins, expert_pad_begins_ref)
        torch.testing.assert_close(sparse_rows, sparse_rows_ref)


def test_edge_cases():
    """Test edge cases like single expert, all tokens to one expert, etc."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = "cuda"

    # Edge case 1: Single expert (degenerate but should work)
    top_experts = torch.zeros((100, 2), dtype=torch.long, device=device)
    expert_ends, expert_pad_begins, sparse_rows = get_sparse_map_scalable(top_experts, 1)
    validate_sparse_map_correctness(sparse_rows, expert_ends, expert_pad_begins, top_experts, 1)

    # Edge case 2: All tokens go to same expert (worst case for load balancing)
    top_experts = torch.full((100, 4), 7, dtype=torch.long, device=device)
    expert_ends, expert_pad_begins, sparse_rows = get_sparse_map_scalable(top_experts, 16)
    validate_sparse_map_correctness(sparse_rows, expert_ends, expert_pad_begins, top_experts, 16)

    # Edge case 3: Perfectly balanced distribution
    num_tokens = 128
    num_experts = 64
    experts_per_token = 2
    # Each token gets consecutive expert pairs: [0,1], [2,3], [4,5], ...
    top_experts = torch.arange(num_tokens * experts_per_token, device=device).view(num_tokens, experts_per_token)
    top_experts = top_experts % num_experts
    expert_ends, expert_pad_begins, sparse_rows = get_sparse_map_scalable(top_experts, num_experts)
    validate_sparse_map_correctness(sparse_rows, expert_ends, expert_pad_begins, top_experts, num_experts)


@pytest.mark.parametrize("num_experts", [32, 64, 128])
def test_deterministic_results(num_experts):
    """Test that kernel produces deterministic results across multiple runs."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = "cuda"
    torch.manual_seed(42)

    top_experts = generate_test_experts(512, num_experts, 4, device)

    # Run multiple times and check results are identical
    results = []
    for _ in range(3):
        expert_ends, expert_pad_begins, sparse_rows = get_sparse_map_scalable(top_experts, num_experts)
        results.append((expert_ends.clone(), expert_pad_begins.clone(), sparse_rows.clone()))

    # All runs should produce identical results
    for i in range(1, len(results)):
        torch.testing.assert_close(results[0][0], results[i][0])
        torch.testing.assert_close(results[0][1], results[i][1])
        torch.testing.assert_close(results[0][2], results[i][2])


def test_automatic_kernel_selection():
    """
    Test that get_sparse_map() automatically selects the correct kernel
    based on num_experts and produces correct results.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = "cuda"

    # Test 1: Small num_experts should use original kernel (num_experts <= 64)
    top_experts_small = generate_test_experts(256, 32, 4, device)
    sparse_map_small = get_sparse_map(top_experts_small, 32)
    validate_sparse_map_correctness(
        sparse_map_small.sparse_rows,
        sparse_map_small.expert_ends,
        sparse_map_small.expert_pad_begins,
        top_experts_small,
        32,
    )

    # Test 2: Large num_experts should use scalable kernel (num_experts > 64)
    top_experts_large = generate_test_experts(256, 128, 4, device)
    sparse_map_large = get_sparse_map(top_experts_large, 128)
    validate_sparse_map_correctness(
        sparse_map_large.sparse_rows,
        sparse_map_large.expert_ends,
        sparse_map_large.expert_pad_begins,
        top_experts_large,
        128,
    )

    # Test 3: Results should match PyTorch reference
    expert_ends_ref, expert_pad_begins_ref, sparse_rows_ref = sparse_map_pytorch(top_experts_large, 128)
    torch.testing.assert_close(sparse_map_large.expert_ends, expert_ends_ref)
    torch.testing.assert_close(sparse_map_large.expert_pad_begins, expert_pad_begins_ref)
    torch.testing.assert_close(sparse_map_large.sparse_rows, sparse_rows_ref)


if __name__ == "__main__":
    # Quick smoke test
    if torch.cuda.is_available():
        print("Running smoke tests...")
        test_scalable_kernel_matches_original_small(16, 256, 2, True)
        print("✓ Small experts test passed")

        test_scalable_kernel_matches_pytorch_large(128, 256, 4, True)
        print("✓ Large experts test passed")

        test_gpt_oss_config(128, 1024, 4)
        print("✓ GPT-OSS config test passed")

        test_edge_cases()
        print("✓ Edge cases test passed")

        test_deterministic_results(128)
        print("✓ Deterministic test passed")

        test_automatic_kernel_selection()
        print("✓ Automatic kernel selection test passed")

        print("\nAll smoke tests passed! ✨")
    else:
        print("CUDA not available, skipping tests")
