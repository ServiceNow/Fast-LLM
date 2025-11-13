#!/usr/bin/env python3
"""
Comprehensive test suite for sparse_map_kernel debugging.

This test compares the Triton kernel output against the PyTorch reference
implementation across various configurations to identify the bug.
"""

import torch
import sys

sys.path.insert(0, '/home/ubuntu/Fast-LLM')
from fast_llm.functional.triton.sparse_copy import get_sparse_map, sparse_map_pytorch


def test_sparse_map_correctness(num_experts, num_rows_dense, num_experts_per_token, seed=42):
    """
    Test that Triton kernel produces same results as PyTorch reference.

    Args:
        num_experts: Number of experts
        num_rows_dense: Number of tokens (dense rows)
        num_experts_per_token: Number of experts selected per token
        seed: Random seed for reproducibility
    """
    torch.manual_seed(seed)
    top_experts = torch.randint(0, num_experts, (num_rows_dense, num_experts_per_token), device='cuda')

    # Get Triton result
    sparse_map_triton = get_sparse_map(top_experts, num_experts=num_experts, use_triton=True)

    # Get PyTorch reference result
    expert_ends_pt, expert_pad_begins_pt, sparse_rows_pt = sparse_map_pytorch(
        top_experts.cpu(), num_experts=num_experts
    )

    # Compare results
    expert_ends_match = torch.equal(sparse_map_triton.expert_ends.cpu(), expert_ends_pt)
    expert_pad_begins_match = torch.equal(sparse_map_triton.expert_pad_begins.cpu(), expert_pad_begins_pt)
    sparse_rows_match = torch.equal(sparse_map_triton.sparse_rows.cpu(), sparse_rows_pt)

    all_match = expert_ends_match and expert_pad_begins_match and sparse_rows_match

    if not all_match:
        print(f"\n{'='*80}")
        print(f"FAILED: experts={num_experts}, rows={num_rows_dense}, experts_per_token={num_experts_per_token}")
        print(f"{'='*80}")

        if not expert_ends_match:
            print(f"\n❌ expert_ends mismatch:")
            print(f"  Triton:  {sparse_map_triton.expert_ends}")
            print(f"  PyTorch: {expert_ends_pt}")

        if not expert_pad_begins_match:
            print(f"\n❌ expert_pad_begins mismatch:")
            print(f"  Triton:  {sparse_map_triton.expert_pad_begins}")
            print(f"  PyTorch: {expert_pad_begins_pt}")

        if not sparse_rows_match:
            print(f"\n❌ sparse_rows mismatch:")
            print(f"  Input top_experts:\n{top_experts}")
            print(f"\n  Triton sparse_rows:\n{sparse_map_triton.sparse_rows}")
            print(f"\n  PyTorch sparse_rows:\n{sparse_rows_pt}")

            # Find first mismatch
            diff = (sparse_map_triton.sparse_rows.cpu() != sparse_rows_pt).nonzero()
            if len(diff) > 0:
                first_diff = diff[0]
                print(f"\n  First mismatch at position {first_diff.tolist()}:")
                print(f"    Triton:  {sparse_map_triton.sparse_rows[first_diff[0], first_diff[1]].item()}")
                print(f"    PyTorch: {sparse_rows_pt[first_diff[0], first_diff[1]].item()}")
    else:
        print(f"✅ PASS: experts={num_experts}, rows={num_rows_dense}, experts_per_token={num_experts_per_token}")

    return all_match


def test_edge_cases():
    """Test various edge cases"""
    print("\n" + "="*80)
    print("Testing Edge Cases")
    print("="*80)

    results = []

    # Test 1: Minimal case
    results.append(("Minimal (2 experts, 1 token)", test_sparse_map_correctness(2, 1, 1)))

    # Test 2: All tokens select same expert
    print("\nTest: All tokens select same expert")
    torch.manual_seed(100)
    top_experts = torch.zeros((4, 2), dtype=torch.int64, device='cuda')  # All select expert 0
    sparse_map_triton = get_sparse_map(top_experts, num_experts=4, use_triton=True)
    _, _, sparse_rows_pt = sparse_map_pytorch(top_experts.cpu(), num_experts=4)
    match = torch.equal(sparse_map_triton.sparse_rows.cpu(), sparse_rows_pt)
    results.append(("All same expert", match))
    if not match:
        print(f"  Triton:  {sparse_map_triton.sparse_rows}")
        print(f"  PyTorch: {sparse_rows_pt}")
    else:
        print("  ✅ PASS")

    # Test 3: Sequential experts
    print("\nTest: Sequential expert selection")
    top_experts = torch.arange(8, device='cuda').view(4, 2) % 4
    sparse_map_triton = get_sparse_map(top_experts, num_experts=4, use_triton=True)
    _, _, sparse_rows_pt = sparse_map_pytorch(top_experts.cpu(), num_experts=4)
    match = torch.equal(sparse_map_triton.sparse_rows.cpu(), sparse_rows_pt)
    results.append(("Sequential experts", match))
    if not match:
        print(f"  Input: {top_experts}")
        print(f"  Triton:  {sparse_map_triton.sparse_rows}")
        print(f"  PyTorch: {sparse_rows_pt}")
    else:
        print("  ✅ PASS")

    return results


def main():
    print("="*80)
    print("SPARSE_MAP_KERNEL COMPREHENSIVE TEST SUITE")
    print("="*80)
    print(f"Device: CUDA")
    print(f"Triton version: {__import__('triton').__version__}")
    print(f"PyTorch version: {torch.__version__}")
    import platform
    print(f"Architecture: {platform.machine()}")

    results = []

    # Test configurations from the actual failing test
    print("\n" + "="*80)
    print("Testing Actual Test Configuration")
    print("="*80)
    results.append(("Actual test config", test_sparse_map_correctness(4, 8, 4)))

    # Test various sizes
    print("\n" + "="*80)
    print("Testing Various Configurations")
    print("="*80)

    test_configs = [
        # Small configs
        (2, 4, 2, "Small: 2 experts, 4 tokens, 2 per token"),
        (4, 4, 2, "Medium: 4 experts, 4 tokens, 2 per token"),
        (4, 8, 2, "Medium: 4 experts, 8 tokens, 2 per token"),

        # Problematic config (experts_per_token=4)
        (4, 16, 4, "Large: 4 experts, 16 tokens, 4 per token"),
        (8, 8, 4, "Large: 8 experts, 8 tokens, 4 per token"),

        # Test with experts_per_token=1
        (4, 8, 1, "Simple: 4 experts, 8 tokens, 1 per token"),
        (8, 16, 1, "Simple: 8 experts, 16 tokens, 1 per token"),

        # Test with experts_per_token=3
        (4, 8, 3, "Medium: 4 experts, 8 tokens, 3 per token"),
        (8, 12, 3, "Medium: 8 experts, 12 tokens, 3 per token"),

        # Test different expert counts
        (16, 32, 2, "Many experts: 16 experts, 32 tokens, 2 per token"),
        (32, 64, 2, "Many experts: 32 experts, 64 tokens, 2 per token"),

        # Test with more tokens
        (4, 32, 4, "More tokens: 4 experts, 32 tokens, 4 per token"),
        (8, 64, 4, "More tokens: 8 experts, 64 tokens, 4 per token"),

        # Power of 2 variations
        (4, 16, 2, "Power of 2: 4 experts, 16 tokens, 2 per token"),
        (8, 16, 2, "Power of 2: 8 experts, 16 tokens, 2 per token"),
        (16, 16, 2, "Power of 2: 16 experts, 16 tokens, 2 per token"),

        # Non-power of 2
        (5, 10, 2, "Non-pow2: 5 experts, 10 tokens, 2 per token"),
        (7, 14, 3, "Non-pow2: 7 experts, 14 tokens, 3 per token"),
        (12, 24, 4, "Non-pow2: 12 experts, 24 tokens, 4 per token"),
    ]

    for num_experts, num_rows, experts_per_token, desc in test_configs:
        results.append((desc, test_sparse_map_correctness(num_experts, num_rows, experts_per_token)))

    # Test edge cases
    edge_results = test_edge_cases()
    results.extend(edge_results)

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n❌ {total - passed} TESTS FAILED")
        print("\nFailed tests:")
        for name, result in results:
            if not result:
                print(f"  - {name}")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
