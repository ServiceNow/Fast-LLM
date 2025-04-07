import torch

from fast_llm.layers.transformer.mixture_of_experts import (
    calculate_mutual_information,
    calculate_normalized_average_entropy,
)


def test_diversity_entropy():
    """
    collapse routing would have low entropy and low mutual information
    """

    collapased_probs = torch.tensor(
        [
            # Batch 1
            [
                [0.99, 0.01, 0.0, 0.0],
                [0.99, 0.01, 0.0, 0.0],
                [0.99, 0.01, 0.0, 0.0],
            ],
            # Batch 2
            [
                [0.99, 0.01, 0.0, 0.0],
                [0.99, 0.01, 0.0, 0.0],
                [0.99, 0.01, 0.0, 0.0],
            ],
        ]
    )
    norm_entropy = calculate_normalized_average_entropy(collapased_probs)
    mutual_info = calculate_mutual_information(collapased_probs)
    assert torch.isclose(norm_entropy, torch.tensor(0.0), atol=1e-1), f"Expected 0.0, got {norm_entropy}"
    assert torch.isclose(mutual_info, torch.tensor(0.0), atol=1e-5), f"Expected 0.0, got {mutual_info}"

    # diverse but no collapse
    # should give low entropy and high mutual information
    diverse_probs = torch.tensor(
        [
            # Batch 1
            [
                [0.99, 0.01, 0.0, 0.0],
                [0.01, 0.99, 0.0, 0.0],
                [0.01, 0.01, 0.99, 0.0],
            ],
            # Batch 2
            [
                [0.01, 0.01, 0.99, 0.0],
                [0.99, 0.01, 0.0, 0.0],
                [0.01, 0.01, 0.01, 0.99],
            ],
        ]
    )
    norm_entropy = calculate_normalized_average_entropy(diverse_probs)
    mutual_info = calculate_mutual_information(diverse_probs)
    assert torch.isclose(norm_entropy, torch.tensor(0.0), atol=1e-1), f"Expected 0.0, got {norm_entropy}"
    assert torch.isclose(mutual_info, torch.tensor(0.9), atol=1e-1), f"Expected 1.0, got {mutual_info}"


def test_calculate_normalized_average_entropy():
    # AI generated test case
    # Create a batch of routing probabilities
    batch_size = 2
    seq_len = 3
    n_experts = 4

    # Test 1: Uniform distribution (should give normalized entropy of 1.0)
    uniform_probs = torch.ones(batch_size, seq_len, n_experts) / n_experts
    norm_entropy = calculate_normalized_average_entropy(uniform_probs)
    assert torch.isclose(norm_entropy, torch.tensor(1.0), atol=1e-5), f"Expected 1.0, got {norm_entropy}"

    # Test 2: One-hot distribution (should give normalized entropy of 0.0)
    one_hot = torch.zeros(batch_size, seq_len, n_experts)
    for b in range(batch_size):
        for s in range(seq_len):
            one_hot[b, s, b % n_experts] = 1.0
    norm_entropy = calculate_normalized_average_entropy(one_hot)
    assert torch.isclose(norm_entropy, torch.tensor(0.0), atol=1e-5), f"Expected 0.0, got {norm_entropy}"

    # Test 3: Mixed distribution
    mixed_probs = torch.tensor(
        [
            # Batch 1
            [
                [0.7, 0.1, 0.1, 0.1],  # Token 1: mostly expert 0
                [0.1, 0.7, 0.1, 0.1],  # Token 2: mostly expert 1
                [0.25, 0.25, 0.25, 0.25],  # Token 3: uniform
            ],
            # Batch 2
            [
                [0.4, 0.4, 0.1, 0.1],  # Token 1: split between experts 0 and 1
                [0.1, 0.1, 0.4, 0.4],  # Token 2: split between experts 2 and 3
                [0.1, 0.1, 0.1, 0.7],  # Token 3: mostly expert 3
            ],
        ]
    )
    norm_entropy = calculate_normalized_average_entropy(mixed_probs)
    # The expected value is between 0 and 1
    assert 0.0 < norm_entropy < 1.0, f"Expected value between 0 and 1, got {norm_entropy}"


def test_calculate_mutual_information():
    # AI generated test cases
    # Create a batch of routing probabilities
    batch_size = 2
    seq_len = 3
    n_experts = 4

    # Test 1: All tokens route to the same expert (low mutual information)
    same_expert = torch.zeros(batch_size, seq_len, n_experts)
    same_expert[:, :, 0] = 1.0  # All tokens route to expert 0
    mutual_info = calculate_mutual_information(same_expert)
    assert torch.isclose(mutual_info, torch.tensor(0.0)), f"Expected 0.0, got {mutual_info}"

    # Test 2: Each token routes to a different expert (high mutual information)
    different_experts = torch.zeros(batch_size, seq_len, n_experts)
    for b in range(batch_size):
        for s in range(seq_len):
            different_experts[b, s, s % n_experts] = 1.0
    mutual_info = calculate_mutual_information(different_experts)
    # The value should be positive and closer to 1
    assert mutual_info > 0.0, f"Expected positive value, got {mutual_info}"

    # Test 3: Mixed routing pattern
    mixed_probs = torch.tensor(
        [
            # Batch 1
            [
                [0.7, 0.1, 0.1, 0.1],  # Token 1: mostly expert 0
                [0.1, 0.7, 0.1, 0.1],  # Token 2: mostly expert 1
                [0.1, 0.1, 0.7, 0.1],  # Token 3: mostly expert 2
            ],
            # Batch 2
            [
                [0.1, 0.1, 0.1, 0.7],  # Token 1: mostly expert 3
                [0.7, 0.1, 0.1, 0.1],  # Token 2: mostly expert 0
                [0.1, 0.7, 0.1, 0.1],  # Token 3: mostly expert 1
            ],
        ]
    )
    mutual_info = calculate_mutual_information(mixed_probs)
    # The expected value is between 0 and 1
    assert 0.0 < mutual_info < 1.0, f"Expected value between 0 and 1, got {mutual_info}"


def test_small_seq_length_batch_size_probabilities():
    # AI generated test cases
    # Test with very small batch and sequence length
    tiny_probs = torch.tensor([[[0.25, 0.25, 0.25, 0.25]]])  # batch=1, seq_len=1, n_experts=4
    norm_entropy = calculate_normalized_average_entropy(tiny_probs)
    mutual_info = calculate_mutual_information(tiny_probs)
    assert torch.isclose(norm_entropy, torch.tensor(1.0)), f"Expected 1.0, got {norm_entropy}"
    assert torch.isclose(mutual_info, torch.tensor(0.0)), f"Expected 0.0, got {mutual_info}"

    # Test with very small probabilities
    small_probs = torch.ones(2, 3, 4) * 1e-8
    small_probs[:, :, 0] = 1.0 - 3e-8  # Make sure they sum to 1
    norm_entropy = calculate_normalized_average_entropy(small_probs)
    mutual_info = calculate_mutual_information(small_probs)
    assert torch.isclose(norm_entropy, torch.tensor(0.0), atol=1e-5), f"Expected ~0.0, got {norm_entropy}"
    assert torch.isclose(mutual_info, torch.tensor(0.0), atol=1e-5), f"Expected ~0.0, got {mutual_info}"
