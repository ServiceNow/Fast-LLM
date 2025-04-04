from unittest import mock

import pytest
import torch

from fast_llm.engine.base_model.base_model import LossDef
from fast_llm.engine.distributed.config import DistributedConfig, PhaseType
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.multi_stage.multi_stage import MultiStageModel
from fast_llm.engine.schedule.config import ScheduleConfig
from fast_llm.engine.schedule.runner import BatchContext, ScheduleRunner
from fast_llm.engine.schedule.schedule import Schedule
from fast_llm.layers.transformer.config import TransformerRoutingMetrics
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


@pytest.fixture
def setup_runner():
    """
    Fixture to set up the test environment.
    TODO: Leave it here for now, but may be moved to common.py
    """
    # Mock objects needed for testing
    distributed_config = DistributedConfig()

    # Mock MultiStageModel with loss_defs
    multi_stage = mock.MagicMock(spec=MultiStageModel)
    multi_stage.base_model.loss_defs = [LossDef(name="test_loss", formatted_name="Test Loss", count=1)]
    multi_stage.base_model.metric_defs = [
        LossDef(
            name=TransformerRoutingMetrics.normalized_average_entropy, formatted_name="Normalized Entropy", count=1
        ),
        LossDef(name=TransformerRoutingMetrics.mutual_info, formatted_name="Mutual Information", count=1),
    ]

    # Create a schedule runner
    schedule_config = ScheduleConfig()
    runner = ScheduleRunner(config=schedule_config, multi_stage=multi_stage, distributed_config=distributed_config)

    # Mock distributed object
    distributed = mock.MagicMock(spec=Distributed)
    distributed.config = distributed_config
    distributed.device = torch.device("cpu")
    distributed.data_group = None
    distributed.pipeline_group = None

    # Setup the runner
    runner._distributed = distributed
    runner.is_initialized = True

    # Create a mock schedule
    schedule = mock.MagicMock(spec=Schedule)
    schedule.phase = PhaseType.training
    schedule.batch_config.num_inputs = 3
    schedule._schedule_config = schedule_config

    # Create a batch context with metrics and losses
    context = BatchContext(
        iteration=1,
        schedule=schedule,
    )

    # Add test metrics
    context.metrics = {
        # Metrics that should be reduced (in TransformerReducedMetrics)
        TransformerRoutingMetrics.normalized_average_entropy: [
            torch.tensor(0.5),
            torch.tensor(0.6),
            torch.tensor(0.7),
        ],
        TransformerRoutingMetrics.mutual_info: [torch.tensor(0.2), torch.tensor(0.3), torch.tensor(0.4)],
        # Metric that should not be reduced
        "non_reduced_metric": [torch.tensor(1.0), torch.tensor(1.0), torch.tensor(1.0)],
    }

    # Add test losses
    context.losses = {"test_loss": [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)]}

    return runner, context, schedule


def test_reduce_losses(setup_runner):
    """Test that _reduce_losses correctly reduces losses"""
    runner, context, _ = setup_runner

    reduced_losses = runner._reduce_losses(context)

    assert "test_loss" in reduced_losses
    assert pytest.approx(reduced_losses["test_loss"], 0.001) == 2.0


if __name__ == "__main__":
    pytest.main([__file__])
