import numpy as np
import pytest
import torch

from fast_llm.layers.language_model.loss.dpo import dpo_loss
from fast_llm.layers.language_model.loss.grpo import grpo_loss
from fast_llm.utils import Assert
from tests.utils.dataset import get_random_spans

VOCAB_SIZE = 100
NUM_TOKENS = 200


def reference_dpo_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    reference_model_logits: torch.Tensor,
    chosen_spans: torch.Tensor,
    rejected_spans: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    # TODO: Too similar to the actual implementation.
    policy_log_probs = (
        torch.nn.functional.log_softmax(logits.float(), dim=-1).gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    )
    policy_chosen_logps = sum(
        policy_log_probs[sample_index, begin:end].sum()
        for sample_index, sample_spans in enumerate(chosen_spans)
        for begin, end in sample_spans
    )
    policy_rejected_logps = sum(
        policy_log_probs[sample_index, begin:end].sum()
        for sample_index, sample_spans in enumerate(rejected_spans)
        for begin, end in sample_spans
    )
    reference_log_probs = (
        torch.nn.functional.log_softmax(reference_model_logits.float(), dim=-1)
        .gather(dim=-1, index=labels.unsqueeze(-1))
        .squeeze(-1)
    )
    reference_chosen_logps = sum(
        reference_log_probs[sample_index, begin:end].sum()
        for sample_index, sample_spans in enumerate(chosen_spans)
        for begin, end in sample_spans
    )
    reference_rejected_logps = sum(
        reference_log_probs[sample_index, begin:end].sum()
        for sample_index, sample_spans in enumerate(rejected_spans)
        for begin, end in sample_spans
    )
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    return -torch.nn.functional.logsigmoid(beta * (pi_logratios - ref_logratios)).mean()


@pytest.mark.skip(reason="DPO loss is broken")
def test_dpo_loss():
    logits = torch.normal(0, 1, (NUM_TOKENS, VOCAB_SIZE))
    reference_model_logits = torch.normal(0, 1, (NUM_TOKENS, VOCAB_SIZE))
    labels = torch.randint(0, VOCAB_SIZE, (NUM_TOKENS,))
    spans = get_random_spans(np.full(10, 50), 0, 10)

    fast_llm_loss = dpo_loss(logits, labels, reference_model_logits, spans[::2], spans[1::2])
    reference_loss = reference_dpo_loss(logits, labels, reference_model_logits, spans[::2], spans[1::2], beta=1)
    Assert.rms_close(fast_llm_loss, reference_loss, 1e-5)


def reference_grpo_loss(
    logits: torch.Tensor,
    loss_mask: "torch.Tensor | None",
    labels: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probabilities: torch.Tensor,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.2,
) -> torch.Tensor:
    # Log probabilities.
    if loss_mask is not None:
        labels = labels * loss_mask
    target_log_probabilities = (
        torch.nn.functional.log_softmax(logits, dim=-1).gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    )
    probability_ratio = torch.exp(target_log_probabilities - old_log_probabilities)
    loss = -torch.min(
        probability_ratio * advantages,
        torch.clamp(probability_ratio, 1 - epsilon_low, 1 + epsilon_high) * advantages,
    )
    if loss_mask is not None:
        loss = loss * loss_mask
    return loss.mean()


def test_grpo_loss():
    logits = torch.normal(0, 1, (NUM_TOKENS, VOCAB_SIZE))
    loss_mask = torch.randint(0, 2, (NUM_TOKENS,), dtype=torch.bool)
    labels = torch.randint(0, VOCAB_SIZE, (NUM_TOKENS,))
    advantages = torch.normal(0, 1, (NUM_TOKENS,))
    old_log_probabilities = torch.normal(0, 1, (NUM_TOKENS,))
    fast_llm_loss = grpo_loss(logits, loss_mask, labels, advantages, old_log_probabilities)
    reference_loss = reference_grpo_loss(logits, loss_mask, labels, advantages, old_log_probabilities)
    Assert.rms_close(fast_llm_loss, reference_loss, 1e-5)
