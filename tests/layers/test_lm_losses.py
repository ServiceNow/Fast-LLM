import pathlib

import numpy as np
import pytest
import torch

from fast_llm.engine.distributed.config import DistributedBackend
from fast_llm.functional.config import TargetFormat
from fast_llm.layers.language_model.loss.dpo import dpo_loss
from fast_llm.layers.language_model.loss.grpo import grpo_loss_forward_backward
from fast_llm.utils import Assert
from tests.functional.test_entropy_loss import compare_losses_and_grads, get_entropy_loss_inputs
from tests.utils.dataset import get_random_spans
from tests.utils.subtest import DistributedTestContext

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
    labels: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probabilities: torch.Tensor,
    grad_output: float | None,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.2,
    logits_scale_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    with torch.set_grad_enabled(grad_output is not None):
        logits_ = logits.float().detach().requires_grad_(grad_output is not None)

        # Log probabilities.
        loss_mask = labels >= 0
        labels = labels * loss_mask
        target_log_probabilities = (
            torch.nn.functional.log_softmax(logits_ * logits_scale_factor, dim=-1)
            .gather(dim=-1, index=labels.unsqueeze(-1))
            .squeeze(-1)
        )
        probability_ratio = torch.exp(target_log_probabilities - old_log_probabilities)
        loss = -torch.min(
            probability_ratio * advantages,
            torch.clamp(probability_ratio, 1 - epsilon_low, 1 + epsilon_high) * advantages,
        )
        loss = (loss * loss_mask).mean()

        if grad_output is None:
            grad = None
        else:
            loss.backward(torch.full_like(loss, grad_output))
            grad = logits_.grad.detach().to(logits.dtype)

    return loss.detach(), grad


def _get_grpo_loss_inputs(num_columns: int, loss_masking: bool, batch_shape: tuple[int] = (256,)):
    logits, target, loss_mask = get_entropy_loss_inputs(num_columns, loss_masking, TargetFormat.labels, batch_shape)
    advantages = torch.randn_like(target, dtype=torch.float32)
    # We want some correlation between the old and new log probabilities for the test to be meaningful.
    old_log_probabilities = (
        torch.nn.functional.log_softmax(logits, dim=-1)
        .gather(dim=-1, index=(target * (target >= 0) if loss_masking else target).unsqueeze(-1))
        .squeeze(-1)
        + torch.randn_like(target, dtype=torch.float32) / 2
    )
    return logits, target, advantages, old_log_probabilities


@pytest.mark.slow
@pytest.mark.parametrize("batch_shape", ((64,), (128,), (16, 8)))
@pytest.mark.parametrize(
    ("num_columns", "grad_output", "logits_scale_factor", "loss_masking"),
    (
        (1000, 1.0, 1.0, False),  # Simple
        (128, 1.0, 1.0, False),
        (1000, None, 1.0, False),  # No grad
        (1000, 1.0, 4.0, False),  # Loss scaling
        (1000, 4.0, 1.0, False),  # Grad scaling
        (1000, 1.0, 1.0, True),  # Loss masking
    ),
)
def test_grpo_loss(batch_shape, num_columns, grad_output, logits_scale_factor, loss_masking):
    logits, target, advantages, old_log_probabilities = _get_grpo_loss_inputs(num_columns, loss_masking, batch_shape)
    out_ref, grad_ref = reference_grpo_loss(
        logits, target, advantages, old_log_probabilities, grad_output, logits_scale_factor=logits_scale_factor
    )
    out_fused, grad_fused = grpo_loss_forward_backward(
        logits, target, advantages, old_log_probabilities, grad_output, logits_scale_factor=logits_scale_factor
    )
    compare_losses_and_grads(out_fused, out_ref, grad_output is not None, grad_fused, grad_ref)


def _grpo_loss_distributed(loss_masking: bool, group: torch.distributed.ProcessGroup):
    # Ensure all workers have the same inputs.
    torch.manual_seed(0)
    rank = group.rank()
    world_size = group.size()
    logits, target, advantages, old_log_probabilities = _get_grpo_loss_inputs(1000, loss_masking)

    out_ref, grad_ref = grpo_loss_forward_backward(logits, target, advantages, old_log_probabilities, 1.0)

    out, grad = grpo_loss_forward_backward(
        logits.chunk(world_size, 1)[rank],
        target,
        advantages,
        old_log_probabilities,
        1.0,
        group=group,
    )
    compare_losses_and_grads(out, out_ref, True, grad, grad_ref.chunk(world_size, 1)[rank], 1e-4)


def _run_grpo_loss_distributed(test_context: DistributedTestContext, base_path: pathlib.Path):
    for loss_masking in [False, True]:
        name = f"grpo_{loss_masking}"
        with test_context.subtest(base_path, name, 2) as subtest:
            if subtest.do_run:
                _grpo_loss_distributed(loss_masking, test_context.group)


@pytest.mark.slow
def test_grpo_loss_distributed_dependency():
    # Mock test so the distributed subtest are placed in the same dependency group.
    pass


@pytest.mark.slow
@pytest.mark.depends_on(on=["test_entropy_loss_distributed_dependency"])
def test_run_grpo_loss_distributed(run_parallel_script, result_path):
    run_parallel_script(
        _run_grpo_loss_distributed,
        (result_path / "test_losses",),
        world_size=2,
        backend=DistributedBackend.gloo,
        use_cuda=False,  # Disable device count check.
    )


# We don't want to depend on `test_run_entropy_loss_distributed` because we still want to run this in cas of failure.
# This should still run after `test_run_entropy_loss_distributed`
@pytest.mark.slow
@pytest.mark.depends_on(on=["test_grpo_loss_distributed_dependency"])
@pytest.mark.parametrize("loss_masking", (False, True))
def test_grpo_loss_distributed(result_path, report_subtest, loss_masking):
    report_subtest(result_path / f"test_losses/grpo_{loss_masking}", 2, use_cuda=False)
