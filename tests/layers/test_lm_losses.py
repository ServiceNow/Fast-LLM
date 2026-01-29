import contextlib
import pathlib
import random

import numpy as np
import pytest
import torch

from fast_llm.core.ops import split_op
from fast_llm.engine.distributed.config import DistributedBackend
from fast_llm.functional.config import EntropyLossImplementation, EntropyLossType, TargetFormat, TritonConfig
from fast_llm.layers.language_model.loss.dpo import dpo_loss
from fast_llm.layers.language_model.loss.entropy_loss import entropy_loss_forward_backward
from fast_llm.layers.language_model.loss.grpo import grpo_loss_forward_backward
from fast_llm.layers.language_model.loss.loss import loss_forward_backward
from fast_llm.layers.language_model.loss.z_loss import z_loss, z_loss_forward_backward
from fast_llm.utils import Assert
from tests.utils.dataset import get_random_spans
from tests.utils.subtest import DistributedTestContext

VOCAB_SIZE = 100
NUM_TOKENS = 200


def _get_lm_loss_inputs(
    num_columns: int, loss_masking: bool, target_format: TargetFormat, batch_shape: tuple[int] = (256,)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # We want something moderately close to the target for the test to be meaningful
    logits_var = torch.randn((*batch_shape, num_columns), dtype=torch.float32, device=device) / 3
    loss_mask = torch.randint(0, 2, batch_shape, dtype=torch.bool, device=device) if loss_masking else None
    if target_format == TargetFormat.labels:
        target = torch.randint(0, num_columns, batch_shape, dtype=torch.int64, device=device)
        logits = torch.nn.functional.one_hot(target, num_columns) + logits_var
        if loss_masking:
            target = torch.where(loss_mask, target, -100)
            loss_mask = None
    else:
        target = torch.randn((*batch_shape, num_columns), dtype=torch.float32, device=device)
        logits = target + logits_var
        if target_format == TargetFormat.probabilities:
            target = torch.softmax(target, -1)
    return logits, target, loss_mask


def _get_grpo_loss_inputs(num_columns: int, loss_masking: bool, batch_shape: tuple[int] = (256,)):
    logits, target, loss_mask = _get_lm_loss_inputs(num_columns, loss_masking, TargetFormat.labels, batch_shape)
    advantages = torch.randn_like(target, dtype=torch.float32)
    # We want some correlation between the old and new log probabilities for the test to be meaningful.
    old_log_probabilities = (
        torch.nn.functional.log_softmax(logits, dim=-1)
        .gather(dim=-1, index=(target * (target >= 0) if loss_masking else target).unsqueeze(-1))
        .squeeze(-1)
        + torch.randn_like(target, dtype=torch.float32) / 2
    )
    return logits, target, advantages, old_log_probabilities


def _compare_losses_and_grads(
    loss: torch.Tensor,
    ref_loss: torch.Tensor,
    has_grad: bool,
    grad: torch.Tensor | None,
    ref_grad: torch.Tensor | None,
    threshold=1e-5,
    loss_min_threshold=1e-6,
    group: torch.distributed.ProcessGroup | None = None,
):
    Assert.rms_close_relative(loss, ref_loss, threshold, loss_min_threshold)
    if has_grad:
        Assert.rms_close_relative(grad, split_op(ref_grad, group, -1), threshold, 1e-8)
    else:
        assert grad is None
        assert ref_grad is None


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


def reference_grpo_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probabilities: torch.Tensor,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.2,
    logits_scale_factor: float = 1.0,
) -> torch.Tensor:
    logits_ = logits.float()

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
    return (loss * loss_mask).mean()


_BATCH_SHAPES = ((64,), (16, 8))
_LOSS_PARAMETERS = (
    (500, 1.0, 1.0, False),  # Simple
    (512, 1.0, 1.0, False),  # Power of 2
    (500, None, 1.0, False),  # No grad
    (500, 1.0, 4.0, False),  # Loss scaling
    (500, 4.0, 1.0, False),  # Grad scaling
    (500, 1.0, 1.0, True),  # Loss masking
    (65538, 1.0, 1.0, False),  # Above max block size
)


def _test_entropy_loss(
    batch_shape,
    num_columns,
    grad_output,
    logits_scale_factor,
    loss_masking,
    target_format,
    entropy_loss_type,
    group=None,
):
    if target_format == TargetFormat.labels and entropy_loss_type == EntropyLossType.reverse_kl:
        pytest.skip(reason="Not implemented")
    # TODO: Test tensor-parallel implementation.
    logits, target, loss_mask = _get_lm_loss_inputs(num_columns, loss_masking, target_format, batch_shape)
    # Torch serves as the reference implementation.
    out_ref, grad_ref = entropy_loss_forward_backward(
        logits=logits,
        target=target,
        loss_mask=loss_mask,
        grad_output=grad_output,
        logits_scale_factor=logits_scale_factor,
        target_format=target_format,
        entropy_loss_type=entropy_loss_type,
        implementation=EntropyLossImplementation.torch,
    )
    out_fused, grad_fused = entropy_loss_forward_backward(
        logits=split_op(logits, group, -1),
        target=target if target_format == TargetFormat.labels else split_op(target, group, -1),
        loss_mask=loss_mask,
        grad_output=grad_output,
        group=group,
        logits_scale_factor=logits_scale_factor,
        target_format=target_format,
        entropy_loss_type=entropy_loss_type,
        implementation=EntropyLossImplementation.fused,
    )

    _compare_losses_and_grads(
        out_fused,
        out_ref,
        grad_output is not None,
        grad_fused,
        grad_ref,
        loss_min_threshold=2e-4 if entropy_loss_type == EntropyLossType.reverse_kl and loss_masking else 5e-6,
        group=group,
    )

    if entropy_loss_type != EntropyLossType.cross_entropy or not torch.cuda.is_available() or group is not None:
        # Triton implementation only supports cross-entropy.
        return
    assert TritonConfig.TRITON_ENABLED
    with pytest.raises(AssertionError) if num_columns > 65536 else contextlib.nullcontext():
        out_triton, grad_triton = entropy_loss_forward_backward(
            logits=logits,
            target=target,
            loss_mask=loss_mask,
            grad_output=grad_output,
            logits_scale_factor=logits_scale_factor,
            target_format=target_format,
            entropy_loss_type=entropy_loss_type,
            implementation=EntropyLossImplementation.triton,
        )
        _compare_losses_and_grads(out_triton, out_ref, grad_output is not None, grad_triton, grad_ref)


def _test_grpo_loss(batch_shape, num_columns, grad_output, logits_scale_factor, loss_masking, group=None):
    logits, target, advantages, old_log_probabilities = _get_grpo_loss_inputs(num_columns, loss_masking, batch_shape)
    out_ref, grad_ref = loss_forward_backward(
        grad_output,
        reference_grpo_loss,
        logits,
        target,
        advantages,
        old_log_probabilities,
        logits_scale_factor=logits_scale_factor,
    )
    out_fused, grad_fused = grpo_loss_forward_backward(
        split_op(logits, group, -1),
        target,
        advantages,
        old_log_probabilities,
        grad_output,
        group=group,
        logits_scale_factor=logits_scale_factor,
    )
    _compare_losses_and_grads(out_fused, out_ref, grad_output is not None, grad_fused, grad_ref, group=group)


def _test_z_loss(batch_shape, num_columns, grad_output, logits_scale_factor, loss_masking, group=None):
    logits, target, loss_mask = _get_lm_loss_inputs(num_columns, loss_masking, TargetFormat.logits, batch_shape)
    out_ref, grad_ref = loss_forward_backward(
        grad_output,
        z_loss,
        logits,
        loss_mask,
        logits_scale_factor=logits_scale_factor,
    )
    out_fused, grad_fused = z_loss_forward_backward(
        split_op(logits, group, -1),
        loss_mask,
        grad_output,
        group=group,
        logits_scale_factor=logits_scale_factor,
    )
    _compare_losses_and_grads(out_fused, out_ref, grad_output is not None, grad_fused, grad_ref, group=group)


@pytest.mark.slow
@pytest.mark.parametrize("batch_shape", _BATCH_SHAPES)
@pytest.mark.parametrize(("num_columns", "grad_output", "logits_scale_factor", "loss_masking"), _LOSS_PARAMETERS)
@pytest.mark.parametrize("target_format", TargetFormat)
@pytest.mark.parametrize("entropy_loss_type", EntropyLossType)
def test_entropy_loss(
    batch_shape, num_columns, grad_output, logits_scale_factor, loss_masking, target_format, entropy_loss_type
):
    _test_entropy_loss(
        batch_shape, num_columns, grad_output, logits_scale_factor, loss_masking, target_format, entropy_loss_type
    )


@pytest.mark.slow
@pytest.mark.parametrize("batch_shape", _BATCH_SHAPES)
@pytest.mark.parametrize(("num_columns", "grad_output", "logits_scale_factor", "loss_masking"), _LOSS_PARAMETERS)
def test_grpo_loss(batch_shape, num_columns, grad_output, logits_scale_factor, loss_masking):
    _test_grpo_loss(batch_shape, num_columns, grad_output, logits_scale_factor, loss_masking)


@pytest.mark.slow
@pytest.mark.parametrize("batch_shape", _BATCH_SHAPES)
@pytest.mark.parametrize(("num_columns", "grad_output", "logits_scale_factor", "loss_masking"), _LOSS_PARAMETERS)
def test_z_loss(batch_shape, num_columns, grad_output, logits_scale_factor, loss_masking):
    _test_z_loss(batch_shape, num_columns, grad_output, logits_scale_factor, loss_masking)


@pytest.mark.skip(reason="DPO loss is broken")
def test_dpo_loss():
    logits = torch.normal(0, 1, (NUM_TOKENS, VOCAB_SIZE))
    reference_model_logits = torch.normal(0, 1, (NUM_TOKENS, VOCAB_SIZE))
    labels = torch.randint(0, VOCAB_SIZE, (NUM_TOKENS,))
    spans = get_random_spans(np.full(10, 50), 0, 10)

    fast_llm_loss = dpo_loss(logits, labels, reference_model_logits, spans[::2], spans[1::2])
    reference_loss = reference_dpo_loss(logits, labels, reference_model_logits, spans[::2], spans[1::2], beta=1)
    Assert.rms_close(fast_llm_loss, reference_loss, 1e-5)


def _run_lm_loss_distributed(test_context: DistributedTestContext, base_path: pathlib.Path, seed: int):
    for batch_shape in _BATCH_SHAPES:
        for num_columns, grad_output, logits_scale_factor, loss_masking in _LOSS_PARAMETERS:
            suffix = f"{num_columns}-{grad_output}-{logits_scale_factor}-{loss_masking}-{"_".join([str(i) for i in batch_shape])}"
            # Entropy loss
            for entropy_loss_type in EntropyLossType:
                for target_format in TargetFormat:
                    if target_format == TargetFormat.labels and entropy_loss_type == EntropyLossType.reverse_kl:
                        continue
                    with test_context.subtest(
                        base_path, f"{entropy_loss_type}-{target_format}-{suffix}", 2
                    ) as subtest:
                        if subtest.do_run:
                            torch.manual_seed((seed + hash(subtest.name)) % 2**32)
                            _test_entropy_loss(
                                batch_shape,
                                num_columns,
                                grad_output,
                                logits_scale_factor,
                                loss_masking,
                                target_format,
                                entropy_loss_type,
                                test_context.group,
                            )
            # GRPO
            with test_context.subtest(base_path, f"z_loss-{suffix}", 2) as subtest:
                if subtest.do_run:
                    torch.manual_seed((seed + hash(subtest.name)) % 2**32)
                    _test_z_loss(
                        batch_shape, num_columns, grad_output, logits_scale_factor, loss_masking, test_context.group
                    )
            # Z loss
            with test_context.subtest(base_path, f"grpo-{suffix}", 2) as subtest:
                if subtest.do_run:
                    torch.manual_seed((seed + hash(subtest.name)) % 2**32)
                    _test_grpo_loss(
                        batch_shape, num_columns, grad_output, logits_scale_factor, loss_masking, test_context.group
                    )


@pytest.mark.slow
def test_lm_loss_distributed_dependency():
    # Mock test so the distributed subtest are placed in the same dependency group.
    pass


# We don't want to depend on `test_run_entropy_loss_distributed` because we still want to run this in cas of failure.
# This should still run after `test_run_entropy_loss_distributed`
@pytest.mark.slow
@pytest.mark.depends_on(on=["test_lm_loss_distributed_dependency"])
def test_run_lm_loss_distributed(run_parallel_script, result_path):
    run_parallel_script(
        _run_lm_loss_distributed,
        (result_path / "test_losses", random.randint(0, 2**32 - 1)),
        world_size=2,
        backend=DistributedBackend.gloo,
        use_cuda=False,  # Disable device count check.
    )


@pytest.mark.slow
@pytest.mark.depends_on(on=["test_lm_loss_distributed_dependency"])
@pytest.mark.parametrize("batch_shape", _BATCH_SHAPES)
@pytest.mark.parametrize(("num_columns", "grad_output", "logits_scale_factor", "loss_masking"), _LOSS_PARAMETERS)
@pytest.mark.parametrize(
    "loss_type",
    (
        *(
            f"{entropy_loss_type}-{target_format}"
            for entropy_loss_type in EntropyLossType
            for target_format in TargetFormat
            if target_format != TargetFormat.labels or entropy_loss_type != EntropyLossType.reverse_kl
        ),
        "grpo",
        "z_loss",
    ),
)
def test_lm_loss_distributed(
    result_path, report_subtest, loss_type, batch_shape, num_columns, grad_output, logits_scale_factor, loss_masking
):
    report_subtest(
        result_path
        / f"test_losses/{loss_type}-{num_columns}-{grad_output}-{logits_scale_factor}-{loss_masking}-{"_".join([str(i) for i in batch_shape])}",
        2,
        use_cuda=False,
    )
