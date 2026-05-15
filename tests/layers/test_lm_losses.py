import pathlib
import random

import numpy as np
import pytest
import torch

from fast_llm.core.ops import split_op
from fast_llm.engine.config_utils import data_type
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.engine.distributed.config import DistributedBackend
from fast_llm.functional.config import EntropyLossType, TargetFormat
from fast_llm.functional.entropy_loss import fused_entropy_loss_forward_backward, torch_entropy_loss_forward_backward
from fast_llm.functional.triton import triton_available
from fast_llm.functional.triton.entropy_loss import triton_entropy_loss_forward_backward
from fast_llm.functional.triton.grpo_loss import triton_grpo_loss_forward_backward
from fast_llm.functional.triton.z_loss import triton_z_loss_forward_backward
from fast_llm.layers.language_model.loss.dpo import dpo_loss
from fast_llm.layers.language_model.loss.grpo import (
    GRPOMetrics,
    compute_grpo_metrics,
    fused_grpo_loss_forward_backward,
)
from fast_llm.layers.language_model.loss.loss import loss_forward_backward
from fast_llm.layers.language_model.loss.z_loss import fused_z_loss_forward_backward, z_loss
from fast_llm.utils import Assert
from tests.utils.dataset import get_random_spans
from tests.utils.subtest import DistributedTestContext


def _get_lm_loss_inputs(
    num_columns: int, loss_masking: bool, target_format: TargetFormat, batch_shape: tuple[int], dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # We want something moderately close to the target for the test to be meaningful
    logits_var = torch.randn((*batch_shape, num_columns), dtype=dtype.torch, device=device) / 3
    loss_mask = torch.randint(0, 2, batch_shape, dtype=torch.bool, device=device) if loss_masking else None
    if target_format == TargetFormat.labels:
        target = torch.randint(0, num_columns, batch_shape, dtype=torch.int64, device=device)
        logits = torch.nn.functional.one_hot(target, num_columns) + logits_var
        if loss_masking:
            target = torch.where(loss_mask, target, -100)
            loss_mask = None
    else:
        # Target logits are typically in training precision, ex. with distillation model.
        target = torch.randn((*batch_shape, num_columns), dtype=dtype.torch, device=device)
        logits = target + logits_var
        if target_format == TargetFormat.probabilities:
            # Probabilities need to be in full precision for accuracy.
            target = torch.softmax(target, -1, dtype=torch.float32)
    return logits, target, loss_mask


def _get_grpo_loss_inputs(num_columns: int, loss_masking: bool, batch_shape: tuple[int], dtype):
    logits, target, loss_mask = _get_lm_loss_inputs(num_columns, loss_masking, TargetFormat.labels, batch_shape, dtype)
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
    group: torch.distributed.ProcessGroup | None = None,
):
    Assert.rms_close_relative(loss, ref_loss, threshold, 1e-6)
    if has_grad:
        Assert.rms_close_relative(
            grad, split_op(ref_grad, group, -1), threshold, 1e-8 if grad.dtype == torch.float32 else 1e-7
        )
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


def reference_grpo_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probabilities: torch.Tensor,
    label_counts: torch.Tensor,
    epsilon_low: float,
    epsilon_high: float,
    logits_scale_factor: float,
    compute_entropy: bool,
) -> GRPOMetrics:
    log_softmax = torch.nn.functional.log_softmax(logits.float() * logits_scale_factor, dim=-1)
    loss_mask = target >= 0
    mask = loss_mask.float()
    masked = mask / label_counts.float().clamp(min=1)

    new_log_probs = log_softmax.gather(-1, (target * loss_mask).unsqueeze(-1)).squeeze(-1)
    log_ratio = new_log_probs - old_log_probabilities.float()
    ratio = log_ratio.exp()
    clipped = (ratio < 1.0 - epsilon_low) | (ratio > 1.0 + epsilon_high)
    kl = ratio - log_ratio - 1.0

    entropy = None
    if compute_entropy:
        entropy_per_token = -(log_softmax.exp() * log_softmax).sum(-1)
        entropy = (entropy_per_token * masked).sum()

    return GRPOMetrics(
        old_logprobs=(old_log_probabilities.float() * masked).sum(),
        ratio_new_old=(ratio * masked).sum(),
        ratio_new_old_sum=(ratio * mask).sum(),
        ratio_new_old_squared_sum=(ratio * ratio * mask).sum(),
        kl_new_old=(kl * masked).sum(),
        clipped_ratio_fraction=(clipped.float() * masked).sum(),
        advantage=(advantages.float() * masked).sum(),
        max_advantage=advantages[loss_mask].max(),
        min_advantage=advantages[loss_mask].min(),
        num_tokens=mask.sum(),
        entropy=entropy,
    )


def reference_grpo_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probabilities: torch.Tensor,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.2,
    logits_scale_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    # new_logprobs: sum of per-sequence mean log-probs
    log_probs = torch.nn.functional.log_softmax(logits_, -1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    new_logprobs = (log_probs * loss_mask).sum() / max(float(loss_mask.sum()), 1.0)
    return (loss * loss_mask).sum() / loss_mask.sum(), new_logprobs


_BATCH_SHAPES = ((64,), (16, 8))
_LOSS_PARAMETERS = (
    (500, 1.0, 1.0, False, DataType.float32, None, False),  # Simple
    (256, 1.0, 1.0, False, DataType.float32, None, False),  # Power of 2
    (500, None, 1.0, False, DataType.float32, None, False),  # No grad
    (500, 1.0, 1.0, False, DataType.float32, None, True),  # Accumulate
    (500, 1.0, 4.0, False, DataType.float32, None, False),  # Loss scaling
    (500, 4.0, 1.0, False, DataType.float32, None, False),  # Grad scaling
    (500, 1.0, 1.0, True, DataType.float32, None, False),  # Loss masking
    (500, 1.0, 1.0, False, DataType.float16, None, False),  # Fp16
    (500, 1.0, 1.0, False, DataType.float32, 256, False),  # Looped
    (1000, 2.0, 3.0, True, DataType.float32, 256, True),  # Hard
)


def _test_entropy_loss(
    batch_shape,
    num_columns,
    grad_output,
    logits_scale_factor,
    loss_masking,
    target_format,
    entropy_loss_type,
    dtype,
    block_size,
    accumulate,
    group=None,
):
    if target_format == TargetFormat.labels and entropy_loss_type == EntropyLossType.reverse_kl:
        pytest.skip(reason="Reverse KL loss not implemented for target labels")
    # TODO: Test tensor-parallel implementation.
    logits, target, loss_mask = _get_lm_loss_inputs(num_columns, loss_masking, target_format, batch_shape, dtype)
    local_logits = split_op(logits, group, -1).contiguous()
    local_target = target if target_format == TargetFormat.labels else split_op(target, group, -1).contiguous()
    # Torch serves as the reference implementation.
    out_ref, grad_ref = torch_entropy_loss_forward_backward(
        logits=logits,
        target=target,
        loss_mask=loss_mask,
        grad_output=grad_output,
        logits_scale_factor=logits_scale_factor,
        target_format=target_format,
        entropy_loss_type=entropy_loss_type,
    )
    if accumulate:
        previous_grad = torch.randn_like(grad_ref)
        grad_ref = grad_ref + previous_grad
        local_previous_grad = split_op(previous_grad, group, -1).contiguous()
    out_fused, grad_fused = fused_entropy_loss_forward_backward(
        logits=local_logits,
        target=local_target,
        loss_mask=loss_mask,
        grad_logits=local_previous_grad.clone() if accumulate else None,
        grad_output=grad_output,
        group=group,
        logits_scale_factor=logits_scale_factor,
        target_format=target_format,
        entropy_loss_type=entropy_loss_type,
    )
    _compare_losses_and_grads(
        out_fused,
        out_ref,
        grad_output is not None,
        grad_fused,
        grad_ref,
        threshold=1e-5 if data_type == DataType.float32 else 1e-4,
        group=group,
    )

    if not triton_available:
        return
    out_triton, grad_triton = triton_entropy_loss_forward_backward(
        logits=local_logits,
        target=local_target,
        loss_mask=loss_mask,
        grad_logits=local_previous_grad.clone() if accumulate else None,
        grad_output=grad_output,
        logits_scale_factor=logits_scale_factor,
        target_format=target_format,
        entropy_loss_type=entropy_loss_type,
        group=group,
        block_size=block_size,
    )
    _compare_losses_and_grads(
        out_triton,
        out_ref,
        grad_output is not None,
        grad_triton,
        grad_ref,
        threshold=1e-5 if target_format != TargetFormat.probabilities and data_type == DataType.float32 else 1e-4,
        group=group,
    )


def _test_grpo_loss(
    batch_shape, num_columns, grad_output, logits_scale_factor, loss_masking, dtype, block_size, accumulate, group=None
):
    logits, target, advantages, old_log_probabilities = _get_grpo_loss_inputs(
        num_columns, loss_masking, batch_shape, dtype
    )
    num_labels = int((target >= 0).sum().item())
    num_labels_in_seq = torch.where(
        target >= 0,
        torch.full(batch_shape, num_labels, dtype=torch.int32, device=target.device),
        torch.zeros(batch_shape, dtype=torch.int32, device=target.device),
    )
    divisor = max(num_labels, 1)
    out_ref, grad_ref = loss_forward_backward(
        grad_output,
        lambda *args, **kwargs: reference_grpo_loss(*args, **kwargs)[0],
        logits,
        target,
        advantages,
        old_log_probabilities,
        logits_scale_factor=logits_scale_factor,
    )
    if accumulate:
        previous_grad = torch.randn_like(grad_ref)
        grad_ref = grad_ref + previous_grad
        local_previous_grad = split_op(previous_grad, group, -1).contiguous()
    out_fused, grad_fused, new_logprobs_fused = fused_grpo_loss_forward_backward(
        split_op(logits, group, -1),
        target,
        advantages,
        old_log_probabilities,
        grad_logits=local_previous_grad.clone() if accumulate else None,
        grad_output=grad_output,
        group=group,
        logits_scale_factor=logits_scale_factor,
        num_labels_in_seq=num_labels_in_seq,
        divisor=divisor,
    )
    _compare_losses_and_grads(out_fused, out_ref, grad_output is not None, grad_fused, grad_ref, group=group)

    if not triton_available:
        return
    out_triton, grad_triton, new_logprobs_triton = triton_grpo_loss_forward_backward(
        split_op(logits, group, -1).contiguous(),
        target,
        advantages,
        old_log_probabilities,
        grad_logits=local_previous_grad.clone() if accumulate else None,
        grad_output=grad_output,
        group=group,
        logits_scale_factor=logits_scale_factor,
        num_labels_in_seq=num_labels_in_seq,
        divisor=divisor,
        block_size=block_size,
    )
    _compare_losses_and_grads(out_triton, out_ref, grad_output is not None, grad_triton, grad_ref, group=group)
    Assert.rms_close_relative(new_logprobs_triton, new_logprobs_fused, 1e-5, 1e-6)


def _check_grpo_metrics(ref: GRPOMetrics, got: GRPOMetrics, threshold: float) -> None:
    for name in GRPOMetrics._fields:
        ref_value = getattr(ref, name)
        got_value = getattr(got, name)
        if ref_value is None:
            assert got_value is None, name
        else:
            Assert.rms_close_relative(got_value, ref_value, threshold, 1e-6)


def _test_grpo_metrics(
    batch_shape, num_columns, logits_scale_factor, loss_masking, dtype, compute_entropy, group=None
):
    logits, target, advantages, old_log_probabilities = _get_grpo_loss_inputs(
        num_columns, loss_masking, batch_shape, dtype
    )
    # Different denominators per position so the per-token-mean broadcasting is exercised.
    label_counts = (torch.arange(target.numel(), device=target.device).reshape(target.shape) % 5 + 1).to(
        torch.int32
    ) * (target >= 0)

    ref = reference_grpo_metrics(
        logits,
        target,
        advantages,
        old_log_probabilities,
        label_counts,
        epsilon_low=0.2,
        epsilon_high=0.2,
        logits_scale_factor=logits_scale_factor,
        compute_entropy=compute_entropy,
    )
    got = compute_grpo_metrics(
        split_op(logits, group, -1).contiguous(),
        target,
        advantages,
        old_log_probabilities,
        label_counts,
        epsilon_low=0.2,
        epsilon_high=0.2,
        logits_scale_factor=logits_scale_factor,
        group=group,
        compute_entropy=compute_entropy,
    )
    _check_grpo_metrics(ref, got, threshold=1e-5 if dtype == DataType.float32 else 1e-4)


def _test_z_loss(
    batch_shape, num_columns, grad_output, logits_scale_factor, loss_masking, dtype, block_size, accumulate, group=None
):
    logits, target, loss_mask = _get_lm_loss_inputs(num_columns, loss_masking, TargetFormat.logits, batch_shape, dtype)
    local_logits = split_op(logits, group, -1).contiguous()
    out_ref, grad_ref = loss_forward_backward(
        grad_output,
        z_loss,
        logits,
        loss_mask=loss_mask,
        logits_scale_factor=logits_scale_factor,
    )
    if accumulate:
        previous_grad = torch.randn_like(grad_ref)
        grad_ref = grad_ref + previous_grad
        local_previous_grad = split_op(previous_grad, group, -1).contiguous()
    out_fused, grad_fused = fused_z_loss_forward_backward(
        logits=local_logits,
        loss_mask=loss_mask,
        grad_logits=local_previous_grad.clone() if accumulate else None,
        grad_output=grad_output,
        group=group,
        logits_scale_factor=logits_scale_factor,
    )
    _compare_losses_and_grads(
        out_fused,
        out_ref,
        grad_output is not None,
        grad_fused,
        grad_ref,
        threshold=1e-5 if data_type == DataType.float32 else 1e-4,
        group=group,
    )
    if not triton_available:
        return
    out_triton, grad_triton = triton_z_loss_forward_backward(
        logits=local_logits,
        loss_mask=loss_mask,
        grad_logits=local_previous_grad.clone() if accumulate else None,
        grad_output=grad_output,
        group=group,
        logits_scale_factor=logits_scale_factor,
        block_size=block_size,
    )
    _compare_losses_and_grads(
        out_triton,
        out_ref,
        grad_output is not None,
        grad_triton,
        grad_ref,
        threshold=1e-5 if data_type == DataType.float32 else 1e-4,
        group=group,
    )


@pytest.mark.slow
@pytest.mark.parametrize("batch_shape", _BATCH_SHAPES)
@pytest.mark.parametrize(
    ("num_columns", "grad_output", "logits_scale_factor", "loss_masking", "dtype", "block_size", "accumulate"),
    _LOSS_PARAMETERS,
)
@pytest.mark.parametrize("target_format", TargetFormat)
@pytest.mark.parametrize("entropy_loss_type", EntropyLossType)
def test_entropy_loss(
    batch_shape,
    num_columns,
    grad_output,
    logits_scale_factor,
    loss_masking,
    target_format,
    entropy_loss_type,
    dtype,
    block_size,
    accumulate,
):
    _test_entropy_loss(
        batch_shape,
        num_columns,
        grad_output,
        logits_scale_factor,
        loss_masking,
        target_format,
        entropy_loss_type,
        dtype,
        block_size,
        accumulate,
    )


@pytest.mark.slow
@pytest.mark.parametrize("batch_shape", _BATCH_SHAPES)
@pytest.mark.parametrize(
    ("num_columns", "grad_output", "logits_scale_factor", "loss_masking", "dtype", "block_size", "accumulate"),
    _LOSS_PARAMETERS,
)
def test_z_loss(
    batch_shape, num_columns, grad_output, logits_scale_factor, loss_masking, dtype, block_size, accumulate
):
    _test_z_loss(
        batch_shape, num_columns, grad_output, logits_scale_factor, loss_masking, dtype, block_size, accumulate
    )


@pytest.mark.slow
@pytest.mark.parametrize("batch_shape", _BATCH_SHAPES)
@pytest.mark.parametrize(
    ("num_columns", "grad_output", "logits_scale_factor", "loss_masking", "dtype", "block_size", "accumulate"),
    _LOSS_PARAMETERS,
)
def test_grpo_loss(
    batch_shape, num_columns, grad_output, logits_scale_factor, loss_masking, dtype, block_size, accumulate
):
    _test_grpo_loss(
        batch_shape, num_columns, grad_output, logits_scale_factor, loss_masking, dtype, block_size, accumulate
    )


@pytest.mark.slow
@pytest.mark.parametrize("batch_shape", _BATCH_SHAPES)
@pytest.mark.parametrize(
    ("num_columns", "grad_output", "logits_scale_factor", "loss_masking", "dtype", "block_size", "accumulate"),
    _LOSS_PARAMETERS,
)
@pytest.mark.parametrize("compute_entropy", (False, True))
def test_grpo_metrics(
    batch_shape,
    num_columns,
    grad_output,
    logits_scale_factor,
    loss_masking,
    dtype,
    block_size,
    accumulate,
    compute_entropy,
):
    _test_grpo_metrics(batch_shape, num_columns, logits_scale_factor, loss_masking, dtype, compute_entropy)


@pytest.mark.skip(reason="DPO loss is broken")
def test_dpo_loss():
    logits = torch.normal(0, 1, (200, 100))
    reference_model_logits = torch.normal(0, 1, (200, 100))
    labels = torch.randint(0, 100, (200,))
    spans = get_random_spans(np.full(10, 50), 0, 10)

    fast_llm_loss = dpo_loss(logits, labels, reference_model_logits, spans[::2], spans[1::2])
    reference_loss = reference_dpo_loss(logits, labels, reference_model_logits, spans[::2], spans[1::2], beta=1)
    Assert.rms_close(fast_llm_loss, reference_loss, 1e-5)


def _run_lm_loss_distributed(test_context: DistributedTestContext, base_path: pathlib.Path, seed: int):
    for batch_shape in _BATCH_SHAPES:
        for (
            num_columns,
            grad_output,
            logits_scale_factor,
            loss_masking,
            dtype,
            block_size,
            accumulate,
        ) in _LOSS_PARAMETERS:
            suffix = f"{num_columns}-{grad_output}-{logits_scale_factor}-{loss_masking}-{dtype}-{block_size}-{accumulate}-{"_".join([str(i) for i in batch_shape])}"
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
                                dtype,
                                block_size,
                                accumulate,
                                test_context.group,
                            )
            # Z loss
            with test_context.subtest(base_path, f"z_loss-{suffix}", 2) as subtest:
                if subtest.do_run:
                    torch.manual_seed((seed + hash(subtest.name)) % 2**32)
                    _test_z_loss(
                        batch_shape,
                        num_columns,
                        grad_output,
                        logits_scale_factor,
                        loss_masking,
                        dtype,
                        block_size,
                        accumulate,
                        test_context.group,
                    )
            # GRPO
            with test_context.subtest(base_path, f"grpo-{suffix}", 2) as subtest:
                if subtest.do_run:
                    torch.manual_seed((seed + hash(subtest.name)) % 2**32)
                    _test_grpo_loss(
                        batch_shape,
                        num_columns,
                        grad_output,
                        logits_scale_factor,
                        loss_masking,
                        dtype,
                        block_size,
                        accumulate,
                        test_context.group,
                    )
            # GRPO metrics
            for compute_entropy in (False, True):
                with test_context.subtest(base_path, f"grpo_metrics-{compute_entropy}-{suffix}", 2) as subtest:
                    if subtest.do_run:
                        torch.manual_seed((seed + hash(subtest.name)) % 2**32)
                        _test_grpo_metrics(
                            batch_shape,
                            num_columns,
                            logits_scale_factor,
                            loss_masking,
                            dtype,
                            compute_entropy,
                            test_context.group,
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
        backend=DistributedBackend.nccl if (use_nccl := torch.cuda.device_count() >= 2) else DistributedBackend.gloo,
        use_cuda=use_nccl,  # Disable device count check.
    )


@pytest.mark.slow
@pytest.mark.depends_on(on=["test_lm_loss_distributed_dependency"])
@pytest.mark.parametrize("batch_shape", _BATCH_SHAPES)
@pytest.mark.parametrize(
    ("num_columns", "grad_output", "logits_scale_factor", "loss_masking", "dtype", "block_size", "accumulate"),
    _LOSS_PARAMETERS,
)
@pytest.mark.parametrize(
    "loss_type",
    (
        *(
            f"{entropy_loss_type}-{target_format}"
            for entropy_loss_type in EntropyLossType
            for target_format in TargetFormat
            if target_format != TargetFormat.labels or entropy_loss_type != EntropyLossType.reverse_kl
        ),
        "z_loss",
        "grpo",
        "grpo_metrics-False",
        "grpo_metrics-True",
    ),
)
def test_lm_loss_distributed(
    result_path,
    report_subtest,
    loss_type,
    batch_shape,
    num_columns,
    grad_output,
    logits_scale_factor,
    loss_masking,
    dtype,
    block_size,
    accumulate,
):
    report_subtest(
        result_path
        / f"test_losses/{loss_type}-{num_columns}-{grad_output}-{logits_scale_factor}-{loss_masking}-{dtype}-{block_size}-{accumulate}-{"_".join([str(i) for i in batch_shape])}",
        2,
        use_cuda=False,
    )


# Validates the parallel-teacher-stream distillation contract: student and
# teacher sequences may have different total lengths, but data prep guarantees
# the masked region (assistant response) is byte-identical on each side. The
# KL loss is therefore computed on exactly N response tokens regardless of how
# much prefix surrounds them.
@pytest.mark.parametrize(
    ("L_student", "L_teacher"),
    [
        pytest.param(96, 96, id="equal_length"),
        pytest.param(96, 130, id="teacher_longer"),
        pytest.param(130, 96, id="teacher_shorter"),
    ],
)
@pytest.mark.parametrize("entropy_loss_type", (EntropyLossType.forward_kl, EntropyLossType.reverse_kl))
@pytest.mark.parametrize("dtype", (DataType.float32, DataType.bfloat16))
def test_distillation_gather_then_kl_parallel_stream(entropy_loss_type, dtype, L_student, L_teacher):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    V = 200
    n_response = 35
    divisor = float(n_response)

    response_seed_state = torch.Generator(device=device).manual_seed(0)
    response_student = torch.randn(n_response, V, dtype=dtype.torch, device=device, generator=response_seed_state) / 3
    response_teacher = torch.randn(n_response, V, dtype=dtype.torch, device=device, generator=response_seed_state)

    prefix_seed_state = torch.Generator(device=device).manual_seed(1)
    prefix_student = torch.randn(
        L_student - n_response, V, dtype=dtype.torch, device=device, generator=prefix_seed_state
    )
    prefix_teacher = torch.randn(
        L_teacher - n_response, V, dtype=dtype.torch, device=device, generator=prefix_seed_state
    )

    student_logits = torch.cat([prefix_student, response_student], dim=0)
    teacher_logits = torch.cat([prefix_teacher, response_teacher], dim=0)

    student_mask = torch.zeros(L_student, dtype=torch.bool, device=device)
    student_mask[L_student - n_response :] = True
    teacher_mask = torch.zeros(L_teacher, dtype=torch.bool, device=device)
    teacher_mask[L_teacher - n_response :] = True

    student_flat = student_logits[student_mask]
    teacher_flat = teacher_logits[teacher_mask]
    # Both sides gather to the same number of tokens regardless of L_student / L_teacher.
    Assert.eq(student_flat.shape[0], n_response)
    Assert.eq(teacher_flat.shape[0], n_response)
    # And those tokens are the byte-identical response logits across all three regimes.
    Assert.eq((student_flat - response_student).abs().sum().item(), 0.0)
    Assert.eq((teacher_flat - response_teacher).abs().sum().item(), 0.0)

    new_loss, new_grad_flat = fused_entropy_loss_forward_backward(
        student_flat,
        teacher_flat,
        None,
        grad_output=1.0,
        target_format=TargetFormat.logits,
        entropy_loss_type=entropy_loss_type,
        divisor=divisor,
    )

    # Reference: pytorch KL on the gathered tensors with autograd.
    s_ref = student_flat.detach().float().requires_grad_(True)
    t_ref = teacher_flat.detach().float()
    s_logp = torch.nn.functional.log_softmax(s_ref, dim=-1)
    t_logp = torch.nn.functional.log_softmax(t_ref, dim=-1)
    if entropy_loss_type == EntropyLossType.forward_kl:
        per_sample = torch.nn.functional.kl_div(s_logp, t_logp, reduction="none", log_target=True).sum(-1)
    else:
        per_sample = torch.nn.functional.kl_div(t_logp, s_logp, reduction="none", log_target=True).sum(-1)
    ref_loss = per_sample.sum() / divisor
    ref_loss.backward()
    ref_grad_flat = s_ref.grad

    threshold = 1e-5 if dtype == DataType.float32 else 1e-3
    Assert.rms_close_relative(new_loss, ref_loss, threshold, 1e-6)
    Assert.rms_close_relative(
        new_grad_flat.float(), ref_grad_flat, threshold, 1e-8 if dtype == DataType.float32 else 1e-6
    )

    # Verify the scatter-back step that `_forward_backward_parallel_stream`
    # uses to match the chunked logits shape expected by the head: the grad
    # appears at masked positions, zeros elsewhere.
    grad_scattered = torch.zeros_like(student_logits)
    grad_scattered[student_mask] = new_grad_flat
    Assert.eq(grad_scattered[~student_mask].abs().sum().item(), 0.0)
    Assert.rms_close_relative(
        grad_scattered[student_mask].float(), ref_grad_flat, threshold, 1e-8 if dtype == DataType.float32 else 1e-6
    )
