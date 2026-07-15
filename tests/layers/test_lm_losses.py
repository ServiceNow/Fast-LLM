import pathlib
import random

import numpy as np
import pytest
import torch

from fast_llm.core.ops import split_op
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.engine.distributed.config import DistributedBackend, DistributedConfig
from fast_llm.functional.config import EntropyLossType, TargetFormat
from fast_llm.functional.entropy_loss import torch_entropy_loss_forward_backward
from fast_llm.functional.triton import triton_available
from fast_llm.functional.triton.entropy_loss import triton_entropy_loss_forward_backward
from fast_llm.functional.triton.grpo_loss import triton_grpo_loss_forward_backward
from fast_llm.functional.triton.gspo_loss import triton_gspo_loss_forward_backward
from fast_llm.functional.triton.monolithic_loss import (
    _monolithic_forward_reduce,
    triton_monolithic_loss_forward_backward,
)
from fast_llm.functional.triton.z_loss import triton_z_loss_forward_backward
from fast_llm.layers.language_model.loss.config import (
    LanguageModelDistillationLossConfig,
    LanguageModelGRPOLossConfig,
    LanguageModelLabelEntropyLossConfig,
    LanguageModelZLossConfig,
)
from fast_llm.layers.language_model.loss.dpo import dpo_loss
from fast_llm.layers.language_model.loss.loss import loss_forward_backward
from fast_llm.layers.language_model.loss.monolithic import _monolithic_core
from fast_llm.layers.language_model.loss.policy_gradient import (
    PolicyMetrics,
    compute_grpo_metrics,
    compute_gspo_metrics,
    fused_gspo_loss_forward_backward,
    gspo_segment_seam,
)
from fast_llm.layers.language_model.loss.z_loss import z_loss
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
    loss_min_threshold: float = 1e-6,
):
    # `loss_min_threshold` raises the absolute floor of the scalar-loss comparison. GSPO's per-segment
    # geometric-mean reduction accumulates more float32 error than a per-token sum, and its loss can be near
    # zero, so the fused-vs-reference difference sits at the abs floor rather than the relative band; the
    # default `1e-6` floor flakes there. The gradient stays tight.
    Assert.rms_close_relative(loss, ref_loss, threshold, loss_min_threshold)
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
) -> PolicyMetrics:
    log_softmax = torch.nn.functional.log_softmax(logits.float() * logits_scale_factor, dim=-1)
    loss_mask = target >= 0
    mask = loss_mask.float()
    masked = mask / label_counts.float().clamp(min=1)

    new_log_probs = log_softmax.gather(-1, (target * loss_mask).unsqueeze(-1)).squeeze(-1)
    log_ratio = new_log_probs - old_log_probabilities.float()
    ratio = log_ratio.exp()
    clipped = (ratio < 1.0 - epsilon_low) | (ratio > 1.0 + epsilon_high)
    kl = ratio - log_ratio - 1.0

    entropy_per_token = -(log_softmax.exp() * log_softmax).sum(-1)
    entropy = (entropy_per_token * masked).sum()

    return PolicyMetrics(
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
        num_segments=None,
        entropy=entropy,
    )


def reference_gspo_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probabilities: torch.Tensor,
    document_index: torch.Tensor,
    num_segments: int,
    num_labels_in_seq: torch.Tensor,
    epsilon_low: float,
    epsilon_high: float,
    logits_scale_factor: float,
) -> PolicyMetrics:
    log_softmax = torch.nn.functional.log_softmax(logits.float() * logits_scale_factor, dim=-1)
    loss_mask = target >= 0
    new_log_probs = log_softmax.gather(-1, (target * loss_mask).unsqueeze(-1)).squeeze(-1)
    log_ratio = new_log_probs - old_log_probabilities.float()

    flat_doc = document_index.reshape(-1).long()
    flat_mask = loss_mask.reshape(-1)
    flat_log_ratio = log_ratio.reshape(-1)
    flat_advantages = advantages.reshape(-1).float()
    flat_old = old_log_probabilities.reshape(-1).float()

    old_logprobs = log_ratio.new_zeros(())
    ratio_sum = log_ratio.new_zeros(())
    ratio_squared_sum = log_ratio.new_zeros(())
    kl_sum = log_ratio.new_zeros(())
    clipped_sum = log_ratio.new_zeros(())
    advantage_sum = log_ratio.new_zeros(())
    num_segments_count = log_ratio.new_zeros(())
    for segment in range(num_segments):
        in_segment = (flat_doc == segment) & flat_mask
        count = in_segment.sum()
        if int(count) == 0:
            continue
        count_float = count.float()
        mean_log_ratio = flat_log_ratio[in_segment].sum() / count_float
        ratio = mean_log_ratio.exp()
        ratio_sum = ratio_sum + ratio
        ratio_squared_sum = ratio_squared_sum + ratio * ratio
        kl_sum = kl_sum + (ratio - mean_log_ratio - 1.0)
        clipped_sum = clipped_sum + ((ratio < 1 - epsilon_low) | (ratio > 1 + epsilon_high)).float()
        advantage_sum = advantage_sum + flat_advantages[in_segment].sum() / count_float
        old_logprobs = old_logprobs + flat_old[in_segment].sum() / count_float
        num_segments_count = num_segments_count + 1.0

    entropy_per_token = -(log_softmax.exp() * log_softmax).sum(-1).reshape(-1)
    masked = flat_mask.float() / num_labels_in_seq.reshape(-1).float().clamp(min=1)
    entropy = (entropy_per_token * masked).sum()

    return PolicyMetrics(
        old_logprobs=old_logprobs,
        ratio_new_old=ratio_sum,
        ratio_new_old_sum=ratio_sum,
        ratio_new_old_squared_sum=ratio_squared_sum,
        kl_new_old=kl_sum,
        clipped_ratio_fraction=clipped_sum,
        advantage=advantage_sum,
        max_advantage=advantages[loss_mask].max(),
        min_advantage=advantages[loss_mask].min(),
        num_tokens=loss_mask.sum(),
        num_segments=num_segments_count,
        entropy=entropy,
    )


def reference_gspo_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probabilities: torch.Tensor,
    document_index: torch.Tensor,
    num_segments: int,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.2,
    logits_scale_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits_ = logits.float()
    loss_mask = labels >= 0
    labels_safe = labels * loss_mask
    target_log_probabilities = (
        torch.nn.functional.log_softmax(logits_ * logits_scale_factor, dim=-1)
        .gather(dim=-1, index=labels_safe.unsqueeze(-1))
        .squeeze(-1)
    )
    log_ratio = target_log_probabilities - old_log_probabilities

    flat_doc = document_index.reshape(-1)
    flat_mask = loss_mask.reshape(-1)
    flat_log_ratio = log_ratio.reshape(-1)
    flat_advantages = advantages.reshape(-1)

    total = log_ratio.new_zeros(())
    new_logprobs = log_ratio.new_zeros(())
    for segment in range(num_segments):
        in_segment = (flat_doc == segment) & flat_mask
        count = in_segment.sum()
        if int(count) == 0:
            continue
        count_float = count.float()
        ratio = (flat_log_ratio[in_segment].sum() / count_float).exp()
        advantage = (flat_advantages[in_segment].sum() / count_float).detach()
        clipped_ratio = ratio.clamp(1 - epsilon_low, 1 + epsilon_high)
        total = total + -torch.minimum(ratio * advantage, clipped_ratio * advantage) * count_float
        # Matches the kernel's `sum_t logprob_t * mask_t / N_d` — sum of per-document mean logprobs.
        new_logprobs = new_logprobs + target_log_probabilities.reshape(-1)[in_segment].sum() / count_float
    total = total / num_segments
    return total, new_logprobs


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
    divisor = local_logits.shape[:-1].numel()
    if target_format == TargetFormat.labels:
        loss = _combinable_loss(
            LanguageModelLabelEntropyLossConfig(loss_type=entropy_loss_type), "ce", logits_scale_factor
        )
        arguments = (local_target, grad_output, divisor)
    else:
        loss = _combinable_loss(
            LanguageModelDistillationLossConfig(loss_type=entropy_loss_type), "distillation", logits_scale_factor
        )
        arguments = (local_target, loss_mask, grad_output, divisor, entropy_loss_type, 1.0)
    out_fused, grad_fused, _ = loss.combinable_forward_backward(
        local_logits, group, local_previous_grad.clone() if accumulate else None, arguments
    )
    _compare_losses_and_grads(
        out_fused,
        out_ref,
        grad_output is not None,
        grad_fused,
        grad_ref,
        threshold=1e-5 if dtype == DataType.float32 else 1e-4,
        group=group,
    )

    if not triton_available:
        return
    out_triton, _, grad_triton = triton_entropy_loss_forward_backward(
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
        threshold=1e-5 if dtype == DataType.float32 else 1e-4,
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
    loss = _combinable_loss(LanguageModelGRPOLossConfig(), "grpo", logits_scale_factor)
    out_fused, grad_fused, (new_logprobs_fused, _) = loss.combinable_forward_backward(
        split_op(logits, group, -1),
        group,
        local_previous_grad.clone() if accumulate else None,
        (target, advantages, old_log_probabilities, grad_output, divisor, 0.2, 0.2, num_labels_in_seq, False),
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


def _test_gspo_loss(
    batch_shape,
    num_columns,
    grad_output,
    logits_scale_factor,
    loss_masking,
    dtype,
    num_segments,
    accumulate,
    group=None,
):
    logits, target, advantages, old_log_probabilities = _get_grpo_loss_inputs(
        num_columns, loss_masking, batch_shape, dtype
    )
    # Build per-token segment IDs by partitioning each batch row into `num_segments` contiguous spans.
    seq_len = batch_shape[-1] if len(batch_shape) > 1 else batch_shape[0]
    span = max(seq_len // num_segments, 1)
    base = torch.arange(seq_len, device=target.device) // span
    document_index = base.clamp(max=num_segments - 1).expand(batch_shape).contiguous()
    # Per-document labeled-token count broadcast per token, matching what the data
    # preprocessor produces.
    flat_doc = document_index.reshape(-1).long()
    flat_target = target.reshape(-1)
    labels_per_document = torch.zeros(num_segments, dtype=torch.int32, device=target.device).scatter_add(
        0, flat_doc, (flat_target >= 0).to(torch.int32)
    )
    num_labels_in_seq = labels_per_document[flat_doc].reshape(target.shape)
    _, ref_new_logprobs = reference_gspo_loss(
        logits,
        target,
        advantages,
        old_log_probabilities,
        document_index,
        num_segments,
        logits_scale_factor=logits_scale_factor,
    )
    out_ref, grad_ref = loss_forward_backward(
        grad_output,
        lambda *args, **kwargs: reference_gspo_loss(*args, **kwargs)[0],
        logits,
        target,
        advantages,
        old_log_probabilities,
        document_index,
        num_segments,
        logits_scale_factor=logits_scale_factor,
    )
    if accumulate:
        previous_grad = torch.randn_like(grad_ref)
        grad_ref = grad_ref + previous_grad
        local_previous_grad = split_op(previous_grad, group, -1).contiguous()
    out_fused, grad_fused, new_logprobs_fused = fused_gspo_loss_forward_backward(
        split_op(logits, group, -1),
        target,
        advantages,
        old_log_probabilities,
        document_index,
        num_segments,
        divisor=num_segments,
        grad_logits=local_previous_grad.clone() if accumulate else None,
        grad_output=grad_output,
        group=group,
        logits_scale_factor=logits_scale_factor,
        num_labels_in_seq=num_labels_in_seq,
    )
    _compare_losses_and_grads(
        out_fused, out_ref, grad_output is not None, grad_fused, grad_ref, group=group, loss_min_threshold=1e-5
    )
    Assert.rms_close_relative(new_logprobs_fused, ref_new_logprobs, 1e-5, 1e-6)

    if not triton_available:
        return
    out_triton, grad_triton, new_logprobs_triton = triton_gspo_loss_forward_backward(
        split_op(logits, group, -1).contiguous(),
        target,
        advantages,
        old_log_probabilities,
        document_index,
        num_segments,
        divisor=num_segments,
        num_labels_in_seq=num_labels_in_seq,
        grad_logits=local_previous_grad.clone() if accumulate else None,
        grad_output=grad_output,
        group=group,
        logits_scale_factor=logits_scale_factor,
    )
    _compare_losses_and_grads(
        out_triton, out_ref, grad_output is not None, grad_triton, grad_ref, group=group, loss_min_threshold=1e-5
    )
    Assert.rms_close_relative(new_logprobs_triton, new_logprobs_fused, 1e-5, 1e-6)


def _check_policy_metrics(ref: PolicyMetrics, got: PolicyMetrics, threshold: float) -> None:
    for name in PolicyMetrics._fields:
        ref_value = getattr(ref, name)
        got_value = getattr(got, name)
        if ref_value is None:
            assert got_value is None, name
        else:
            Assert.rms_close_relative(got_value, ref_value, threshold, 1e-6)


def _test_grpo_metrics(batch_shape, num_columns, logits_scale_factor, loss_masking, dtype, group=None):
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
    )
    _check_policy_metrics(ref, got, threshold=5e-5 if dtype == DataType.float32 else 1e-4)


def _test_gspo_metrics(batch_shape, num_columns, logits_scale_factor, loss_masking, dtype, num_segments, group=None):
    logits, target, advantages, old_log_probabilities = _get_grpo_loss_inputs(
        num_columns, loss_masking, batch_shape, dtype
    )
    # Per-token segment IDs + per-document labeled-token counts, mirroring `_test_gspo_loss`.
    seq_len = batch_shape[-1] if len(batch_shape) > 1 else batch_shape[0]
    span = max(seq_len // num_segments, 1)
    base = torch.arange(seq_len, device=target.device) // span
    document_index = base.clamp(max=num_segments - 1).expand(batch_shape).contiguous()
    flat_doc = document_index.reshape(-1).long()
    flat_target = target.reshape(-1)
    labels_per_document = torch.zeros(num_segments, dtype=torch.int32, device=target.device).scatter_add(
        0, flat_doc, (flat_target >= 0).to(torch.int32)
    )
    num_labels_in_seq = labels_per_document[flat_doc].reshape(target.shape)

    ref = reference_gspo_metrics(
        logits,
        target,
        advantages,
        old_log_probabilities,
        document_index,
        num_segments,
        num_labels_in_seq,
        epsilon_low=0.2,
        epsilon_high=0.2,
        logits_scale_factor=logits_scale_factor,
    )
    got = compute_gspo_metrics(
        split_op(logits, group, -1).contiguous(),
        target,
        advantages,
        old_log_probabilities,
        document_index,
        num_segments,
        num_labels_in_seq,
        epsilon_low=0.2,
        epsilon_high=0.2,
        logits_scale_factor=logits_scale_factor,
        group=group,
    )
    _check_policy_metrics(ref, got, threshold=5e-5 if dtype == DataType.float32 else 1e-4)


def _combinable_loss(config, name: str, logits_scale_factor: float):
    # Build the loss object so its `combinable_forward_backward` method is exercised directly. The tensor-
    # parallel `group` is passed per call, so a trivial single-rank distributed config suffices even for the
    # distributed subtests.
    distributed_config = DistributedConfig()
    distributed_config.validate()
    return config.get_layer(distributed_config, name=name, logits_scale_factor=logits_scale_factor)


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
    loss = _combinable_loss(LanguageModelZLossConfig(), "z_loss", logits_scale_factor)
    out_fused, grad_fused, _ = loss.combinable_forward_backward(
        local_logits,
        group,
        local_previous_grad.clone() if accumulate else None,
        (loss_mask, grad_output, local_logits.shape[:-1].numel()),
    )
    _compare_losses_and_grads(
        out_fused,
        out_ref,
        grad_output is not None,
        grad_fused,
        grad_ref,
        threshold=1e-5 if dtype == DataType.float32 else 1e-4,
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
        threshold=1e-5 if dtype == DataType.float32 else 1e-4,
        group=group,
    )


def _test_monolithic_loss(
    batch_shape, num_columns, grad_output, logits_scale_factor, loss_masking, dtype, accumulate, group=None
):
    # A cross-entropy (labels) + z-loss composite sharing one softmax, checked against the same two losses run
    # standalone. This exercises the shared, tensor-parallel-reduced softmax and the fp32 gradient accumulation
    # against the already-validated single-loss path.
    logits, target, _ = _get_lm_loss_inputs(num_columns, loss_masking, TargetFormat.labels, batch_shape, dtype)
    local_logits = split_op(logits, group, -1).contiguous()
    divisor = max(int((target >= 0).sum().item()), 1)
    children = (
        _combinable_loss(LanguageModelLabelEntropyLossConfig(), "cross_entropy", logits_scale_factor),
        _combinable_loss(LanguageModelZLossConfig(), "z_loss", logits_scale_factor),
    )
    arguments = ((target, grad_output, divisor), (None, grad_output, local_logits.shape[:-1].numel()))
    previous_grad = torch.randn_like(local_logits) if accumulate else None

    # Reference: run each loss standalone, accumulating into one gradient buffer.
    grad_ref = previous_grad.clone() if accumulate else None
    losses_ref = []
    for child, child_arguments in zip(children, arguments, strict=True):
        loss_ref, grad_ref, _ = child.combinable_forward_backward(local_logits, group, grad_ref, child_arguments)
        losses_ref.append(loss_ref)

    results, grad_fused = _monolithic_core(
        children, local_logits, group, logits_scale_factor, previous_grad.clone() if accumulate else None, arguments
    )

    threshold = 1e-5 if dtype == DataType.float32 else 1e-4
    for (loss_fused, _), loss_ref in zip(results, losses_ref, strict=True):
        Assert.rms_close_relative(loss_fused, loss_ref, threshold, 1e-6)
    if grad_output is None:
        assert grad_fused is None and grad_ref is None
    else:
        # The composite sums child gradients in fp32 and casts once; the standalone path casts each child
        # gradient before adding. In fp16 the two differ by up to a rounding step, so allow a wider abs floor.
        Assert.rms_close_relative(grad_fused, grad_ref, threshold, 1e-8 if grad_fused.dtype == torch.float32 else 1e-6)


def _test_monolithic_loss_triton(
    batch_shape, num_columns, grad_output, logits_scale_factor, loss_masking, dtype, block_size, accumulate, group=None
):
    # The triton monolithic kernel (cross-entropy + z-loss over one softmax) against the sum of the standalone
    # triton kernels. Both use the same label-count divisor, as the head threads the same divisor to each
    # combinable child. Tensor-parallel `group` is passed per call for the distributed subtest (case B).
    if not triton_available:
        return
    logits, target, _ = _get_lm_loss_inputs(num_columns, loss_masking, TargetFormat.labels, batch_shape, dtype)
    local_logits = split_op(logits, group, -1).contiguous()
    divisor = max(int((target >= 0).sum().item()), 1)
    previous_grad = torch.randn_like(local_logits) if accumulate else None

    # Reference: standalone triton cross-entropy then z-loss, accumulating into one gradient buffer.
    grad_ref = previous_grad.clone() if accumulate else None
    ce_ref, _, grad_ref = triton_entropy_loss_forward_backward(
        local_logits,
        target,
        None,
        grad_logits=grad_ref,
        grad_output=grad_output,
        group=group,
        logits_scale_factor=logits_scale_factor,
        target_format=TargetFormat.labels,
        divisor=divisor,
        block_size=block_size,
    )
    z_ref, grad_ref = triton_z_loss_forward_backward(
        local_logits,
        None,
        grad_logits=grad_ref,
        grad_output=grad_output,
        group=group,
        logits_scale_factor=logits_scale_factor,
        divisor=divisor,
        block_size=block_size,
    )

    ce_fused, z_fused, _, _, grad_fused, _, _ = triton_monolithic_loss_forward_backward(
        local_logits,
        target.contiguous(),
        previous_grad.clone() if accumulate else None,
        logits_scale_factor,
        group,
        divisor,
        ce=(grad_output,),
        z=(None, grad_output),
        block_size=block_size,
    )

    threshold = 1e-5 if dtype == DataType.float32 else 1e-4
    Assert.rms_close_relative(ce_fused, ce_ref, threshold, 1e-6)
    Assert.rms_close_relative(z_fused, z_ref, threshold, 1e-6)
    if grad_output is None:
        assert grad_fused is None and grad_ref is None
    else:
        # The kernel superposes both losses' gradients in fp32 before the single cast; the standalone path
        # rounds cross-entropy's gradient before z-loss adds to it, so the two differ by up to a rounding step.
        Assert.rms_close_relative(grad_fused, grad_ref, threshold, 1e-8 if grad_fused.dtype == torch.float32 else 1e-6)


def _test_monolithic_single_loss_triton(
    batch_shape, num_columns, grad_output, logits_scale_factor, loss_masking, dtype, block_size, accumulate, group=None
):
    # A single cross-entropy loss through the monolithic kernel. With masking this exercises the masked-row
    # early-return (no z-loss / GSPO / metrics need the softmax on such a row), which must match the standalone
    # triton cross-entropy exactly — the skip only avoids work that would have summed to zero.
    if not triton_available:
        return
    logits, target, _ = _get_lm_loss_inputs(num_columns, loss_masking, TargetFormat.labels, batch_shape, dtype)
    local_logits = split_op(logits, group, -1).contiguous()
    divisor = max(int((target >= 0).sum().item()), 1)
    previous_grad = torch.randn_like(local_logits) if accumulate else None

    ce_ref, _, grad_ref = triton_entropy_loss_forward_backward(
        local_logits,
        target,
        None,
        grad_logits=previous_grad.clone() if accumulate else None,
        grad_output=grad_output,
        group=group,
        logits_scale_factor=logits_scale_factor,
        target_format=TargetFormat.labels,
        divisor=divisor,
        block_size=block_size,
    )
    ce_fused, _, _, _, grad_fused, _, _ = triton_monolithic_loss_forward_backward(
        local_logits,
        target.contiguous(),
        previous_grad.clone() if accumulate else None,
        logits_scale_factor,
        group,
        divisor,
        ce=(grad_output,),
        block_size=block_size,
    )
    threshold = 1e-5 if dtype == DataType.float32 else 1e-4
    Assert.rms_close_relative(ce_fused, ce_ref, threshold, 1e-6)
    if grad_output is None:
        assert grad_fused is None and grad_ref is None
    else:
        Assert.rms_close_relative(grad_fused, grad_ref, threshold, 1e-8 if grad_fused.dtype == torch.float32 else 1e-7)


def _test_monolithic_grpo_loss_triton(
    batch_shape, num_columns, grad_output, logits_scale_factor, loss_masking, dtype, block_size, accumulate, group=None
):
    # A single GRPO loss through the monolithic kernel's `grpo` slot (loss / gradient / new-log-probs), against
    # the python reference. The ce+z monolithic tests never pass `grpo=`, so this is the only low-level cover of
    # that slot; the object-dispatch driver around it is exercised by the head tests.
    if not triton_available:
        return
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
    _, _, grpo_fused, _, grad_fused, _, _ = triton_monolithic_loss_forward_backward(
        split_op(logits, group, -1).contiguous(),
        target.contiguous(),
        local_previous_grad.clone() if accumulate else None,
        logits_scale_factor,
        group,
        divisor,
        grpo=(advantages, old_log_probabilities, grad_output, 0.2, 0.2, num_labels_in_seq),
        block_size=block_size,
    )
    _compare_losses_and_grads(grpo_fused, out_ref, grad_output is not None, grad_fused, grad_ref, group=group)


def _test_monolithic_gspo_loss_triton(
    batch_shape,
    num_columns,
    grad_output,
    logits_scale_factor,
    loss_masking,
    dtype,
    num_segments,
    accumulate,
    group=None,
):
    # A single GSPO loss through the monolithic path: the eager segment seam produces the per-token backward
    # coefficient, which the kernel's `gspo_coeff` slot then superposes. Checked against the python reference.
    # The ce+z monolithic tests never pass `gspo_coeff=`.
    if not triton_available:
        return
    logits, target, advantages, old_log_probabilities = _get_grpo_loss_inputs(
        num_columns, loss_masking, batch_shape, dtype
    )
    seq_len = batch_shape[-1] if len(batch_shape) > 1 else batch_shape[0]
    span = max(seq_len // num_segments, 1)
    document_index = (torch.arange(seq_len, device=target.device) // span).clamp(max=num_segments - 1)
    document_index = document_index.expand(batch_shape).contiguous()
    flat_doc = document_index.reshape(-1).long()
    labels_per_document = torch.zeros(num_segments, dtype=torch.int32, device=target.device).scatter_add(
        0, flat_doc, (target.reshape(-1) >= 0).to(torch.int32)
    )
    num_labels_in_seq = labels_per_document[flat_doc].reshape(target.shape)
    out_ref, grad_ref = loss_forward_backward(
        grad_output,
        lambda *args, **kwargs: reference_gspo_loss(*args, **kwargs)[0],
        logits,
        target,
        advantages,
        old_log_probabilities,
        document_index,
        num_segments,
        logits_scale_factor=logits_scale_factor,
    )
    if accumulate:
        previous_grad = torch.randn_like(grad_ref)
        grad_ref = grad_ref + previous_grad
        local_previous_grad = split_op(previous_grad, group, -1).contiguous()
    local_logits = split_op(logits, group, -1).contiguous()
    # Eager pre-kernel phases (forward reduce + segment seam), as the driver runs them, produce the coefficient.
    softmax = _monolithic_forward_reduce(local_logits, target.contiguous(), group, logits_scale_factor)
    max_logits, sum_exp_logits, predicted_logits = softmax
    new_log_probs = (predicted_logits - max_logits - sum_exp_logits.log()).reshape(target.shape)
    gspo_loss, _, gspo_coeff = gspo_segment_seam(
        new_log_probs,
        target >= 0,
        advantages,
        old_log_probabilities,
        document_index,
        num_segments,
        num_labels_in_seq,
        num_segments,  # divisor
        grad_output,
        None,  # sdp_group
        None,  # sp_group: the vocab-parallel path shards the vocab, not the sequence
        0.2,
        0.2,
        logits_scale_factor,
    )
    _, _, _, _, grad_fused, _, _ = triton_monolithic_loss_forward_backward(
        local_logits,
        target.contiguous(),
        local_previous_grad.clone() if accumulate else None,
        logits_scale_factor,
        group,
        1.0,  # GSPO pre-scales its coefficient; no in-kernel divisor
        gspo_coeff=None if gspo_coeff is None else gspo_coeff.reshape(-1).contiguous(),
        softmax=softmax,
    )
    _compare_losses_and_grads(
        gspo_loss, out_ref, grad_output is not None, grad_fused, grad_ref, group=group, loss_min_threshold=1e-5
    )


def _test_monolithic_metrics_without_grad_triton(batch_shape, num_columns, loss_masking, dtype, group=None):
    # A zero-weight policy loss (grad_output=None) that still logs metrics: `compute_metrics=True` with no
    # gradient must emit the same forward loss / new-log-probs / Σ exp·logits_norm as the with-gradient call,
    # not crash or skip. The kernel runs the backward re-stream for the metrics alone.
    if not triton_available:
        return
    logits, target, advantages, old_log_probabilities = _get_grpo_loss_inputs(
        num_columns, loss_masking, batch_shape, dtype
    )
    num_labels = int((target >= 0).sum().item())
    num_labels_in_seq = torch.where(
        target >= 0,
        torch.full(batch_shape, num_labels, dtype=torch.int32, device=target.device),
        torch.zeros(batch_shape, dtype=torch.int32, device=target.device),
    )
    local_logits = split_op(logits, group, -1).contiguous()

    def run(grad_output):
        return triton_monolithic_loss_forward_backward(
            local_logits,
            target.contiguous(),
            None,
            1.0,
            group,
            max(num_labels, 1),
            grpo=(advantages, old_log_probabilities, grad_output, 0.2, 0.2, num_labels_in_seq),
            compute_metrics=True,
        )

    _, _, loss_grad, new_logprobs_grad, grad_grad, _, weighted_grad = run(1.0)
    _, _, loss_nograd, new_logprobs_nograd, grad_nograd, _, weighted_nograd = run(None)

    assert grad_grad is not None and grad_nograd is None
    Assert.rms_close_relative(loss_nograd, loss_grad, 1e-5, 1e-6)
    Assert.rms_close_relative(new_logprobs_nograd, new_logprobs_grad, 1e-5, 1e-6)
    Assert.rms_close_relative(weighted_nograd, weighted_grad, 1e-5, 1e-6)


def _test_monolithic_distillation_loss_triton(
    batch_shape,
    num_columns,
    grad_output,
    logits_scale_factor,
    loss_masking,
    dtype,
    entropy_loss_type,
    block_size,
    accumulate,
    group=None,
):
    # Distillation (from a teacher distribution) fused with z-loss over one shared student softmax — the
    # distribution-family kernel — against the sum of the standalone triton distillation and z-loss kernels
    # accumulating into one gradient buffer. Covers each KL direction. Distillation and z-loss share the loss
    # mask and the divisor (both are threaded identically by the head), and a non-unit temperature checks that
    # it stays confined to the teacher term (z-loss is on the student softmax, so it must be temperature-free).
    if not triton_available:
        return
    temperature = 1.5
    logits, target, loss_mask = _get_lm_loss_inputs(num_columns, loss_masking, TargetFormat.logits, batch_shape, dtype)
    local_logits = split_op(logits, group, -1).contiguous()
    local_target = split_op(target, group, -1).contiguous()
    divisor = local_logits.shape[:-1].numel()
    previous_grad = torch.randn_like(local_logits) if accumulate else None

    # Reference: standalone triton distillation then z-loss, accumulating into one gradient buffer.
    grad_ref = previous_grad.clone() if accumulate else None
    dist_ref, _, grad_ref = triton_entropy_loss_forward_backward(
        local_logits,
        local_target,
        loss_mask,
        grad_logits=grad_ref,
        grad_output=grad_output,
        group=group,
        logits_scale_factor=logits_scale_factor,
        temperature=temperature,
        target_format=TargetFormat.logits,
        entropy_loss_type=entropy_loss_type,
        divisor=divisor,
        block_size=block_size,
    )
    z_ref, grad_ref = triton_z_loss_forward_backward(
        local_logits,
        loss_mask,
        grad_logits=grad_ref,
        grad_output=grad_output,
        group=group,
        logits_scale_factor=logits_scale_factor,
        divisor=divisor,
        block_size=block_size,
    )

    dist_fused, z_fused, grad_fused = triton_entropy_loss_forward_backward(
        local_logits,
        local_target,
        loss_mask,
        grad_logits=previous_grad.clone() if accumulate else None,
        grad_output=grad_output,
        group=group,
        logits_scale_factor=logits_scale_factor,
        temperature=temperature,
        target_format=TargetFormat.logits,
        entropy_loss_type=entropy_loss_type,
        z=(grad_output,),
        divisor=divisor,
        block_size=block_size,
    )

    threshold = 1e-5 if dtype == DataType.float32 else 1e-4
    Assert.rms_close_relative(dist_fused, dist_ref, threshold, 1e-6)
    Assert.rms_close_relative(z_fused, z_ref, threshold, 1e-6)
    if grad_output is None:
        assert grad_fused is None and grad_ref is None
    else:
        # The kernel superposes both losses' gradients in fp32 before the single cast; the standalone path
        # rounds distillation's gradient before z-loss adds to it, so the two differ by up to a rounding step.
        Assert.rms_close_relative(grad_fused, grad_ref, threshold, 1e-8 if grad_fused.dtype == torch.float32 else 1e-6)


@pytest.mark.slow
@pytest.mark.parametrize("batch_shape", _BATCH_SHAPES)
@pytest.mark.parametrize("loss_masking", (False, True))
def test_monolithic_metrics_without_grad_triton(batch_shape, loss_masking):
    _test_monolithic_metrics_without_grad_triton(batch_shape, 500, loss_masking, DataType.float32)


@pytest.mark.slow
@pytest.mark.parametrize("batch_shape", _BATCH_SHAPES)
@pytest.mark.parametrize(
    ("num_columns", "grad_output", "logits_scale_factor", "loss_masking", "dtype", "block_size", "accumulate"),
    _LOSS_PARAMETERS,
)
@pytest.mark.parametrize("target_format", (TargetFormat.labels, TargetFormat.logits))
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
def test_monolithic_loss(
    batch_shape, num_columns, grad_output, logits_scale_factor, loss_masking, dtype, block_size, accumulate
):
    _test_monolithic_loss(batch_shape, num_columns, grad_output, logits_scale_factor, loss_masking, dtype, accumulate)


@pytest.mark.slow
@pytest.mark.parametrize("batch_shape", _BATCH_SHAPES)
@pytest.mark.parametrize(
    ("num_columns", "grad_output", "logits_scale_factor", "loss_masking", "dtype", "block_size", "accumulate"),
    _LOSS_PARAMETERS,
)
def test_monolithic_loss_triton(
    batch_shape, num_columns, grad_output, logits_scale_factor, loss_masking, dtype, block_size, accumulate
):
    _test_monolithic_loss_triton(
        batch_shape, num_columns, grad_output, logits_scale_factor, loss_masking, dtype, block_size, accumulate
    )


@pytest.mark.slow
@pytest.mark.parametrize("batch_shape", _BATCH_SHAPES)
@pytest.mark.parametrize(
    ("num_columns", "grad_output", "logits_scale_factor", "loss_masking", "dtype", "block_size", "accumulate"),
    _LOSS_PARAMETERS,
)
@pytest.mark.parametrize("entropy_loss_type", EntropyLossType)
def test_monolithic_distillation_loss_triton(
    batch_shape,
    num_columns,
    grad_output,
    logits_scale_factor,
    loss_masking,
    dtype,
    block_size,
    accumulate,
    entropy_loss_type,
):
    _test_monolithic_distillation_loss_triton(
        batch_shape,
        num_columns,
        grad_output,
        logits_scale_factor,
        loss_masking,
        dtype,
        entropy_loss_type,
        block_size,
        accumulate,
    )


@pytest.mark.slow
@pytest.mark.parametrize("batch_shape", _BATCH_SHAPES)
@pytest.mark.parametrize(
    ("num_columns", "grad_output", "logits_scale_factor", "loss_masking", "dtype", "block_size", "accumulate"),
    _LOSS_PARAMETERS,
)
def test_monolithic_single_loss_triton(
    batch_shape, num_columns, grad_output, logits_scale_factor, loss_masking, dtype, block_size, accumulate
):
    _test_monolithic_single_loss_triton(
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


_GSPO_PARAMETERS = (
    # (num_columns, grad_output, logits_scale_factor, loss_masking, dtype, num_segments, accumulate)
    (500, 1.0, 1.0, False, DataType.float32, 4, False),  # Simple
    (256, 1.0, 1.0, False, DataType.float32, 4, False),  # Power of 2
    (500, None, 1.0, False, DataType.float32, 4, False),  # No grad
    (500, 1.0, 1.0, False, DataType.float32, 4, True),  # Accumulate
    (500, 1.0, 4.0, False, DataType.float32, 4, False),  # Loss scaling
    (500, 4.0, 1.0, False, DataType.float32, 4, False),  # Grad scaling
    (500, 1.0, 1.0, True, DataType.float32, 4, False),  # Loss masking
    (500, 1.0, 1.0, False, DataType.float16, 4, False),  # Fp16
    (500, 1.0, 1.0, False, DataType.float32, 1, False),  # One segment
    (500, 1.0, 1.0, True, DataType.float32, 16, True),  # Many segments + masking + accumulate
)


@pytest.mark.slow
@pytest.mark.parametrize("batch_shape", _BATCH_SHAPES)
@pytest.mark.parametrize(
    ("num_columns", "grad_output", "logits_scale_factor", "loss_masking", "dtype", "num_segments", "accumulate"),
    _GSPO_PARAMETERS,
)
def test_gspo_loss(
    batch_shape, num_columns, grad_output, logits_scale_factor, loss_masking, dtype, num_segments, accumulate
):
    _test_gspo_loss(
        batch_shape, num_columns, grad_output, logits_scale_factor, loss_masking, dtype, num_segments, accumulate
    )


@pytest.mark.slow
@pytest.mark.parametrize("batch_shape", _BATCH_SHAPES)
@pytest.mark.parametrize(
    ("num_columns", "grad_output", "logits_scale_factor", "loss_masking", "dtype", "block_size", "accumulate"),
    _LOSS_PARAMETERS,
)
def test_monolithic_grpo_loss_triton(
    batch_shape, num_columns, grad_output, logits_scale_factor, loss_masking, dtype, block_size, accumulate
):
    _test_monolithic_grpo_loss_triton(
        batch_shape, num_columns, grad_output, logits_scale_factor, loss_masking, dtype, block_size, accumulate
    )


@pytest.mark.slow
@pytest.mark.parametrize("batch_shape", _BATCH_SHAPES)
@pytest.mark.parametrize(
    ("num_columns", "grad_output", "logits_scale_factor", "loss_masking", "dtype", "num_segments", "accumulate"),
    _GSPO_PARAMETERS,
)
def test_monolithic_gspo_loss_triton(
    batch_shape, num_columns, grad_output, logits_scale_factor, loss_masking, dtype, num_segments, accumulate
):
    _test_monolithic_gspo_loss_triton(
        batch_shape, num_columns, grad_output, logits_scale_factor, loss_masking, dtype, num_segments, accumulate
    )


@pytest.mark.slow
@pytest.mark.parametrize("batch_shape", _BATCH_SHAPES)
@pytest.mark.parametrize(
    ("num_columns", "grad_output", "logits_scale_factor", "loss_masking", "dtype", "block_size", "accumulate"),
    _LOSS_PARAMETERS,
)
def test_grpo_metrics(
    batch_shape,
    num_columns,
    grad_output,
    logits_scale_factor,
    loss_masking,
    dtype,
    block_size,
    accumulate,
):
    _test_grpo_metrics(batch_shape, num_columns, logits_scale_factor, loss_masking, dtype)


@pytest.mark.slow
@pytest.mark.parametrize("batch_shape", _BATCH_SHAPES)
@pytest.mark.parametrize(
    ("num_columns", "grad_output", "logits_scale_factor", "loss_masking", "dtype", "num_segments", "accumulate"),
    _GSPO_PARAMETERS,
)
def test_gspo_metrics(
    batch_shape,
    num_columns,
    grad_output,
    logits_scale_factor,
    loss_masking,
    dtype,
    num_segments,
    accumulate,
):
    _test_gspo_metrics(batch_shape, num_columns, logits_scale_factor, loss_masking, dtype, num_segments)


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
                for target_format in (TargetFormat.labels, TargetFormat.logits):
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
            # GSPO (tensor-parallel vocab path; segment seam runs eagerly per rank)
            with test_context.subtest(base_path, f"gspo-{suffix}", 2) as subtest:
                if subtest.do_run:
                    torch.manual_seed((seed + hash(subtest.name)) % 2**32)
                    _test_gspo_loss(
                        batch_shape,
                        num_columns,
                        grad_output,
                        logits_scale_factor,
                        loss_masking,
                        dtype,
                        4,  # num_segments
                        accumulate,
                        test_context.group,
                    )
            # GRPO metrics
            with test_context.subtest(base_path, f"grpo_metrics-{suffix}", 2) as subtest:
                if subtest.do_run:
                    torch.manual_seed((seed + hash(subtest.name)) % 2**32)
                    _test_grpo_metrics(
                        batch_shape,
                        num_columns,
                        logits_scale_factor,
                        loss_masking,
                        dtype,
                        test_context.group,
                    )
            # GSPO metrics
            num_segments = 4
            with test_context.subtest(base_path, f"gspo_metrics-{suffix}", 2) as subtest:
                if subtest.do_run:
                    torch.manual_seed((seed + hash(subtest.name)) % 2**32)
                    _test_gspo_metrics(
                        batch_shape,
                        num_columns,
                        logits_scale_factor,
                        loss_masking,
                        dtype,
                        num_segments,
                        test_context.group,
                    )
            # Monolithic composite: multiple losses share one tensor-parallel-reduced softmax.
            with test_context.subtest(base_path, f"monolithic-{suffix}", 2) as subtest:
                if subtest.do_run:
                    torch.manual_seed((seed + hash(subtest.name)) % 2**32)
                    _test_monolithic_loss(
                        batch_shape,
                        num_columns,
                        grad_output,
                        logits_scale_factor,
                        loss_masking,
                        dtype,
                        accumulate,
                        test_context.group,
                    )
            # Triton monolithic composite: the vocab-parallel reduced-softmax path (case B).
            with test_context.subtest(base_path, f"monolithic-triton-{suffix}", 2) as subtest:
                if subtest.do_run:
                    torch.manual_seed((seed + hash(subtest.name)) % 2**32)
                    _test_monolithic_loss_triton(
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
            # Triton monolithic GRPO / GSPO: the policy-loss slots over the tensor-parallel reduced softmax.
            with test_context.subtest(base_path, f"monolithic-grpo-triton-{suffix}", 2) as subtest:
                if subtest.do_run:
                    torch.manual_seed((seed + hash(subtest.name)) % 2**32)
                    _test_monolithic_grpo_loss_triton(
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
            with test_context.subtest(base_path, f"monolithic-gspo-triton-{suffix}", 2) as subtest:
                if subtest.do_run:
                    torch.manual_seed((seed + hash(subtest.name)) % 2**32)
                    _test_monolithic_gspo_loss_triton(
                        batch_shape,
                        num_columns,
                        grad_output,
                        logits_scale_factor,
                        loss_masking,
                        dtype,
                        4,  # num_segments
                        accumulate,
                        test_context.group,
                    )
            # Triton monolithic distribution family: distillation fused with z-loss over one softmax.
            for entropy_loss_type in EntropyLossType:
                with test_context.subtest(
                    base_path, f"monolithic-distillation-{entropy_loss_type}-triton-{suffix}", 2
                ) as subtest:
                    if subtest.do_run:
                        torch.manual_seed((seed + hash(subtest.name)) % 2**32)
                        _test_monolithic_distillation_loss_triton(
                            batch_shape,
                            num_columns,
                            grad_output,
                            logits_scale_factor,
                            loss_masking,
                            dtype,
                            entropy_loss_type,
                            block_size,
                            accumulate,
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
            for target_format in (TargetFormat.labels, TargetFormat.logits)
            if target_format != TargetFormat.labels or entropy_loss_type != EntropyLossType.reverse_kl
        ),
        "z_loss",
        "grpo",
        "gspo",
        "grpo_metrics",
        "gspo_metrics",
        "monolithic",
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
