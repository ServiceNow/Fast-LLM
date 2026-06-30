import typing

import torch

from fast_llm.core.distributed import ProcessGroup
from fast_llm.functional.config import EntropyLossType, TargetFormat
from fast_llm.functional.entropy_loss import (
    cross_entropy_from_distribution_core,
    predicted_logits_from_labels,
    reverse_kl_from_distribution_core,
    softmax_base,
)
from fast_llm.layers.language_model.loss.grpo_metrics import GRPOMetrics, grpo_metrics_core
from fast_llm.utils import Assert


class MonolithicLossSpec(typing.NamedTuple):
    """
    Per-loss inputs gathered by `LanguageModelLoss.get_monolithic_spec` and consumed by the monolithic
    head-loss kernel. `kind` selects which branch of the kernel computes the loss and its gradient.
    The math fields not used by a given `kind` are left at their defaults.
    """

    kind: str
    name: str
    weight: float
    logits_scale_factor: float
    grad_output: float | None
    divisor: float
    target: torch.Tensor | None = (
        None  # Cross-entropy / GRPO / GSPO labels, or a teacher distribution (logits/probabilities).
    )
    loss_mask: torch.Tensor | None = None  # z-loss / distribution-loss mask (cross-entropy / GRPO derive their own).
    # Distribution losses (cross-entropy / forward-KL / reverse-KL from a teacher distribution) only.
    target_format: TargetFormat = TargetFormat.logits
    entropy_loss_type: EntropyLossType = EntropyLossType.cross_entropy
    temperature: float = 1.0
    # Policy-gradient (GRPO / GSPO) only.
    advantages: torch.Tensor | None = None
    old_log_probabilities: torch.Tensor | None = None
    epsilon_low: float = 0.2
    epsilon_high: float = 0.2
    num_labels_in_seq: torch.Tensor | None = None  # Per-sequence label count; enables the `new_logprobs_mean` metric.
    compute_metrics: bool = False  # Emit the GRPO metric family from the shared softmax.
    compute_entropy: bool = False  # Also emit per-token entropy (the only metric needing the vocab axis).
    # Sequence-policy-gradient (GSPO) only.
    document_index: torch.Tensor | None = None  # Per-token segment ID (0-based).
    num_segments: int = 0  # Per-segment buffer size, ≥ document_index.max() + 1.
    sdp_group: ProcessGroup | None = None  # Sequence-data group for cross-rank segment aggregation.
    sp_group: ProcessGroup | None = None  # TP group when sequence-parallel shards the sequence.


class MonolithicLossOutput(typing.NamedTuple):
    """Per-loss outputs returned by the monolithic kernel, registered via `register_monolithic_outputs`."""

    loss: torch.Tensor | None = None
    new_logprobs_mean: torch.Tensor | None = None  # GRPO / GSPO only.
    metrics: GRPOMetrics | None = None  # GRPO only.


def _apply_combinable_losses(
    logits_norm: torch.Tensor,  # (*batch, vocab)
    exp_logits: torch.Tensor,  # (*batch, vocab)
    sum_exp_logits: torch.Tensor,  # (*batch,)
    logits_max: torch.Tensor,  # (*batch,)
    group: ProcessGroup | None,
    logits_scale_factor: float,
    ce_target: torch.Tensor | None,
    ce_grad_output: float | None,
    ce_divisor: float,
    z_loss_enabled: bool,
    z_loss_mask: torch.Tensor | None,
    z_loss_grad_output: float | None,
    z_loss_divisor: float,
    distribution_target: torch.Tensor | None,
    distribution_grad_output: float | None,
    distribution_divisor: float,
    distribution_loss_mask: torch.Tensor | None,
    distribution_target_format: TargetFormat,
    distribution_entropy_loss_type: EntropyLossType,
    distribution_temperature: float,
    grpo_target: torch.Tensor | None,
    grpo_advantages: torch.Tensor | None,
    grpo_old_log_probabilities: torch.Tensor | None,
    grpo_grad_output: float | None,
    grpo_divisor: float,
    grpo_epsilon_low: float,
    grpo_epsilon_high: float,
    grpo_num_labels_in_seq: torch.Tensor | None,
    grpo_compute_metrics: bool,
    grpo_compute_entropy: bool,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    "GRPOMetrics | None",
    torch.Tensor | None,
]:
    """
    The per-loss scalar + fp32 gradient-accumulator math shared by the single-pass (`_monolithic_core`)
    and GSPO-split (`_monolithic_gspo_forward_core`) paths. Takes the already-computed shared softmax
    tensors and returns each enabled loss's scalar plus the summed gradient, accumulated in fp32 and
    *not* cast (the caller casts once). Plain (un-decorated) so it inlines and fuses into its compiled
    callers; a disabled loss is dead-code-eliminated on its `None`/static flag.
    """
    grad = None

    cross_entropy_loss = None
    if ce_target is not None:
        loss_mask = ce_target >= 0
        predicted_logits, target_masked, target_mask = predicted_logits_from_labels(
            logits_norm, ce_target, loss_mask, group
        )
        cross_entropy_loss = ((sum_exp_logits.log() - predicted_logits) * loss_mask).sum() / ce_divisor
        if ce_grad_output is not None:
            grad_output = ce_grad_output / ce_divisor * logits_scale_factor
            cross_entropy_grad = exp_logits.scatter_add(
                -1,
                target_masked.unsqueeze(-1),
                (
                    -sum_exp_logits.unsqueeze(-1)
                    if target_mask is None
                    else -(target_mask * sum_exp_logits).unsqueeze(-1)
                ),
            ) * (grad_output / sum_exp_logits.unsqueeze(-1))
            cross_entropy_grad = cross_entropy_grad * loss_mask.unsqueeze(-1)
            grad = cross_entropy_grad if grad is None else grad + cross_entropy_grad

    z_loss = None
    if z_loss_enabled:
        # z-loss needs the un-regularized log-sum-exp, so it adds back `logits_max` (cross-entropy cancels it).
        log_sum_exp_logits = sum_exp_logits.log() + logits_max
        z_loss_term = log_sum_exp_logits**2
        if z_loss_mask is not None:
            z_loss_term = z_loss_term * z_loss_mask
        z_loss = z_loss_term.sum() / z_loss_divisor
        if z_loss_grad_output is not None:
            grad_output = z_loss_grad_output / z_loss_divisor * logits_scale_factor
            z_loss_grad_base = 2 * grad_output * (log_sum_exp_logits / sum_exp_logits)
            if z_loss_mask is not None:
                z_loss_grad_base = z_loss_grad_base * z_loss_mask
            z_loss_grad = z_loss_grad_base.unsqueeze(-1) * exp_logits
            grad = z_loss_grad if grad is None else grad + z_loss_grad

    distribution_loss = None
    if distribution_target is not None:
        distribution_grad_output_scaled = (
            None
            if distribution_grad_output is None
            else distribution_grad_output / distribution_divisor * logits_scale_factor
        )
        if distribution_entropy_loss_type == EntropyLossType.reverse_kl:
            per_sample_loss, distribution_grad = reverse_kl_from_distribution_core(
                logits_norm,
                exp_logits,
                sum_exp_logits,
                distribution_target,
                distribution_grad_output_scaled,
                logits_scale_factor,
                distribution_target_format,
                group,
                distribution_temperature,
            )
        else:
            per_sample_loss, distribution_grad = cross_entropy_from_distribution_core(
                logits_norm,
                exp_logits,
                sum_exp_logits,
                distribution_target,
                distribution_grad_output_scaled,
                logits_scale_factor,
                distribution_target_format,
                group,
                distribution_temperature,
                return_kl_loss=distribution_entropy_loss_type == EntropyLossType.forward_kl,
            )
        if distribution_loss_mask is not None:
            per_sample_loss = per_sample_loss * distribution_loss_mask
        distribution_loss = per_sample_loss.sum() / distribution_divisor
        if distribution_grad is not None:
            if distribution_loss_mask is not None:
                distribution_grad = distribution_grad * distribution_loss_mask.unsqueeze(-1)
            grad = distribution_grad if grad is None else grad + distribution_grad

    grpo_loss = None
    grpo_new_logprobs_mean = None
    grpo_metrics = None
    if grpo_target is not None:
        grpo_loss_mask = grpo_target >= 0
        grpo_predicted_logits, grpo_target_masked, grpo_target_mask = predicted_logits_from_labels(
            logits_norm, grpo_target, grpo_loss_mask, group
        )
        new_log_probs = grpo_predicted_logits - sum_exp_logits.log()
        probability_ratio = (new_log_probs - grpo_old_log_probabilities).exp()
        grpo_losses = -torch.min(
            probability_ratio * grpo_advantages,
            torch.clamp(probability_ratio, 1 - grpo_epsilon_low, 1 + grpo_epsilon_high) * grpo_advantages,
        )
        grpo_loss = (grpo_losses * grpo_loss_mask).sum() / grpo_divisor
        if grpo_num_labels_in_seq is not None:
            # Sum of per-sequence mean log-probs; clamp avoids 0/0 at fully-masked documents (also loss-masked).
            grpo_new_logprobs_mean = (new_log_probs * grpo_loss_mask / grpo_num_labels_in_seq.clamp(min=1)).sum()
        if grpo_compute_metrics:
            # The metric family reuses this branch's softmax + new_log_probs — no second softmax pass.
            grpo_metrics = grpo_metrics_core(
                logits_norm,
                exp_logits,
                sum_exp_logits,
                new_log_probs,
                grpo_advantages,
                grpo_old_log_probabilities,
                grpo_loss_mask,
                grpo_num_labels_in_seq,
                grpo_epsilon_low,
                grpo_epsilon_high,
                group,
                grpo_compute_entropy,
            )
        if grpo_grad_output is not None:
            grad_output = grpo_grad_output / grpo_divisor * logits_scale_factor
            # grad[a>=0] = -a * (ratio <= 1 + epsilon_high); grad[a<=0] = a * (ratio >= 1 - epsilon_low).
            probability_ratio_grad = (
                grad_output
                * (
                    torch.clamp_min(grpo_advantages, 0) * (probability_ratio <= 1 + grpo_epsilon_high)
                    + torch.clamp_max(grpo_advantages, 0) * (probability_ratio >= 1 - grpo_epsilon_low)
                )
                * grpo_loss_mask
            )
            # d(probability_ratio)/d(logits) = -probability_ratio * (predicted_probabilities - target_probabilities).
            predicted_probabilities = exp_logits / sum_exp_logits.unsqueeze(-1)
            grpo_grad = (probability_ratio_grad * probability_ratio).unsqueeze(
                -1
            ) * predicted_probabilities.scatter_add(
                -1,
                grpo_target_masked.unsqueeze(-1),
                -(grpo_loss_mask if grpo_target_mask is None else grpo_target_mask).unsqueeze(-1).to(torch.float32),
            )
            grad = grpo_grad if grad is None else grad + grpo_grad

    return cross_entropy_loss, z_loss, distribution_loss, grpo_loss, grpo_new_logprobs_mean, grpo_metrics, grad


@torch.compile
def _monolithic_core(
    logits: torch.Tensor,  # (*batch, vocab)
    group: ProcessGroup | None,
    logits_scale_factor: float,
    grad_logits: torch.Tensor | None,
    ce_target: torch.Tensor | None,
    ce_grad_output: float | None,
    ce_divisor: float,
    z_loss_enabled: bool,
    z_loss_mask: torch.Tensor | None,
    z_loss_grad_output: float | None,
    z_loss_divisor: float,
    distribution_target: torch.Tensor | None,
    distribution_grad_output: float | None,
    distribution_divisor: float,
    distribution_loss_mask: torch.Tensor | None,
    distribution_target_format: TargetFormat,
    distribution_entropy_loss_type: EntropyLossType,
    distribution_temperature: float,
    grpo_target: torch.Tensor | None,
    grpo_advantages: torch.Tensor | None,
    grpo_old_log_probabilities: torch.Tensor | None,
    grpo_grad_output: float | None,
    grpo_divisor: float,
    grpo_epsilon_low: float,
    grpo_epsilon_high: float,
    grpo_num_labels_in_seq: torch.Tensor | None,
    grpo_compute_metrics: bool,
    grpo_compute_entropy: bool,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    "GRPOMetrics | None",
    torch.Tensor | None,
]:
    """
    The single-pass monolithic head-loss kernel: one shared softmax over the logits, then every enabled
    loss's scalar and gradient contribution, fused in one `@torch.compile` boundary (it inlines the plain
    per-loss core). Disabled losses gate on their `None`/static flag and are dead-code-eliminated.
    Gradients accumulate in fp32 and are cast to `logits.dtype` once at the end. GSPO is *not* handled
    here — its eager segment seam requires the three-phase split (`_monolithic_gspo_forward_core`).
    """
    logits_norm, exp_logits, sum_exp_logits, logits_max = softmax_base(logits, logits_scale_factor, group)
    cross_entropy_loss, z_loss, distribution_loss, grpo_loss, grpo_new_logprobs_mean, grpo_metrics, grad = (
        _apply_combinable_losses(
            logits_norm,
            exp_logits,
            sum_exp_logits,
            logits_max,
            group,
            logits_scale_factor,
            ce_target,
            ce_grad_output,
            ce_divisor,
            z_loss_enabled,
            z_loss_mask,
            z_loss_grad_output,
            z_loss_divisor,
            distribution_target,
            distribution_grad_output,
            distribution_divisor,
            distribution_loss_mask,
            distribution_target_format,
            distribution_entropy_loss_type,
            distribution_temperature,
            grpo_target,
            grpo_advantages,
            grpo_old_log_probabilities,
            grpo_grad_output,
            grpo_divisor,
            grpo_epsilon_low,
            grpo_epsilon_high,
            grpo_num_labels_in_seq,
            grpo_compute_metrics,
            grpo_compute_entropy,
        )
    )

    if grad is not None:
        grad = grad.to(logits.dtype)
        if grad_logits is None:
            grad_logits = grad
        else:
            grad_logits.add_(grad)

    return cross_entropy_loss, z_loss, distribution_loss, grpo_loss, grpo_new_logprobs_mean, grpo_metrics, grad_logits


@torch.compile
def _monolithic_gspo_forward_core(
    logits: torch.Tensor,  # (*batch, vocab)
    group: ProcessGroup | None,
    logits_scale_factor: float,
    ce_target: torch.Tensor | None,
    ce_grad_output: float | None,
    ce_divisor: float,
    z_loss_enabled: bool,
    z_loss_mask: torch.Tensor | None,
    z_loss_grad_output: float | None,
    z_loss_divisor: float,
    distribution_target: torch.Tensor | None,
    distribution_grad_output: float | None,
    distribution_divisor: float,
    distribution_loss_mask: torch.Tensor | None,
    distribution_target_format: TargetFormat,
    distribution_entropy_loss_type: EntropyLossType,
    distribution_temperature: float,
    gspo_target: torch.Tensor,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    """
    GSPO-present forward: one shared softmax, every enabled non-GSPO loss's scalar + fp32 gradient
    (uncast), and GSPO's per-token `new_log_probs`. The softmax tensors and the uncast gradient cross the
    eager segment seam to `gspo_backward_core` (the `index_add_`/`num_segments` seam can't be compiled).
    """
    logits_norm, exp_logits, sum_exp_logits, logits_max = softmax_base(logits, logits_scale_factor, group)
    cross_entropy_loss, z_loss, distribution_loss, _, _, _, grad = _apply_combinable_losses(
        logits_norm,
        exp_logits,
        sum_exp_logits,
        logits_max,
        group,
        logits_scale_factor,
        ce_target,
        ce_grad_output,
        ce_divisor,
        z_loss_enabled,
        z_loss_mask,
        z_loss_grad_output,
        z_loss_divisor,
        distribution_target,
        distribution_grad_output,
        distribution_divisor,
        distribution_loss_mask,
        distribution_target_format,
        distribution_entropy_loss_type,
        distribution_temperature,
        None,
        None,
        None,
        None,
        1.0,
        0.2,
        0.2,
        None,
        False,
        False,
    )
    gspo_loss_mask = gspo_target >= 0
    predicted_logits, target_masked, target_mask = predicted_logits_from_labels(
        logits_norm, gspo_target, gspo_loss_mask, group
    )
    new_log_probs = predicted_logits - sum_exp_logits.log()
    return (
        cross_entropy_loss,
        z_loss,
        distribution_loss,
        grad,
        new_log_probs,
        exp_logits,
        sum_exp_logits,
        target_masked,
        target_mask,
    )


def gspo_segment_seam(
    new_log_probs: torch.Tensor,  # (*batch,)
    loss_mask: torch.Tensor,  # (*batch,) bool
    advantages: torch.Tensor,  # (*batch,)
    old_log_probabilities: torch.Tensor,  # (*batch,)
    document_index_zero_based: torch.Tensor,  # (*batch,) int
    num_segments: int,
    num_labels_in_seq: torch.Tensor,  # (*batch,)
    divisor: float,
    grad_output: float | None,
    sdp_group: ProcessGroup | None,
    sp_group: ProcessGroup | None,
    epsilon_low: float,
    epsilon_high: float,
    logits_scale_factor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Eager segment seam between the compiled forward and backward. The `index_add_` segment
    aggregation and the symbolic `num_segments` live here so they never enter a compiled boundary
    (no per-`num_segments` recompiles). Returns the loss, the `new_logprobs` metric, and the per-token
    backward coefficient `effective_grad = grad_output_scaled · clip_factor · loss_weight · R_s`
    (None when no gradient is requested)."""
    log_ratio = new_log_probs - old_log_probabilities

    flat_document_index = document_index_zero_based.reshape(-1).long()
    flat_mask = loss_mask.reshape(-1).to(log_ratio.dtype)
    # Per-token weight: mask / per-document label count, from the preprocessor.
    # Each labeled token contributes `1 / N_d` so all of doc d's tokens sum to 1 (across
    # SDP/SP ranks too), regardless of how the doc is sharded.
    mean_token_weight = flat_mask / num_labels_in_seq.reshape(-1).to(log_ratio.dtype).clamp(min=1)
    # Pre-divide the per-token contributions by the per-doc label count, then sum per segment.
    # All tokens in a segment share the same N_d, so this is mathematically equivalent to
    # `log_ratio_sum / N_d` but avoids any per-segment denominator extraction.
    mean_log_ratio_per_segment = log_ratio.new_zeros(num_segments).index_add_(
        0, flat_document_index, log_ratio.reshape(-1) * mean_token_weight
    )
    # Accumulate in `log_ratio.dtype` (fp32). Casting the product back to `advantages.dtype`
    # before summing would round each token's contribution to a possibly-low input dtype.
    mean_advantage_per_segment = log_ratio.new_zeros(num_segments).index_add_(
        0, flat_document_index, advantages.reshape(-1).to(log_ratio.dtype) * mean_token_weight
    )
    for reduce_group in (sdp_group, sp_group):
        if reduce_group is not None:
            torch.distributed.all_reduce(
                mean_log_ratio_per_segment, op=torch.distributed.ReduceOp.SUM, group=reduce_group
            )
            torch.distributed.all_reduce(
                mean_advantage_per_segment, op=torch.distributed.ReduceOp.SUM, group=reduce_group
            )

    segment_ratio = mean_log_ratio_per_segment.exp()  # (num_segments,) — geometric-mean IS ratio
    segment_advantage = mean_advantage_per_segment.detach()  # (num_segments,) — no grad through A

    probability_ratio = segment_ratio[flat_document_index].reshape(log_ratio.shape)
    advantage_per_token = segment_advantage[flat_document_index].reshape(log_ratio.shape)
    loss_weight = loss_mask.to(log_ratio.dtype)

    losses = -torch.min(
        probability_ratio * advantage_per_token,
        torch.clamp(probability_ratio, 1 - epsilon_low, 1 + epsilon_high) * advantage_per_token,
    )
    loss = (losses * loss_weight).sum() / divisor

    new_logprobs_mean = (new_log_probs * loss_mask / num_labels_in_seq.clamp(min=1)).sum()

    if grad_output is None:
        return loss, new_logprobs_mean, None

    grad_output_scaled = grad_output / divisor * logits_scale_factor
    probability_ratio_grad = (
        grad_output_scaled
        * (
            torch.clamp_min(advantage_per_token, 0) * (probability_ratio <= 1 + epsilon_high)
            + torch.clamp_max(advantage_per_token, 0) * (probability_ratio >= 1 - epsilon_low)
        )
        * loss_weight
    )
    effective_grad = probability_ratio_grad * probability_ratio
    return loss, new_logprobs_mean, effective_grad


@torch.compile
def gspo_backward_core(
    exp_logits: torch.Tensor,  # (*batch, vocab)
    sum_exp_logits: torch.Tensor,  # (*batch,)
    target_masked: torch.Tensor,  # (*batch,)
    target_mask: torch.Tensor | None,  # (*batch,) or None (no TP)
    loss_mask: torch.Tensor,  # (*batch,) bool
    effective_grad: torch.Tensor,  # (*batch,) — per-token backward coefficient from the seam
    logits_dtype: torch.dtype,
    grad_logits: torch.Tensor | None,
    grad_accumulator: torch.Tensor | None = None,  # fp32 grad of co-resident losses to combine before the cast
) -> torch.Tensor:
    """GSPO compiled backward: the per-token coefficient times the softmax chain rule, fused into one
    kernel. Any `grad_accumulator` (other losses' fp32 grad from the shared forward) is added before the
    single cast. `sum_exp_logits.unsqueeze` is out-of-place (the standalone eager kernel mutates it)."""
    predicted_probabilities = exp_logits / sum_exp_logits.unsqueeze(-1)
    grad = effective_grad.unsqueeze(-1) * predicted_probabilities.scatter_add(
        -1,
        target_masked.unsqueeze(-1),
        -(loss_mask if target_mask is None else target_mask).unsqueeze(-1).to(torch.float32),
    )
    if grad_accumulator is not None:
        grad = grad + grad_accumulator
    grad = grad.to(logits_dtype)
    if grad_logits is None:
        grad_logits = grad
    else:
        grad_logits.add_(grad)
    return grad_logits


def monolithic_head_loss_forward_backward(
    logits: torch.Tensor,
    specs: list[MonolithicLossSpec],
    *,
    group: ProcessGroup | None,
    grad_logits: torch.Tensor | None = None,
) -> tuple[list[MonolithicLossOutput], torch.Tensor | None]:
    """
    Marshal the per-loss specs into the flat arguments of the kernel, run it once over a single shared
    softmax, and map its outputs back to one `MonolithicLossOutput` per spec (in input order). When a
    GSPO spec is present the kernel splits into the three-phase path (compiled forward → eager segment
    seam → compiled backward); otherwise it is the single-pass `_monolithic_core`. All specs must share
    one effective `logits_scale_factor` (validated at config time and asserted here).
    """
    logits_scale_factor = specs[0].logits_scale_factor
    specs_by_kind: dict[str, MonolithicLossSpec] = {}
    for spec in specs:
        Assert.eq(spec.logits_scale_factor, logits_scale_factor)
        if spec.kind not in ("cross_entropy", "z_loss", "entropy_from_distribution", "grpo", "gspo"):
            raise NotImplementedError(spec.kind)
        assert spec.kind not in specs_by_kind, spec.kind
        specs_by_kind[spec.kind] = spec

    cross_entropy_spec = specs_by_kind.get("cross_entropy")
    z_loss_spec = specs_by_kind.get("z_loss")
    distribution_spec = specs_by_kind.get("entropy_from_distribution")
    grpo_spec = specs_by_kind.get("grpo")
    gspo_spec = specs_by_kind.get("gspo")
    # GRPO and GSPO are mutually exclusive RL objectives (both consume the same labels/advantages).
    assert grpo_spec is None or gspo_spec is None

    # Shared (non-GSPO) loss arguments.
    ce_target = None if cross_entropy_spec is None else cross_entropy_spec.target
    ce_grad_output = None if cross_entropy_spec is None else cross_entropy_spec.grad_output
    ce_divisor = 1.0 if cross_entropy_spec is None else cross_entropy_spec.divisor
    z_loss_mask = None if z_loss_spec is None else z_loss_spec.loss_mask
    z_loss_grad_output = None if z_loss_spec is None else z_loss_spec.grad_output
    z_loss_divisor = 1.0 if z_loss_spec is None else z_loss_spec.divisor
    distribution_target = None if distribution_spec is None else distribution_spec.target
    distribution_grad_output = None if distribution_spec is None else distribution_spec.grad_output
    distribution_divisor = 1.0 if distribution_spec is None else distribution_spec.divisor
    distribution_loss_mask = None if distribution_spec is None else distribution_spec.loss_mask
    distribution_target_format = TargetFormat.logits if distribution_spec is None else distribution_spec.target_format
    distribution_entropy_loss_type = (
        EntropyLossType.cross_entropy if distribution_spec is None else distribution_spec.entropy_loss_type
    )
    distribution_temperature = 1.0 if distribution_spec is None else distribution_spec.temperature

    grpo_new_logprobs_mean = None
    grpo_metrics = None
    gspo_loss = None
    gspo_new_logprobs_mean = None

    if gspo_spec is None:
        cross_entropy_loss, z_loss, distribution_loss, grpo_loss, grpo_new_logprobs_mean, grpo_metrics, grad_logits = (
            _monolithic_core(
                logits,
                group,
                logits_scale_factor,
                grad_logits,
                ce_target=ce_target,
                ce_grad_output=ce_grad_output,
                ce_divisor=ce_divisor,
                z_loss_enabled=z_loss_spec is not None,
                z_loss_mask=z_loss_mask,
                z_loss_grad_output=z_loss_grad_output,
                z_loss_divisor=z_loss_divisor,
                distribution_target=distribution_target,
                distribution_grad_output=distribution_grad_output,
                distribution_divisor=distribution_divisor,
                distribution_loss_mask=distribution_loss_mask,
                distribution_target_format=distribution_target_format,
                distribution_entropy_loss_type=distribution_entropy_loss_type,
                distribution_temperature=distribution_temperature,
                grpo_target=None if grpo_spec is None else grpo_spec.target,
                grpo_advantages=None if grpo_spec is None else grpo_spec.advantages,
                grpo_old_log_probabilities=None if grpo_spec is None else grpo_spec.old_log_probabilities,
                grpo_grad_output=None if grpo_spec is None else grpo_spec.grad_output,
                grpo_divisor=1.0 if grpo_spec is None else grpo_spec.divisor,
                grpo_epsilon_low=0.2 if grpo_spec is None else grpo_spec.epsilon_low,
                grpo_epsilon_high=0.2 if grpo_spec is None else grpo_spec.epsilon_high,
                grpo_num_labels_in_seq=None if grpo_spec is None else grpo_spec.num_labels_in_seq,
                grpo_compute_metrics=False if grpo_spec is None else grpo_spec.compute_metrics,
                grpo_compute_entropy=False if grpo_spec is None else grpo_spec.compute_entropy,
            )
        )
    else:
        grpo_loss = None
        (
            cross_entropy_loss,
            z_loss,
            distribution_loss,
            grad,
            gspo_new_log_probs,
            exp_logits,
            sum_exp_logits,
            target_masked,
            target_mask,
        ) = _monolithic_gspo_forward_core(
            logits,
            group,
            logits_scale_factor,
            ce_target,
            ce_grad_output,
            ce_divisor,
            z_loss_spec is not None,
            z_loss_mask,
            z_loss_grad_output,
            z_loss_divisor,
            distribution_target,
            distribution_grad_output,
            distribution_divisor,
            distribution_loss_mask,
            distribution_target_format,
            distribution_entropy_loss_type,
            distribution_temperature,
            gspo_spec.target,
        )
        gspo_loss_mask = gspo_spec.target >= 0
        gspo_loss, gspo_new_logprobs_mean, effective_grad = gspo_segment_seam(
            gspo_new_log_probs,
            gspo_loss_mask,
            gspo_spec.advantages,
            gspo_spec.old_log_probabilities,
            gspo_spec.document_index,
            gspo_spec.num_segments,
            gspo_spec.num_labels_in_seq,
            gspo_spec.divisor,
            gspo_spec.grad_output,
            gspo_spec.sdp_group,
            gspo_spec.sp_group,
            gspo_spec.epsilon_low,
            gspo_spec.epsilon_high,
            logits_scale_factor,
        )
        if effective_grad is not None:
            grad_logits = gspo_backward_core(
                exp_logits,
                sum_exp_logits,
                target_masked,
                target_mask,
                gspo_loss_mask,
                effective_grad,
                logits.dtype,
                grad_logits,
                grad_accumulator=grad,
            )
        elif grad is not None:
            # No GSPO gradient (e.g. eval), but co-resident losses produced one — finalize it here.
            grad = grad.to(logits.dtype)
            if grad_logits is None:
                grad_logits = grad
            else:
                grad_logits.add_(grad)

    loss_by_kind = {
        "cross_entropy": cross_entropy_loss,
        "z_loss": z_loss,
        "entropy_from_distribution": distribution_loss,
        "grpo": grpo_loss,
        "gspo": gspo_loss,
    }
    return [
        MonolithicLossOutput(
            loss=loss_by_kind[spec.kind],
            new_logprobs_mean=(
                grpo_new_logprobs_mean
                if spec.kind == "grpo"
                else gspo_new_logprobs_mean if spec.kind == "gspo" else None
            ),
            metrics=grpo_metrics if spec.kind == "grpo" else None,
        )
        for spec in specs
    ], grad_logits
