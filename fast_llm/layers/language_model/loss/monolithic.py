import typing

import torch

from fast_llm.core.distributed import ProcessGroup
from fast_llm.functional.config import EntropyLossType, TargetFormat
from fast_llm.functional.entropy_loss import (
    _cross_entropy_from_distribution_core,
    _predicted_logits_from_labels,
    _reverse_kl_from_distribution_core,
    _softmax_base,
)
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
    target: torch.Tensor | None = None  # Cross-entropy labels, or a teacher distribution (logits/probabilities).
    loss_mask: torch.Tensor | None = None  # z-loss / distribution-loss mask (cross-entropy derives its own).
    # Distribution losses (cross-entropy / forward-KL / reverse-KL from a teacher distribution) only.
    target_format: TargetFormat = TargetFormat.logits
    entropy_loss_type: EntropyLossType = EntropyLossType.cross_entropy
    temperature: float = 1.0


class MonolithicLossOutput(typing.NamedTuple):
    """Per-loss outputs returned by the monolithic kernel, registered via `register_monolithic_outputs`."""

    loss: torch.Tensor | None = None


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
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """
    The monolithic head-loss kernel: one shared softmax over the logits, then every enabled loss's scalar
    and gradient contribution. Losses with a required tensor input are toggled off by passing it as `None`
    (e.g. cross-entropy's `ce_target`); losses without one (z-loss) toggle on a static `*_enabled` bool. In
    both cases `torch.compile` specializes on the flag and dead-code-eliminates the disabled branch. Because
    this is a single `@torch.compile` boundary that calls the plain (un-compiled) softmax cores, compile
    fuses the work across all enabled losses.

    Gradients accumulate in fp32 and are cast to `logits.dtype` once at the end.
    """
    logits_norm, exp_logits, sum_exp_logits, logits_max = _softmax_base(logits, logits_scale_factor, group)
    grad = None

    cross_entropy_loss = None
    if ce_target is not None:
        loss_mask = ce_target >= 0
        predicted_logits, target_masked, target_mask = _predicted_logits_from_labels(
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
            per_sample_loss, distribution_grad = _reverse_kl_from_distribution_core(
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
            per_sample_loss, distribution_grad = _cross_entropy_from_distribution_core(
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

    if grad is not None:
        grad = grad.to(logits.dtype)
        if grad_logits is None:
            grad_logits = grad
        else:
            grad_logits.add_(grad)

    return cross_entropy_loss, z_loss, distribution_loss, grad_logits


def monolithic_head_loss_forward_backward(
    logits: torch.Tensor,
    specs: list[MonolithicLossSpec],
    *,
    group: ProcessGroup | None,
    grad_logits: torch.Tensor | None = None,
) -> tuple[list[MonolithicLossOutput], torch.Tensor | None]:
    """
    Marshal the per-loss specs into the flat arguments of `_monolithic_core`, run it once, and map its
    outputs back to one `MonolithicLossOutput` per spec (in input order). All specs must share a single
    effective `logits_scale_factor` (the shared softmax cannot serve two scales); this is validated at
    config time and asserted here.
    """
    logits_scale_factor = specs[0].logits_scale_factor
    specs_by_kind: dict[str, MonolithicLossSpec] = {}
    for spec in specs:
        Assert.eq(spec.logits_scale_factor, logits_scale_factor)
        if spec.kind not in ("cross_entropy", "z_loss", "entropy_from_distribution"):
            raise NotImplementedError(spec.kind)
        assert spec.kind not in specs_by_kind, spec.kind
        specs_by_kind[spec.kind] = spec

    cross_entropy_spec = specs_by_kind.get("cross_entropy")
    z_loss_spec = specs_by_kind.get("z_loss")
    distribution_spec = specs_by_kind.get("entropy_from_distribution")

    cross_entropy_loss, z_loss, distribution_loss, grad_logits = _monolithic_core(
        logits,
        group,
        logits_scale_factor,
        grad_logits,
        ce_target=None if cross_entropy_spec is None else cross_entropy_spec.target,
        ce_grad_output=None if cross_entropy_spec is None else cross_entropy_spec.grad_output,
        ce_divisor=1.0 if cross_entropy_spec is None else cross_entropy_spec.divisor,
        z_loss_enabled=z_loss_spec is not None,
        z_loss_mask=None if z_loss_spec is None else z_loss_spec.loss_mask,
        z_loss_grad_output=None if z_loss_spec is None else z_loss_spec.grad_output,
        z_loss_divisor=1.0 if z_loss_spec is None else z_loss_spec.divisor,
        distribution_target=None if distribution_spec is None else distribution_spec.target,
        distribution_grad_output=None if distribution_spec is None else distribution_spec.grad_output,
        distribution_divisor=1.0 if distribution_spec is None else distribution_spec.divisor,
        distribution_loss_mask=None if distribution_spec is None else distribution_spec.loss_mask,
        distribution_target_format=(
            TargetFormat.logits if distribution_spec is None else distribution_spec.target_format
        ),
        distribution_entropy_loss_type=(
            EntropyLossType.cross_entropy if distribution_spec is None else distribution_spec.entropy_loss_type
        ),
        distribution_temperature=1.0 if distribution_spec is None else distribution_spec.temperature,
    )

    loss_by_kind = {
        "cross_entropy": cross_entropy_loss,
        "z_loss": z_loss,
        "entropy_from_distribution": distribution_loss,
    }
    return [MonolithicLossOutput(loss=loss_by_kind[spec.kind]) for spec in specs], grad_logits
