import typing

import torch

from fast_llm.core.distributed import ProcessGroup
from fast_llm.functional.entropy_loss import _predicted_logits_from_labels, _softmax_base
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
    # Cross-entropy (from labels).
    target: torch.Tensor | None = None
    loss_mask: torch.Tensor | None = None


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
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """
    The monolithic head-loss kernel: one shared softmax over the logits, then every enabled loss's scalar
    and gradient contribution. Disabled losses are toggled off by passing their inputs as `None`, so
    `torch.compile` dead-code-eliminates their branch. Because this is a single `@torch.compile` boundary
    that calls the plain (un-compiled) softmax cores, compile fuses the work across all enabled losses.

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

    if grad is not None:
        grad = grad.to(logits.dtype)
        if grad_logits is None:
            grad_logits = grad
        else:
            grad_logits.add_(grad)

    return cross_entropy_loss, grad_logits


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
    cross_entropy_spec = None
    for spec in specs:
        Assert.eq(spec.logits_scale_factor, logits_scale_factor)
        if spec.kind == "cross_entropy":
            assert cross_entropy_spec is None
            cross_entropy_spec = spec
        else:
            raise NotImplementedError(spec.kind)

    cross_entropy_loss, grad_logits = _monolithic_core(
        logits,
        group,
        logits_scale_factor,
        grad_logits,
        ce_target=None if cross_entropy_spec is None else cross_entropy_spec.target,
        ce_grad_output=None if cross_entropy_spec is None else cross_entropy_spec.grad_output,
        ce_divisor=1.0 if cross_entropy_spec is None else cross_entropy_spec.divisor,
    )

    outputs = []
    for spec in specs:
        if spec.kind == "cross_entropy":
            outputs.append(MonolithicLossOutput(loss=cross_entropy_loss))
    return outputs, grad_logits
