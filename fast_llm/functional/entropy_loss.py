import torch

from fast_llm.core.distributed import ProcessGroup, ReduceOp, all_reduce
from fast_llm.functional.config import EntropyLossType, TargetFormat
from fast_llm.utils import Assert


@torch.compile
def torch_entropy_loss_forward_backward(
    logits: torch.Tensor,  # (*batch, vocab)
    target: torch.Tensor,  # (*batch,) or (*batch, vocab)
    loss_mask: torch.Tensor | None,  # (*batch,)
    grad_output: float | None,
    logits_scale_factor: float,
    target_format: TargetFormat,
    entropy_loss_type: EntropyLossType,
    group: ProcessGroup | None = None,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:  # (), (*batch, vocab)
    """
    A wrapper for the pytorch implementation of cross-entropy.
    The cross-entropy kernels themselves are well-optimized, but the need for explicit casting
    and separate forward and backward kernels lead to poor performance.
    """
    assert group is None
    # Torch methods require flattened batch dimension.
    target = target.flatten() if target_format == TargetFormat.labels else target.flatten(0, -2)
    if target_format == TargetFormat.labels:
        assert loss_mask is None
        loss_mask = target >= 0
    else:
        target = target.float()
        if loss_mask is not None:
            loss_mask = loss_mask.flatten()

    # Torch compile doesn't understand this.
    with torch.set_grad_enabled(grad_output is not None):
        logits_ = logits.float().detach().requires_grad_(grad_output is not None)

        logits_scaled = (logits_ if logits_scale_factor == 1.0 else logits_ * logits_scale_factor).flatten(0, -2)
        if target_format == TargetFormat.logits:
            target_scale = logits_scale_factor / temperature
            target = target if target_scale == 1.0 else target * target_scale
        else:
            Assert.eq(temperature, 1.0)

        if entropy_loss_type == EntropyLossType.cross_entropy:
            if target_format == TargetFormat.logits:
                target = torch.softmax(target, dim=-1)
            per_sample_loss = torch.nn.functional.cross_entropy(logits_scaled, target, reduction="none")
        else:
            predicted_log_probability = torch.nn.functional.log_softmax(logits_scaled, dim=-1)
            if target_format == TargetFormat.logits:
                target_log_probability = torch.nn.functional.log_softmax(target, dim=-1)
            elif target_format == TargetFormat.probabilities:
                target_log_probability = target.log()
            else:
                target_probability = torch.nn.functional.one_hot(
                    torch.clamp_min(target, 0), num_classes=logits_scaled.size(-1)
                )
                if loss_mask is not None:
                    target_probability = target_probability * loss_mask.unsqueeze(-1)
                target_log_probability = target_probability.add(1.0e-10).log()
            if entropy_loss_type == EntropyLossType.forward_kl:
                per_sample_loss = torch.nn.functional.kl_div(
                    predicted_log_probability,
                    target_log_probability,
                    reduction="none",
                    log_target=True,
                )
            elif entropy_loss_type == EntropyLossType.reverse_kl:
                per_sample_loss = torch.nn.functional.kl_div(
                    target_log_probability,
                    predicted_log_probability,
                    reduction="none",
                    log_target=True,
                )
            else:
                raise NotImplementedError(entropy_loss_type)
            per_sample_loss = per_sample_loss.sum(dim=-1)

        if loss_mask is not None:
            per_sample_loss = per_sample_loss * loss_mask
        loss = per_sample_loss.mean()

        if grad_output is None:
            grad = None
        else:
            loss.backward(torch.full_like(loss, grad_output))
            grad = logits_.grad.detach().to(logits.dtype)
    return loss.detach_(), grad


def softmax_base(
    logits: torch.Tensor,  # (*batch, vocab)
    logits_scale_factor: float = 1.0,
    group: ProcessGroup | None = None,
    dim: int = -1,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:  # (*batch, vocab), (*batch, vocab), (*batch,), (*batch,)
    """
    Calculate the required inputs for softmax computation, mainly sum_exp_logits,
    in a numerically stable way and with tensor-parallel support.
    Warning: The returned values are regularized by `logits_max`.
        The regularization typically but not always cancels out in derived quantities.

    Un-compiled so it can be inlined into a `@torch.compile` boundary that fuses several losses over a
    single softmax; `fused_softmax_base` is the compiled standalone wrapper.
    """
    logits = logits.float()
    if logits_scale_factor != 1.0:
        logits = logits * logits_scale_factor
    logits_max = logits.max(dim=dim)[0]
    if group is not None:
        all_reduce(logits_max, op=ReduceOp.MAX, group=group)
    logits_norm = (logits - logits_max.unsqueeze(-1)).float()
    exp_logits = logits_norm.exp()
    sum_exp_logits = exp_logits.sum(dim=dim)
    if group is not None:
        all_reduce(sum_exp_logits, op=ReduceOp.SUM, group=group)
    return logits_norm, exp_logits, sum_exp_logits, logits_max


fused_softmax_base = torch.compile(softmax_base)


def reverse_kl_from_distribution_core(
    logits_norm: torch.Tensor,  # (*batch, vocab)
    exp_logits: torch.Tensor,  # (*batch, vocab)
    sum_exp_logits: torch.Tensor,  # (*batch,)
    target: torch.Tensor,  # (*batch, vocab)
    grad_output: float | None,
    logits_scale_factor: float,
    target_format: TargetFormat,
    group: ProcessGroup | None = None,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:  # (*batch,), (*batch, vocab)
    """Reverse-KL from a precomputed student softmax (adding a teacher softmax when the target is logits).
    Un-compiled core, inlined into a `@torch.compile` boundary."""
    assert target_format in (TargetFormat.logits, TargetFormat.probabilities)
    predicted_log_probability = logits_norm - sum_exp_logits.log().unsqueeze(-1)
    predicted_probability = exp_logits / sum_exp_logits.unsqueeze(-1)

    if target_format == TargetFormat.logits:
        target_logits_norm, _, sum_exp_target_logits, _ = softmax_base(
            target, logits_scale_factor / temperature, group
        )
        target_log_probability = target_logits_norm - sum_exp_target_logits.log().unsqueeze(-1)
    else:
        target_log_probability = torch.log(target)

    # Compute loss terms: student_probs * log_ratio, then sum over vocab
    # This is equivalent to kl_div(..., log_target=True) but more memory efficient
    log_ratio = predicted_log_probability - target_log_probability
    per_sample_loss = (predicted_probability * log_ratio).sum(dim=-1)
    if group is not None:
        all_reduce(per_sample_loss, op=ReduceOp.SUM, group=group)

    if grad_output is None:
        grad = None
    else:
        # Gradient: d/d(logits) KL(q||p) = q * (log(q/p) - E_q[log(q/p)])
        # where E_q[log(q/p)] is the expected log ratio under the student distribution
        grad = (log_ratio - per_sample_loss.unsqueeze(-1)) * predicted_probability * grad_output

    return per_sample_loss, grad


def cross_entropy_from_distribution_core(
    logits_norm: torch.Tensor,  # (*batch, vocab)
    exp_logits: torch.Tensor,  # (*batch, vocab)
    sum_exp_logits: torch.Tensor,  # (*batch,)
    target: torch.Tensor,  # (*batch, vocab)
    grad_output: float | None,
    logits_scale_factor: float,
    target_format: TargetFormat,
    group: ProcessGroup | None = None,
    temperature: float = 1.0,
    return_kl_loss: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:  # (*batch,), (*batch, vocab)
    """Cross-entropy / forward-KL from a precomputed student softmax (adding a teacher softmax when the
    target is logits). Un-compiled core, inlined into a `@torch.compile` boundary."""
    if target_format == TargetFormat.logits:
        target_logits_norm, exp_logits_targets, sum_exp_target_logits, _ = softmax_base(
            target, logits_scale_factor / temperature, group
        )
        target = exp_logits_targets / sum_exp_target_logits.unsqueeze(-1)

    # CE loss = mean(log(sum_exp_logits) - sum(probabilities * logits))
    # KL loss = mean(log(sum_exp_logits) - sum(probabilities * (logits - log_probabilities))
    if return_kl_loss:
        if target_format == TargetFormat.logits:
            target_log_probability = target_logits_norm
        else:
            target_log_probability = torch.log(target)
        logits_norm = logits_norm - target_log_probability
    predicted_logits = (target * logits_norm).sum(dim=-1)
    if group is not None:
        # We need to sum the over the tensor-parallel group,
        # but this is handled in the final averaging provided we multiply by the group size.
        all_reduce(predicted_logits, op=ReduceOp.SUM, group=group)

    per_sample_loss = sum_exp_logits.log() - predicted_logits
    if return_kl_loss and target_format == TargetFormat.logits:
        per_sample_loss = per_sample_loss - sum_exp_target_logits.log()

    if grad_output is None:
        grad = None
    else:
        # grad / grad_output = exp_logits / sum_exp_logits - target_probabilities.
        grad = (exp_logits - sum_exp_logits.unsqueeze(-1) * target) * (grad_output / sum_exp_logits.unsqueeze(-1))

    return per_sample_loss, grad


def predicted_logits_from_labels(
    logits: torch.Tensor,  # (*batch, vocab)
    target: torch.Tensor,  # (*batch,)
    loss_mask: torch.Tensor,  # (*batch,), == target>=0
    group: ProcessGroup | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # (*batch,), (*batch,), (*batch,)
    """
    Recover the value of the logits at the target index, with support for masking (target < 0) and tensor parallelism.
    In the simple case, equivalent to `logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)`

    May also recover probabilities or log probabilities:
    `predicted_probabilities = predicted_logits.exp() / sum_exp_logits`
    `predicted_log_probabilities = predicted_logits / sum_exp_logits.log()`

    Un-compiled core, inlined into a `@torch.compile` boundary; `fused_predicted_logits_from_labels` is the
    compiled standalone wrapper.
    """

    if group is None:
        # Keep values within range for scatter and gather ops to work.
        target_masked = target * loss_mask
        target_mask = None
    else:
        # Mask the target (fused).
        # TODO: Could mask earlier on cpu or overlap with reduce?
        vocab_start_index = logits.size(-1) * group.rank()
        target_mask = (target >= vocab_start_index) * (target < vocab_start_index + logits.size(-1))
        target_masked = (target - vocab_start_index) * target_mask

    # CE loss = mean(log(sum_exp_logits) - sum(probabilities * logits))
    # KL loss is the same because P * log(P) == 0.
    predicted_logits = logits.gather(-1, target_masked.unsqueeze(-1)).squeeze(-1)
    if group is not None:
        predicted_logits = target_mask * predicted_logits
        all_reduce(predicted_logits, op=ReduceOp.SUM, group=group)
    return predicted_logits, target_masked, target_mask


fused_predicted_logits_from_labels = torch.compile(predicted_logits_from_labels)


def cross_entropy_from_labels_core(
    logits_norm: torch.Tensor,  # (*batch, vocab), == logits - max
    exp_logits: torch.Tensor,  # (*batch, vocab)
    sum_exp_logits: torch.Tensor,  # (*batch,)
    target: torch.Tensor,  # (*batch,)
    loss_mask: torch.Tensor,  # (*batch,), == target>=0
    grad_output: float | None,  # already normalized: raw_grad_output / divisor * logits_scale_factor
    group: ProcessGroup | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:  # (*batch,) unmasked, (*batch, vocab) unmasked
    """
    Cross-entropy from labels, taking the already-computed shared softmax tensors. Returns the unmasked
    per-sample loss and (when `grad_output` is given) the unmasked gradient; the caller applies the loss
    mask, reduction, and dtype cast. Un-compiled core, inlined into a `@torch.compile` boundary.
    """
    predicted_logits, target_masked, target_mask = predicted_logits_from_labels(logits_norm, target, loss_mask, group)

    # CE loss = mean(log(sum_exp_logits) - sum(probabilities * logits))
    # KL loss is the same because P * log(P) == 0.
    per_sample_loss = sum_exp_logits.log() - predicted_logits

    if grad_output is None:
        grad = None
    else:
        # grad / grad_output = exp_logits / sum_exp_logits - target_probabilities.
        grad = exp_logits.scatter_add(
            -1,
            target_masked.unsqueeze(-1),
            -sum_exp_logits.unsqueeze(-1) if target_mask is None else -(target_mask * sum_exp_logits).unsqueeze(-1),
        ) * (grad_output / sum_exp_logits.unsqueeze(-1))

    return per_sample_loss, grad


def z_loss_core(
    exp_logits: torch.Tensor,  # (*batch, vocab)
    sum_exp_logits: torch.Tensor,  # (*batch,)
    logits_max: torch.Tensor,  # (*batch,)
    grad_output: float | None,  # already normalized: raw_grad_output / divisor * logits_scale_factor
) -> tuple[torch.Tensor, torch.Tensor | None]:  # (*batch,) unmasked, (*batch, vocab) unmasked
    """
    Z-loss from the already-computed shared softmax tensors. Returns the unmasked per-sample loss term
    (`log_sum_exp ** 2`) and (when `grad_output` is given) the unmasked gradient; the caller applies the
    loss mask, reduction, and dtype cast. z-loss needs the un-regularized log-sum-exp, so it adds back
    `logits_max` (cross-entropy cancels it). Un-compiled core, inlined into a `@torch.compile` boundary.
    """
    log_sum_exp_logits = sum_exp_logits.log() + logits_max
    loss_term = log_sum_exp_logits**2
    if grad_output is None:
        grad = None
    else:
        grad = (2 * grad_output * (log_sum_exp_logits / sum_exp_logits)).unsqueeze(-1) * exp_logits
    return loss_term, grad
