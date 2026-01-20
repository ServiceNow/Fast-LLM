import torch

from fast_llm.core.distributed import ProcessGroup, ReduceOp, all_reduce
from fast_llm.functional.config import EntropyLossImplementation, EntropyLossType, TargetFormat
from fast_llm.functional.triton.cross_entropy import triton_cross_entropy_forward_backward
from fast_llm.utils import Assert


def _torch_entropy_loss_forward_backward(
    logits: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor | None,
    grad_output: float | None,
    logits_scale_factor: float,
    target_format: TargetFormat,
    entropy_loss_type: EntropyLossType,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    A wrapper for the pytorch implementation of cross-entropy.
    The cross-entropy kernels themselves are well-optimized, but the need for explicit casting
    and separate forward and backward kernels lead to poor performance.
    TODO: loss masking only works for with labels format and if the masking index is set to -100.
    """
    # Torch compile doesn't understand this.
    with torch.set_grad_enabled(grad_output is not None):
        logits_ = logits.float().detach().requires_grad_(grad_output is not None)
        logits_scaled = logits_ if logits_scale_factor == 1.0 else logits_ * logits_scale_factor
        if target_format == TargetFormat.logits:
            target_scale = logits_scale_factor / temperature
            target = target if target_scale == 1.0 else target * target_scale
        else:
            Assert.eq(temperature, 1.0)

        if entropy_loss_type == EntropyLossType.cross_entropy:
            if target_format == TargetFormat.logits:
                target = torch.softmax(target, dim=-1)
            loss = torch.nn.functional.cross_entropy(
                logits_scaled, target, reduction="mean" if loss_mask is None else "none"
            )
        else:
            predicted_log_probability = torch.nn.functional.log_softmax(logits_scaled, dim=-1)
            if target_format == TargetFormat.logits:
                target_log_probability = torch.nn.functional.log_softmax(target, dim=-1)
            elif target_format == TargetFormat.probabilities:
                target_log_probability = target.log()
            else:
                target_log_probability = (
                    torch.nn.functional.one_hot(target, num_classes=logits_scaled.size(-1)).add(1.0e-10).log()
                )
            if entropy_loss_type == EntropyLossType.forward_kl:
                loss = torch.nn.functional.kl_div(
                    predicted_log_probability,
                    target_log_probability,
                    reduction="batchmean" if loss_mask is None else "none",
                    log_target=True,
                )
            elif entropy_loss_type == EntropyLossType.reverse_kl:
                loss = torch.nn.functional.kl_div(
                    target_log_probability,
                    predicted_log_probability,
                    reduction="batchmean" if loss_mask is None else "none",
                    log_target=True,
                )
            else:
                raise NotImplementedError(entropy_loss_type)
            if loss_mask is not None:
                loss = loss.sum(dim=-1)

        if loss_mask is not None:
            loss = (loss * loss_mask).mean()

        if grad_output is None:
            grad = None
        else:
            loss.backward(torch.full_like(loss, grad_output))
            grad = logits_.grad.detach().to(logits.dtype)
    return loss.detach_(), grad


@torch.compile
def _fused_softmax_base(
    logits: torch.Tensor, logits_scale_factor: float = 1.0, group: ProcessGroup | None = None, dim: int = -1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = logits.float()
    if logits_scale_factor != 1.0:
        logits = logits * logits_scale_factor
    logits_max = torch.max(logits, dim=dim, keepdim=True)[0]
    if group is not None:
        all_reduce(logits_max, op=ReduceOp.MAX, group=group)
    logits_norm = (logits - logits_max).float()
    exp_logits = logits_norm.exp()
    sum_exp_logits = exp_logits.sum(dim=dim, keepdim=True)
    if group is not None:
        all_reduce(sum_exp_logits, op=ReduceOp.SUM, group=group)
    return logits_norm, exp_logits, sum_exp_logits


@torch.compile
def _fused_reverse_kl_base(
    logits: torch.Tensor,
    target: torch.Tensor,
    grad_output: float | None,
    logits_scale_factor: float,
    target_format: TargetFormat,
    group: ProcessGroup | None = None,
    temperature: float = 1.0,
):
    logits_norm, exp_logits, sum_exp_logits = _fused_softmax_base(logits, logits_scale_factor, group)
    predicted_log_probability = logits_norm - sum_exp_logits.log()
    predicted_probability = exp_logits / sum_exp_logits

    if target_format == TargetFormat.logits:
        target_logits_norm, _, sum_exp_target_logits = _fused_softmax_base(
            target, logits_scale_factor / temperature, group
        )
        target_log_probability = target_logits_norm - sum_exp_target_logits.log()
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


@torch.compile
def _fused_cross_entropy_base(
    logits: torch.Tensor,
    target: torch.Tensor,
    grad_output: float | None,
    logits_scale_factor: float,
    target_format: TargetFormat,
    group: ProcessGroup | None = None,
    temperature: float = 1.0,
    return_kl_loss: bool = False,
):
    logits_norm, exp_logits, sum_exp_logits = _fused_softmax_base(logits, logits_scale_factor, group)

    if target_format == TargetFormat.logits:
        target_logits_norm, exp_logits_targets, sum_exp_target_logits = _fused_softmax_base(
            target, logits_scale_factor / temperature, group
        )
        target = exp_logits_targets / sum_exp_target_logits

    # CE loss = mean(log(sum_exp_logits) - sum(probabilities * logits))
    # KL loss = mean(log(sum_exp_logits) - sum(probabilities * (logits - log_probabilities))
    if return_kl_loss:
        if target_format == TargetFormat.logits:
            target_log_probability = target_logits_norm - sum_exp_target_logits.log()
        else:
            target_log_probability = torch.log(target)
        logits_norm = logits_norm - target_log_probability
    predicted_logits = (target * logits_norm).sum(dim=-1, keepdim=True)
    if group is not None:
        # We need to sum the over the tensor-parallel group,
        # but this is handled in the final averaging provided we multiply by the group size.
        all_reduce(predicted_logits, op=ReduceOp.SUM, group=group)

    per_sample_loss = sum_exp_logits.log() - predicted_logits

    if grad_output is None:
        grad = None
    else:
        # grad / grad_output = exp_logits / sum_exp_logits - target_probabilities.
        grad = (exp_logits - sum_exp_logits * target) * (grad_output / sum_exp_logits)

    return per_sample_loss, grad


@torch.compile
def _fused_cross_entropy_base_from_labels(
    logits: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
    grad_output: float | None,
    logits_scale_factor: float,
    group: ProcessGroup | None = None,
):
    logits_norm, exp_logits, sum_exp_logits = _fused_softmax_base(logits, logits_scale_factor, group)

    target = target.unsqueeze(-1)

    if group is None:
        # Keep values within range for scatter and gather ops to work.
        target = target * loss_mask.unsqueeze(-1)
        target_mask = None
    else:
        # Mask the target (fused)
        # TODO: Could mask earlier on cpu or overlap with reduce?
        vocab_start_index = logits.size(-1) * group.rank()
        target_mask = (target >= vocab_start_index) * (target < vocab_start_index + logits.size(-1))
        target = (target - vocab_start_index) * target_mask

    # CE loss = mean(log(sum_exp_logits) - sum(probabilities * logits))
    # KL loss is the same because P * log(P) == 0.
    predicted_logits = logits_norm.gather(1, target)
    if group is not None:
        predicted_logits = target_mask * predicted_logits
        all_reduce(predicted_logits, op=ReduceOp.SUM, group=group)
    per_sample_loss = sum_exp_logits.log() - predicted_logits

    if grad_output is None:
        grad = None
    else:
        # grad / grad_output = exp_logits / sum_exp_logits - target_probabilities.
        grad = exp_logits.scatter_add(
            1, target, -sum_exp_logits if target_mask is None else -(target_mask * sum_exp_logits)
        ) * (grad_output / sum_exp_logits)

    return per_sample_loss, grad


@torch.compile
def _fused_entropy_loss_forward_backward(
    logits: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor | None,
    grad_output: float | None,
    logits_scale_factor: float,
    target_format: TargetFormat,
    entropy_loss_type: EntropyLossType,
    group: ProcessGroup | None = None,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    A fused implementation of cross-entropy with torch compile.
    It is an improvement over the pytorch implementation because of the fused casting, both in speed and memory,
    but still suboptimal because it needs multiple kernels.
    """
    grad_output = None if grad_output is None else grad_output / logits.size(0) * logits_scale_factor
    if target_format == TargetFormat.labels:
        assert entropy_loss_type in (EntropyLossType.cross_entropy, EntropyLossType.forward_kl)
        if loss_mask is None:
            loss_mask = target >= 0
        per_sample_loss, grad = _fused_cross_entropy_base_from_labels(
            logits,
            target,
            loss_mask,
            grad_output,
            logits_scale_factor,
            group,
        )
    elif entropy_loss_type in (EntropyLossType.cross_entropy, EntropyLossType.forward_kl):
        per_sample_loss, grad = _fused_cross_entropy_base(
            logits,
            target,
            grad_output,
            logits_scale_factor,
            target_format,
            group,
            temperature,
            return_kl_loss=entropy_loss_type == EntropyLossType.forward_kl,
        )
    elif entropy_loss_type == EntropyLossType.reverse_kl:
        per_sample_loss, grad = _fused_reverse_kl_base(
            logits,
            target,
            grad_output,
            logits_scale_factor,
            target_format,
            group,
            temperature,
        )
    else:
        raise NotImplementedError(entropy_loss_type)

    if loss_mask is not None:
        per_sample_loss = per_sample_loss * loss_mask.unsqueeze(-1)
    loss = per_sample_loss.mean()

    if grad is not None:
        if loss_mask is not None:
            grad = grad * loss_mask.unsqueeze(-1)
        grad = grad.to(logits.dtype)

    return loss, grad


_ENTROPY_LOSS_IMPLEMENTATIONS = {
    EntropyLossImplementation.torch: _torch_entropy_loss_forward_backward,
    EntropyLossImplementation.fused: _fused_entropy_loss_forward_backward,
    EntropyLossImplementation.triton: triton_cross_entropy_forward_backward,
}


def entropy_loss_forward_backward(
    logits: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor | None,
    grad_output: float | None,
    group: ProcessGroup | None = None,
    implementation: EntropyLossImplementation = EntropyLossImplementation.fused,
    logits_scale_factor: float = 1.0,
    temperature: float = 1.0,
    target_format: TargetFormat = TargetFormat.labels,
    entropy_loss_type: EntropyLossType = EntropyLossType.cross_entropy,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Select the appropriate implementation of cross-entropy.
    The triton implementation from the triton submodule is the fastest and recommended one.
    It doesn't have a tensor-parallel implementation, but can be computed in a sequence-tensor-parallel way,
    which is faster and has a relatively small memory overhead.
    """
    if target_format == TargetFormat.labels:
        Assert.eq(target.shape, logits.shape[:-1])
        Assert.eq(target.dtype, torch.int64)
        assert loss_mask is None
    else:
        Assert.eq(target.shape, logits.shape)
        assert target.dtype.is_floating_point, target.dtype
        if loss_mask is not None:
            Assert.eq(loss_mask.shape, logits.shape[:-1])
    if group:
        Assert.eq(implementation, EntropyLossImplementation.fused)
        return _fused_entropy_loss_forward_backward(
            logits,
            target,
            loss_mask,
            grad_output,
            logits_scale_factor,
            target_format,
            entropy_loss_type,
            group,
            temperature,
        )
    else:
        return _ENTROPY_LOSS_IMPLEMENTATIONS[implementation](
            logits,
            target,
            loss_mask,
            grad_output,
            logits_scale_factor,
            target_format,
            entropy_loss_type,
            temperature=temperature,
        )
