import torch

from fast_llm.core.distributed import ProcessGroup, ReduceOp, all_reduce
from fast_llm.functional.config import CrossEntropyImpl, ReverseKLImpl, TargetFormat
from fast_llm.functional.triton.cross_entropy import triton_cross_entropy_forward_backward
from fast_llm.utils import Assert


def _torch_cross_entropy_forward_backward(
    logits: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor | None,
    grad_output: float | None,
    logits_scale_factor: float,
    target_format: TargetFormat,
    teacher_softmax_temperature: float = 1.0,
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
        if target_format == TargetFormat.logits:
            if logits_scale_factor != 1.0:
                target = target * logits_scale_factor
            if teacher_softmax_temperature != 1.0:
                target = target / teacher_softmax_temperature
            target = torch.softmax(target, dim=-1)
        if loss_mask is None:
            loss = torch.nn.functional.cross_entropy(
                logits_ if logits_scale_factor == 1 else logits_ * logits_scale_factor, target
            )
        else:
            loss = (
                torch.nn.functional.cross_entropy(
                    logits_ if logits_scale_factor == 1 else logits_ * logits_scale_factor, target, reduction="none"
                )
                * loss_mask
            ).mean()
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
        logits *= logits_scale_factor
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
def _fused_softmax(
    logits: torch.Tensor, logits_scale_factor: float = 1.0, group: ProcessGroup = None, dim: int = -1
) -> torch.Tensor:
    _, exp_logits, sum_exp_logits = _fused_softmax_base(logits, logits_scale_factor, group, dim)
    return exp_logits / sum_exp_logits


# @torch.compile
def _fused_cross_entropy_forward_backward(
    logits: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor | None,
    grad_output: float | None,
    logits_scale_factor: float,
    target_format: TargetFormat,
    group: ProcessGroup | None = None,
    teacher_softmax_temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    A fused implementation of cross-entropy with torch compile.
    It is an improvement over the pytorch implementation because of the fused casting, both in speed and memory,
    but still suboptimal because it needs multiple kernels.
    """
    # Do the forward and backward passes all at once, and fused with dtype conversion.
    # Way faster and more memory-efficient than the pytorch version.

    logits_norm, exp_logits, sum_exp_logits = _fused_softmax_base(logits, logits_scale_factor, group)

    if target_format == TargetFormat.logits:
        target = _fused_softmax(target, logits_scale_factor / teacher_softmax_temperature, group)

    if target_format == TargetFormat.labels:
        target = target.unsqueeze(-1)
        loss_mask = target >= 0
        if group is None:
            # Keep values within range for scatter and gather ops to work.
            target = target * loss_mask
            target_mask = None
        else:
            # Mask the target (fused)
            # TODO: Could mask earlier on cpu or overlap with reduce?
            vocab_start_index = logits.size(-1) * group.rank()
            target_mask = (target >= vocab_start_index) * (target < vocab_start_index + logits.size(-1))
            target = (target - vocab_start_index) * target_mask
    else:
        # Target should be tensor-parallel already, no further manipulation needed.
        target_mask = None
        if loss_mask is not None:
            loss_mask = loss_mask.unsqueeze(-1)

    if grad_output is None:
        grad = None
    else:
        # grad / grad_output = exp_logits / sum_exp_logits - target_probabilities.
        if target_format == TargetFormat.labels:
            grad_base = exp_logits.scatter_add(
                1, target, -sum_exp_logits if target_mask is None else -(target_mask * sum_exp_logits)
            )
        else:
            grad_base = exp_logits - sum_exp_logits * target

        grad = grad_base.mul((grad_output / logits.size(0)) / sum_exp_logits)
        if logits_scale_factor != 1.0:
            grad *= logits_scale_factor
        if loss_mask is not None:
            grad *= loss_mask
        grad = grad.to(logits.dtype)

    # loss = mean(log(sum_exp_logits) - sum(probabilities * logits))
    if target_format == TargetFormat.labels:
        predicted_logits = logits_norm.gather(1, target)
        if group is not None:
            predicted_logits = target_mask * predicted_logits
            all_reduce(predicted_logits, op=ReduceOp.SUM, group=group)
    else:
        predicted_logits = (target * logits_norm).sum(dim=-1, keepdim=True)

    # shouldn't the predicted_logits be scaled by the number of ranks so that the average loss is correct? i.e.
    # i.e. on each rank we calculate log Z - sum_i t_i * z_i, z_i is logit.
    # then we average: 1/K sum_ranks (log Z - sum_i t_i * z_i)
    # = log Z - 1/K sum_ranks (sum_i t_i * z_i)
    # but sum_ranks (sum_i t_i * z_i) = sum_i t_i * z_i (over all vocab), so we need to divide predicted_logits by K to match?

    per_sample_loss = sum_exp_logits.log() - predicted_logits
    if loss_mask is not None:
        per_sample_loss = per_sample_loss * loss_mask

    loss = per_sample_loss.mean()
    if target_format != TargetFormat.labels and group is not None:
        all_reduce(loss, op=ReduceOp.AVG, group=group)

    return loss, grad


_CROSS_ENTROPY_IMPLEMENTATIONS = {
    CrossEntropyImpl.torch: _torch_cross_entropy_forward_backward,
    CrossEntropyImpl.fused: _fused_cross_entropy_forward_backward,
    CrossEntropyImpl.triton: triton_cross_entropy_forward_backward,
}


def cross_entropy_forward_backward(
    logits: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor | None,
    grad_output: float | None,
    group: ProcessGroup | None = None,
    implementation: CrossEntropyImpl = CrossEntropyImpl.fused,
    logits_scale_factor: float = 1.0,
    teacher_softmax_temperature: float = 1.0,
    target_format: TargetFormat = TargetFormat.labels,
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
        Assert.eq(implementation, CrossEntropyImpl.fused)
        return _fused_cross_entropy_forward_backward(
            logits,
            target,
            loss_mask,
            grad_output,
            logits_scale_factor,
            target_format,
            group,
            teacher_softmax_temperature,
        )
    else:
        return _CROSS_ENTROPY_IMPLEMENTATIONS[implementation](
            logits,
            target,
            loss_mask,
            grad_output,
            logits_scale_factor,
            target_format,
            teacher_softmax_temperature=teacher_softmax_temperature,
        )


def distributed_log_softmax(
    logits: torch.Tensor, logits_scale_factor: float = 1.0, group: ProcessGroup | None = None, dim: int = -1
):
    logits_norm, _, sum_exp_logits = _fused_softmax_base(logits, logits_scale_factor, group=group, dim=dim)

    return logits_norm - sum_exp_logits.log()  # log_softmax


def _torch_reverse_kl_forward_backward(
    logits: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor | None,
    grad_output: float | None,
    target_format: TargetFormat,
    group: ProcessGroup | None = None,
    logits_scale_factor: float = 1.0,
    teacher_softmax_temperature: float = 1.0,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Reverse KL using PyTorch's native kl_div function.
    This is used for TP version where we split accross vocab dimantion. KL is additive over partitions of the vocab.

    Takes:
        logits: [BxS, V] or [B, S, V]
        target: [BxS, V] or [B, S, V] (logits format)
        loss_mask: [BxS] or [B, S] or None
        ...
    """
    Assert.eq(
        teacher_softmax_temperature,
        1,
        msg="Teacher softmax temperature must be 1 for sequence-tensor-parallel reverse KL",
    )
    Assert.eq(logits_scale_factor, 1, msg="Logits scale factor must be 1 for sequence-tensor-parallel reverse KL")
    Assert.eq(target.shape, logits.shape)
    assert target.dtype.is_floating_point, target.dtype
    if loss_mask is not None:
        Assert.eq(loss_mask.shape, logits.shape[:-1])

    # Compute log probabilities
    teacher_log_probs = distributed_log_softmax(target.float(), group=group)
    # batch_size = logits.shape[0]
    with torch.enable_grad():
        logits_ = logits.float().detach().requires_grad_(grad_output is not None)
        student_log_probs = distributed_log_softmax(logits_, group=group)

        # Reverse KL: input=teacher_log_probs, target=student_probs
        loss_terms = torch.nn.functional.kl_div(
            teacher_log_probs,  # input = log(p)
            student_log_probs,  # target = log(q)
            reduction="none",
            log_target=True,
        ).sum(dim=-1)
        if loss_mask is not None:
            # loss mask is the same on all ranks for TP over vocab.
            valid = loss_mask.to(loss_terms.dtype)
            loss_terms = loss_terms * valid
            valid_tokens = torch.tensor(valid.sum(), device=loss_terms.device, dtype=loss_terms.dtype)
        else:
            valid_tokens = torch.prod(torch.tensor(loss_terms.shape, device=loss_terms.device, dtype=loss_terms.dtype))
        loss = loss_terms.sum()  # sums over batch and seq. len.

        if group is not None:
            all_reduce(loss, op=ReduceOp.SUM, group=group)
        loss /= valid_tokens

        if grad_output is not None:
            loss.backward(torch.full_like(loss, grad_output))
            grad = logits_.grad.to(logits.dtype)
        else:
            grad = None

    return loss.detach_(), grad


REVERSE_KL_IMPLEMENTATIONS = {
    ReverseKLImpl.no_tp: _torch_reverse_kl_forward_backward,
    ReverseKLImpl.tp: _torch_reverse_kl_forward_backward,
}


def reverse_kl_forward_backward(
    logits: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor | None,
    grad_output: float | None,
    group: ProcessGroup | None = None,
    logits_scale_factor: float = 1.0,
    teacher_softmax_temperature: float = 1.0,
    target_format: TargetFormat = TargetFormat.labels,
    sequence_parallel_logits: bool = False,
    group_size: int = None,
    vocab_size: int = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Compute reverse KL divergence: KL(q||p) where q is the predicted distribution (student) and p is the target (teacher).
    This is mode-seeking (vs. mode-covering for forward KL) and useful for:
    - Encouraging the model to focus on the modes of the target distribution
    - Avoiding probability mass on low-probability regions of the target
    - Distillation scenarios where you want sharp, focused predictions

    Key differences from standard cross-entropy:
    - Standard CE: KL(p||q) = mode-covering (spreads mass broadly)
    - Reverse KL: KL(q||p) = mode-seeking (focuses on target modes)

    Takes:
        logits: [BxS, V] or [B, S, V], where V is local vocab size
        target: [BxS, V] or [B, S, V] (logits format)
        loss_mask: [BxS] or [B, S] or None
        ...

    Returns:
        loss: Reverse KL divergence loss
        grad: Gradients w.r.t. logits
    """

    if logits.shape[-1] != vocab_size:
        reverse_kl_impl = ReverseKLImpl.tp
    elif sequence_parallel_logits:
        # TODO: see hybrid dev branch where it is implemented
        raise NotImplementedError("Sequence-parallel reverse KL is not implemented yet, set vocab_parallel true")
    else:
        reverse_kl_impl = ReverseKLImpl.no_tp

    Assert.eq(target_format, TargetFormat.logits, msg="Reverse KL only supports logits format")
    Assert.eq(target.shape, logits.shape)
    assert target.dtype.is_floating_point, target.dtype
    if loss_mask is not None:
        Assert.eq(loss_mask.shape, logits.shape[:-1])

    # TODO: implement fused?
    distillation_loss, distillation_grad = REVERSE_KL_IMPLEMENTATIONS[reverse_kl_impl](
        logits=logits,
        target=target,
        loss_mask=loss_mask,
        grad_output=grad_output,
        logits_scale_factor=logits_scale_factor,
        target_format=target_format,
        teacher_softmax_temperature=teacher_softmax_temperature,
        group=group,
    )
    return distillation_loss, distillation_grad
