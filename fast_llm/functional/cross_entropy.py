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


def distributed_log_softmax(logits: torch.Tensor, group: ProcessGroup, dim: int = -1):
    logits = logits.float()
    local_max = logits.max(dim=dim, keepdim=True)[0]
    all_reduce(local_max, op=ReduceOp.MAX, group=group)

    logits_shifted = logits - local_max
    exp_logits = torch.exp(logits_shifted)
    sum_exp = exp_logits.sum(dim=dim, keepdim=True)
    all_reduce(sum_exp, op=ReduceOp.SUM, group=group)

    return logits_shifted - sum_exp.log()  # log_softmax


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

    per_sample_loss = sum_exp_logits.log() - predicted_logits
    if loss_mask is not None:
        per_sample_loss = per_sample_loss[loss_mask]

    loss = per_sample_loss.mean()
    if target_format != TargetFormat.labels and group is not None:
        all_reduce(loss, op=ReduceOp.SUM, group=group)
        loss /= group.size()

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


def _torch_reverse_kl_forward_backward_vocab_parallel(
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
    This is used for TP version where we split accross vocab dimantion.
    This works with sequence-tensor-parallel (distributing over the sequence dimention) as well as a non-TP case.
    In sequence-tensor-parallel, where we split along sequence dim., we compute per split loss and then average the loss.
    """
    Assert.eq(
        teacher_softmax_temperature,
        1,
        msg="Teacher softmax temperature must be 1 for sequence-tensor-parallel reverse KL",
    )
    Assert.eq(logits_scale_factor, 1, msg="Logits scale factor must be 1 for sequence-tensor-parallel reverse KL")
    # TODO: merge into single function _torch_reverse_kl_forward_backward
    Assert.eq(target_format, TargetFormat.logits, msg="Reverse KL only supports logits format")
    Assert.eq(target.shape, logits.shape)
    assert target.dtype.is_floating_point, target.dtype
    if loss_mask is not None:
        Assert.eq(loss_mask.shape, logits.shape[:-1])

    # Compute log probabilities - let _fused_softmax handle scaling internally
    teacher_log_probs = distributed_log_softmax(target, group=group)
    batch_size = logits.shape[0]
    with torch.enable_grad():
        logits_ = logits.detach().requires_grad_(grad_output is not None)
        student_log_probs = distributed_log_softmax(logits_, group=group)

        # Reverse KL: input=teacher_log_probs, target=student_probs
        if loss_mask is None:
            loss = torch.nn.functional.kl_div(
                teacher_log_probs,  # input = log(p)
                student_log_probs,  # target = log(q)
                reduction="sum",
                log_target=True,
            )
        else:
            # Apply loss mask - this requires some reshaping
            raise NotImplementedError("Loss mask not implemented with TP for reverse KL , it must be doublechecked")
            loss_per_sample = torch.nn.functional.kl_div(
                teacher_log_probs, student_log_probs, reduction="none", log_target=True
            ).sum(dim=-1)
            loss = (loss_per_sample * loss_mask).sum()

        if group is not None and target_format != TargetFormat.labels:
            all_reduce(loss, op=ReduceOp.SUM, group=group)
            loss /= batch_size

        if grad_output is not None:
            loss.backward(torch.full_like(loss, grad_output))
            grad = logits_.grad.to(logits.dtype)
        else:
            grad = None

    return loss.detach_(), grad


def _torch_reverse_kl_forward_backward_no_tp(
    logits: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor | None,
    grad_output: float | None,
    logits_scale_factor: float,
    target_format: TargetFormat,
    teacher_softmax_temperature: float = 1.0,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Reverse KL using PyTorch's native kl_div function.
    THis is only used for no-TP case.
    """
    Assert.eq(target_format, TargetFormat.logits, msg="Reverse KL only supports logits format")
    Assert.eq(target.shape, logits.shape)
    assert target.dtype.is_floating_point, target.dtype
    if loss_mask is not None:
        Assert.eq(loss_mask.shape, logits.shape[:-1])
    # Scale target logits more carefully
    scaled_target = target * (logits_scale_factor / teacher_softmax_temperature)
    # Clamp to prevent extreme values that cause NaNs in log_softmax
    scaled_target = torch.clamp(scaled_target, min=-100.0, max=100.0)

    teacher_log_probs = torch.log_softmax(scaled_target, dim=-1)

    # For reverse KL: KL(q||p) = Σ q * log(q/p) = Σ q * (log(q) - log(p))
    # Use kl_div with: input=log(p), target=q, log_target=False
    # This gives: Σ q * (log(q) - log(p)) = exactly what we want!

    with torch.enable_grad():
        logits_ = logits.detach().requires_grad_(grad_output is not None)

        scaled_logits = logits_ * logits_scale_factor
        # Clamp to prevent extreme values that cause NaNs in log_softmax
        scaled_logits = torch.clamp(scaled_logits, min=-100.0, max=100.0)
        student_log_probs = torch.log_softmax(scaled_logits, dim=-1)

        # Reverse KL: input=teacher_log_probs, target=student_probs
        if loss_mask is None:
            loss = torch.nn.functional.kl_div(
                teacher_log_probs,  # input = log(p)
                student_log_probs,  # target = log(q)
                reduction="batchmean",
                log_target=True,
            )
        else:
            # Apply loss mask - this requires some reshaping
            loss_per_sample = torch.nn.functional.kl_div(
                teacher_log_probs, student_log_probs, reduction="none", log_target=True
            ).sum(dim=-1)
            loss = (loss_per_sample * loss_mask).mean()

        if grad_output is not None:
            # note, we never get here in TP over seq. dim.
            loss.backward(torch.full_like(loss, grad_output))
            grad = logits_.grad.to(logits.dtype)
        else:
            grad = None

    return loss.detach_(), grad


def _torch_reverse_kl_forward_backward_sequence_tensor_parallel(
    logits: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor | None,
    grad_output: float | None,
    logits_scale_factor: float,
    target_format: TargetFormat,
    teacher_softmax_temperature: float = 1.0,
    total_valid_tokens: int | None = None,  # total number of unmasked tokens in the batch
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Reverse KL using PyTorch's native kl_div function.
    THis is only used for sequence-tensor-parallel case where we split over sequence dimension.
    """
    Assert.eq(
        total_valid_tokens is not None,
        msg="Total valid tokens must be provided for sequence-tensor-parallel reverse KL",
    )
    Assert.eq(
        teacher_softmax_temperature,
        1,
        msg="Teacher softmax temperature must be 1 for sequence-tensor-parallel reverse KL",
    )
    Assert.eq(logits_scale_factor, 1, msg="Logits scale factor must be 1 for sequence-tensor-parallel reverse KL")
    Assert.eq(target_format, TargetFormat.logits, msg="Reverse KL only supports logits format")
    Assert.eq(target.shape, logits.shape)
    assert target.dtype.is_floating_point, target.dtype
    if loss_mask is not None:
        Assert.eq(loss_mask.shape, logits.shape[:-1])
    # Scale target logits more carefully
    scaled_target = target * (logits_scale_factor / teacher_softmax_temperature)
    # Clamp to prevent extreme values that cause NaNs in log_softmax
    scaled_target = torch.clamp(scaled_target, min=-100.0, max=100.0)

    teacher_log_probs = torch.log_softmax(scaled_target, dim=-1)

    # For reverse KL: KL(q||p) = Σ q * log(q/p) = Σ q * (log(q) - log(p))
    # Use kl_div with: input=log(p), target=q, log_target=False
    # This gives: Σ q * (log(q) - log(p)) = exactly what we want!

    with torch.enable_grad():
        logits_ = logits.detach().requires_grad_(grad_output is not None)

        scaled_logits = logits_ * logits_scale_factor
        # Clamp to prevent extreme values that cause NaNs in log_softmax
        scaled_logits = torch.clamp(scaled_logits, min=-100.0, max=100.0)
        student_log_probs = torch.log_softmax(scaled_logits, dim=-1)

        # Reverse KL: input=teacher_log_probs, target=student_probs
        if loss_mask is None:
            loss = torch.nn.functional.kl_div(
                teacher_log_probs,  # input = log(p)
                student_log_probs,  # target = log(q)
                reduction="sum",
                log_target=True,
            )
        else:
            # Apply loss mask - this requires some reshaping
            loss_per_sample = torch.nn.functional.kl_div(
                teacher_log_probs, student_log_probs, reduction="none", log_target=True
            ).sum(dim=-1)
            loss = (loss_per_sample * loss_mask).sum()  # this can be 0.0 if all tokens are masked

        if grad_output is not None:
            # note, if we compute gradient w.r.t sum of losses,
            # and grad_output should reflect the scaling by 1/valid samples
            loss.backward(torch.full_like(loss, grad_output))
            grad = logits_.grad.to(logits.dtype)
        else:
            grad = None

    return loss.detach_(), grad


REVERSE_KL_IMPLEMENTATIONS = {
    ReverseKLImpl.no_tp: _torch_reverse_kl_forward_backward_no_tp,
    ReverseKLImpl.tp: _torch_reverse_kl_forward_backward_vocab_parallel,
    ReverseKLImpl.stp: _torch_reverse_kl_forward_backward_sequence_tensor_parallel,
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
    reverse_kl_impl: ReverseKLImpl = ReverseKLImpl.no_tp,
    total_valid_tokens: int | None = None,
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

    Args:
        logits: Model predictions [batch_size, ..., vocab_size]
        target: Target distribution or labels
        loss_mask: Optional mask for loss computation
        grad_output: Gradient output scale factor
        group: Process group for tensor parallelism
        logits_scale_factor: Temperature scaling factor (1/T)
        target_format: Format of target (labels or logits)

    Returns:
        loss: Reverse KL divergence loss
        grad: Gradients w.r.t. logits

    Example usage:
        # Replace standard cross-entropy with reverse KL
        # loss, grad = cross_entropy_forward_backward(logits, target, ...)
        loss, grad = reverse_kl_forward_backward(logits, target,
                                               loss_mask=None,
                                               grad_output=1.0,
                                               logits_scale_factor=1.0,
                                               target_format=TargetFormat.labels)
    """
    if target_format == TargetFormat.labels:
        Assert.eq(target.shape, logits.shape[:-1])
        Assert.eq(target.dtype, torch.int64)
    else:
        Assert.eq(target.shape, logits.shape)
        assert target.dtype.is_floating_point, target.dtype
        if loss_mask is not None:
            Assert.eq(loss_mask.shape, logits.shape[:-1])
    # TODO: implement fused reverse KL?
    return REVERSE_KL_IMPLEMENTATIONS[reverse_kl_impl](
        logits=logits,
        target=target,
        loss_mask=loss_mask,
        grad_output=grad_output,
        logits_scale_factor=logits_scale_factor,
        target_format=target_format,
        teacher_softmax_temperature=teacher_softmax_temperature,
        group=group,
        total_valid_tokens=total_valid_tokens,
    )
