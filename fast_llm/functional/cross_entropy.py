import torch
import torch._dynamo  # noqa
import torch.autograd

from fast_llm.core.distributed import ProcessGroup, ReduceOp, all_reduce
from fast_llm.functional.config import CrossEntropyImpl, TargetFormat
from fast_llm.functional.triton.cross_entropy import triton_cross_entropy_forward_backward
from fast_llm.utils import Assert


def _torch_cross_entropy_forward_backward(
    logits: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor | None,
    grad_output: float | None,
    logits_scale_factor: float,
    target_format: TargetFormat,
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
        target = _fused_softmax(target, logits_scale_factor, group)

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
        loss = (per_sample_loss * loss_mask).sum() / torch.maximum(loss_mask.sum(), 1)
    else:
        loss = per_sample_loss.mean()
    if target_format != TargetFormat.labels and group is not None:
        all_reduce(loss, op=ReduceOp.MEAN, group=group)

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
            logits, target, loss_mask, grad_output, logits_scale_factor, target_format, group
        )
    else:
        return _CROSS_ENTROPY_IMPLEMENTATIONS[implementation](
            logits, target, loss_mask, grad_output, logits_scale_factor, target_format
        )
