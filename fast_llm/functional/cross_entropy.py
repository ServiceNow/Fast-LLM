import torch
import torch._dynamo  # noqa
import torch.autograd

from fast_llm.core.distributed import ProcessGroup, ReduceOp, all_reduce
from fast_llm.functional.config import CrossEntropyImpl
from fast_llm.functional.triton.cross_entropy import triton_cross_entropy_forward_backward
from fast_llm.utils import Assert


def torch_cross_entropy_forward_backward(logits, target, grad_output):
    """
    A wrapper for the pytorch implementation of cross-entropy.
    The cross-entropy kernels themselves are well-optimized, but the need for explicit casting
    and separate forward and backward kernels lead to poor performance.
    """
    # Torch compile doesn't understand this.
    with torch.enable_grad():
        logits_ = logits.float().detach().requires_grad_()
        loss = torch.nn.functional.cross_entropy(logits_, target).mean()
        loss.backward(torch.full_like(loss, grad_output))
    return loss.detach(), logits_.grad.detach().to(logits.dtype)


@torch.compile
def fused_cross_entropy_forward_backward(logits, target, grad_output: float | None):
    """
    A fused implementation of cross-entropy with torch compile.
    It is an improvement over the pytorch implementation because of the fused casting, both in speed and memory,
    but still suboptimal because it needs multiple kernels.
    """
    # Do the forward and backward passes all at once, and fused with dtype conversion.
    # Way faster and more memory-efficient than the pytorch version.
    target = target.unsqueeze(1)
    logits = logits.sub(torch.max(logits, dim=-1)[0].unsqueeze(dim=-1))
    exp_logits = logits.float().exp()
    sum_exp_logits = exp_logits.sum(dim=-1)

    if grad_output is None:
        grad = None
    else:
        exp_logits = exp_logits.scatter(1, target, exp_logits.gather(1, target) - sum_exp_logits.unsqueeze(dim=-1))
        # exp_logits[torch.arange(0, logits.size(0), device=logits.device), target.squeeze(dim=-1)]-=sum_exp_logits
        exp_logits = exp_logits.mul((grad_output / logits.size(0)) / sum_exp_logits.unsqueeze(dim=-1))
        grad = exp_logits.to(logits.dtype)

    loss = sum_exp_logits.log().sub(logits.gather(1, target).squeeze(1)).mean()

    return loss, grad


@torch.compile
def parallel_cross_entropy_forward_backward(logits, target, grad_output: float | None, group: ProcessGroup):
    """
    A fused implementation of cross-entropy with torch compile, with support for tensor parallelism.
    Comes with a noticeable overhead, but reduces memory usage.
    """
    # TODO: Compiled version incorrect for some inputs (32 bit indexing issue?).
    # TODO: Optimize, overlap/combine reductions
    target = target.unsqueeze(1)

    logits_max = torch.max(logits, dim=-1)[0]
    all_reduce(logits_max, op=ReduceOp.MAX, group=group)
    logits_norm = logits.sub(logits_max.unsqueeze(dim=-1))

    exp_logits = logits_norm.float().exp()
    sum_exp_logits = exp_logits.sum(dim=-1)
    all_reduce(sum_exp_logits, op=ReduceOp.SUM, group=group)

    # Mask the target (fused)
    # TODO: Could mask earlier on cpu or overlap with reduce?
    vocab_start_index = logits_norm.size(-1) * group.rank()
    target_mask = (target >= vocab_start_index) * (target < vocab_start_index + logits_norm.size(-1))
    target = (target - vocab_start_index) * target_mask

    if grad_output is None:
        grad = None
    else:
        exp_logits1 = exp_logits.scatter(
            1, target, exp_logits.gather(1, target) - target_mask * sum_exp_logits.unsqueeze(dim=-1)
        )
        exp_logits2 = exp_logits1.mul((grad_output / logits_norm.size(0)) / sum_exp_logits.unsqueeze(dim=-1))
        grad = exp_logits2.to(logits_norm.dtype)

    predicted_logits = (target_mask * logits_norm.gather(1, target)).squeeze(1)
    all_reduce(predicted_logits, op=ReduceOp.SUM, group=group)
    loss = sum_exp_logits.log().sub(predicted_logits).mean()

    return loss, grad


_CROSS_ENTROPY_IMPLEMENTATIONS = {
    CrossEntropyImpl.torch: torch_cross_entropy_forward_backward,
    CrossEntropyImpl.fused: fused_cross_entropy_forward_backward,
    CrossEntropyImpl.triton: triton_cross_entropy_forward_backward,
}


def cross_entropy_forward_backward(
    logits,
    target,
    grad_output: float | None,
    group: ProcessGroup | None,
    implementation: CrossEntropyImpl = CrossEntropyImpl.fused,
):
    """
    Select the appropriate implementation of cross-entropy.
    The triton implementation from the triton submodule is the fastest and recommended one.
    It doesn't have a tensor-parallel implementation, but can be computed in a sequence-tensor-parallel way,
    which is faster and has a relatively small memory overhead.
    """
    if group:
        Assert.eq(implementation, CrossEntropyImpl.fused)
        return parallel_cross_entropy_forward_backward(logits, target, grad_output, group)
    else:
        return _CROSS_ENTROPY_IMPLEMENTATIONS[implementation](logits, target, grad_output)
