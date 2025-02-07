import torch
import triton
import triton.language as tl

from fast_llm.functional.config import TritonConfig


@triton.jit
def triton_cross_entropy_forward_backward_kernel(
    logits_ptr,
    labels_ptr,
    grad_logits_ptr,
    losses_ptr,
    grad_losses,
    n_cols,
    logits_stride_0,
    grad_logits_stride_0,
    logits_scale_factor: tl.constexpr,
    block_size: tl.constexpr,
):
    # TODO: Int64 ptr only if needed?
    block_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, block_size)
    logits_ptr = logits_ptr + block_idx * logits_stride_0
    mask = col_offsets < n_cols

    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)
    if logits_scale_factor != 1.0:
        logits *= logits_scale_factor

    max_logits = tl.max(logits, 0)
    exp_logits = tl.exp(logits - max_logits)
    sum_exp_logits = tl.sum(exp_logits, 0)

    label_idx = tl.load(labels_ptr + block_idx)

    label_logits = tl.load(logits_ptr + label_idx).to(tl.float32)
    if label_idx < 0:
        loss = 0.0
    else:
        loss = tl.log(sum_exp_logits) + max_logits - label_logits
    tl.store(losses_ptr + block_idx, loss)

    grad_logits_ptr = grad_logits_ptr + block_idx * grad_logits_stride_0
    col_offsets = tl.arange(0, block_size)
    label_idx = tl.load(labels_ptr + block_idx)
    exp_logits = exp_logits / sum_exp_logits
    if logits_scale_factor != 1.0:
        exp_logits *= logits_scale_factor
    if label_idx < 0:
        grad_losses = 0.0
    grad_logits = grad_losses * tl.where(col_offsets == label_idx, exp_logits - 1.0, exp_logits)
    tl.store(grad_logits_ptr + col_offsets, grad_logits, mask=mask)


def triton_cross_entropy_forward_backward(
    logits, target, grad_output: float | None, logits_scale_factor: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    A fast triton implementation of cross-entropy, which combines the casting and forward and backward passes,
    all in a single kernel.
     Compared to a standard pytorch implementation, this reduces memory usage (of logits) by 3x and memory I/O by 5x.
    TODO: Better handling of `grad_output = None`
    """
    assert TritonConfig.TRITON_ENABLED
    # TODO: Improve assumptions.
    assert logits.is_contiguous()
    assert target.is_contiguous()
    n_rows, n_cols = logits.shape
    assert target.shape == (n_rows,)
    block_size = triton.next_power_of_2(n_cols)
    assert block_size <= TritonConfig.MAX_BLOCK_SIZE_BYTES
    num_warps = 4 if block_size < 2048 else (8 if block_size < 8192 else 16)
    losses = torch.empty(n_rows, dtype=torch.float, device=logits.device)
    # TODO: Safe to do inplace?
    grad_logits = torch.empty_like(logits)
    triton_cross_entropy_forward_backward_kernel[(n_rows,)](
        logits,
        target,
        grad_logits,
        losses,
        1 if grad_output is None else grad_output / n_rows,
        n_cols,
        logits.stride(0),
        grad_logits.stride(0),
        logits_scale_factor,
        block_size=block_size,
        num_warps=num_warps,
    )
    return losses.mean(), None if grad_output is None else grad_logits
