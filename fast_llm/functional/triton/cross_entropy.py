import torch

from fast_llm.functional.config import TargetFormat, TritonConfig
from fast_llm.functional.triton import tl, tl_constexpr, triton, triton_jit


@triton_jit()
def triton_cross_entropy_forward_backward_kernel(
    logits_ptr,
    labels_ptr,
    grad_logits_ptr,
    losses_ptr,
    grad_losses,
    n_cols: tl_constexpr,
    logits_stride_0: tl_constexpr,
    grad_logits_stride_0: tl_constexpr,
    logits_scale_factor: tl_constexpr,
    block_size: tl_constexpr,
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

    if label_idx < 0:
        # Loss mask
        loss = 0.0
    else:
        label_logits = tl.load(logits_ptr + label_idx).to(tl.float32)
        if logits_scale_factor != 1.0:
            label_logits *= logits_scale_factor
        loss = tl.log(sum_exp_logits) + max_logits - label_logits
    tl.store(losses_ptr + block_idx, loss)

    if grad_losses is not None:
        if label_idx < 0:
            grad_losses = 0.0
        grad_base = exp_logits / sum_exp_logits
        grad_logits = grad_losses * tl.where(col_offsets == label_idx, grad_base - 1.0, grad_base)
        if logits_scale_factor != 1.0:
            grad_logits *= logits_scale_factor
        tl.store(grad_logits_ptr + block_idx * grad_logits_stride_0 + col_offsets, grad_logits, mask=mask)


@triton_jit()
def triton_cross_entropy_from_distribution_forward_backward_kernel(
    logits_ptr,
    target_ptr,
    loss_mask_ptr,
    grad_logits_ptr,
    losses_ptr,
    grad_losses,
    n_cols: tl_constexpr,
    logits_stride_0: tl_constexpr,
    target_stride_0: tl_constexpr,
    grad_logits_stride_0: tl_constexpr,
    logits_scale_factor: tl_constexpr,
    from_logits: tl_constexpr,
    block_size: tl_constexpr,
):
    # TODO: Int64 ptr only if needed?
    block_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, block_size)
    mask = col_offsets < n_cols

    if loss_mask_ptr is not None:
        loss_mask = tl.load(loss_mask_ptr + block_idx)
        if loss_mask == 0:
            tl.store(losses_ptr + block_idx, 0)
            if grad_losses is not None:
                tl.store(grad_logits_ptr + block_idx * grad_logits_stride_0 + col_offsets, 0, mask=mask)
            return

    logits = tl.load(logits_ptr + block_idx * logits_stride_0 + col_offsets, mask=mask, other=-float("inf")).to(
        tl.float32
    )
    if logits_scale_factor != 1.0:
        logits *= logits_scale_factor

    max_logits = tl.max(logits, 0)
    logits_norm = logits - max_logits
    exp_logits = tl.exp(logits_norm)
    sum_exp_logits = tl.sum(exp_logits, 0)

    target = tl.load(target_ptr + block_idx * target_stride_0 + col_offsets, mask=mask, other=-float("inf")).to(
        tl.float32
    )
    if from_logits:
        if logits_scale_factor != 1.0:
            target *= logits_scale_factor
        max_target_logits = tl.max(target, 0)
        exp_target_logits = tl.exp(target - max_target_logits)
        sum_exp_target_logits = tl.sum(exp_target_logits, 0)
        target = exp_target_logits / sum_exp_target_logits

    # per_sample_loss = log(sum_exp_logits) - sum(probabilities * logits)
    loss = tl.log(sum_exp_logits) - tl.sum(tl.where(mask, target * logits_norm, 0), 0)
    tl.store(losses_ptr + block_idx, loss)

    if grad_losses is not None:
        # grad / grad_output = exp_logits / sum_exp_logits - target_probabilities.
        grad_logits = grad_losses * (exp_logits / sum_exp_logits - target)
        if logits_scale_factor != 1.0:
            grad_logits *= logits_scale_factor
        if loss_mask_ptr is not None:
            grad_logits = grad_logits
        tl.store(grad_logits_ptr + block_idx * grad_logits_stride_0 + col_offsets, grad_logits, mask=mask)


def triton_cross_entropy_forward_backward(
    logits: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor | None,
    grad_output: float | None,
    logits_scale_factor: float,
    target_format: TargetFormat,
    teacher_softmax_temperature: float = 1.0,
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
    block_size = triton.next_power_of_2(n_cols)
    assert block_size <= TritonConfig.MAX_BLOCK_SIZE_BYTES
    num_warps = 4 if block_size < 2048 else (8 if block_size < 8192 else 16)
    losses = torch.empty(n_rows, dtype=torch.float, device=logits.device)
    # TODO: Safe to do inplace?
    grad_logits = None if grad_output is None else torch.empty_like(logits)
    if target_format == TargetFormat.labels:
        triton_cross_entropy_forward_backward_kernel[(n_rows,)](
            logits,
            target,
            grad_logits,
            losses,
            None if grad_output is None else grad_output / n_rows,
            n_cols,
            logits.stride(0),
            None if grad_output is None else grad_logits.stride(0),
            logits_scale_factor,
            block_size=block_size,
            num_warps=num_warps,
        )
    else:
        if loss_mask is not None:
            assert loss_mask.is_contiguous()
        triton_cross_entropy_from_distribution_forward_backward_kernel[(n_rows,)](
            logits,
            target / teacher_softmax_temperature,
            loss_mask,
            grad_logits,
            losses,
            None if grad_output is None else grad_output / n_rows,
            n_cols,
            logits.stride(0),
            target.stride(0),
            None if grad_output is None else grad_logits.stride(0),
            logits_scale_factor,
            block_size=block_size,
            num_warps=num_warps,
            from_logits=target_format == TargetFormat.logits,
        )
    return losses.mean(), grad_logits
