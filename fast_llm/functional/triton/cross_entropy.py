import torch

from fast_llm.functional.config import EntropyLossType, TargetFormat, TritonConfig
from fast_llm.functional.triton import tl, tl_constexpr, triton, triton_jit
from fast_llm.utils import Assert


@triton_jit()
def triton_fused_softmax_base(
    logits_ptr,
    n_cols: tl_constexpr,
    logits_scale_factor: tl_constexpr,
    block_size: tl_constexpr,
):
    for col_offset in tl.static_range(0, n_cols, block_size):
        col_offsets = tl.arange(col_offset, col_offset + block_size)
        mask = col_offsets < n_cols
        logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)
        if logits_scale_factor != 1.0:
            logits *= logits_scale_factor

        if col_offset == 0:
            max_logits = tl.max(logits, 0)
            exp_logits = tl.exp(logits - max_logits)
            sum_exp_logits = tl.sum(exp_logits, 0)
        else:
            new_max_logits = tl.maximum(tl.max(logits, 0), max_logits)
            exp_logits = tl.exp(logits - new_max_logits)
            sum_exp_logits = tl.sum(exp_logits, 0) + sum_exp_logits * tl.exp(max_logits - new_max_logits)
            max_logits = new_max_logits
    return exp_logits, sum_exp_logits, max_logits, mask


@triton_jit()
def triton_cross_entropy_forward_parallel_kernel(
    logits_ptr,
    labels_ptr,
    max_logits_ptr,
    sum_exp_logits_ptr,
    predicted_logits_ptr,
    col_min: tl_constexpr,
    n_cols: tl_constexpr,
    logits_stride_0: tl_constexpr,
    logits_scale_factor: tl_constexpr,
    block_size: tl_constexpr,
):
    # TODO: Int64 ptr only if needed?
    block_idx = tl.program_id(0).to(tl.int64)
    logits_ptr = logits_ptr + block_idx * logits_stride_0

    exp_logits, sum_exp_logits, max_logits, mask = triton_fused_softmax_base(
        logits_ptr, n_cols, logits_scale_factor, block_size
    )

    if labels_ptr is not None and predicted_logits_ptr is not None:
        label_idx = tl.load(labels_ptr + block_idx) - col_min
        if label_idx < 0 or label_idx >= n_cols:
            # Loss mask
            predicted_logits = 0.0
        else:
            predicted_logits = tl.load(logits_ptr + label_idx).to(tl.float32)
            if logits_scale_factor != 1.0:
                predicted_logits *= logits_scale_factor
        tl.store(predicted_logits_ptr + block_idx, predicted_logits)

    tl.store(max_logits_ptr + block_idx, max_logits)
    tl.store(sum_exp_logits_ptr + block_idx, sum_exp_logits)


@triton_jit()
def triton_cross_entropy_forward_backward_kernel(
    logits_ptr,
    labels_ptr,
    grad_logits_ptr,
    losses_ptr,
    max_logits_ptr,
    sum_exp_logits_ptr,
    grad_losses,
    col_min: tl_constexpr,
    n_cols: tl_constexpr,
    logits_stride_0: tl_constexpr,
    grad_logits_stride_0: tl_constexpr,
    logits_scale_factor: tl_constexpr,
    block_size: tl_constexpr,
):
    # TODO: Int64 ptr only if needed?
    block_idx = tl.program_id(0).to(tl.int64)
    logits_ptr = logits_ptr + block_idx * logits_stride_0

    if max_logits_ptr is None or sum_exp_logits_ptr is None:
        exp_logits, sum_exp_logits, max_logits, mask = triton_fused_softmax_base(
            logits_ptr, n_cols, logits_scale_factor, block_size
        )
    else:
        max_logits = tl.load(max_logits_ptr + block_idx)
        sum_exp_logits = tl.load(sum_exp_logits_ptr + block_idx)

    label_idx = tl.load(labels_ptr + block_idx) - col_min

    if losses_ptr is not None:
        if label_idx < 0 or label_idx >= n_cols:
            # Loss mask
            loss = 0.0
            predicted_logits = 0.0
        else:
            predicted_logits = tl.load(logits_ptr + label_idx).to(tl.float32)
            if logits_scale_factor != 1.0:
                predicted_logits *= logits_scale_factor
            loss = tl.log(sum_exp_logits) + max_logits - predicted_logits
        tl.store(losses_ptr + block_idx, loss)

    if grad_losses is not None:
        # Run in reverse order to maximize input and cache reuse.
        for col_offset in tl.static_range((n_cols - 1) // block_size * block_size, -1, -block_size):
            if max_logits_ptr is None or sum_exp_logits_ptr is None or col_offset != n_cols - block_size:
                col_offsets = tl.arange(col_offset, col_offset + block_size)
                mask = col_offsets < n_cols
                logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)
                if logits_scale_factor != 1.0:
                    logits *= logits_scale_factor
                exp_logits = tl.exp(logits - max_logits)

            if label_idx < -col_min:
                grad_losses = 0.0
            elif logits_scale_factor != 1.0:
                grad_losses *= logits_scale_factor
            grad_base = exp_logits / sum_exp_logits
            if label_idx < 0 or label_idx >= n_cols:
                grad_logits = grad_base
            else:
                grad_logits = tl.where(col_offsets == label_idx, grad_base - 1.0, grad_base)
            tl.store(
                grad_logits_ptr + block_idx * grad_logits_stride_0 + col_offsets, grad_logits * grad_losses, mask=mask
            )


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


@torch.compile
def _rescale_sum_exp_logits(
    sum_exp_logits: torch.Tensor,
    local_max_logits: torch.Tensor,
    max_logits: torch.Tensor,
) -> torch.Tensor:
    return sum_exp_logits * (local_max_logits - max_logits).exp()


@torch.compile
def _calculate_loss(
    predicted_logits: torch.Tensor,
    target: torch.Tensor,
    sum_exp_logits: torch.Tensor,
    max_logits: torch.Tensor,
) -> torch.Tensor:
    return torch.where(target.flatten() >= 0, sum_exp_logits.log() + max_logits - predicted_logits, 0).mean()


def triton_cross_entropy_forward_backward(
    logits: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor | None,
    grad_output: float | None,
    logits_scale_factor: float,
    target_format: TargetFormat,
    entropy_loss_type: EntropyLossType,
    group: torch.distributed.ProcessGroup | None = None,
    temperature: float = 1.0,
    block_size: int | None = None,
    num_warps: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    A fast triton implementation of cross-entropy, which combines the casting and forward and backward passes,
    all in a single kernel.
     Compared to a standard pytorch implementation, this reduces memory usage (of logits) by 3x and memory I/O by 5x.
    TODO: Better handling of `grad_output = None`
    """
    assert TritonConfig.TRITON_ENABLED
    Assert.eq(entropy_loss_type, EntropyLossType.cross_entropy)
    # TODO: Improve assumptions.
    assert logits.is_contiguous()
    assert target.is_contiguous()
    n_rows = logits.shape[:-1].numel()
    n_cols = logits.size(-1)
    if block_size is None:
        block_size = min(triton.next_power_of_2(n_cols), 32768)
    if num_warps is None:
        num_warps = 4 if block_size < 2048 else (8 if block_size < 8192 else 16)
    # TODO: Safe to do inplace?
    grad_logits = None if grad_output is None else torch.empty_like(logits)
    if target_format == TargetFormat.labels:
        if group is None:
            losses = torch.empty(n_rows, dtype=torch.float, device=logits.device)
            triton_cross_entropy_forward_backward_kernel[(n_rows,)](
                logits,
                target,
                grad_logits,
                losses,
                None,
                None,
                None if grad_output is None else grad_output / n_rows,
                0,
                n_cols,
                logits.stride(-2),
                None if grad_output is None else grad_logits.stride(-2),
                logits_scale_factor,
                block_size=block_size,
                num_warps=num_warps,
            )
            loss = losses.mean()
        else:
            predicted_logits = torch.empty(n_rows, dtype=torch.float, device=logits.device)
            local_max_logits = torch.empty_like(predicted_logits)
            sum_exp_logits = torch.empty_like(predicted_logits)
            triton_cross_entropy_forward_parallel_kernel[(n_rows,)](
                logits,
                target,
                local_max_logits,
                sum_exp_logits,
                predicted_logits,
                n_cols * group.rank(),
                n_cols,
                logits.stride(-2),
                logits_scale_factor,
                block_size=block_size,
            )
            max_logits = local_max_logits.clone()
            torch.distributed.all_reduce(max_logits, op=torch.distributed.ReduceOp.MAX, group=group)
            sum_exp_logits = _rescale_sum_exp_logits(sum_exp_logits, local_max_logits, max_logits)
            torch.distributed.all_reduce(sum_exp_logits, op=torch.distributed.ReduceOp.SUM, group=group)
            torch.distributed.all_reduce(predicted_logits, op=torch.distributed.ReduceOp.SUM, group=group)
            loss = _calculate_loss(predicted_logits, target, sum_exp_logits, max_logits)
            triton_cross_entropy_forward_backward_kernel[(n_rows,)](
                logits,
                target,
                grad_logits,
                None,
                max_logits,
                sum_exp_logits,
                None if grad_output is None else grad_output / n_rows,
                n_cols * group.rank(),
                n_cols,
                logits.stride(-2),
                None if grad_output is None else grad_logits.stride(-2),
                logits_scale_factor,
                block_size=block_size,
                num_warps=num_warps,
            )
    else:
        assert group is None
        losses = torch.empty(n_rows, dtype=torch.float, device=logits.device)
        if loss_mask is not None:
            assert loss_mask.is_contiguous()
        triton_cross_entropy_from_distribution_forward_backward_kernel[(n_rows,)](
            logits,
            target / temperature,
            loss_mask,
            grad_logits,
            losses,
            None if grad_output is None else grad_output / n_rows,
            n_cols,
            logits.stride(-2),
            target.stride(-2),
            None if grad_output is None else grad_logits.stride(-2),
            logits_scale_factor,
            block_size=block_size,
            num_warps=num_warps,
            from_logits=target_format == TargetFormat.logits,
        )
        loss = losses.mean()
    return loss, grad_logits
