import torch

from fast_llm.functional.config import EntropyLossType, TargetFormat
from fast_llm.functional.triton import tl, tl_arange, tl_constexpr, triton, triton_jit


@triton_jit()
def triton_fused_softmax_iter_base(
    logits_ptr,
    col_offset: tl.constexpr,
    n_cols: tl_constexpr,
    block_size: tl_constexpr,
    max_logits=None,
    sum_exp_logits=None,
    col_offsets=None,
    mask=None,
    logits_scale_factor: tl_constexpr = 1.0,
):
    if col_offsets is None:
        col_offsets = tl_arange(col_offset, col_offset + block_size)
    if mask is None:
        mask = col_offsets < n_cols
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)
    if logits_scale_factor != 1.0:
        logits *= logits_scale_factor

    if col_offset == 0:
        new_max_logits = tl.max(logits, 0)
        exp_logits = tl.exp(logits - new_max_logits)
        sum_exp_logits = tl.sum(exp_logits, 0)
    else:
        new_max_logits = tl.maximum(tl.max(logits, 0), max_logits)
        exp_logits = tl.exp(logits - new_max_logits)
        sum_exp_logits = tl.sum(exp_logits, 0) + sum_exp_logits * tl.exp(max_logits - new_max_logits)
    return logits, exp_logits, sum_exp_logits, new_max_logits, col_offsets, mask


@triton_jit()
def triton_fused_softmax_base(
    logits_ptr,
    n_cols: tl_constexpr,
    block_size: tl_constexpr,
    logits_scale_factor: tl_constexpr = 1.0,
):
    exp_logits = None
    sum_exp_logits = None
    max_logits = None
    for col_offset in tl.static_range(0, n_cols, block_size):
        logits, exp_logits, sum_exp_logits, max_logits, col_offsets, mask = triton_fused_softmax_iter_base(
            logits_ptr,
            col_offset=col_offset,
            n_cols=n_cols,
            block_size=block_size,
            max_logits=max_logits,
            sum_exp_logits=sum_exp_logits,
            logits_scale_factor=logits_scale_factor,
        )
    return exp_logits, sum_exp_logits, max_logits, col_offsets, mask


@triton_jit()
def triton_cross_entropy_forward_from_labels_parallel_kernel(
    logits_ptr,
    labels_ptr,
    n_cols: tl_constexpr,
    logits_stride_0: tl_constexpr,
    block_size: tl_constexpr,
    max_logits_ptr=None,
    sum_exp_logits_ptr=None,
    predicted_logits_ptr=None,
    col_min: tl_constexpr = 0,
    logits_scale_factor: tl_constexpr = 1.0,
):
    # TODO: Int64 ptr only if needed?
    block_idx = tl.program_id(0).to(tl.int64)
    logits_ptr = logits_ptr + block_idx * logits_stride_0

    exp_logits, sum_exp_logits, max_logits, _, _ = triton_fused_softmax_base(
        logits_ptr, n_cols=n_cols, block_size=block_size, logits_scale_factor=logits_scale_factor
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

    if max_logits_ptr is not None:
        tl.store(max_logits_ptr + block_idx, max_logits)
    if sum_exp_logits_ptr is not None:
        tl.store(sum_exp_logits_ptr + block_idx, sum_exp_logits)


@triton_jit()
def triton_cross_entropy_forward_backward_from_labels_kernel(
    logits_ptr,
    labels_ptr,
    n_cols: tl_constexpr,
    logits_stride_0: tl_constexpr,
    block_size: tl_constexpr,
    losses_ptr=None,
    max_logits_ptr=None,
    sum_exp_logits_ptr=None,
    grad_losses=None,
    grad_logits_ptr=None,
    grad_logits_stride_0: tl_constexpr = None,
    col_min: tl_constexpr = 0,
    logits_scale_factor: tl_constexpr = 1.0,
    accumulate: tl_constexpr = False,
):
    # TODO: Int64 ptr only if needed?
    block_idx = tl.program_id(0).to(tl.int64)
    logits_ptr = logits_ptr + block_idx * logits_stride_0

    label_idx = tl.load(labels_ptr + block_idx)
    if label_idx < 0:
        # This entry is masked, ignore.
        if losses_ptr is not None:
            tl.store(losses_ptr + block_idx, 0)
        if grad_losses is not None and not accumulate:
            for col_offset in tl.static_range(0, n_cols, block_size):
                col_offsets = tl_arange(int(col_offset), int(col_offset + block_size))
                tl.store(
                    grad_logits_ptr + block_idx * grad_logits_stride_0 + col_offsets, 0, mask=col_offsets < n_cols
                )
        return

    label_idx -= col_min

    if max_logits_ptr is None or sum_exp_logits_ptr is None:
        exp_logits, sum_exp_logits, max_logits, col_offsets, mask = triton_fused_softmax_base(
            logits_ptr, n_cols=n_cols, block_size=block_size, logits_scale_factor=logits_scale_factor
        )
    else:
        max_logits = tl.load(max_logits_ptr + block_idx)
        sum_exp_logits = tl.load(sum_exp_logits_ptr + block_idx)

    if losses_ptr is not None:
        if label_idx < 0 or label_idx >= n_cols:
            # Loss mask
            loss = 0.0
        else:
            predicted_logits = tl.load(logits_ptr + label_idx).to(tl.float32)
            if logits_scale_factor != 1.0:
                predicted_logits *= logits_scale_factor
            loss = tl.log(sum_exp_logits) + max_logits - predicted_logits
        tl.store(losses_ptr + block_idx, loss)

    if grad_losses is not None:
        if logits_scale_factor != 1.0:
            grad_losses *= logits_scale_factor
        # Run in reverse order to maximize input and cache reuse.
        col_offset_start: tl.constexpr = (n_cols - 1) // block_size * block_size
        for col_offset in tl.static_range(col_offset_start, -1, -block_size):
            if max_logits_ptr is not None or sum_exp_logits_ptr is not None or col_offset != col_offset_start:
                col_offsets = tl_arange(col_offset, col_offset + block_size)
                mask = col_offsets < n_cols
                logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)
                if logits_scale_factor != 1.0:
                    logits *= logits_scale_factor
                exp_logits = tl.exp(logits - max_logits)

            grad_base = exp_logits / sum_exp_logits
            if label_idx < 0 or label_idx >= n_cols:
                grad_logits = grad_base
            else:
                grad_logits = tl.where(col_offsets == label_idx, grad_base - 1.0, grad_base)
            grad_logits *= grad_losses
            grad_logits_col_ptr = grad_logits_ptr + block_idx * grad_logits_stride_0 + col_offsets
            if accumulate:
                grad_logits += tl.load(grad_logits_col_ptr, mask=mask)
            tl.store(grad_logits_col_ptr, grad_logits, mask=mask)


@triton_jit()
def triton_predicted_logits_from_distribution(
    logits_ptr,
    target_ptr,
    n_cols: tl_constexpr,
    block_size: tl_constexpr,
    from_logits: tl_constexpr = True,
    target_logits_scale_factor: tl_constexpr = 1.0,
    logits_scale_factor: tl_constexpr = 1.0,
    return_partial_loss: tl_constexpr = False,  # Skip division by sum_exp_logits in the logits case.
    return_kl_loss: tl.constexpr = False,
):
    max_logits = None
    sum_exp_logits = None
    if from_logits:
        target_max_logits = None
        target_sum_exp_logits = None
    else:
        target_max_logits = 0
        target_sum_exp_logits = 0

    for col_offset in tl.static_range(0, n_cols, block_size):
        logits, exp_logits, sum_exp_logits, max_logits, col_offsets, mask = triton_fused_softmax_iter_base(
            logits_ptr,
            col_offset=col_offset,
            n_cols=n_cols,
            block_size=block_size,
            max_logits=max_logits,
            sum_exp_logits=sum_exp_logits,
            logits_scale_factor=logits_scale_factor,
        )

        if from_logits:
            target_logits, target_exp_logits, target_sum_exp_logits, target_new_max_logits, _, _ = (
                triton_fused_softmax_iter_base(
                    target_ptr,
                    col_offset=col_offset,
                    n_cols=n_cols,
                    block_size=block_size,
                    max_logits=target_max_logits,
                    sum_exp_logits=target_sum_exp_logits,
                    logits_scale_factor=target_logits_scale_factor,
                    col_offsets=col_offsets,
                    mask=mask,
                )
            )
            if col_offset == 0:
                logits_shifted = logits - target_logits if return_kl_loss else logits
                predicted_logits = tl.sum(tl.where(mask, target_exp_logits * logits_shifted, 0))
            else:
                logits_shifted = logits - target_logits if return_kl_loss else logits
                predicted_logits = predicted_logits * tl.exp(target_max_logits - target_new_max_logits) + tl.sum(
                    tl.where(mask, target_exp_logits * logits_shifted, 0)
                )
            target_max_logits = target_new_max_logits
        else:
            target = tl.load(target_ptr + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)
            if col_offset == 0:
                logits_shifted = logits - tl.log(target) if return_kl_loss else logits
                predicted_logits = tl.sum(tl.where(mask, target * logits_shifted, 0))
            else:
                logits_shifted = logits - tl.log(target) if return_kl_loss else logits
                predicted_logits += tl.sum(tl.where(mask, target * logits_shifted, 0))

    if from_logits:
        target = target_exp_logits
        if not return_partial_loss:
            predicted_logits /= target_sum_exp_logits
            target /= target_sum_exp_logits
            if return_kl_loss:
                predicted_logits += tl.log(target_sum_exp_logits) + target_max_logits

    return predicted_logits, exp_logits, sum_exp_logits, max_logits, target_sum_exp_logits, target_max_logits, target


@triton_jit()
def triton_cross_entropy_from_distribution_forward_parallel_kernel(
    logits_ptr,
    target_ptr,
    n_cols: tl_constexpr,
    logits_stride_0: tl_constexpr,
    target_stride_0: tl_constexpr,
    block_size: tl_constexpr,
    loss_mask_ptr=None,
    max_logits_ptr=None,
    sum_exp_logits_ptr=None,
    target_max_logits_ptr=None,
    target_sum_exp_logits_ptr=None,
    partial_losses_ptr=None,
    from_logits: tl_constexpr = True,
    logits_scale_factor: tl_constexpr = 1.0,
    target_logits_scale_factor: tl_constexpr = 1.0,
    return_kl_loss: tl.constexpr = False,
):
    # TODO: Int64 ptr only if needed?
    block_idx = tl.program_id(0).to(tl.int64)
    logits_ptr = logits_ptr + block_idx * logits_stride_0
    target_ptr = target_ptr + block_idx * target_stride_0

    if loss_mask_ptr is not None and tl.load(loss_mask_ptr + block_idx) == 0:
        # This entry is masked, ignore.
        tl.store(partial_losses_ptr + block_idx, 0)
        return

    predicted_logits, _, sum_exp_logits, max_logits, target_sum_exp_logits, target_max_logits, target = (
        triton_predicted_logits_from_distribution(
            logits_ptr,
            target_ptr,
            n_cols=n_cols,
            block_size=block_size,
            from_logits=from_logits,
            logits_scale_factor=logits_scale_factor,
            target_logits_scale_factor=target_logits_scale_factor,
            return_partial_loss=True,
            return_kl_loss=return_kl_loss,
        )
    )
    if partial_losses_ptr is not None:
        tl.store(partial_losses_ptr + block_idx, predicted_logits)
    if max_logits_ptr is not None:
        tl.store(max_logits_ptr + block_idx, max_logits)
    if sum_exp_logits_ptr is not None:
        tl.store(sum_exp_logits_ptr + block_idx, sum_exp_logits)

    if target_max_logits_ptr is not None:
        tl.store(target_max_logits_ptr + block_idx, target_max_logits)
    if target_sum_exp_logits_ptr is not None:
        tl.store(target_sum_exp_logits_ptr + block_idx, target_sum_exp_logits)


@triton_jit()
def triton_cross_entropy_from_distribution_forward_backward_kernel(
    logits_ptr,
    target_ptr,
    n_cols: tl_constexpr,
    logits_stride_0: tl_constexpr,
    target_stride_0: tl_constexpr,
    block_size: tl_constexpr,
    loss_mask_ptr=None,
    partial_losses_ptr=None,
    losses_ptr=None,
    max_logits_ptr=None,
    sum_exp_logits_ptr=None,
    target_max_logits_ptr=None,
    target_sum_exp_logits_ptr=None,
    from_logits: tl_constexpr = True,
    grad_losses=None,
    grad_logits_ptr=None,
    grad_logits_stride_0: tl_constexpr = None,
    logits_scale_factor: tl_constexpr = 1.0,
    target_logits_scale_factor: tl_constexpr = 1.0,
    return_kl_loss: tl.constexpr = False,
    accumulate: tl_constexpr = False,
):
    # TODO: Int64 ptr only if needed?
    block_idx = tl.program_id(0).to(tl.int64)
    logits_ptr = logits_ptr + block_idx * logits_stride_0
    target_ptr = target_ptr + block_idx * target_stride_0

    if loss_mask_ptr is not None and tl.load(loss_mask_ptr + block_idx) == 0:
        # This entry is masked, ignore.
        if losses_ptr is not None:
            tl.store(losses_ptr + block_idx, 0)
        if grad_losses is not None and not accumulate:
            for col_offset in tl.static_range(0, n_cols, block_size):
                col_offsets = tl_arange(int(col_offset), int(col_offset + block_size))
                tl.store(
                    grad_logits_ptr + block_idx * grad_logits_stride_0 + col_offsets, 0, mask=col_offsets < n_cols
                )
        return

    if max_logits_ptr is None or sum_exp_logits_ptr is None:
        predicted_logits, exp_logits, sum_exp_logits, max_logits, target_sum_exp_logits, target_max_logits, target = (
            triton_predicted_logits_from_distribution(
                logits_ptr,
                target_ptr,
                n_cols=n_cols,
                block_size=block_size,
                from_logits=from_logits,
                logits_scale_factor=logits_scale_factor,
                target_logits_scale_factor=target_logits_scale_factor,
                return_kl_loss=return_kl_loss,
            )
        )
    else:
        max_logits = tl.load(max_logits_ptr + block_idx)
        sum_exp_logits = tl.load(sum_exp_logits_ptr + block_idx)
        if from_logits:
            target_max_logits = tl.load(target_max_logits_ptr + block_idx)
            target_sum_exp_logits = tl.load(target_sum_exp_logits_ptr + block_idx)

    if losses_ptr is not None:
        # if return_kl_loss:
        #    predicted_logits = predicted_logits + target_sum_exp_logits.log() + target_max_logits
        # per_sample_losses = sum_exp_logits.log() + max_logits - predicted_logits
        # if loss_mask is not None:
        #    per_sample_losses = torch.where(loss_mask.flatten(), per_sample_losses, 0)
        if partial_losses_ptr is not None:
            predicted_logits = tl.load(partial_losses_ptr + block_idx)
            if from_logits:
                predicted_logits /= target_sum_exp_logits
                if return_kl_loss:
                    predicted_logits += tl.log(target_sum_exp_logits) + target_max_logits
        # per_sample_loss = log(sum_exp_logits) - sum(probabilities * logits)
        loss = tl.log(sum_exp_logits) + max_logits - predicted_logits
        tl.store(losses_ptr + block_idx, loss)

    if grad_losses is not None:
        if logits_scale_factor != 1.0:
            grad_losses *= logits_scale_factor
        # grad / grad_output = exp_logits / sum_exp_logits - target_probabilities.
        col_offset_start: tl.constexpr = (n_cols - 1) // block_size * block_size
        for col_offset in tl.static_range(col_offset_start, -1, -block_size):
            col_offsets = tl_arange(col_offset, col_offset + block_size)
            mask = col_offsets < n_cols
            if max_logits_ptr is not None or sum_exp_logits_ptr is not None or col_offset != col_offset_start:
                logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)
                if logits_scale_factor != 1.0:
                    logits *= logits_scale_factor
                exp_logits = tl.exp(logits - max_logits)
                if from_logits:
                    target_logits = tl.load(target_ptr + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)
                    if target_logits_scale_factor != 1.0:
                        target_logits *= target_logits_scale_factor
                    target = tl.exp(target_logits - target_max_logits) / target_sum_exp_logits
                else:
                    target = tl.load(target_ptr + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)

            grad_logits = grad_losses * (exp_logits / sum_exp_logits - target)
            grad_logits_col_ptr = grad_logits_ptr + block_idx * grad_logits_stride_0 + col_offsets
            if accumulate:
                grad_logits += tl.load(grad_logits_col_ptr, mask=mask)
            tl.store(grad_logits_col_ptr, grad_logits, mask=mask)


@triton_jit()
def triton_reverse_kl_forward_from_distribution(
    logits_ptr,
    target_ptr,
    n_cols: tl_constexpr,
    block_size: tl_constexpr,
    from_logits: tl_constexpr = True,
    target_logits_scale_factor: tl_constexpr = 1.0,
    logits_scale_factor: tl_constexpr = 1.0,
    return_partial_loss: tl_constexpr = False,
):
    max_logits = None
    sum_exp_logits = None
    if from_logits:
        target_max_logits = None
        target_sum_exp_logits = None
    else:
        target_max_logits = 0
        target_sum_exp_logits = 0

    for col_offset in tl.static_range(0, n_cols, block_size):
        logits, exp_logits, sum_exp_logits, new_max_logits, col_offsets, mask = triton_fused_softmax_iter_base(
            logits_ptr,
            col_offset=col_offset,
            n_cols=n_cols,
            block_size=block_size,
            max_logits=max_logits,
            sum_exp_logits=sum_exp_logits,
            logits_scale_factor=logits_scale_factor,
        )
        if from_logits:
            # log_target excludes the log_sum_exp term to be added later
            log_target, _, target_sum_exp_logits, target_new_max_logits, _, _ = triton_fused_softmax_iter_base(
                target_ptr,
                col_offset=col_offset,
                n_cols=n_cols,
                block_size=block_size,
                max_logits=target_max_logits,
                sum_exp_logits=target_sum_exp_logits,
                logits_scale_factor=target_logits_scale_factor,
                col_offsets=col_offsets,
                mask=mask,
            )
            target = log_target
        else:
            target = tl.load(target_ptr + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)
            log_target = tl.log(target)
        if col_offset == 0:
            loss = tl.sum(tl.where(mask, exp_logits * (logits - log_target), 0))

        else:
            loss = loss * tl.exp(max_logits - new_max_logits) + tl.sum(
                tl.where(mask, exp_logits * (logits - log_target), 0)
            )
        max_logits = new_max_logits
        if from_logits:
            target_max_logits = target_new_max_logits

    if not return_partial_loss:
        loss = loss / sum_exp_logits - tl.log(sum_exp_logits) - max_logits
        if from_logits:
            loss = loss + tl.log(target_sum_exp_logits) + target_max_logits

    return loss, logits, target, sum_exp_logits, max_logits, target_sum_exp_logits, target_max_logits


@triton_jit()
def triton_reverse_kl_forward_kernel_from_distribution(
    logits_ptr,
    target_ptr,
    n_cols: tl_constexpr,
    logits_stride_0: tl_constexpr,
    target_stride_0: tl_constexpr,
    block_size: tl_constexpr,
    loss_mask_ptr=None,
    max_logits_ptr=None,
    sum_exp_logits_ptr=None,
    target_max_logits_ptr=None,
    target_sum_exp_logits_ptr=None,
    partial_losses_ptr=None,
    from_logits: tl_constexpr = True,
    logits_scale_factor: tl_constexpr = 1.0,
    target_logits_scale_factor: tl_constexpr = 1.0,
):
    # TODO: Int64 ptr only if needed?
    block_idx = tl.program_id(0).to(tl.int64)
    logits_ptr = logits_ptr + block_idx * logits_stride_0
    target_ptr = target_ptr + block_idx * target_stride_0

    if loss_mask_ptr is not None and tl.load(loss_mask_ptr + block_idx) == 0:
        # This entry is masked, ignore.
        tl.store(partial_losses_ptr + block_idx, 0)
        return

    partial_loss, _, _, sum_exp_logits, max_logits, target_sum_exp_logits, target_max_logits = (
        triton_reverse_kl_forward_from_distribution(
            logits_ptr,
            target_ptr,
            n_cols=n_cols,
            block_size=block_size,
            from_logits=from_logits,
            logits_scale_factor=logits_scale_factor,
            target_logits_scale_factor=target_logits_scale_factor,
            return_partial_loss=True,
        )
    )
    if partial_losses_ptr is not None:
        tl.store(partial_losses_ptr + block_idx, partial_loss)
    if max_logits_ptr is not None:
        tl.store(max_logits_ptr + block_idx, max_logits)
    if sum_exp_logits_ptr is not None:
        tl.store(sum_exp_logits_ptr + block_idx, sum_exp_logits)

    if target_max_logits_ptr is not None:
        tl.store(target_max_logits_ptr + block_idx, target_max_logits)
    if target_sum_exp_logits_ptr is not None:
        tl.store(target_sum_exp_logits_ptr + block_idx, target_sum_exp_logits)


@triton_jit()
def triton_reverse_kl_forward_backward_kernel_from_distribution(
    logits_ptr,
    target_ptr,
    n_cols: tl_constexpr,
    logits_stride_0: tl_constexpr,
    target_stride_0: tl_constexpr,
    block_size: tl_constexpr,
    loss_mask_ptr=None,
    partial_losses_ptr=None,
    losses_ptr=None,
    max_logits_ptr=None,
    sum_exp_logits_ptr=None,
    target_max_logits_ptr=None,
    target_sum_exp_logits_ptr=None,
    from_logits: tl_constexpr = True,
    grad_losses=None,
    grad_logits_ptr=None,
    grad_logits_stride_0: tl_constexpr = None,
    logits_scale_factor: tl_constexpr = 1.0,
    target_logits_scale_factor: tl_constexpr = 1.0,
    accumulate: tl_constexpr = False,
):
    # TODO: Int64 ptr only if needed?
    block_idx = tl.program_id(0).to(tl.int64)
    logits_ptr = logits_ptr + block_idx * logits_stride_0
    target_ptr = target_ptr + block_idx * target_stride_0

    if loss_mask_ptr is not None and tl.load(loss_mask_ptr + block_idx) == 0:
        # This entry is masked, ignore.
        if losses_ptr is not None:
            tl.store(losses_ptr + block_idx, 0)
        if grad_losses is not None and not accumulate:
            for col_offset in tl.static_range(0, n_cols, block_size):
                col_offsets = tl_arange(int(col_offset), int(col_offset + block_size))
                tl.store(
                    grad_logits_ptr + block_idx * grad_logits_stride_0 + col_offsets, 0, mask=col_offsets < n_cols
                )
        return

    if max_logits_ptr is None or sum_exp_logits_ptr is None:
        loss, logits, target, sum_exp_logits, max_logits, target_sum_exp_logits, target_max_logits = (
            triton_reverse_kl_forward_from_distribution(
                logits_ptr,
                target_ptr,
                n_cols=n_cols,
                block_size=block_size,
                from_logits=from_logits,
                logits_scale_factor=logits_scale_factor,
                target_logits_scale_factor=target_logits_scale_factor,
            )
        )
    else:
        max_logits = tl.load(max_logits_ptr + block_idx)
        sum_exp_logits = tl.load(sum_exp_logits_ptr + block_idx)
        if from_logits:
            target_max_logits = tl.load(target_max_logits_ptr + block_idx)
            target_sum_exp_logits = tl.load(target_sum_exp_logits_ptr + block_idx)

    if losses_ptr is not None:
        if partial_losses_ptr is not None:
            loss = tl.load(partial_losses_ptr + block_idx)
            loss = loss / sum_exp_logits - tl.log(sum_exp_logits) - max_logits
            if from_logits:
                loss = loss + tl.log(target_sum_exp_logits) + target_max_logits
        tl.store(losses_ptr + block_idx, loss)

    if grad_losses is not None:
        if logits_scale_factor != 1.0:
            grad_losses *= logits_scale_factor
        col_offset_start: tl.constexpr = (n_cols - 1) // block_size * block_size
        for col_offset in tl.static_range(col_offset_start, -1, -block_size):
            col_offsets = tl_arange(col_offset, col_offset + block_size)
            mask = col_offsets < n_cols
            if max_logits_ptr is not None or sum_exp_logits_ptr is not None or col_offset != col_offset_start:
                logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)
                if logits_scale_factor != 1.0:
                    logits *= logits_scale_factor
                target = tl.load(target_ptr + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)
                if from_logits and target_logits_scale_factor != 1.0:
                    target *= target_logits_scale_factor
            if from_logits:
                target_log_probability = target - target_max_logits - tl.log(target_sum_exp_logits)
            else:
                target_log_probability = tl.log(target)

            predicted_probability = tl.exp(logits - max_logits) / sum_exp_logits
            predicted_log_probability = logits - max_logits - tl.log(sum_exp_logits)
            grad_logits = (
                grad_losses * (predicted_log_probability - target_log_probability - loss) * predicted_probability
            )
            grad_logits_col_ptr = grad_logits_ptr + block_idx * grad_logits_stride_0 + col_offsets
            if accumulate:
                grad_logits += tl.load(grad_logits_col_ptr, mask=mask)
            tl.store(grad_logits_col_ptr, grad_logits, mask=mask)


@torch.compile
def _rescale_sum_exp_logits(
    sum_exp_logits: torch.Tensor,
    local_max_logits: torch.Tensor,
    max_logits: torch.Tensor,
) -> torch.Tensor:
    return sum_exp_logits * (local_max_logits - max_logits).exp()


def parallel_sum_exp_logits(
    sum_exp_logits: torch.Tensor,
    local_max_logits: torch.Tensor,
    group: torch.distributed.ProcessGroup | None,
) -> torch.Tensor:
    max_logits = local_max_logits.clone()
    torch.distributed.all_reduce(max_logits, op=torch.distributed.ReduceOp.MAX, group=group)
    sum_exp_logits = _rescale_sum_exp_logits(sum_exp_logits, local_max_logits, max_logits)
    torch.distributed.all_reduce(sum_exp_logits, op=torch.distributed.ReduceOp.SUM, group=group)
    return max_logits, sum_exp_logits


@torch.compile
def _cross_entropy_loss_from_labels(
    predicted_logits: torch.Tensor,
    target: torch.Tensor,
    sum_exp_logits: torch.Tensor,
    max_logits: torch.Tensor,
) -> torch.Tensor:
    return torch.where(target.flatten() >= 0, sum_exp_logits.log() + max_logits - predicted_logits, 0).mean()


@torch.compile
def _rescale_predicted_logits(
    predicted_logits: torch.Tensor,
    local_target_max_logits: torch.Tensor,
    target_max_logits: torch.Tensor,
):
    # We skipped the division by `target_sum_exp_logits` in the triton kernel so we do it here.
    return predicted_logits * torch.exp(local_target_max_logits - target_max_logits)


@torch.compile
def _cross_entropy_loss_from_distribution(
    predicted_logits: torch.Tensor,
    loss_mask: torch.Tensor | None,
    sum_exp_logits: torch.Tensor,
    max_logits: torch.Tensor,
    target_sum_exp_logits: torch.Tensor | None,
    target_max_logits: torch.Tensor | None,
    return_kl_loss: bool = False,
) -> torch.Tensor:
    if return_kl_loss:
        predicted_logits = predicted_logits + target_sum_exp_logits.log() + target_max_logits
    per_sample_losses = sum_exp_logits.log() + max_logits - predicted_logits
    if loss_mask is not None:
        per_sample_losses = torch.where(loss_mask.flatten(), per_sample_losses, 0)
    return per_sample_losses.mean()


def triton_entropy_loss_forward_backward(
    logits: torch.Tensor,  # (*batch, vocab)
    target: torch.Tensor,  # (*batch,) or (*batch, vocab)
    loss_mask: torch.Tensor | None,  # (*batch,)
    grad_logits: torch.Tensor | None = None,
    grad_output: float | None = None,
    group: torch.distributed.ProcessGroup | None = None,
    logits_scale_factor: float = 1.0,
    temperature: float = 1.0,
    target_format: TargetFormat = TargetFormat.labels,
    entropy_loss_type: EntropyLossType = EntropyLossType.cross_entropy,
    block_size: int | None = None,
    num_warps: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    A fast triton implementation of cross-entropy, which combines the casting and forward and backward passes,
    all in a single kernel.
     Compared to a standard pytorch implementation, this reduces memory usage (of logits) by 3x and memory I/O by 5x.
    TODO: Better handling of `grad_output = None`
    """
    # TODO: Improve assumptions.
    assert logits.is_contiguous()
    assert target.is_contiguous()
    n_rows = logits.shape[:-1].numel()
    n_cols = logits.size(-1)
    if block_size is None:
        block_size = min(triton.next_power_of_2(n_cols), 32768)
    if num_warps is None:
        num_warps = 4 if block_size < 2048 else (8 if block_size < 8192 else 16)
    kwargs = {
        "logits_stride_0": logits.stride(-2),
        "n_cols": n_cols,
        "logits_scale_factor": logits_scale_factor,
        "block_size": block_size,
        "num_warps": num_warps,
    }
    if grad_output is None:
        backward_kwargs = {}
    else:
        accumulate = grad_logits is not None
        grad_logits = torch.empty_like(logits) if grad_logits is None else grad_logits
        backward_kwargs = {
            "grad_logits_ptr": grad_logits,
            "grad_losses": grad_output / n_rows,
            "grad_logits_stride_0": grad_logits.stride(-2),
            "accumulate": accumulate,
        }
    if target_format == TargetFormat.labels:
        assert entropy_loss_type != EntropyLossType.reverse_kl
        if group is None:
            losses = torch.empty(n_rows, dtype=torch.float, device=logits.device)
            triton_cross_entropy_forward_backward_from_labels_kernel[(n_rows,)](
                logits,
                target,
                losses_ptr=losses,
                **kwargs,
                **backward_kwargs,
            )
            loss = losses.mean()
        else:
            partial_losses = torch.empty(n_rows, dtype=torch.float, device=logits.device)
            local_max_logits = torch.empty_like(partial_losses)
            sum_exp_logits = torch.empty_like(partial_losses)
            triton_cross_entropy_forward_from_labels_parallel_kernel[(n_rows,)](
                logits,
                target,
                max_logits_ptr=local_max_logits,
                sum_exp_logits_ptr=sum_exp_logits,
                predicted_logits_ptr=partial_losses,
                col_min=n_cols * group.rank(),
                **kwargs,
            )
            max_logits, sum_exp_logits = parallel_sum_exp_logits(sum_exp_logits, local_max_logits, group)
            torch.distributed.all_reduce(partial_losses, op=torch.distributed.ReduceOp.SUM, group=group)
            loss = _cross_entropy_loss_from_labels(partial_losses, target, sum_exp_logits, max_logits)
            if grad_output is not None:
                triton_cross_entropy_forward_backward_from_labels_kernel[(n_rows,)](
                    logits,
                    target,
                    max_logits_ptr=max_logits,
                    sum_exp_logits_ptr=sum_exp_logits,
                    col_min=n_cols * group.rank(),
                    **kwargs,
                    **backward_kwargs,
                )
    else:
        if loss_mask is not None:
            assert loss_mask.is_contiguous()
        if entropy_loss_type == EntropyLossType.reverse_kl:
            kernel = triton_reverse_kl_forward_backward_kernel_from_distribution
        else:
            kernel = triton_cross_entropy_from_distribution_forward_backward_kernel
            kwargs["return_kl_loss"] = entropy_loss_type == EntropyLossType.forward_kl

        if group is None:
            losses = torch.empty(n_rows, dtype=torch.float, device=logits.device)
            kernel[(n_rows,)](
                logits,
                target,
                loss_mask_ptr=loss_mask,
                losses_ptr=losses,
                max_logits_ptr=None,
                sum_exp_logits_ptr=None,
                target_max_logits_ptr=None,
                target_sum_exp_logits_ptr=None,
                target_stride_0=target.stride(-2),
                target_logits_scale_factor=logits_scale_factor / temperature,
                from_logits=target_format == TargetFormat.logits,
                **kwargs,
                **backward_kwargs,
            )
            loss = losses.mean()
        else:
            partial_losses = torch.empty(n_rows, dtype=torch.float, device=logits.device)
            local_max_logits = torch.empty_like(partial_losses)
            sum_exp_logits = torch.empty_like(partial_losses)
            if target_format == TargetFormat.logits:
                local_target_max_logits = torch.empty_like(partial_losses)
                target_sum_exp_logits = torch.empty_like(partial_losses)
            else:
                local_target_max_logits = target_sum_exp_logits = None

            forward_kernel = (
                triton_reverse_kl_forward_kernel_from_distribution
                if entropy_loss_type == EntropyLossType.reverse_kl
                else triton_cross_entropy_from_distribution_forward_parallel_kernel
            )

            forward_kernel[(n_rows,)](
                logits,
                target,
                loss_mask_ptr=loss_mask,
                max_logits_ptr=local_max_logits,
                sum_exp_logits_ptr=sum_exp_logits,
                target_max_logits_ptr=local_target_max_logits,
                target_sum_exp_logits_ptr=target_sum_exp_logits,
                partial_losses_ptr=partial_losses,
                target_stride_0=target.stride(-2),
                target_logits_scale_factor=logits_scale_factor / temperature,
                from_logits=target_format == TargetFormat.logits,
                **kwargs,
            )
            max_logits, sum_exp_logits = parallel_sum_exp_logits(sum_exp_logits, local_max_logits, group)
            if target_format == TargetFormat.logits:
                target_max_logits, target_sum_exp_logits = parallel_sum_exp_logits(
                    target_sum_exp_logits, local_target_max_logits, group
                )
                if entropy_loss_type != EntropyLossType.reverse_kl:
                    partial_losses = _rescale_predicted_logits(
                        partial_losses, local_target_max_logits, target_max_logits
                    )
            else:
                target_max_logits = None
            if entropy_loss_type == EntropyLossType.reverse_kl:
                partial_losses = _rescale_predicted_logits(partial_losses, local_max_logits, max_logits)
            torch.distributed.all_reduce(partial_losses, op=torch.distributed.ReduceOp.SUM, group=group)

            kernel[(n_rows,)](
                logits,
                target,
                loss_mask_ptr=loss_mask,
                max_logits_ptr=max_logits,
                sum_exp_logits_ptr=sum_exp_logits,
                target_max_logits_ptr=target_max_logits,
                target_sum_exp_logits_ptr=target_sum_exp_logits,
                partial_losses_ptr=partial_losses,
                losses_ptr=partial_losses,
                target_stride_0=target.stride(-2),
                target_logits_scale_factor=logits_scale_factor / temperature,
                from_logits=target_format == TargetFormat.logits,
                **kwargs,
                **backward_kwargs,
            )
            loss = partial_losses.mean()
    return loss, grad_logits
