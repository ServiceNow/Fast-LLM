import torch

from fast_llm.functional.triton import tl, tl_arange, tl_constexpr, triton, triton_jit
from fast_llm.functional.triton.entropy_loss import (
    parallel_sum_exp_logits,
    triton_cross_entropy_forward_from_labels_parallel_kernel,
    triton_fused_softmax_base,
)


@triton_jit()
def triton_z_loss_forward_backward_kernel(
    logits_ptr,
    loss_mask_ptr,
    n_cols: tl_constexpr,
    logits_stride_0: tl_constexpr,
    block_size: tl_constexpr,
    losses_ptr=None,
    max_logits_ptr=None,
    sum_exp_logits_ptr=None,
    grad_losses=None,
    grad_logits_ptr=None,
    grad_logits_stride_0: tl_constexpr = None,
    logits_scale_factor: tl_constexpr = 1.0,
    accumulate: tl_constexpr = False,
):
    # TODO: Int64 ptr only if needed?
    block_idx = tl.program_id(0).to(tl.int64)
    logits_ptr = logits_ptr + block_idx * logits_stride_0

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
        exp_logits, sum_exp_logits, max_logits, col_offsets, mask = triton_fused_softmax_base(
            logits_ptr, n_cols=n_cols, block_size=block_size, logits_scale_factor=logits_scale_factor
        )
    else:
        max_logits = tl.load(max_logits_ptr + block_idx)
        sum_exp_logits = tl.load(sum_exp_logits_ptr + block_idx)

    log_sum_exp_logits = tl.log(sum_exp_logits) + max_logits

    if losses_ptr is not None:
        tl.store(losses_ptr + block_idx, log_sum_exp_logits * log_sum_exp_logits)

    if grad_losses is not None:
        if logits_scale_factor != 1.0:
            grad_losses *= logits_scale_factor
        grad_losses *= 2 * log_sum_exp_logits / sum_exp_logits
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
            grad_logits = exp_logits * grad_losses
            grad_logits_col_ptr = grad_logits_ptr + block_idx * grad_logits_stride_0 + col_offsets
            if accumulate:
                grad_logits += tl.load(grad_logits_col_ptr, mask=mask)
            tl.store(grad_logits_col_ptr, grad_logits, mask=mask)


def triton_z_loss_forward_backward(
    logits: torch.Tensor,
    loss_mask: torch.Tensor | None,
    grad_logits: torch.Tensor | None = None,
    grad_output: float | None = None,
    group: torch.distributed.ProcessGroup | None = None,
    logits_scale_factor: float = 1.0,
    block_size: int | None = None,
    num_warps: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert logits.is_contiguous()
    if loss_mask is not None:
        assert loss_mask.is_contiguous()
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
    losses = torch.empty(n_rows, dtype=torch.float, device=logits.device)
    if group is None:
        triton_z_loss_forward_backward_kernel[(n_rows,)](
            logits,
            loss_mask_ptr=loss_mask,
            losses_ptr=losses,
            **kwargs,
            **backward_kwargs,
        )
    else:
        local_max_logits = torch.empty(n_rows, dtype=torch.float, device=logits.device)
        sum_exp_logits = torch.empty_like(local_max_logits)
        triton_cross_entropy_forward_from_labels_parallel_kernel[(n_rows,)](
            logits,
            None,
            max_logits_ptr=local_max_logits,
            sum_exp_logits_ptr=sum_exp_logits,
            **kwargs,
        )
        max_logits, sum_exp_logits = parallel_sum_exp_logits(sum_exp_logits, local_max_logits, group)
        triton_z_loss_forward_backward_kernel[(n_rows,)](
            logits,
            loss_mask_ptr=loss_mask,
            losses_ptr=losses,
            max_logits_ptr=max_logits,
            sum_exp_logits_ptr=sum_exp_logits,
            **kwargs,
            **backward_kwargs,
        )
    return losses.mean(), grad_logits
