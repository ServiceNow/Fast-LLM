import torch

from fast_llm.functional.triton import tl, tl_arange, tl_constexpr, triton, triton_jit
from fast_llm.functional.triton.entropy_loss import (
    parallel_sum_exp_logits,
    triton_cross_entropy_forward_from_labels_parallel_kernel,
    triton_fused_softmax_base,
)
from fast_llm.functional.utils import reduce_losses


@triton_jit()
def triton_grpo_loss_forward_backward_kernel(
    logits_ptr,
    labels_ptr,
    advantages_ptr,
    old_log_probs_ptr,
    n_cols: tl_constexpr,
    logits_stride_0: tl_constexpr,
    block_size: tl_constexpr,
    losses_ptr=None,
    new_logprobs_mean_parts_ptr=None,
    num_labels_in_seq_ptr=None,
    max_logits_ptr=None,
    sum_exp_logits_ptr=None,
    predicted_logits_ptr=None,
    grad_losses=None,
    grad_logits_ptr=None,
    grad_logits_stride_0: tl_constexpr = None,
    col_min: tl_constexpr = 0,
    logits_scale_factor: tl_constexpr = 1.0,
    epsilon_low: tl_constexpr = 0.2,
    epsilon_high: tl_constexpr = 0.2,
    accumulate: tl_constexpr = False,
):
    block_idx = tl.program_id(0).to(tl.int64)
    logits_ptr = logits_ptr + block_idx * logits_stride_0

    label_idx = tl.load(labels_ptr + block_idx)
    if label_idx < 0:
        # Masked position.
        if losses_ptr is not None:
            tl.store(losses_ptr + block_idx, 0)
        if new_logprobs_mean_parts_ptr is not None:
            tl.store(new_logprobs_mean_parts_ptr + block_idx, 0)
        if grad_losses is not None and not accumulate:
            for col_offset in tl.static_range(0, n_cols, block_size):
                col_offsets = tl_arange(int(col_offset), int(col_offset + block_size))
                tl.store(
                    grad_logits_ptr + block_idx * grad_logits_stride_0 + col_offsets, 0, mask=col_offsets < n_cols
                )
        return

    label_idx -= col_min

    if max_logits_ptr is None or sum_exp_logits_ptr is None:
        # Non-parallel: compute softmax and predicted logit in one forward pass.
        exp_logits, sum_exp_logits, max_logits, col_offsets, mask = triton_fused_softmax_base(
            logits_ptr, n_cols=n_cols, block_size=block_size, logits_scale_factor=logits_scale_factor
        )
        if label_idx >= 0 and label_idx < n_cols:
            predicted_logit = tl.load(logits_ptr + label_idx).to(tl.float32)
            if logits_scale_factor != 1.0:
                predicted_logit *= logits_scale_factor
        else:
            # Parallel case only: target not in local vocab shard.
            predicted_logit = 0.0
    else:
        # Parallel case: use globally reduced values from the first pass.
        max_logits = tl.load(max_logits_ptr + block_idx)
        sum_exp_logits = tl.load(sum_exp_logits_ptr + block_idx)
        predicted_logit = tl.load(predicted_logits_ptr + block_idx)

    # new_log_prob = log_softmax(logits * scale)[label]
    #              = logits[label]*scale - (max_logits + log(sum_exp_logits))
    new_log_prob = predicted_logit - max_logits - tl.log(sum_exp_logits)
    old_log_prob = tl.load(old_log_probs_ptr + block_idx).to(tl.float32)
    advantage = tl.load(advantages_ptr + block_idx).to(tl.float32)

    ratio = tl.exp(new_log_prob - old_log_prob)
    clipped_ratio = tl.minimum(tl.maximum(ratio, 1.0 - epsilon_low), 1.0 + epsilon_high)
    loss = -tl.minimum(ratio * advantage, clipped_ratio * advantage)

    if losses_ptr is not None:
        tl.store(losses_ptr + block_idx, loss)

    if new_logprobs_mean_parts_ptr is not None:
        num_labels = tl.load(num_labels_in_seq_ptr + block_idx).to(tl.float32)
        tl.store(new_logprobs_mean_parts_ptr + block_idx, new_log_prob / tl.maximum(num_labels, 1.0))

    if grad_losses is not None:
        if logits_scale_factor != 1.0:
            grad_losses *= logits_scale_factor
        # effective_grad = probability_ratio_grad * ratio
        # = (clamp_min(adv, 0) * (ratio <= 1+eps_high) + clamp_max(adv, 0) * (ratio >= 1-eps_low)) * ratio * grad_losses
        effective_grad = (
            (
                tl.maximum(advantage, 0.0) * (ratio <= 1.0 + epsilon_high)
                + tl.minimum(advantage, 0.0) * (ratio >= 1.0 - epsilon_low)
            )
            * ratio
            * grad_losses
        )

        # grad_logits_i = effective_grad * (p_i - delta_{i, label})
        col_offset_start: tl.constexpr = (n_cols - 1) // block_size * block_size
        for col_offset in tl.static_range(col_offset_start, -1, -block_size):
            if max_logits_ptr is not None or sum_exp_logits_ptr is not None or col_offset != col_offset_start:
                col_offsets = tl_arange(col_offset, col_offset + block_size)
                mask = col_offsets < n_cols
                logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)
                if logits_scale_factor != 1.0:
                    logits *= logits_scale_factor
                exp_logits = tl.exp(logits - max_logits)
            prob = exp_logits / sum_exp_logits
            if label_idx < 0 or label_idx >= n_cols:
                # Target not in local vocab shard (parallel case): no delta term.
                grad_logits = effective_grad * prob
            else:
                grad_logits = effective_grad * tl.where(col_offsets == label_idx, prob - 1.0, prob)
            grad_logits_col_ptr = grad_logits_ptr + block_idx * grad_logits_stride_0 + col_offsets
            if accumulate:
                grad_logits += tl.load(grad_logits_col_ptr, mask=mask)
            tl.store(grad_logits_col_ptr, grad_logits, mask=mask)


def triton_grpo_loss_forward_backward(
    logits: torch.Tensor,  # (*batch, vocab)
    target: torch.Tensor,  # (*batch,)
    advantages: torch.Tensor,  # (*batch,)
    old_log_probabilities: torch.Tensor,  # (*batch,)
    grad_logits: torch.Tensor | None = None,
    grad_output: float | None = None,
    group: torch.distributed.ProcessGroup | None = None,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.2,
    logits_scale_factor: float = 1.0,
    num_labels_in_seq: torch.Tensor | None = None,
    divisor: float | None = None,
    block_size: int | None = None,
    num_warps: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    assert logits.is_contiguous()
    assert target.is_contiguous()
    assert advantages.is_contiguous()
    assert old_log_probabilities.is_contiguous()
    n_rows = logits.shape[:-1].numel()
    n_cols = logits.size(-1)
    if divisor is None:
        divisor = n_rows
    if block_size is None:
        block_size = min(triton.next_power_of_2(n_cols), 32768)
    if num_warps is None:
        num_warps = 4 if block_size < 2048 else (8 if block_size < 8192 else 16)
    shared_kwargs = {
        "logits_stride_0": logits.stride(-2),
        "n_cols": n_cols,
        "logits_scale_factor": logits_scale_factor,
        "block_size": block_size,
        "num_warps": num_warps,
    }
    kwargs = {
        **shared_kwargs,
        "epsilon_low": epsilon_low,
        "epsilon_high": epsilon_high,
    }
    if grad_output is None:
        backward_kwargs = {}
    else:
        accumulate = grad_logits is not None
        grad_logits = torch.empty_like(logits) if grad_logits is None else grad_logits
        backward_kwargs = {
            "grad_logits_ptr": grad_logits,
            "grad_losses": grad_output / divisor,
            "grad_logits_stride_0": grad_logits.stride(-2),
            "accumulate": accumulate,
        }
    losses = torch.empty(n_rows, dtype=torch.float, device=logits.device)
    if num_labels_in_seq is not None:
        assert num_labels_in_seq.is_contiguous()
        new_logprobs_mean_parts = torch.empty(n_rows, dtype=torch.float, device=logits.device)
        new_logprobs_mean_kwargs = {
            "new_logprobs_mean_parts_ptr": new_logprobs_mean_parts,
            "num_labels_in_seq_ptr": num_labels_in_seq,
        }
    else:
        new_logprobs_mean_kwargs = {}

    if group is None:
        triton_grpo_loss_forward_backward_kernel[(n_rows,)](
            logits,
            target,
            advantages,
            old_log_probabilities,
            losses_ptr=losses,
            **kwargs,
            **backward_kwargs,
            **new_logprobs_mean_kwargs,
        )
    else:
        local_max_logits = torch.empty(n_rows, dtype=torch.float, device=logits.device)
        sum_exp_logits = torch.empty_like(local_max_logits)
        predicted_logits_local = torch.empty_like(local_max_logits)
        triton_cross_entropy_forward_from_labels_parallel_kernel[(n_rows,)](
            logits,
            target,
            max_logits_ptr=local_max_logits,
            sum_exp_logits_ptr=sum_exp_logits,
            predicted_logits_ptr=predicted_logits_local,
            col_min=n_cols * group.rank(),
            **shared_kwargs,
        )
        max_logits, sum_exp_logits = parallel_sum_exp_logits(sum_exp_logits, local_max_logits, group)
        torch.distributed.all_reduce(predicted_logits_local, op=torch.distributed.ReduceOp.SUM, group=group)
        triton_grpo_loss_forward_backward_kernel[(n_rows,)](
            logits,
            target,
            advantages,
            old_log_probabilities,
            losses_ptr=losses,
            max_logits_ptr=max_logits,
            sum_exp_logits_ptr=sum_exp_logits,
            predicted_logits_ptr=predicted_logits_local,
            col_min=n_cols * group.rank(),
            **kwargs,
            **backward_kwargs,
            **new_logprobs_mean_kwargs,
        )

    loss = reduce_losses(losses, divisor)
    new_logprobs_mean = new_logprobs_mean_parts.sum() if num_labels_in_seq is not None else None
    return loss, grad_logits, new_logprobs_mean
