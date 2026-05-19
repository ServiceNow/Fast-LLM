import torch

from fast_llm.core.distributed import ReduceOp, all_reduce
from fast_llm.functional.triton import tl, tl_arange, tl_constexpr, triton, triton_jit
from fast_llm.functional.triton.entropy_loss import (
    parallel_sum_exp_logits,
    triton_cross_entropy_forward_from_labels_parallel_kernel,
)


@triton_jit()
def triton_gspo_loss_backward_kernel(
    logits_ptr,
    labels_ptr,
    max_logits_ptr,
    sum_exp_logits_ptr,
    probability_ratio_ptr,
    seg_advantage_ptr,
    token_weight_ptr,
    grad_logits_ptr,
    n_cols: tl_constexpr,
    logits_stride_0: tl_constexpr,
    grad_logits_stride_0: tl_constexpr,
    block_size: tl_constexpr,
    grad_losses,
    col_min: tl_constexpr = 0,
    logits_scale_factor: tl_constexpr = 1.0,
    epsilon_low: tl_constexpr = 0.2,
    epsilon_high: tl_constexpr = 0.2,
    accumulate: tl_constexpr = False,
):
    block_idx = tl.program_id(0).to(tl.int64)

    # token_weight = mask_t / token_count_{s(t)}; zero for masked tokens and empty segments.
    token_weight = tl.load(token_weight_ptr + block_idx).to(tl.float32)
    if token_weight == 0.0:
        if not accumulate:
            for col_offset in tl.static_range(0, n_cols, block_size):
                col_offsets = tl_arange(int(col_offset), int(col_offset + block_size))
                tl.store(
                    grad_logits_ptr + block_idx * grad_logits_stride_0 + col_offsets, 0, mask=col_offsets < n_cols
                )
        return

    label_idx = tl.load(labels_ptr + block_idx) - col_min
    max_logits = tl.load(max_logits_ptr + block_idx)
    sum_exp_logits = tl.load(sum_exp_logits_ptr + block_idx)
    probability_ratio = tl.load(probability_ratio_ptr + block_idx).to(tl.float32)
    seg_advantage = tl.load(seg_advantage_ptr + block_idx).to(tl.float32)

    # effective_grad = grad_losses * scale * weight * R_s * clip_factor
    # clip_factor = clamp_min(A_s, 0) * (R_s <= 1+eps_h) + clamp_max(A_s, 0) * (R_s >= 1-eps_l)
    grad_scale = grad_losses
    if logits_scale_factor != 1.0:
        grad_scale *= logits_scale_factor
    effective_grad = (
        (
            tl.maximum(seg_advantage, 0.0) * (probability_ratio <= 1.0 + epsilon_high)
            + tl.minimum(seg_advantage, 0.0) * (probability_ratio >= 1.0 - epsilon_low)
        )
        * probability_ratio
        * grad_scale
        * token_weight
    )

    logits_ptr = logits_ptr + block_idx * logits_stride_0

    # grad_logit_i = effective_grad * (softmax_i - delta_{i, label})
    col_offset_start: tl.constexpr = (n_cols - 1) // block_size * block_size
    for col_offset in tl.static_range(col_offset_start, -1, -block_size):
        col_offsets = tl_arange(col_offset, col_offset + block_size)
        mask = col_offsets < n_cols
        logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)
        if logits_scale_factor != 1.0:
            logits *= logits_scale_factor
        prob = tl.exp(logits - max_logits) / sum_exp_logits
        if label_idx < 0 or label_idx >= n_cols:
            # Target not in this TP shard.
            grad = effective_grad * prob
        else:
            grad = effective_grad * tl.where(col_offsets == label_idx, prob - 1.0, prob)
        grad_col_ptr = grad_logits_ptr + block_idx * grad_logits_stride_0 + col_offsets
        if accumulate:
            grad += tl.load(grad_col_ptr, mask=mask)
        tl.store(grad_col_ptr, grad, mask=mask)


def triton_gspo_loss_forward_backward(
    logits: torch.Tensor,  # (*batch, vocab)
    target: torch.Tensor,  # (*batch,)
    advantages: torch.Tensor,  # (*batch,)
    old_log_probabilities: torch.Tensor,  # (*batch,)
    document_index: torch.Tensor,  # (*batch,) int — 0-based segment ID per token
    num_segments: int,  # buffer size, >= document_index.max() + 1
    grad_logits: torch.Tensor | None = None,
    grad_output: float | None = None,
    group: torch.distributed.ProcessGroup | None = None,
    sdp_group: torch.distributed.ProcessGroup | None = None,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.2,
    logits_scale_factor: float = 1.0,
    num_labels_in_seq: torch.Tensor | None = None,
    divisor: float | None = None,
    block_size: int | None = None,
    num_warps: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Triton GSPO loss. Forward fuses softmax + predicted-logit lookup; backward fuses the
    softmax chain rule with the per-token GSPO gradient factor (R_s * clip * token_weight).
    Segment aggregation, loss, and the SDP all-reduce live in PyTorch between the two passes.

    See `fused_gspo_loss_forward_backward` in policy_gradient.py for the math derivation;
    this kernel produces identical outputs.
    """
    assert logits.is_contiguous()
    assert target.is_contiguous()
    assert advantages.is_contiguous()
    assert old_log_probabilities.is_contiguous()
    assert document_index.is_contiguous()

    n_rows = logits.shape[:-1].numel()
    n_cols = logits.size(-1)
    if divisor is None:
        divisor = float(num_segments) if num_segments > 0 else 1.0
    if block_size is None:
        block_size = min(triton.next_power_of_2(n_cols), 32768)
    if num_warps is None:
        num_warps = 4 if block_size < 2048 else (8 if block_size < 8192 else 16)
    col_min = n_cols * group.rank() if group is not None else 0

    # === Forward (Triton): per-token softmax, save max / sum / predicted_logit ===
    max_logits = torch.empty(n_rows, dtype=torch.float, device=logits.device)
    sum_exp_logits = torch.empty_like(max_logits)
    predicted_logits = torch.empty_like(max_logits)
    triton_cross_entropy_forward_from_labels_parallel_kernel[(n_rows,)](
        logits,
        target,
        max_logits_ptr=max_logits,
        sum_exp_logits_ptr=sum_exp_logits,
        predicted_logits_ptr=predicted_logits,
        col_min=col_min,
        n_cols=n_cols,
        logits_stride_0=logits.stride(-2),
        block_size=block_size,
        num_warps=num_warps,
        logits_scale_factor=logits_scale_factor,
    )
    if group is not None:
        # Merge per-shard local max / sum_exp into global values.
        max_logits, sum_exp_logits = parallel_sum_exp_logits(sum_exp_logits, max_logits, group)
        all_reduce(predicted_logits, op=ReduceOp.SUM, group=group)

    # === Segment aggregation (PyTorch) ===
    flat_target = target.reshape(-1)
    flat_doc = document_index.reshape(-1).long()
    flat_advantages = advantages.reshape(-1).float()
    loss_mask = (flat_target >= 0).to(max_logits.dtype)

    new_log_probs = predicted_logits - max_logits - sum_exp_logits.log()
    log_ratio = (new_log_probs - old_log_probabilities.reshape(-1).float()) * loss_mask

    new_logprobs_mean = (
        (new_log_probs * loss_mask / num_labels_in_seq.reshape(-1).clamp(min=1).to(new_log_probs.dtype)).sum()
        if num_labels_in_seq is not None
        else None
    )

    log_ratio_sum = log_ratio.new_zeros(num_segments).index_add_(0, flat_doc, log_ratio)
    advantage_sum = log_ratio.new_zeros(num_segments).index_add_(0, flat_doc, flat_advantages * loss_mask)
    token_count = log_ratio.new_zeros(num_segments).index_add_(0, flat_doc, loss_mask)
    if sdp_group is not None:
        torch.distributed.all_reduce(log_ratio_sum, op=torch.distributed.ReduceOp.SUM, group=sdp_group)
        torch.distributed.all_reduce(advantage_sum, op=torch.distributed.ReduceOp.SUM, group=sdp_group)
        torch.distributed.all_reduce(token_count, op=torch.distributed.ReduceOp.SUM, group=sdp_group)

    safe_count = token_count.clamp(min=1)
    segment_ratio = (log_ratio_sum / safe_count).exp()
    segment_advantage = advantage_sum / safe_count
    inv_token_count = torch.where(token_count > 0, 1.0 / safe_count, token_count.new_zeros(()))

    probability_ratio = segment_ratio[flat_doc].contiguous()
    seg_advantage = segment_advantage[flat_doc].contiguous()
    token_weight = (loss_mask * inv_token_count[flat_doc]).contiguous()

    losses = -torch.min(
        probability_ratio * seg_advantage,
        torch.clamp(probability_ratio, 1 - epsilon_low, 1 + epsilon_high) * seg_advantage,
    )
    loss = (losses * token_weight).sum() / divisor

    if grad_output is None:
        return loss, grad_logits, new_logprobs_mean

    # === Backward (Triton) ===
    accumulate = grad_logits is not None
    grad_logits = torch.empty_like(logits) if grad_logits is None else grad_logits
    triton_gspo_loss_backward_kernel[(n_rows,)](
        logits,
        target,
        max_logits_ptr=max_logits,
        sum_exp_logits_ptr=sum_exp_logits,
        probability_ratio_ptr=probability_ratio,
        seg_advantage_ptr=seg_advantage,
        token_weight_ptr=token_weight,
        grad_logits_ptr=grad_logits,
        n_cols=n_cols,
        logits_stride_0=logits.stride(-2),
        grad_logits_stride_0=grad_logits.stride(-2),
        block_size=block_size,
        grad_losses=grad_output / divisor,
        col_min=col_min,
        logits_scale_factor=logits_scale_factor,
        epsilon_low=epsilon_low,
        epsilon_high=epsilon_high,
        accumulate=accumulate,
        num_warps=num_warps,
    )

    return loss, grad_logits, new_logprobs_mean
