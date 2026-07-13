import torch

from fast_llm.core.distributed import ReduceOp, all_reduce
from fast_llm.functional.triton import tl, tl_arange, tl_constexpr, triton, triton_jit
from fast_llm.functional.triton.entropy_loss import (
    parallel_sum_exp_logits,
    triton_cross_entropy_forward_from_labels_parallel_kernel,
    triton_fused_softmax_base,
)
from fast_llm.functional.utils import reduce_losses


@triton_jit()
def triton_monolithic_loss_forward_backward_kernel(
    logits_ptr,
    labels_ptr,
    n_cols: tl_constexpr,
    logits_stride_0: tl_constexpr,
    block_size: tl_constexpr,
    ce_losses_ptr=None,
    z_losses_ptr=None,
    grpo_losses_ptr=None,
    new_logprobs_mean_parts_ptr=None,
    z_loss_mask_ptr=None,
    advantages_ptr=None,
    old_log_probs_ptr=None,
    num_labels_in_seq_ptr=None,
    gspo_coeff_ptr=None,
    max_logits_ptr=None,
    sum_exp_logits_ptr=None,
    predicted_logits_ptr=None,
    weighted_logits_sum_ptr=None,
    grad_logits_ptr=None,
    grad_logits_stride_0: tl_constexpr = None,
    grad_losses_ce=0.0,
    grad_losses_z=0.0,
    grad_losses_grpo=0.0,
    col_min: tl_constexpr = 0,
    logits_scale_factor: tl_constexpr = 1.0,
    epsilon_low: tl_constexpr = 0.2,
    epsilon_high: tl_constexpr = 0.2,
    accumulate: tl_constexpr = False,
):
    """One shared softmax feeding several label-based losses over the same logits row. Each enabled loss
    (selected by the presence of its output/input pointers) stores its own forward scalar, but their
    gradients superpose into two per-row coefficients: `grad_j = prob_coeff * softmax_j - label_coeff *
    delta_{j, label}`. The softmax is computed in-kernel when `max_logits_ptr`/`sum_exp_logits_ptr` are
    absent (single-pass, no tensor parallelism), or loaded from a reduced forward pass otherwise."""
    block_idx = tl.program_id(0).to(tl.int64)
    logits_ptr = logits_ptr + block_idx * logits_stride_0

    # The shared label feeds cross-entropy, GRPO, and GSPO; `labels_ptr` is set whenever any of them is
    # present. The defaults keep both variables defined on the label-free path — triton compiles every branch,
    # so a variable used later must be defined on all of them.
    label_valid = False
    label_idx = 0
    if labels_ptr is not None:
        label_idx = tl.load(labels_ptr + block_idx)
        label_valid = label_idx >= 0
        label_idx -= col_min

    # A masked row of a single label-based loss (nothing else needs the softmax on a label-less row) contributes
    # only zeros. The active-loss *set* is compile-time; only the per-token mask is runtime. Skip such a row's
    # softmax and backward `exp` — but via a branch, not an early `return` (a mid-kernel return compiles into
    # heavy register spills in this large kernel), so the masked-row work drops without hurting the valid path.
    row_inactive = (
        (not label_valid)
        and z_losses_ptr is None
        and gspo_coeff_ptr is None
        and weighted_logits_sum_ptr is None
        and (ce_losses_ptr is not None or grpo_losses_ptr is not None)
    )
    # Defaults keep the softmax outputs defined on the inactive path (whose backward branch never reads them).
    exp_logits = tl.zeros([block_size], tl.float32)
    max_logits = 0.0
    sum_exp_logits = 1.0
    if row_inactive:
        pass
    elif max_logits_ptr is None or sum_exp_logits_ptr is None:
        exp_logits, sum_exp_logits, max_logits, col_offsets, mask = triton_fused_softmax_base(
            logits_ptr, n_cols=n_cols, block_size=block_size, logits_scale_factor=logits_scale_factor
        )
    else:
        max_logits = tl.load(max_logits_ptr + block_idx)
        sum_exp_logits = tl.load(sum_exp_logits_ptr + block_idx)

    log_sum_exp_logits = tl.log(sum_exp_logits) + max_logits

    # Target-index logit shared by cross-entropy and GRPO; the defaults keep it defined when neither is present.
    predicted_logit = 0.0
    new_log_prob = 0.0
    if ce_losses_ptr is not None or grpo_losses_ptr is not None:
        if predicted_logits_ptr is not None:
            predicted_logit = tl.load(predicted_logits_ptr + block_idx)
        elif label_valid and label_idx >= 0 and label_idx < n_cols:
            predicted_logit = tl.load(logits_ptr + label_idx).to(tl.float32)
            if logits_scale_factor != 1.0:
                predicted_logit *= logits_scale_factor
        else:
            predicted_logit = 0.0
        new_log_prob = predicted_logit - log_sum_exp_logits

    prob_coeff = 0.0
    label_coeff = 0.0

    if ce_losses_ptr is not None:
        if label_valid:
            tl.store(ce_losses_ptr + block_idx, log_sum_exp_logits - predicted_logit)
            grad_losses = grad_losses_ce * logits_scale_factor if logits_scale_factor != 1.0 else grad_losses_ce
            prob_coeff += grad_losses
            label_coeff += grad_losses
        else:
            tl.store(ce_losses_ptr + block_idx, 0.0)

    if z_losses_ptr is not None:
        if z_loss_mask_ptr is None or tl.load(z_loss_mask_ptr + block_idx) != 0:
            tl.store(z_losses_ptr + block_idx, log_sum_exp_logits * log_sum_exp_logits)
            grad_losses = grad_losses_z * logits_scale_factor if logits_scale_factor != 1.0 else grad_losses_z
            prob_coeff += 2.0 * grad_losses * log_sum_exp_logits
        else:
            tl.store(z_losses_ptr + block_idx, 0.0)

    if grpo_losses_ptr is not None:
        if label_valid:
            old_log_prob = tl.load(old_log_probs_ptr + block_idx).to(tl.float32)
            advantage = tl.load(advantages_ptr + block_idx).to(tl.float32)
            ratio = tl.exp(new_log_prob - old_log_prob)
            clipped_ratio = tl.minimum(tl.maximum(ratio, 1.0 - epsilon_low), 1.0 + epsilon_high)
            tl.store(grpo_losses_ptr + block_idx, -tl.minimum(ratio * advantage, clipped_ratio * advantage))
            grad_losses = grad_losses_grpo * logits_scale_factor if logits_scale_factor != 1.0 else grad_losses_grpo
            # clip_factor = clamp_min(A, 0) * (ratio <= 1 + eps_h) + clamp_max(A, 0) * (ratio >= 1 - eps_l)
            coeff = (
                tl.maximum(advantage, 0.0) * (ratio <= 1.0 + epsilon_high)
                + tl.minimum(advantage, 0.0) * (ratio >= 1.0 - epsilon_low)
            ) * (ratio * grad_losses)
            prob_coeff += coeff
            label_coeff += coeff
            if new_logprobs_mean_parts_ptr is not None:
                num_labels = tl.load(num_labels_in_seq_ptr + block_idx).to(tl.float32)
                tl.store(new_logprobs_mean_parts_ptr + block_idx, new_log_prob / tl.maximum(num_labels, 1.0))
        else:
            tl.store(grpo_losses_ptr + block_idx, 0.0)
            if new_logprobs_mean_parts_ptr is not None:
                tl.store(new_logprobs_mean_parts_ptr + block_idx, 0.0)

    if gspo_coeff_ptr is not None:
        # The GSPO per-token coefficient is fully scaled by its eager segment seam.
        coeff = tl.load(gspo_coeff_ptr + block_idx).to(tl.float32)
        prob_coeff += coeff
        label_coeff += coeff

    if grad_logits_ptr is not None:
        weighted_logits_sum = 0.0
        col_offset_start: tl.constexpr = (n_cols - 1) // block_size * block_size
        for col_offset in tl.static_range(col_offset_start, -1, -block_size):
            col_offsets = tl_arange(col_offset, col_offset + block_size)
            mask = col_offsets < n_cols
            if row_inactive:
                # No softmax was computed for this row; its gradient is exactly zero.
                grad_logits = col_offsets * 0.0
            else:
                if max_logits_ptr is not None or sum_exp_logits_ptr is not None or col_offset != col_offset_start:
                    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)
                    if logits_scale_factor != 1.0:
                        logits *= logits_scale_factor
                    exp_logits = tl.exp(logits - max_logits)
                    if weighted_logits_sum_ptr is not None:
                        # Local (per-rank) Σ exp·logits_norm feeding the policy entropy; the caller all-reduces
                        # over the vocab group. Zero `logits_norm` on masked columns first (they load as -inf),
                        # so the product is 0 there rather than 0·-inf = nan. `max_logits` is final here (this is
                        # the backward re-stream), so the accumulation needs no online rescaling.
                        weighted_logits_sum += tl.sum(exp_logits * tl.where(mask, logits - max_logits, 0.0), 0)
                grad_logits = prob_coeff * (exp_logits / sum_exp_logits)
                if label_valid:
                    grad_logits = tl.where(col_offsets == label_idx, grad_logits - label_coeff, grad_logits)
            grad_logits_col_ptr = grad_logits_ptr + block_idx * grad_logits_stride_0 + col_offsets
            if accumulate:
                grad_logits += tl.load(grad_logits_col_ptr, mask=mask)
            tl.store(grad_logits_col_ptr, grad_logits, mask=mask)
        if weighted_logits_sum_ptr is not None:
            tl.store(weighted_logits_sum_ptr + block_idx, weighted_logits_sum)


def _monolithic_forward_reduce(
    logits: torch.Tensor,
    labels: torch.Tensor | None,
    group: torch.distributed.ProcessGroup | None,
    logits_scale_factor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Explicit forward pass: per-row softmax and target-index logit, reduced over the vocab group to the
    global values the monolithic kernel then loads. Reuses the shared cross-entropy forward kernel. Used for
    tensor parallelism and — regardless of parallelism — to feed the GSPO segment seam before the backward."""
    n_rows = logits.shape[:-1].numel()
    n_cols = logits.size(-1)
    block_size = min(triton.next_power_of_2(n_cols), 32768)
    num_warps = 4 if block_size < 2048 else (8 if block_size < 8192 else 16)
    local_max_logits = torch.empty(n_rows, dtype=torch.float, device=logits.device)
    sum_exp_logits = torch.empty_like(local_max_logits)
    predicted_logits = torch.empty_like(local_max_logits) if labels is not None else None
    triton_cross_entropy_forward_from_labels_parallel_kernel[(n_rows,)](
        logits,
        labels,
        max_logits_ptr=local_max_logits,
        sum_exp_logits_ptr=sum_exp_logits,
        predicted_logits_ptr=predicted_logits,
        col_min=n_cols * group.rank() if group is not None else 0,
        n_cols=n_cols,
        logits_stride_0=logits.stride(-2),
        block_size=block_size,
        num_warps=num_warps,
        logits_scale_factor=logits_scale_factor,
    )
    if group is None:
        return local_max_logits, sum_exp_logits, predicted_logits
    max_logits, sum_exp_logits = parallel_sum_exp_logits(sum_exp_logits, local_max_logits, group)
    if predicted_logits is not None:
        all_reduce(predicted_logits, op=ReduceOp.SUM, group=group)
    return max_logits, sum_exp_logits, predicted_logits


def triton_monolithic_loss_forward_backward(
    logits: torch.Tensor,  # (*batch, vocab)
    labels: torch.Tensor | None,  # (*batch,) — shared by cross-entropy / GRPO / GSPO
    grad_logits: torch.Tensor | None,
    logits_scale_factor: float,
    group: torch.distributed.ProcessGroup | None,
    divisor: float,
    *,
    ce: tuple[float | None] | None = None,  # cross-entropy: (weighted grad_output,); `None` => absent
    z: tuple[torch.Tensor | None, float | None] | None = None,  # (loss_mask, weighted grad_output)
    # GRPO: (advantages, old_log_probabilities, weighted grad_output, epsilon_low, epsilon_high, num_labels_in_seq)
    grpo: tuple | None = None,
    gspo_coeff: torch.Tensor | None = None,  # per-token backward coefficient from the eager segment seam
    softmax: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None = None,  # precomputed (max, sum, predicted)
    compute_metrics: bool = False,  # also emit the reduced softmax and Σ exp·logits_norm for policy metrics
    block_size: int | None = None,
    num_warps: int | None = None,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None,
    torch.Tensor | None,
]:
    """Dispatch the monolithic label-loss kernel over a single shared softmax. Returns the reduced per-loss
    scalars `(cross_entropy, z, grpo, grpo_new_logprobs_mean)` (each `None` when the loss is absent), the
    accumulated `grad_logits`, and — for policy metrics — the reduced `softmax` (max, sum, predicted) and the
    all-reduced per-row `weighted_logits_sum` (`Σ exp·logits_norm`, both `None` unless `compute_metrics`).
    `softmax` is provided by the caller when the forward was already run (the GSPO seam); otherwise it is
    computed in-kernel (`group is None`, no metrics) or by a reduced forward pass (tensor parallel or metrics).
    A present loss whose weighted grad_output is `None` still emits its forward scalar but no gradient term."""
    assert logits.is_contiguous()
    if labels is not None:
        assert labels.is_contiguous()
    n_rows = logits.shape[:-1].numel()
    n_cols = logits.size(-1)
    if block_size is None:
        block_size = min(triton.next_power_of_2(n_cols), 32768)
    if num_warps is None:
        num_warps = 4 if block_size < 2048 else (8 if block_size < 8192 else 16)

    # Forward-scalar buffers (needed for the total loss even when nothing is registered this step).
    ce_losses = torch.empty(n_rows, dtype=torch.float, device=logits.device) if ce is not None else None
    z_losses = torch.empty(n_rows, dtype=torch.float, device=logits.device) if z is not None else None
    grpo_losses = torch.empty(n_rows, dtype=torch.float, device=logits.device) if grpo is not None else None

    ce_grad_output = ce[0] if ce is not None else None
    z_loss_mask, z_grad_output = z if z is not None else (None, None)
    if grpo is not None:
        advantages, old_log_probabilities, grpo_grad_output, epsilon_low, epsilon_high, num_labels_in_seq = grpo
    else:
        advantages = old_log_probabilities = num_labels_in_seq = grpo_grad_output = None
        epsilon_low = epsilon_high = 0.2
    new_logprobs_mean_parts = (
        torch.empty(n_rows, dtype=torch.float, device=logits.device)
        if grpo is not None and num_labels_in_seq is not None
        else None
    )
    for tensor in (z_loss_mask, advantages, old_log_probabilities, num_labels_in_seq, gspo_coeff):
        if tensor is not None:
            assert tensor.is_contiguous()

    # Metrics reuse the reduced softmax, so run the explicit forward pass even without tensor parallelism.
    if softmax is None and (group is not None or compute_metrics):
        softmax = _monolithic_forward_reduce(logits, labels, group, logits_scale_factor)
    max_logits, sum_exp_logits, predicted_logits = softmax if softmax is not None else (None, None, None)

    has_grad = (
        ce_grad_output is not None
        or z_grad_output is not None
        or grpo_grad_output is not None
        or gspo_coeff is not None
    )
    # The entropy's `Σ exp·logits_norm` is accumulated for free in the backward pass, so metrics need it.
    assert has_grad or not compute_metrics
    weighted_logits_sum = torch.empty(n_rows, dtype=torch.float, device=logits.device) if compute_metrics else None
    if has_grad:
        accumulate = grad_logits is not None
        grad_logits = torch.empty_like(logits) if grad_logits is None else grad_logits
        backward_kwargs = {
            "grad_logits_ptr": grad_logits,
            "grad_logits_stride_0": grad_logits.stride(-2),
            "accumulate": accumulate,
            "grad_losses_ce": 0.0 if ce_grad_output is None else ce_grad_output / divisor,
            "grad_losses_z": 0.0 if z_grad_output is None else z_grad_output / divisor,
            "grad_losses_grpo": 0.0 if grpo_grad_output is None else grpo_grad_output / divisor,
            "weighted_logits_sum_ptr": weighted_logits_sum,
        }
    else:
        backward_kwargs = {}

    triton_monolithic_loss_forward_backward_kernel[(n_rows,)](
        logits,
        labels,
        n_cols=n_cols,
        logits_stride_0=logits.stride(-2),
        block_size=block_size,
        num_warps=num_warps,
        logits_scale_factor=logits_scale_factor,
        col_min=n_cols * group.rank() if group is not None else 0,
        epsilon_low=epsilon_low,
        epsilon_high=epsilon_high,
        ce_losses_ptr=ce_losses,
        z_losses_ptr=z_losses,
        grpo_losses_ptr=grpo_losses,
        new_logprobs_mean_parts_ptr=new_logprobs_mean_parts,
        z_loss_mask_ptr=z_loss_mask,
        advantages_ptr=advantages,
        old_log_probs_ptr=old_log_probabilities,
        num_labels_in_seq_ptr=num_labels_in_seq,
        gspo_coeff_ptr=gspo_coeff,
        max_logits_ptr=max_logits,
        sum_exp_logits_ptr=sum_exp_logits,
        predicted_logits_ptr=predicted_logits,
        **backward_kwargs,
    )

    if weighted_logits_sum is not None and group is not None:
        all_reduce(weighted_logits_sum, op=ReduceOp.SUM, group=group)

    return (
        None if ce_losses is None else reduce_losses(ce_losses, divisor),
        None if z_losses is None else reduce_losses(z_losses, divisor),
        None if grpo_losses is None else reduce_losses(grpo_losses, divisor),
        None if new_logprobs_mean_parts is None else new_logprobs_mean_parts.sum(),
        grad_logits,
        softmax,
        weighted_logits_sum,
    )
