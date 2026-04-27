import dataclasses

import torch
import torch.distributed

from fast_llm.functional.entropy_loss import fused_predicted_logits_from_labels, fused_softmax_base


@dataclasses.dataclass
class PolicyGradientMetrics:
    """
    Scalar metrics for policy-gradient losses (GRPO, PPO, …).

    All per-token-mean fields use the same normalization as new_logprobs_mean:
      sum(value * mask / label_counts.clamp(1))
    The caller must then divide by num_documents_in_batch for the final logged value.

    ratio_new_old_sum / ratio_new_old_squared_sum are raw masked sums (no label_counts division) for ESS.

    max_advantage / min_advantage are raw per-local-batch extrema; the caller must
    all_reduce them with ReduceOp.MAX / ReduceOp.MIN across SDP ranks.
    """

    old_logprobs: torch.Tensor  # per-token mean (label_counts normalised)
    ratio_new_old: torch.Tensor  # per-token mean IS ratio
    ratio_new_old_sum: torch.Tensor  # raw masked sum (ESS numerator)
    ratio_new_old_squared_sum: torch.Tensor  # raw masked sum (ESS denominator)
    kl_new_old: torch.Tensor  # per-token mean Schulman KL approx
    clamp_log_ratio_new_old_indicator: torch.Tensor  # per-token mean clipping indicator
    advantage: torch.Tensor  # per-token mean
    max_advantage: torch.Tensor  # max over masked tokens (caller does MAX all-reduce)
    min_advantage: torch.Tensor  # min over masked tokens (caller does MIN all-reduce)
    num_tokens: torch.Tensor  # raw masked sum
    entropy: torch.Tensor | None  # per-token mean entropy; None when not requested


@torch.compile
def _compute_pg_base_metrics(
    logits: torch.Tensor,  # (*batch, vocab_local)
    target: torch.Tensor,  # (*batch,)
    old_log_probabilities: torch.Tensor,  # (*batch,)
    advantages: torch.Tensor,  # (*batch,)
    label_counts: torch.Tensor,  # (*batch,) global per-seq count, broadcast per token
    epsilon_low: float,
    epsilon_high: float,
    logits_scale_factor: float,
    group: torch.distributed.ProcessGroup | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Compute all non-entropy policy-gradient metrics in a single fused pass."""
    loss_mask = target >= 0
    mask = loss_mask.float()
    denom = label_counts.float().clamp(min=1)

    logits_norm, _, sum_exp_logits, _ = fused_softmax_base(logits, logits_scale_factor, group)
    predicted_logits, _, _ = fused_predicted_logits_from_labels(logits_norm, target, loss_mask, group)
    new_log_probs = predicted_logits - sum_exp_logits.log()

    log_ratio = new_log_probs - old_log_probabilities
    ratio = log_ratio.exp()
    clipped = (ratio < 1.0 - epsilon_low) | (ratio > 1.0 + epsilon_high)

    # Schulman KL approximation: exp(r) - r - 1
    kl = ratio - log_ratio - 1.0

    old_lp = (old_log_probabilities * mask / denom).sum()
    ratio_new_old_mean = (ratio * mask / denom).sum()
    ratio_new_old_sum = (ratio * mask).sum()
    ratio_new_old_squared_sum = (ratio * ratio * mask).sum()
    kl_mean = (kl * mask / denom).sum()
    clamp_indicator_mean = (clipped.float() * mask / denom).sum()
    adv_mean = (advantages * mask / denom).sum()
    num_tokens = mask.sum()

    # max/min over masked positions; fill non-masked with sentinel values
    neg_inf = advantages.new_full((), float("-inf"))
    pos_inf = advantages.new_full((), float("inf"))
    max_adv = torch.where(loss_mask, advantages, neg_inf).max()
    min_adv = torch.where(loss_mask, advantages, pos_inf).min()

    return (
        old_lp,
        ratio_new_old_mean,
        ratio_new_old_sum,
        ratio_new_old_squared_sum,
        kl_mean,
        clamp_indicator_mean,
        adv_mean,
        max_adv,
        min_adv,
        num_tokens,
    )


def compute_chunked_entropy(
    logits: torch.Tensor,  # (*batch, vocab_local)
    target: torch.Tensor,  # (*batch,) — used only for loss_mask
    label_counts: torch.Tensor,  # (*batch,)
    logits_scale_factor: float,
    group: torch.distributed.ProcessGroup | None,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """
    Compute per-token entropy -Σ p log p, chunked over the batch dimension to
    limit peak memory.  Supports vocab-parallel via all-reduce per chunk.

    Returns a scalar using the same label_counts normalisation as other mean metrics
    (sum of per-sequence mean entropies). Caller must divide by num_documents_in_batch.

    Memory per chunk: chunk_size × vocab_local × 4 bytes.
    At chunk_size=4096, vocab_local=19K (8-way TP): ~300 MB.

    Entropy formula (numerically stable):
      entropy_i = log(Σ exp(x_j - x_max)) - Σ(exp(x_j - x_max) * (x_j - x_max)) / Σ exp(x_j - x_max)
                = log(sum_exp) - (exp_logits · logits_norm).sum() / sum_exp
    """
    loss_mask = target >= 0
    mask = loss_mask.float()
    denom = label_counts.float().clamp(min=1)

    batch_size = logits.shape[0]
    total = logits.new_zeros(())

    for start in range(0, batch_size, chunk_size):
        sl = slice(start, start + chunk_size)
        logits_chunk = logits[sl]

        # Recompute softmax base for this chunk only.
        # Scale here since fused_softmax_base expects the full tensor for max/all-reduce;
        # we handle it manually to avoid a full-tensor pass.
        if logits_scale_factor != 1.0:
            logits_chunk = logits_chunk * logits_scale_factor

        logits_max = logits_chunk.float().max(dim=-1).values
        if group is not None:
            torch.distributed.all_reduce(logits_max, op=torch.distributed.ReduceOp.MAX, group=group)

        logits_norm_chunk = logits_chunk.float() - logits_max.unsqueeze(-1)
        exp_chunk = logits_norm_chunk.exp()
        sum_exp_chunk = exp_chunk.sum(dim=-1)
        if group is not None:
            torch.distributed.all_reduce(sum_exp_chunk, op=torch.distributed.ReduceOp.SUM, group=group)

        # entropy_i = log(sum_exp) - (exp · logits_norm).sum(-1) / sum_exp
        entropy_chunk = sum_exp_chunk.log() - (exp_chunk * logits_norm_chunk).sum(-1) / sum_exp_chunk

        total = total + (entropy_chunk * mask[sl] / denom[sl]).sum()

    return total


def compute_policy_gradient_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    old_log_probabilities: torch.Tensor,
    advantages: torch.Tensor,
    label_counts: torch.Tensor,
    epsilon_low: float,
    epsilon_high: float,
    logits_scale_factor: float,
    vocab_parallel_group: torch.distributed.ProcessGroup | None,
    compute_entropy: bool = False,
    entropy_chunk_size: int = 4096,
) -> PolicyGradientMetrics:
    (
        old_lp,
        ratio_new_old_mean,
        ratio_new_old_sum,
        ratio_new_old_squared_sum,
        kl_mean,
        clamp_indicator_mean,
        adv_mean,
        max_adv,
        min_adv,
        num_tokens,
    ) = _compute_pg_base_metrics(
        logits,
        target,
        old_log_probabilities,
        advantages,
        label_counts,
        epsilon_low,
        epsilon_high,
        logits_scale_factor,
        vocab_parallel_group,
    )

    entropy = None
    if compute_entropy:
        entropy = compute_chunked_entropy(
            logits,
            target,
            label_counts,
            logits_scale_factor,
            vocab_parallel_group,
            entropy_chunk_size,
        )

    return PolicyGradientMetrics(
        old_logprobs=old_lp,
        ratio_new_old=ratio_new_old_mean,
        ratio_new_old_sum=ratio_new_old_sum,
        ratio_new_old_squared_sum=ratio_new_old_squared_sum,
        kl_new_old=kl_mean,
        clamp_log_ratio_new_old_indicator=clamp_indicator_mean,
        advantage=adv_mean,
        max_advantage=max_adv,
        min_advantage=min_adv,
        num_tokens=num_tokens,
        entropy=entropy,
    )
