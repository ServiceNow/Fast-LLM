import typing

import torch

from fast_llm.core.distributed import ProcessGroup, ReduceOp, all_reduce


class GRPOMetrics(typing.NamedTuple):
    old_logprobs: torch.Tensor
    ratio_new_old: torch.Tensor
    ratio_new_old_sum: torch.Tensor
    ratio_new_old_squared_sum: torch.Tensor
    kl_new_old: torch.Tensor
    clipped_ratio_fraction: torch.Tensor
    advantage: torch.Tensor
    max_advantage: torch.Tensor
    min_advantage: torch.Tensor
    num_tokens: torch.Tensor
    entropy: torch.Tensor | None


def grpo_metrics_core(
    logits_norm: torch.Tensor,  # (*batch, vocab)
    exp_logits: torch.Tensor,  # (*batch, vocab)
    sum_exp_logits: torch.Tensor,  # (*batch,)
    new_log_probs: torch.Tensor,  # (*batch,) — predicted_logits - log(sum_exp_logits)
    advantages: torch.Tensor,  # (*batch,)
    old_log_probabilities: torch.Tensor,  # (*batch,)
    loss_mask: torch.Tensor,  # (*batch,) bool, == target >= 0
    label_counts: torch.Tensor,  # (*batch,) — global per-sequence count broadcast per token
    epsilon_low: float,
    epsilon_high: float,
    group: ProcessGroup | None = None,
    compute_entropy: bool = False,
) -> GRPOMetrics:
    """
    GRPO metric family from a precomputed student softmax and per-token new log-probs. Entropy is the only
    term needing the full vocab axis (a sum over the local slice plus a tensor-parallel all-reduce); every
    other term is per-token.

    This plain (un-compiled) core is shared between the public `compute_grpo_metrics` wrapper and the
    monolithic head-loss kernel, which inlines it inside its own `@torch.compile` boundary — so the metrics
    reuse the loss kernel's softmax instead of recomputing it.
    """
    mask = loss_mask.float()
    masked = mask / label_counts.float().clamp(min=1)

    log_ratio = new_log_probs - old_log_probabilities
    ratio = log_ratio.exp()
    clipped = (ratio < 1.0 - epsilon_low) | (ratio > 1.0 + epsilon_high)
    kl = ratio - log_ratio - 1.0

    neg_inf = advantages.new_full((), float("-inf"))
    pos_inf = advantages.new_full((), float("inf"))

    entropy: torch.Tensor | None = None
    if compute_entropy:
        # exp_logits and logits_norm are local vocab slices — sum over the local slice, then all-reduce
        # across the tensor-parallel group to recover the global E_p[logit_norm] before dividing by the
        # already-global sum_exp_logits.
        weighted_logits_sum = (exp_logits * logits_norm).sum(-1)
        if group is not None:
            all_reduce(weighted_logits_sum, op=ReduceOp.SUM, group=group)
        entropy_per_token = sum_exp_logits.log() - weighted_logits_sum / sum_exp_logits
        entropy = (entropy_per_token * masked).sum()

    return GRPOMetrics(
        old_logprobs=(old_log_probabilities * masked).sum(),
        ratio_new_old=(ratio * masked).sum(),
        ratio_new_old_sum=(ratio * mask).sum(),
        ratio_new_old_squared_sum=(ratio * ratio * mask).sum(),
        kl_new_old=(kl * masked).sum(),
        clipped_ratio_fraction=(clipped.float() * masked).sum(),
        advantage=(advantages * masked).sum(),
        max_advantage=torch.where(loss_mask, advantages, neg_inf).max(),
        min_advantage=torch.where(loss_mask, advantages, pos_inf).min(),
        num_tokens=mask.sum(),
        entropy=entropy,
    )
