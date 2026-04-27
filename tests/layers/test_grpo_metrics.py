"""
Unit tests for pg_metrics.py — PolicyGradientMetrics computation.

All tests run on CPU (or GPU if available) without distributed communication
(vocab_parallel_group=None).  Distributed reduction is exercised conceptually
via the mock-SDP and mock-vocab-parallel sections.
"""

import math

import torch

from fast_llm.layers.language_model.loss.pg_metrics import (
    compute_chunked_entropy,
    compute_policy_gradient_metrics,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"


def _manual_metrics(logits, target, old_log_probs, advantages, label_counts, eps_lo, eps_hi):
    """Reference implementation (pure PyTorch, no compilation)."""
    loss_mask = target >= 0
    mask = loss_mask.float()
    denom = label_counts.float().clamp(min=1)

    log_softmax = torch.log_softmax(logits.float(), dim=-1)
    new_log_probs = log_softmax.gather(-1, (target * loss_mask).unsqueeze(-1)).squeeze(-1)

    log_ratio = new_log_probs - old_log_probs.float()
    ratio = log_ratio.exp()
    clipped = (ratio < 1.0 - eps_lo) | (ratio > 1.0 + eps_hi)
    kl = ratio - log_ratio - 1.0

    old_lp = (old_log_probs.float() * mask / denom).sum()
    ratio_mean = (ratio * mask / denom).sum()
    ratio_sum = (ratio * mask).sum()
    ratio_sq_sum = (ratio * ratio * mask).sum()
    kl_mean = (kl * mask / denom).sum()
    clamp_mean = (clipped.float() * mask / denom).sum()
    adv_mean = (advantages.float() * mask / denom).sum()
    max_adv = advantages.float()[loss_mask].max()
    min_adv = advantages.float()[loss_mask].min()
    num_tokens = mask.sum()

    probs = log_softmax.exp()
    entropy_per_token = -(probs * log_softmax).sum(-1)
    entropy_mean = (entropy_per_token * mask / denom).sum()

    return dict(
        old_logprobs=old_lp,
        ratio=ratio_mean,
        ratio_sum=ratio_sum,
        ratio_sq_sum=ratio_sq_sum,
        kl_new_old=kl_mean,
        clamp_frac=clamp_mean,
        advantage=adv_mean,
        max_advantage=max_adv,
        min_advantage=min_adv,
        num_tokens=num_tokens,
        entropy=entropy_mean,
    )


def _run_metrics(logits, target, old_log_probs, advantages, label_counts, eps_lo=0.2, eps_hi=0.2, chunk_size=4096):
    return compute_policy_gradient_metrics(
        logits,
        target,
        old_log_probs,
        advantages,
        label_counts,
        eps_lo,
        eps_hi,
        logits_scale_factor=1.0,
        vocab_parallel_group=None,
        compute_entropy=True,
        entropy_chunk_size=chunk_size,
    )


def _assert_close(a, b, msg="", atol=1e-5):
    assert abs(a.item() - b.item()) < atol, f"{msg}: got {a.item():.8f}, expected {b.item():.8f}"


# ---------------------------------------------------------------------------
# 1. Single sequence — all metrics match manual computation
# ---------------------------------------------------------------------------


def test_single_sequence_all_metrics():
    torch.manual_seed(0)
    seq_len, vocab = 12, 8
    logits = torch.randn(seq_len, vocab, device=device)
    target = torch.randint(0, vocab, (seq_len,), device=device)
    old_log_probs = torch.randn(seq_len, device=device) - 3.0
    advantages = torch.randn(seq_len, device=device)
    label_counts = torch.full((seq_len,), seq_len, device=device)  # all tokens in one seq

    ref = _manual_metrics(logits, target, old_log_probs, advantages, label_counts, 0.2, 0.2)
    got = _run_metrics(logits, target, old_log_probs, advantages, label_counts)

    for key in ref:
        _assert_close(getattr(got, key), ref[key], msg=key)


# ---------------------------------------------------------------------------
# 2. Packed multi-sequence — per-sequence normalization
# ---------------------------------------------------------------------------


def test_packed_multi_sequence():
    """
    Three sequences of lengths [4, 6, 5] packed into one flat batch (15 tokens).
    label_counts broadcasts the global per-sequence count.
    """
    torch.manual_seed(1)
    lengths = [4, 6, 5]
    total = sum(lengths)
    vocab = 10

    logits = torch.randn(total, vocab, device=device)
    target = torch.randint(0, vocab, (total,), device=device)
    old_log_probs = torch.randn(total, device=device) - 2.0
    advantages = torch.randn(total, device=device)
    label_counts = torch.tensor([l for l in lengths for _ in range(l)], dtype=torch.long, device=device)

    ref = _manual_metrics(logits, target, old_log_probs, advantages, label_counts, 0.2, 0.2)
    got = _run_metrics(logits, target, old_log_probs, advantages, label_counts)

    for key in ref:
        _assert_close(getattr(got, key), ref[key], msg=key)


# ---------------------------------------------------------------------------
# 3. Masked tokens — masked-out tokens must not contribute
# ---------------------------------------------------------------------------


def test_masked_tokens_do_not_contribute():
    """
    A batch where half the tokens are masked (target=-100).
    Metrics computed on full batch should equal metrics on unmasked subset only.
    """
    torch.manual_seed(2)
    seq_len, vocab = 20, 16
    logits = torch.randn(seq_len, vocab, device=device)
    target_full = torch.randint(0, vocab, (seq_len,), device=device)

    # mask the first half
    mask_bool = torch.ones(seq_len, dtype=torch.bool, device=device)
    mask_bool[: seq_len // 2] = False
    target_masked = torch.where(mask_bool, target_full, torch.full_like(target_full, -100))

    old_log_probs = torch.randn(seq_len, device=device) - 2.0
    advantages = torch.randn(seq_len, device=device)
    label_counts = torch.full((seq_len,), mask_bool.sum().item(), device=device)

    # reference: only the unmasked slice
    half = seq_len // 2
    ref = _manual_metrics(
        logits[half:],
        target_full[half:],
        old_log_probs[half:],
        advantages[half:],
        label_counts[half:],
        0.2,
        0.2,
    )
    got = _run_metrics(logits, target_masked, old_log_probs, advantages, label_counts)

    for key in ref:
        _assert_close(getattr(got, key), ref[key], msg=f"masked_{key}")


# ---------------------------------------------------------------------------
# 4. Clamp fraction — known ratios → known clamp_frac
# ---------------------------------------------------------------------------


def test_clamp_fraction_known():
    """
    Construct logits so that probability_ratio is exactly known.
    With eps_lo=0.1, eps_hi=0.1 and 5 tokens:
      2 tokens outside the clip range, 3 inside → clamp_frac = 2/5.
    """
    seq_len, vocab = 5, 4
    # uniform logits → probabilities = 1/vocab for any label
    logits = torch.zeros(seq_len, vocab, device=device)
    target = torch.zeros(seq_len, dtype=torch.long, device=device)  # all label=0
    # p_new = 1/4, so new_log_prob = log(0.25)
    new_lp = math.log(1.0 / vocab)

    # Set old_log_probs so ratio = exp(new - old) is known per token
    # ratios: [0.85, 1.0, 1.05, 1.2, 0.75]  (eps=0.1 → clip outside (0.9, 1.1))
    # clipped: True, False, False, True, True → 3 clipped
    ratios = torch.tensor([0.85, 1.0, 1.05, 1.2, 0.75], device=device)
    old_log_probs = torch.full((seq_len,), new_lp, device=device) - ratios.log()

    advantages = torch.ones(seq_len, device=device)
    label_counts = torch.full((seq_len,), seq_len, device=device)

    got = _run_metrics(logits, target, old_log_probs, advantages, label_counts, eps_lo=0.1, eps_hi=0.1)

    expected_clamp_frac = 3.0 / seq_len  # 3 out of 5 tokens clipped
    _assert_close(got.clamp_frac, torch.tensor(expected_clamp_frac), msg="clamp_frac", atol=1e-5)


# ---------------------------------------------------------------------------
# 5. Entropy correctness — small vocab, verify chunked vs reference
# ---------------------------------------------------------------------------


def test_entropy_matches_manual():
    """Small vocab so we can compute entropy exactly by hand."""
    torch.manual_seed(3)
    seq_len, vocab = 8, 6
    logits = torch.randn(seq_len, vocab, device=device)
    target = torch.randint(0, vocab, (seq_len,), device=device)
    old_log_probs = torch.randn(seq_len, device=device) - 2.0
    advantages = torch.randn(seq_len, device=device)
    label_counts = torch.full((seq_len,), seq_len, device=device)

    # Reference entropy
    ref = _manual_metrics(logits, target, old_log_probs, advantages, label_counts, 0.2, 0.2)

    # Test with different chunk sizes (including chunk_size=1 and chunk_size>seq_len)
    for chunk_size in (1, 3, seq_len, seq_len + 10):
        got = _run_metrics(logits, target, old_log_probs, advantages, label_counts, chunk_size=chunk_size)
        _assert_close(got.entropy, ref["entropy"], msg=f"entropy chunk_size={chunk_size}")


# ---------------------------------------------------------------------------
# 6. Mock SDP — split batch in half, verify sum/max/min consistency
# ---------------------------------------------------------------------------


def test_mock_sdp_split():
    """
    Simulate two SDP ranks each holding half the batch.
    SUM metrics on full batch == sum of the two halves.
    MAX/MIN metrics on full batch == max/min of the two halves.
    """
    torch.manual_seed(4)
    seq_len, vocab = 18, 12
    logits = torch.randn(seq_len, vocab, device=device)
    target = torch.randint(0, vocab, (seq_len,), device=device)
    old_log_probs = torch.randn(seq_len, device=device) - 2.0
    advantages = torch.randn(seq_len, device=device)
    label_counts = torch.full((seq_len,), seq_len // 2, device=device)

    half = seq_len // 2

    full = _run_metrics(logits, target, old_log_probs, advantages, label_counts)
    lo = _run_metrics(logits[:half], target[:half], old_log_probs[:half], advantages[:half], label_counts[:half])
    hi = _run_metrics(logits[half:], target[half:], old_log_probs[half:], advantages[half:], label_counts[half:])

    # SUM metrics accumulate across both halves
    for attr in (
        "old_logprobs",
        "ratio",
        "ratio_sum",
        "ratio_sq_sum",
        "kl_new_old",
        "clamp_frac",
        "advantage",
        "num_tokens",
    ):
        combined = getattr(lo, attr) + getattr(hi, attr)
        _assert_close(getattr(full, attr), combined, msg=f"sdp_{attr}")

    # MAX/MIN are extrema across both halves
    _assert_close(full.max_advantage, torch.max(lo.max_advantage, hi.max_advantage), msg="sdp_max_adv")
    _assert_close(full.min_advantage, torch.min(lo.min_advantage, hi.min_advantage), msg="sdp_min_adv")

    # Entropy (SUM metric)
    _assert_close(full.entropy, lo.entropy + hi.entropy, msg="sdp_entropy")


# ---------------------------------------------------------------------------
# 7. Mock vocab-parallel entropy — split logits along vocab dim
# ---------------------------------------------------------------------------


def test_mock_vocab_parallel_entropy():
    """
    Simulate 2-way vocab-parallel: split logits along the vocab dim.
    Each "rank" computes a partial softmax; the global entropy should
    match single-rank computation (all-reduce simulated manually).
    """
    torch.manual_seed(5)
    seq_len, vocab = 10, 16
    logits = torch.randn(seq_len, vocab, device=device)
    target = torch.randint(0, vocab, (seq_len,), device=device)
    label_counts = torch.full((seq_len,), seq_len, device=device)
    mask = torch.ones(seq_len, dtype=torch.bool, device=device)

    # Reference: single rank, full vocab
    ref_entropy = compute_chunked_entropy(
        logits,
        target,
        label_counts,
        logits_scale_factor=1.0,
        group=None,
        chunk_size=seq_len,
    )

    # Simulate vocab-parallel: split vocab into [0:8] and [8:16]
    # Both ranks see the same sequence but different vocab shards.
    # global max is needed for numerical stability:
    logits_max = logits.float().max(dim=-1).values  # (seq_len,)

    half_v = vocab // 2
    logits_lo = logits[:, :half_v]
    logits_hi = logits[:, half_v:]

    # Per rank: compute local sum_exp relative to global max
    exp_lo = (logits_lo.float() - logits_max.unsqueeze(-1)).exp()
    exp_hi = (logits_hi.float() - logits_max.unsqueeze(-1)).exp()
    sum_exp_lo = exp_lo.sum(-1)
    sum_exp_hi = exp_hi.sum(-1)
    sum_exp_global = sum_exp_lo + sum_exp_hi  # simulated SUM all-reduce

    logits_norm_lo = logits_lo.float() - logits_max.unsqueeze(-1)
    logits_norm_hi = logits_hi.float() - logits_max.unsqueeze(-1)

    # entropy = log(sum_exp_global) - (exp · logits_norm).sum(-1) / sum_exp_global
    dot_lo = (exp_lo * logits_norm_lo).sum(-1)
    dot_hi = (exp_hi * logits_norm_hi).sum(-1)
    dot_global = dot_lo + dot_hi  # simulated SUM all-reduce

    entropy_per_tok = sum_exp_global.log() - dot_global / sum_exp_global
    denom = label_counts.float().clamp(min=1)
    manual_vp_entropy = (entropy_per_tok * mask.float() / denom).sum()

    _assert_close(ref_entropy, manual_vp_entropy, msg="vocab_parallel_entropy")


# ---------------------------------------------------------------------------
# 8. Consistency with new_logprobs_mean normalization
# ---------------------------------------------------------------------------


def test_old_logprobs_normalization_matches_new_logprobs_pattern():
    """
    old_logprobs metric uses the same normalization as new_logprobs_mean:
      sum(value * mask / label_counts.clamp(1))
    Verify that when old == new (zero perturbation), old_logprobs == new_logprobs_mean.
    """
    torch.manual_seed(6)
    seq_len, vocab = 14, 20
    logits = torch.randn(seq_len, vocab, device=device)
    target = torch.randint(0, vocab, (seq_len,), device=device)
    label_counts = torch.full((seq_len,), seq_len, device=device)

    # old_log_probs = actual new_log_probs (no perturbation)
    with torch.no_grad():
        new_lp = torch.log_softmax(logits.float(), dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)

    old_log_probs = new_lp.detach()
    advantages = torch.randn(seq_len, device=device)

    got = _run_metrics(logits, target, old_log_probs, advantages, label_counts)

    # new_logprobs_mean pattern (from grpo.py fused function)
    mask = (target >= 0).float()
    denom = label_counts.float().clamp(min=1)
    expected_new_lp_mean = (new_lp * mask / denom).sum()

    _assert_close(got.old_logprobs, expected_new_lp_mean, msg="old_logprobs_vs_new_logprobs_mean")

    # ratio should be ~1 everywhere, kl should be ~0
    _assert_close(got.ratio, torch.tensor(1.0) * (mask / denom).sum(), msg="ratio_at_1", atol=1e-4)
    _assert_close(got.kl_new_old, torch.zeros(()), msg="kl_at_zero", atol=1e-4)
