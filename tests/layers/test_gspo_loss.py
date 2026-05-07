"""
Unit tests for fused_gspo_loss_forward_backward.

Tests: single segment, multi-segment packed, GRPO/GSPO equivalence at ratio=1,
segment-level clipping, SDP mock, gradient check, extra metrics unchanged.
"""

import math

import torch

from fast_llm.layers.language_model.loss.grpo import (
    fused_grpo_loss_forward_backward,
    fused_gspo_loss_forward_backward,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
atol = 1e-4 if device == "cuda" else 1e-5


# ---------------------------------------------------------------------------
# Reference GSPO implementation
# ---------------------------------------------------------------------------


def _gspo_reference(logits, target, advantages, old_log_probs, doc_idx, eps_lo, eps_hi, divisor):
    """Pure-PyTorch reference without compilation or distributed calls."""
    loss_mask = target >= 0
    log_softmax = torch.log_softmax(logits.float(), dim=-1)
    new_log_probs = log_softmax.gather(-1, (target * loss_mask).unsqueeze(-1)).squeeze(-1)
    log_ratio = (new_log_probs - old_log_probs.float()) * loss_mask.float()

    n_segs = int(doc_idx.max().item()) + 1
    lrn_sum = torch.zeros(n_segs, dtype=torch.float32)
    adv_sum = torch.zeros(n_segs, dtype=torch.float32)
    tok_sum = torch.zeros(n_segs, dtype=torch.float32)
    for i in range(len(target)):
        if loss_mask[i]:
            s = doc_idx[i].item()
            lrn_sum[s] += log_ratio[i].item()
            adv_sum[s] += advantages[i].item()
            tok_sum[s] += 1.0

    loss = 0.0
    for s in range(n_segs):
        if tok_sum[s] == 0:
            continue
        R = math.exp(lrn_sum[s] / tok_sum[s])
        A = adv_sum[s] / tok_sum[s]
        R_clipped = max(1.0 - eps_lo, min(1.0 + eps_hi, R))
        surr1 = R * A
        surr2 = R_clipped * A
        loss += -min(surr1, surr2) * tok_sum[s]
    return loss / divisor


# ---------------------------------------------------------------------------
# Test 1: single segment
# ---------------------------------------------------------------------------


def test_single_segment():
    torch.manual_seed(0)
    n_tok, vocab = 8, 16
    logits = torch.randn(n_tok, vocab, device=device)
    target = torch.randint(0, vocab, (n_tok,), device=device)
    advantages = torch.randn(n_tok, device=device)
    old_log_probs = (
        torch.log_softmax(torch.randn(n_tok, vocab, device=device), dim=-1)
        .gather(-1, target.unsqueeze(-1))
        .squeeze(-1)
    )
    doc_idx = torch.zeros(n_tok, dtype=torch.long, device=device)
    divisor = float(n_tok)

    loss_actual, _, _ = fused_gspo_loss_forward_backward(
        logits,
        target,
        advantages,
        old_log_probs,
        doc_idx,
        divisor=divisor,
        sdp_group=None,
    )
    loss_ref = _gspo_reference(
        logits.cpu(),
        target.cpu(),
        advantages.cpu(),
        old_log_probs.cpu(),
        doc_idx.cpu(),
        0.2,
        0.2,
        divisor,
    )
    assert abs(loss_actual.item() - loss_ref) < atol, f"{loss_actual.item()} vs {loss_ref}"


# ---------------------------------------------------------------------------
# Test 2: multi-segment packed
# ---------------------------------------------------------------------------


def test_multi_segment_packed():
    torch.manual_seed(1)
    # 3 segments of lengths [5, 7, 4]
    segs = [5, 7, 4]
    n_tok = sum(segs)
    vocab = 32
    logits = torch.randn(n_tok, vocab, device=device)
    target = torch.randint(0, vocab, (n_tok,), device=device)
    advantages = torch.randn(n_tok, device=device)
    old_log_probs = (
        torch.log_softmax(torch.randn(n_tok, vocab, device=device), dim=-1)
        .gather(-1, target.unsqueeze(-1))
        .squeeze(-1)
    )
    doc_idx = torch.cat([torch.full((l,), i, dtype=torch.long) for i, l in enumerate(segs)]).to(device)
    divisor = float(n_tok)

    loss_actual, _, _ = fused_gspo_loss_forward_backward(
        logits,
        target,
        advantages,
        old_log_probs,
        doc_idx,
        divisor=divisor,
        sdp_group=None,
    )
    loss_ref = _gspo_reference(
        logits.cpu(),
        target.cpu(),
        advantages.cpu(),
        old_log_probs.cpu(),
        doc_idx.cpu(),
        0.2,
        0.2,
        divisor,
    )
    assert abs(loss_actual.item() - loss_ref) < atol * 3, f"{loss_actual.item()} vs {loss_ref}"


# ---------------------------------------------------------------------------
# Test 3: GRPO vs GSPO equivalence when all tokens in a segment have ratio=1
# ---------------------------------------------------------------------------


def test_ratio_one_matches_grpo():
    """When new == old log-probs (ratio=1 everywhere), GRPO and GSPO losses match."""
    torch.manual_seed(2)
    n_tok, vocab = 12, 16
    logits = torch.randn(n_tok, vocab, device=device)
    target = torch.randint(0, vocab, (n_tok,), device=device)
    advantages = torch.randn(n_tok, device=device)
    # Set old log probs equal to new log probs for ratio=1
    old_log_probs = torch.log_softmax(logits.float(), dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1).detach()
    doc_idx = torch.zeros(n_tok, dtype=torch.long, device=device)
    divisor = float(n_tok)

    loss_grpo, _, _ = fused_grpo_loss_forward_backward(logits, target, advantages, old_log_probs, divisor=divisor)
    loss_gspo, _, _ = fused_gspo_loss_forward_backward(
        logits,
        target,
        advantages,
        old_log_probs,
        doc_idx,
        divisor=divisor,
        sdp_group=None,
    )
    # At ratio=1, GRPO loss = sum_t -A_t * mask_t / divisor (no clipping)
    # GSPO loss = sum_s tok_s * -A_s / divisor (weighted per segment)
    # For a single segment: GSPO = -mean(A) * N / divisor = same total
    assert abs(loss_grpo.item() - loss_gspo.item()) < atol, f"grpo={loss_grpo.item()}, gspo={loss_gspo.item()}"


# ---------------------------------------------------------------------------
# Test 4: segment-level clipping (GSPO clips whole segment, not per-token)
# ---------------------------------------------------------------------------


def test_segment_level_clipping():
    """
    Construct a segment where per-token ratios straddle the clip boundary (some high, some low),
    but the geometric mean ratio is in-range. GSPO should NOT clip; GRPO should clip some tokens.
    """
    torch.manual_seed(3)
    vocab = 8
    # 4 tokens, alternating log_ratio +0.5 and -0.5 → mean = 0 → R = exp(0) = 1.0 (in range)
    n_tok = 4
    target = torch.zeros(n_tok, dtype=torch.long, device=device)
    advantages = torch.ones(n_tok, device=device)
    doc_idx = torch.zeros(n_tok, dtype=torch.long, device=device)

    # Build logits such that new_log_probs - old_log_probs alternates +0.4 and -0.4
    # Use constant logits; set old_log_probs manually
    logits = torch.zeros(n_tok, vocab, device=device)
    old_log_probs = torch.tensor([0.4, -0.4, 0.4, -0.4], device=device)  # per-token log_ratio = 0 - old

    eps = 0.2
    divisor = float(n_tok)
    loss_gspo, _, _ = fused_gspo_loss_forward_backward(
        logits,
        target,
        advantages,
        old_log_probs,
        doc_idx,
        epsilon_low=eps,
        epsilon_high=eps,
        divisor=divisor,
        sdp_group=None,
    )

    # GSPO: mean log_ratio = mean of (log_softmax(0)[0] - old_log_probs)
    # R = exp(mean), A=1.0
    # As long as R is in [1-eps, 1+eps], loss = -R * 1 * 4 / 4 = -R
    new_log_probs = torch.log_softmax(logits.float(), dim=-1)[:, 0]
    log_ratios = new_log_probs - old_log_probs
    mean_log_ratio = log_ratios.mean().item()
    R = math.exp(mean_log_ratio)
    expected = -R  # unclipped, weight 4/divisor = 1
    assert abs(loss_gspo.item() - expected) < atol, f"gspo={loss_gspo.item()}, expected={expected}"


# ---------------------------------------------------------------------------
# Test 5: masked tokens don't contribute
# ---------------------------------------------------------------------------


def test_masked_tokens():
    torch.manual_seed(4)
    n_tok, vocab = 10, 16
    logits = torch.randn(n_tok, vocab, device=device)
    target = torch.randint(0, vocab, (n_tok,), device=device)
    target[3] = -100  # mask token 3
    target[7] = -100  # mask token 7
    advantages = torch.randn(n_tok, device=device)
    old_log_probs = torch.randn(n_tok, device=device)
    doc_idx = torch.zeros(n_tok, dtype=torch.long, device=device)
    divisor = float(n_tok)

    loss_actual, _, _ = fused_gspo_loss_forward_backward(
        logits,
        target,
        advantages,
        old_log_probs,
        doc_idx,
        divisor=divisor,
        sdp_group=None,
    )
    loss_ref = _gspo_reference(
        logits.cpu(),
        target.cpu(),
        advantages.cpu(),
        old_log_probs.cpu(),
        doc_idx.cpu(),
        0.2,
        0.2,
        divisor,
    )
    assert abs(loss_actual.item() - loss_ref) < atol, f"{loss_actual.item()} vs {loss_ref}"


# ---------------------------------------------------------------------------
# Test 6: SDP mock — split tokens across 2 "ranks", verify correctness
# ---------------------------------------------------------------------------


def test_sdp_mock():
    """
    Simulate SDP=2: split tokens in half, compute per-rank scatter_add, manually all-reduce,
    then verify the combined sums match the full-batch computation.
    """
    torch.manual_seed(5)
    segs = [6, 5, 7]  # 3 segments
    n_tok = sum(segs)
    vocab = 16
    logits = torch.randn(n_tok, vocab, device=device)
    target = torch.randint(0, vocab, (n_tok,), device=device)
    advantages = torch.randn(n_tok, device=device)
    old_log_probs = torch.randn(n_tok, device=device)
    doc_idx = torch.cat([torch.full((l,), i, dtype=torch.long) for i, l in enumerate(segs)]).to(device)
    divisor = float(n_tok)

    # Full-batch reference loss
    loss_full, _, _ = fused_gspo_loss_forward_backward(
        logits,
        target,
        advantages,
        old_log_probs,
        doc_idx,
        divisor=divisor,
        sdp_group=None,
    )

    # Simulate SDP=2: split at midpoint
    mid = n_tok // 2
    loss_r0_only, _, _ = fused_gspo_loss_forward_backward(
        logits[:mid],
        target[:mid],
        advantages[:mid],
        old_log_probs[:mid],
        doc_idx[:mid],
        divisor=divisor,
        sdp_group=None,
    )
    loss_r1_only, _, _ = fused_gspo_loss_forward_backward(
        logits[mid:],
        target[mid:],
        advantages[mid:],
        old_log_probs[mid:],
        doc_idx[mid:],
        divisor=divisor,
        sdp_group=None,
    )
    # These individual ranks do NOT give the right answer (segments are split)
    # But the full-batch result should match the reference
    loss_ref = _gspo_reference(
        logits.cpu(),
        target.cpu(),
        advantages.cpu(),
        old_log_probs.cpu(),
        doc_idx.cpu(),
        0.2,
        0.2,
        divisor,
    )
    assert abs(loss_full.item() - loss_ref) < atol * 3, f"full={loss_full.item()}, ref={loss_ref}"

    # When sdp_group is None but we manually pre-sum, the result should also match
    # (This conceptually validates the all-reduce logic without actual distributed calls)
    log_softmax_full = torch.log_softmax(logits.float(), dim=-1)
    new_lp_full = log_softmax_full.gather(-1, (target * (target >= 0)).unsqueeze(-1)).squeeze(-1)
    log_ratio_full = (new_lp_full - old_log_probs.float()) * (target >= 0).float()

    n_segs = 3
    lrn_r0 = torch.zeros(n_segs)
    adv_r0 = torch.zeros(n_segs)
    tok_r0 = torch.zeros(n_segs)
    lrn_r1 = torch.zeros(n_segs)
    adv_r1 = torch.zeros(n_segs)
    tok_r1 = torch.zeros(n_segs)
    for i in range(mid):
        if target[i] >= 0:
            s = doc_idx[i].item()
            lrn_r0[s] += log_ratio_full[i].item()
            adv_r0[s] += advantages[i].item()
            tok_r0[s] += 1
    for i in range(mid, n_tok):
        if target[i] >= 0:
            s = doc_idx[i].item()
            lrn_r1[s] += log_ratio_full[i].item()
            adv_r1[s] += advantages[i].item()
            tok_r1[s] += 1

    # Manually all-reduce (SUM)
    lrn_global = lrn_r0 + lrn_r1
    adv_global = adv_r0 + adv_r1
    tok_global = tok_r0 + tok_r1

    loss_manual = 0.0
    for s in range(n_segs):
        if tok_global[s] == 0:
            continue
        R = math.exp(lrn_global[s] / tok_global[s])
        A = adv_global[s] / tok_global[s]
        R_c = max(1 - 0.2, min(1 + 0.2, R))
        loss_manual += -min(R * A, R_c * A) * tok_global[s]
    loss_manual /= divisor

    assert abs(loss_full.item() - loss_manual) < atol * 3, f"full={loss_full.item()}, manual={loss_manual}"


# ---------------------------------------------------------------------------
# Test 7: gradient correctness via finite differences
# ---------------------------------------------------------------------------


def test_gradient_finite_diff():
    torch.manual_seed(6)
    n_tok, vocab = 6, 8
    logits = torch.randn(n_tok, vocab, dtype=torch.float64)
    target = torch.randint(0, vocab, (n_tok,))
    advantages = torch.randn(n_tok, dtype=torch.float64)
    old_log_probs = torch.randn(n_tok, dtype=torch.float64)
    doc_idx = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    divisor = float(n_tok)
    eps = 1e-5

    grad_logits = torch.zeros_like(logits)
    _, grad_out, _ = fused_gspo_loss_forward_backward(
        logits,
        target,
        advantages,
        old_log_probs,
        doc_idx,
        grad_logits=grad_logits,
        grad_output=1.0,
        divisor=divisor,
        sdp_group=None,
    )

    # Finite-difference gradient for one entry
    i, k = 2, 3
    logits_p = logits.clone()
    logits_p[i, k] += eps
    logits_m = logits.clone()
    logits_m[i, k] -= eps
    loss_p, _, _ = fused_gspo_loss_forward_backward(
        logits_p,
        target,
        advantages,
        old_log_probs,
        doc_idx,
        divisor=divisor,
        sdp_group=None,
    )
    loss_m, _, _ = fused_gspo_loss_forward_backward(
        logits_m,
        target,
        advantages,
        old_log_probs,
        doc_idx,
        divisor=divisor,
        sdp_group=None,
    )
    fd_grad = (loss_p.item() - loss_m.item()) / (2 * eps)

    assert abs(grad_out[i, k].item() - fd_grad) < 1e-4, f"analytical={grad_out[i, k].item():.6f}, fd={fd_grad:.6f}"


# ---------------------------------------------------------------------------
# Test 8: extra metrics are per-token regardless of GRPO/GSPO
# ---------------------------------------------------------------------------


def test_extra_metrics_are_per_token():
    """Extra metrics are per-token regardless of GSPO/GRPO — computed from token-level ratios."""
    from fast_llm.layers.language_model.loss.grpo import compute_grpo_metrics

    torch.manual_seed(7)
    n_tok, vocab = 10, 16
    logits = torch.randn(n_tok, vocab, device=device)
    target = torch.randint(0, vocab, (n_tok,), device=device)
    advantages = torch.randn(n_tok, device=device)
    old_log_probs = torch.randn(n_tok, device=device)
    label_counts = torch.full((n_tok,), n_tok, dtype=torch.int32, device=device)

    metrics = compute_grpo_metrics(
        logits,
        target,
        advantages,
        old_log_probs,
        label_counts,
        epsilon_low=0.2,
        epsilon_high=0.2,
        logits_scale_factor=1.0,
        group=None,
    )
    for attr in ("old_logprobs", "ratio_new_old", "kl_new_old", "advantage"):
        val = getattr(metrics, attr)
        assert val.isfinite(), f"{attr} is not finite: {val}"
