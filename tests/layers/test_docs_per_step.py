"""
Unit tests for docs_per_step / normalize_by_documents features.

Covers:
  1. Divisor scaling in fused_grpo_loss_forward_backward and fused_gspo_loss_forward_backward
  2. normalize_by_documents flag in LanguageModelGRPOLoss (GRPO and GSPO policy_loss)
  3. Schedule._eff_depth_first / _eff_sequential_micro_batches / _eff_num_inputs properties
  4. Trainer._prefetch_to_doc_target accumulation logic
"""

import dataclasses
import types

import pytest
import torch

from fast_llm.engine.schedule.config import ScheduleConfig
from fast_llm.engine.schedule.schedule import Schedule
from fast_llm.layers.language_model.config import LanguageModelKwargs
from fast_llm.layers.language_model.loss.config import LanguageModelGRPOLossConfig, LanguageModelLossKwargs
from fast_llm.layers.language_model.loss.grpo import (
    fused_grpo_loss_forward_backward,
    fused_gspo_loss_forward_backward,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
_atol = 1e-4 if device == "cuda" else 1e-5


# ---------------------------------------------------------------------------
# 1. Divisor-scaling correctness in raw kernels
# ---------------------------------------------------------------------------


def test_grpo_divisor_scales_loss():
    """Halving the divisor should double the loss."""
    torch.manual_seed(10)
    n_tok, vocab = 16, 32
    logits = torch.randn(n_tok, vocab, device=device)
    target = torch.randint(0, vocab, (n_tok,), device=device)
    advantages = torch.randn(n_tok, device=device)
    old_lp = torch.randn(n_tok, device=device) - 2.0

    d1 = float(n_tok)
    d2 = float(n_tok) * 2

    loss1, _, _ = fused_grpo_loss_forward_backward(logits, target, advantages, old_lp, divisor=d1)
    loss2, _, _ = fused_grpo_loss_forward_backward(logits, target, advantages, old_lp, divisor=d2)

    assert (
        abs(loss1.item() - 2.0 * loss2.item()) < _atol * 10
    ), f"Expected loss(d1) ≈ 2*loss(d2), got {loss1.item():.6f} vs {2*loss2.item():.6f}"


def test_gspo_divisor_scales_loss():
    """Halving the divisor should double the GSPO loss."""
    torch.manual_seed(11)
    n_tok, vocab = 12, 16
    logits = torch.randn(n_tok, vocab, device=device)
    target = torch.randint(0, vocab, (n_tok,), device=device)
    advantages = torch.randn(n_tok, device=device)
    old_lp = torch.randn(n_tok, device=device) - 2.0
    doc_idx = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2], dtype=torch.long, device=device)

    d1 = float(n_tok)
    d2 = float(n_tok) * 2

    loss1, _, _ = fused_gspo_loss_forward_backward(
        logits, target, advantages, old_lp, doc_idx, divisor=d1, sdp_group=None
    )
    loss2, _, _ = fused_gspo_loss_forward_backward(
        logits, target, advantages, old_lp, doc_idx, divisor=d2, sdp_group=None
    )

    assert (
        abs(loss1.item() - 2.0 * loss2.item()) < _atol * 10
    ), f"Expected loss(d1) ≈ 2*loss(d2), got {loss1.item():.6f} vs {2*loss2.item():.6f}"


# ---------------------------------------------------------------------------
# 2. normalize_by_documents flag in LanguageModelGRPOLoss
# ---------------------------------------------------------------------------


def _make_grpo_loss(normalize_by_documents: bool, policy_loss: str = "grpo"):
    """Instantiate a LanguageModelGRPOLoss with minimal (single-GPU) DistributedConfig."""
    from fast_llm.engine.distributed.config import DistributedConfig
    from fast_llm.layers.language_model.loss.grpo import LanguageModelGRPOLoss

    dist_cfg = DistributedConfig()
    cfg = LanguageModelGRPOLossConfig(
        normalize_by_documents=normalize_by_documents,
        policy_loss=policy_loss,
    )
    return LanguageModelGRPOLoss(cfg, dist_cfg, name="grpo", prediction_distance=1, prediction_heads=1)


def _make_grpo_kwargs(logits, target, advantages, old_lp, doc_idx, n_labels, n_docs):
    """Build the kwargs dict expected by LanguageModelGRPOLoss._forward_backward."""
    return {
        LanguageModelLossKwargs.labels: [target],
        LanguageModelLossKwargs.advantages: [advantages],
        LanguageModelLossKwargs.old_log_probabilities: [old_lp],
        LanguageModelLossKwargs.label_counts: [torch.full_like(target, n_labels, dtype=torch.int32)],
        LanguageModelKwargs.num_labels_in_batch: [n_labels],
        LanguageModelKwargs.num_documents_in_batch: n_docs,
        LanguageModelKwargs.document_index: [doc_idx],
    }


def test_normalize_by_documents_grpo():
    """normalize_by_documents=True → divisor=n_docs; False → divisor=n_labels.

    With n_docs ≠ n_labels, loss ratio must equal n_labels / n_docs.
    """
    torch.manual_seed(20)
    n_tok, vocab = 12, 16
    n_docs, n_labels = 3, n_tok

    logits = torch.randn(n_tok, vocab, device=device)
    target = torch.randint(0, vocab, (n_tok,), device=device)
    advantages = torch.randn(n_tok, device=device)
    old_lp = torch.randn(n_tok, device=device) - 2.0
    doc_idx = torch.zeros(n_tok, dtype=torch.long, device=device)

    kwargs = _make_grpo_kwargs(logits, target, advantages, old_lp, doc_idx, n_labels, n_docs)

    loss_by_tokens, _ = _make_grpo_loss(normalize_by_documents=False)._forward_backward(logits, kwargs)
    loss_by_docs, _ = _make_grpo_loss(normalize_by_documents=True)._forward_backward(logits, kwargs)

    expected_ratio = float(n_labels) / float(n_docs)
    actual_ratio = loss_by_docs.item() / loss_by_tokens.item()
    assert (
        abs(actual_ratio - expected_ratio) < 1e-4
    ), f"Expected loss_docs/loss_tokens ≈ {expected_ratio:.4f}, got {actual_ratio:.4f}"


def test_normalize_by_documents_gspo():
    """Same test for GSPO policy_loss."""
    torch.manual_seed(21)
    n_tok, vocab = 12, 16
    n_docs, n_labels = 3, n_tok

    logits = torch.randn(n_tok, vocab, device=device)
    target = torch.randint(0, vocab, (n_tok,), device=device)
    advantages = torch.randn(n_tok, device=device)
    old_lp = torch.randn(n_tok, device=device) - 2.0
    # 3 equal segments → n_docs=3
    doc_idx = torch.cat([torch.full((n_tok // n_docs,), i, dtype=torch.long) for i in range(n_docs)]).to(device)

    kwargs = _make_grpo_kwargs(logits, target, advantages, old_lp, doc_idx, n_labels, n_docs)

    loss_by_tokens, _ = _make_grpo_loss(normalize_by_documents=False, policy_loss="gspo")._forward_backward(
        logits, kwargs
    )
    loss_by_docs, _ = _make_grpo_loss(normalize_by_documents=True, policy_loss="gspo")._forward_backward(
        logits, kwargs
    )

    expected_ratio = float(n_labels) / float(n_docs)
    actual_ratio = loss_by_docs.item() / loss_by_tokens.item()
    assert (
        abs(actual_ratio - expected_ratio) < 1e-4
    ), f"Expected loss_docs/loss_tokens ≈ {expected_ratio:.4f}, got {actual_ratio:.4f}"


# ---------------------------------------------------------------------------
# 3. Schedule._eff_* properties
# ---------------------------------------------------------------------------


def _make_bare_schedule(depth_first: int, breadth_first: int, splits: int, override: int | None) -> Schedule:
    """Create a Schedule with __init__ bypassed to test the _eff_* properties only."""
    config = ScheduleConfig(
        depth_first_micro_batches=depth_first,
        breadth_first_micro_batches=breadth_first,
        micro_batch_splits=splits,
    )
    sched = object.__new__(Schedule)
    # Minimal attributes used by the three _eff_* properties.
    object.__setattr__(sched, "_config", config)
    object.__setattr__(sched, "_depth_first_override", override)
    # samples_per_batch also needs _distributed_config.batch_data_parallel
    fake_distributed = types.SimpleNamespace(batch_data_parallel=1)
    object.__setattr__(sched, "_distributed_config", fake_distributed)
    return sched


def test_schedule_eff_properties_no_override():
    sched = _make_bare_schedule(depth_first=4, breadth_first=2, splits=3, override=None)
    assert sched._eff_depth_first == 4
    assert sched._eff_sequential_micro_batches == 8  # 4 * 2
    assert sched._eff_num_inputs == 24  # 8 * 3
    assert sched.samples_per_batch == 8  # 8 * dp=1


def test_schedule_eff_properties_with_override():
    sched = _make_bare_schedule(depth_first=4, breadth_first=2, splits=3, override=7)
    assert sched._eff_depth_first == 7  # override wins
    assert sched._eff_sequential_micro_batches == 14  # 7 * 2
    assert sched._eff_num_inputs == 42  # 14 * 3
    assert sched.samples_per_batch == 14  # 14 * dp=1


def test_schedule_eff_properties_override_equals_config():
    """Override equal to config value → same result as no override."""
    sched_no = _make_bare_schedule(depth_first=3, breadth_first=2, splits=1, override=None)
    sched_yes = _make_bare_schedule(depth_first=3, breadth_first=2, splits=1, override=3)
    assert sched_no._eff_depth_first == sched_yes._eff_depth_first
    assert sched_no._eff_sequential_micro_batches == sched_yes._eff_sequential_micro_batches
    assert sched_no._eff_num_inputs == sched_yes._eff_num_inputs


def test_schedule_samples_per_batch_uses_eff():
    """samples_per_batch should scale with _eff_sequential, not config.sequential."""
    sched = _make_bare_schedule(depth_first=2, breadth_first=2, splits=1, override=5)
    # Config says depth_first=2 → sequential=4; override=5 → eff_sequential=10
    assert sched._eff_sequential_micro_batches == 10
    assert sched.samples_per_batch == 10  # dp=1


# ---------------------------------------------------------------------------
# 4. _prefetch_to_doc_target accumulation logic
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _FakeMicrobatch:
    """Stub for a single split of one microbatch."""

    num_documents: int
    num_documents_in_batch: int | None = None

    @classmethod
    def share_batch_data(cls, inputs, distributed):
        """Mimic TokenModelInput.share_batch_data with group=None (single process)."""
        if inputs[0].num_documents_in_batch is None:
            total = sum(inp.num_documents for inp in inputs)
            for inp in inputs:
                inp.num_documents_in_batch = total


def _fake_iterator(doc_counts: list[int]):
    """Yield [_FakeMicrobatch(n)] for each n in doc_counts."""
    for n in doc_counts:
        yield [_FakeMicrobatch(num_documents=n)]


class _StubTrainer:
    """Concrete stub that exposes only the interface _prefetch_to_doc_target needs."""

    # Borrow the method directly so it runs against this stub's attributes.
    from fast_llm.engine.training.trainer import Trainer as _Trainer

    _prefetch_to_doc_target = _Trainer._prefetch_to_doc_target


def _make_fake_trainer(docs_per_step: int, bfmb: int = 1):
    """Create a _StubTrainer with the attributes _prefetch_to_doc_target reads."""
    schedule_cfg = types.SimpleNamespace(
        docs_per_step=docs_per_step,
        breadth_first_micro_batches=bfmb,
    )
    config = types.SimpleNamespace(schedule=schedule_cfg)
    distributed = types.SimpleNamespace(batch_data_group=None)

    trainer = _StubTrainer()
    trainer._config = config
    trainer._distributed = distributed
    return trainer


def test_prefetch_stops_at_target():
    """Buffer should stop growing once cumulative docs ≥ docs_per_step."""
    trainer = _make_fake_trainer(docs_per_step=6, bfmb=1)
    # Each microbatch has 2 docs; need ≥6 → expect 3 microbatches
    it = _fake_iterator([2, 2, 2, 2, 2])
    buffer = trainer._prefetch_to_doc_target(it)

    assert len(buffer) == 3, f"Expected 3 microbatches, got {len(buffer)}"


def test_prefetch_resets_num_documents_in_batch():
    """After the call, every microbatch input has num_documents_in_batch = step total."""
    trainer = _make_fake_trainer(docs_per_step=5, bfmb=1)
    # 3 docs, 3 docs → total=6 (overshoots 5, stops after 2nd)
    it = _fake_iterator([3, 3, 3])
    buffer = trainer._prefetch_to_doc_target(it)

    step_total = sum(mb[0].num_documents for mb in buffer)
    for mb in buffer:
        for mi in mb:
            assert (
                mi.num_documents_in_batch == step_total
            ), f"Expected num_documents_in_batch={step_total}, got {mi.num_documents_in_batch}"


def test_prefetch_overshoot_is_included():
    """A microbatch that pushes the total over the target IS included (not dropped)."""
    trainer = _make_fake_trainer(docs_per_step=5, bfmb=1)
    it = _fake_iterator([4, 4])  # 4 < 5, then 8 ≥ 5 → 2 microbatches
    buffer = trainer._prefetch_to_doc_target(it)
    assert len(buffer) == 2
    assert buffer[-1][0].num_documents_in_batch == 8  # step total = 4+4


def test_prefetch_divisibility_check():
    """Raises when fetched count is not divisible by breadth_first_micro_batches."""
    trainer = _make_fake_trainer(docs_per_step=4, bfmb=2)
    # Each microbatch has 5 docs → only 1 mb needed, but 1 % 2 != 0
    it = _fake_iterator([5, 5, 5])
    with pytest.raises(Exception):
        trainer._prefetch_to_doc_target(it)


def test_prefetch_exact_divisibility():
    """No error when fetched count is exactly divisible by breadth_first_micro_batches."""
    trainer = _make_fake_trainer(docs_per_step=4, bfmb=2)
    # 2 docs each → need ≥4 → fetch 2 microbatches → 2 % 2 == 0
    it = _fake_iterator([2, 2, 2, 2])
    buffer = trainer._prefetch_to_doc_target(it)
    assert len(buffer) == 2
