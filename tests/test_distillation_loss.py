import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from fast_llm.functional.config import CrossEntropyImpl, TargetFormat
from fast_llm.functional.cross_entropy import cross_entropy_forward_backward, reverse_kl_forward_backward


def _mp_worker(rank: int, world_size: int, init_method: str, fn_name: str, fn_args: tuple):
    fn = _WORKERS[fn_name]
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size, init_method=init_method)
    try:
        fn(rank, dist.group.WORLD, *fn_args)
    finally:
        dist.destroy_process_group()


def _spawn_dist(world_size: int, fn, *fn_args):
    """
    Run `fn(rank, group, *fn_args)` across `world_size` ranks using torch.multiprocessing.
    """
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        init_method = f"file://{tmp.name}"

    try:
        mp.spawn(
            _mp_worker,
            args=(world_size, init_method, fn.__name__, fn_args),
            nprocs=world_size,
            join=True,
            start_method="spawn",
        )
    finally:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)


def _assert_loss_and_grad(logits, loss, grad):
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert grad is None or grad.shape == logits.shape
    assert torch.isfinite(loss)
    if grad is not None:
        assert torch.isfinite(grad).all()


@pytest.mark.parametrize("use_mask", [False, True])
def test_reverse_kl_no_tp(use_mask):
    torch.manual_seed(0)
    batch_size, seq_len, vocab_size = 2, 3, 5
    logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    target = torch.randn(batch_size, seq_len, vocab_size)
    loss_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]) if use_mask else None

    loss, grad = reverse_kl_forward_backward(
        logits=logits,
        target=target,
        loss_mask=loss_mask,
        grad_output=1.0,
        group=None,
        target_format=TargetFormat.logits,
        sequence_parallel_logits=False,
        group_size=None,
        vocab_size=vocab_size,
    )
    _assert_loss_and_grad(logits, loss, grad)

    # Manual reference: sum over vocab then average over valid tokens.
    teacher_log_probs = torch.log_softmax(target, dim=-1)
    student_log_probs = torch.log_softmax(logits, dim=-1)
    per_sample = torch.nn.functional.kl_div(
        teacher_log_probs, student_log_probs, reduction="none", log_target=True
    ).sum(dim=-1)
    if loss_mask is not None:
        per_sample = per_sample * loss_mask
        valid_tokens = loss_mask.sum()
    else:
        valid_tokens = logits.shape[0] * logits.shape[1]
    reference = per_sample.sum() / valid_tokens
    torch.testing.assert_close(loss, reference, atol=1e-6, rtol=1e-6)


def _vocab_tp_worker(rank: int, group: dist.ProcessGroup, use_mask: bool):
    torch.manual_seed(0)
    world_size = dist.get_world_size(group)

    batch_size, seq_len, vocab_per_rank = 2, 3, 5
    full_vocab = vocab_per_rank * world_size
    full_logits = torch.randn(batch_size, seq_len, full_vocab)
    full_target = torch.randn(batch_size, seq_len, full_vocab)
    full_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]) if use_mask else None

    start = rank * vocab_per_rank
    end = start + vocab_per_rank
    logits = full_logits[:, :, start:end].clone().requires_grad_(True)
    target = full_target[:, :, start:end].clone()
    loss_mask = full_mask.clone() if full_mask is not None else None

    loss, grad = reverse_kl_forward_backward(
        logits=logits,
        target=target,
        loss_mask=loss_mask,
        grad_output=None,
        group=group,
        target_format=TargetFormat.logits,
        sequence_parallel_logits=False,
        group_size=world_size,
        vocab_size=full_vocab,
    )
    _assert_loss_and_grad(logits, loss, grad)

    if rank == 0:
        ref_loss, _ = reverse_kl_forward_backward(
            logits=full_logits.clone(),
            target=full_target.clone(),
            loss_mask=full_mask.clone() if full_mask is not None else None,
            grad_output=None,
            group=None,
            target_format=TargetFormat.logits,
            sequence_parallel_logits=False,
            group_size=None,
            vocab_size=full_vocab,
        )
    else:
        ref_loss = torch.zeros_like(loss)
    dist.broadcast(ref_loss, src=0, group=group)
    torch.testing.assert_close(loss, ref_loss, atol=1e-6, rtol=1e-6)


def _ce_vocab_tp_worker(rank: int, group: dist.ProcessGroup, use_mask: bool):
    torch.manual_seed(0)
    world_size = dist.get_world_size(group)

    batch_size, seq_len, vocab_per_rank = 2, 3, 5
    full_vocab = vocab_per_rank * world_size
    full_logits = torch.randn(batch_size, seq_len, full_vocab)
    full_target = torch.randn(batch_size, seq_len, full_vocab)
    full_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]) if use_mask else None

    start = rank * vocab_per_rank
    end = start + vocab_per_rank
    logits = full_logits[:, :, start:end].clone().requires_grad_(True)
    target = full_target[:, :, start:end].clone()
    loss_mask = full_mask.clone() if full_mask is not None else None

    loss, grad = cross_entropy_forward_backward(
        logits=logits,
        target=target,
        loss_mask=loss_mask,
        grad_output=None,
        group=group,
        implementation=CrossEntropyImpl.fused,
        target_format=TargetFormat.logits,
        logits_scale_factor=1.0,
    )
    _assert_loss_and_grad(logits, loss, grad)

    if rank == 0:
        ref_loss, _ = cross_entropy_forward_backward(
            logits=full_logits.clone(),
            target=full_target.clone(),
            loss_mask=full_mask.clone() if full_mask is not None else None,
            grad_output=None,
            group=None,
            implementation=CrossEntropyImpl.fused,
            target_format=TargetFormat.logits,
            logits_scale_factor=1.0,
        )
    else:
        ref_loss = torch.zeros_like(loss)
    dist.broadcast(ref_loss, src=0, group=group)
    torch.testing.assert_close(loss, ref_loss, atol=1e-6, rtol=1e-6)


_WORKERS = {"_vocab_tp_worker": _vocab_tp_worker, "_ce_vocab_tp_worker": _ce_vocab_tp_worker}


@pytest.mark.parametrize("use_mask", [True, False])
def test_reverse_kl_vocab_tp_two_ranks(use_mask):
    _spawn_dist(2, _vocab_tp_worker, use_mask)


@pytest.mark.parametrize("use_mask", [True, False])
def test_cross_entropy_vocab_tp_two_ranks(use_mask):
    _spawn_dist(2, _ce_vocab_tp_worker, use_mask)


if __name__ == "__main__":
    pytest.main([__file__])
