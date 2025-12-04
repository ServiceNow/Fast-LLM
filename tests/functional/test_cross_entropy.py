import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from fast_llm.functional.config import CrossEntropyImpl, TargetFormat, TritonConfig
from fast_llm.functional.cross_entropy import cross_entropy_forward_backward, reverse_kl_forward_backward
from fast_llm.utils import Assert
from tests.utils.utils import requires_cuda


def _mp_worker(rank: int, world_size: int, init_method: str, fn_args: tuple):
    fn = combined_worker
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
            args=(world_size, init_method, fn_args),
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


def combined_worker(rank: int, group: dist.ProcessGroup, use_mask: bool):
    _vocab_tp_worker(rank, group, use_mask)
    _ce_vocab_tp_worker(rank, group, use_mask)


# TODO: maybe merge these tests using same parametrization
@pytest.mark.slow
@pytest.mark.parametrize("use_mask", [True, False])
def test_distillation_losses(use_mask):
    _spawn_dist(2, combined_worker, use_mask)


@requires_cuda
@pytest.mark.slow
@pytest.mark.parametrize(
    ("num_columns", "grad_output", "logits_scale_factor", "loss_masking"),
    (
        (8192, 1.0, 1.0, False),  # Simple
        (5000, 1.0, 1.0, False),  # Not a power of 2
        (5000, None, 1.0, False),  # No grad
        (5000, 1.0, 4.0, False),  # Loss scaling
        (5000, 4.0, 1.0, False),  # Grad scaling
        (5000, 1.0, 1.0, True),  # Loss masking
        (65536, 1.0, 1.0, False),  # Max block size
        (65537, 1.0, 1.0, False),  # Above max block size
    ),
)
@pytest.mark.parametrize("target_format", (TargetFormat.labels, TargetFormat.logits, TargetFormat.probabilities))
def test_cross_entropy(num_columns, grad_output, logits_scale_factor, loss_masking, target_format):
    # TODO: Test tensor-parallel implementation.
    assert TritonConfig.TRITON_ENABLED
    # We want something moderately close to the target for the test to be meaningful
    logits_var = torch.randn(256, num_columns, dtype=torch.bfloat16, device="cuda") / 3
    loss_mask = torch.randint(0, 2, (256,), dtype=torch.bool, device="cuda") if loss_masking else None
    if target_format == TargetFormat.labels:
        target = torch.randint(0, num_columns, (256,), dtype=torch.int64, device="cuda")
        logits = (torch.nn.functional.one_hot(target, num_columns) + logits_var).requires_grad_()
        if loss_masking:
            logits = torch.where(loss_mask.unsqueeze(-1), logits, -100)
            loss_mask = None
    else:
        target = torch.randn(256, num_columns, dtype=torch.bfloat16, device="cuda")
        logits = (target + logits_var).requires_grad_()
        if target_format == TargetFormat.probabilities:
            target = torch.softmax(target, -1)

    kwargs = {
        "logits": logits,
        "target": target,
        "loss_mask": loss_mask,
        "grad_output": grad_output,
        "logits_scale_factor": logits_scale_factor,
        "target_format": target_format,
    }
    # Torch serves as the reference implementation.
    out_torch, grad_torch = cross_entropy_forward_backward(**kwargs, implementation=CrossEntropyImpl.torch)

    out_fused, grad_fused = cross_entropy_forward_backward(**kwargs, implementation=CrossEntropyImpl.fused)
    Assert.rms_close(out_fused, out_torch, 5e-3)
    if grad_output is None:
        assert grad_torch is None
        assert grad_fused is None
    else:
        Assert.rms_close(grad_fused, grad_torch, 5e-3)

    if num_columns > 65536:
        with pytest.raises(AssertionError):
            cross_entropy_forward_backward(**kwargs, implementation=CrossEntropyImpl.triton)
    else:
        out_triton, grad_triton = cross_entropy_forward_backward(**kwargs, implementation=CrossEntropyImpl.triton)
        if grad_output is None:
            assert grad_triton is None
        else:
            Assert.rms_close(grad_triton, grad_torch, 5e-3)
        Assert.rms_close(out_triton, out_torch, 5e-3)


if __name__ == "__main__":
    pytest.main([__file__])
