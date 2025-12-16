import os
import sys
import tempfile
import traceback
import typing

import pytest
import torch

from fast_llm.functional.config import CrossEntropyImpl, TargetFormat, TritonConfig
from fast_llm.functional.cross_entropy import cross_entropy_forward_backward, reverse_kl_forward_backward
from fast_llm.utils import Assert
from tests.utils.utils import requires_cuda


def _get_cross_entropy_inputs(
    num_columns: int, loss_masking: bool, target_format: TargetFormat, device="cuda"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # We want something moderately close to the target for the test to be meaningful
    logits_var = torch.randn(256, num_columns, dtype=torch.bfloat16, device=device) / 3
    loss_mask = torch.randint(0, 2, (256,), dtype=torch.bool, device=device) if loss_masking else None
    if target_format == TargetFormat.labels:
        target = torch.randint(0, num_columns, (256,), dtype=torch.int64, device=device)
        logits = torch.nn.functional.one_hot(target, num_columns) + logits_var
        if loss_masking:
            logits = torch.where(loss_mask.unsqueeze(-1), logits, -100)
            loss_mask = None
    else:
        target = torch.randn(256, num_columns, dtype=torch.bfloat16, device=device)
        logits = target + logits_var
        if target_format == TargetFormat.probabilities:
            target = torch.softmax(target, -1)
    return logits, target, loss_mask


def _compare_cross_entropy_outputs(
    loss: torch.Tensor,
    ref_loss: torch.Tensor,
    has_grad: bool,
    grad: torch.Tensor | None,
    ref_grad: torch.Tensor | None,
    threshold=1e-5,
):
    Assert.rms_close_relative(loss, ref_loss, threshold, 1e-6)
    if has_grad:
        Assert.rms_close_relative(grad, ref_grad, threshold, 1e-8)
    else:
        assert grad is None
        assert ref_grad is None


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
    logits, target, loss_mask = _get_cross_entropy_inputs(num_columns, loss_masking, target_format)
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

    # TODO: Why is the error so high with logit scaling?
    threshold = 2e-5 if logits_scale_factor == 1.0 else 1e-2
    _compare_cross_entropy_outputs(out_fused, out_torch, grad_output is not None, grad_fused, grad_torch, threshold)

    if num_columns > 65536:
        with pytest.raises(AssertionError):
            cross_entropy_forward_backward(**kwargs, implementation=CrossEntropyImpl.triton)
    else:
        out_triton, grad_triton = cross_entropy_forward_backward(**kwargs, implementation=CrossEntropyImpl.triton)
        _compare_cross_entropy_outputs(
            out_triton, out_torch, grad_output is not None, grad_triton, grad_torch, threshold
        )


def _reverse_kl_forward_backward_torch(target: torch.Tensor, logits: torch.Tensor, loss_mask: torch.Tensor | None):
    # Manual reference: sum over vocab then average over valid tokens.
    logits = logits.detach().requires_grad_()
    per_sample = torch.nn.functional.kl_div(
        torch.log_softmax(target.float(), dim=-1),
        torch.log_softmax(logits.float(), dim=-1),
        reduction="none",
        log_target=True,
    ).sum(dim=-1)
    output = per_sample.mean() if loss_mask is None else (per_sample * loss_mask).sum() / loss_mask.sum()
    output.backward()
    return output, logits.grad


@requires_cuda
@pytest.mark.slow
# TODO: Support the same parameterization as above in the reference implementation.
@pytest.mark.parametrize("loss_masking", [False, True])
@pytest.mark.parametrize("target_format", (TargetFormat.logits,))
def test_reverse_kl(loss_masking, target_format):
    logits, target, loss_mask = _get_cross_entropy_inputs(1000, loss_masking, target_format)
    out_ref, grad_ref = _reverse_kl_forward_backward_torch(target, logits, loss_mask)
    out, grad = reverse_kl_forward_backward(
        logits=logits,
        target=target,
        loss_mask=loss_mask,
        grad_output=1.0,
        target_format=TargetFormat.logits,
    )
    _compare_cross_entropy_outputs(out, out_ref, True, grad, grad_ref, 1e-3)


def _mp_worker(rank: int, world_size: int, init_method: str, fn_args: tuple):
    try:
        torch.distributed.init_process_group(backend="gloo", rank=rank, world_size=world_size, init_method=init_method)
        fn_args[0](rank, torch.distributed.group.WORLD, *fn_args[1:])
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _spawn_dist(world_size: int, *fn_args):
    """
    Run `fn(rank, group, *fn_args)` across `world_size` ranks using torch.multiprocessing.
    """
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        init_method = f"file://{tmp.name}"

    try:
        torch.multiprocessing.spawn(
            _mp_worker,
            args=(world_size, init_method, fn_args),
            nprocs=world_size,
            join=True,
            start_method="spawn",
        )
    finally:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)


def _compare_parallel_cross_entropy(
    rank: int,
    group: torch.distributed.ProcessGroup,
    target_format: TargetFormat,
    function: typing.Callable,
    loss_masking: bool,
):
    # Ensure all workers have the same inputs.
    torch.manual_seed(0)
    world_size = torch.distributed.get_world_size(group)
    logits, target, loss_mask = _get_cross_entropy_inputs(1000, loss_masking, target_format)

    out, grad = function(
        logits=logits.chunk(world_size, 1)[rank],
        target=target.chunk(world_size, 1)[rank],
        loss_mask=loss_mask,
        grad_output=1,
        group=group,
        target_format=target_format,
    )

    out_ref, grad_ref = function(
        logits=logits,
        target=target,
        loss_mask=loss_mask,
        grad_output=1,
        target_format=target_format,
    )
    _compare_cross_entropy_outputs(out, out_ref, True, grad, grad_ref.chunk(world_size, 1)[rank], 1e-4)


def compare_parallel_cross_entropy(rank: int, group: torch.distributed.ProcessGroup):
    success = True
    for function in (reverse_kl_forward_backward, cross_entropy_forward_backward):
        for target_format in (TargetFormat.logits,):
            for loss_masking in [False, True]:
                try:
                    _compare_parallel_cross_entropy(rank, group, target_format, function, loss_masking)
                except Exception:
                    print(
                        f" >>>>>> Failed {function.__name__}, target_format, use_mask={loss_masking}", file=sys.stderr
                    )
                    traceback.print_exc()
                    success = False
    if not success:
        raise RuntimeError("Test failed")


@requires_cuda
@pytest.mark.slow
def test_distillation_losses():
    _spawn_dist(2, compare_parallel_cross_entropy)
