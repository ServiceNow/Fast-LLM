import pathlib

import pytest
import torch

from fast_llm.engine.distributed.config import DistributedBackend
from fast_llm.functional.config import EntropyLossImplementation, EntropyLossType, TargetFormat, TritonConfig
from fast_llm.functional.entropy_loss import entropy_loss_forward_backward
from fast_llm.utils import Assert
from tests.utils.subtest import DistributedTestContext


def _get_cross_entropy_inputs(
    num_columns: int, loss_masking: bool, target_format: TargetFormat
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # We want something moderately close to the target for the test to be meaningful
    logits_var = torch.randn(256, num_columns, dtype=torch.float32, device=device) / 3
    loss_mask = torch.randint(0, 2, (256,), dtype=torch.bool, device=device) if loss_masking else None
    if target_format == TargetFormat.labels:
        target = torch.randint(0, num_columns, (256,), dtype=torch.int64, device=device)
        logits = torch.nn.functional.one_hot(target, num_columns) + logits_var
        if loss_masking:
            logits = torch.where(loss_mask.unsqueeze(-1), logits, -100)
            loss_mask = None
    else:
        target = torch.randn(256, num_columns, dtype=torch.float32, device=device)
        logits = target + logits_var
        if target_format == TargetFormat.probabilities:
            target = torch.softmax(target, -1)
    return logits, target, loss_mask


def _compare_entropy_loss_outputs(
    loss: torch.Tensor,
    ref_loss: torch.Tensor,
    has_grad: bool,
    grad: torch.Tensor | None,
    ref_grad: torch.Tensor | None,
    threshold=1e-5,
    loss_min_threshold=1e-6,
):
    Assert.rms_close_relative(loss, ref_loss, threshold, loss_min_threshold)
    if has_grad:
        Assert.rms_close_relative(grad, ref_grad, threshold, 1e-8)
    else:
        assert grad is None
        assert ref_grad is None


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
@pytest.mark.parametrize("target_format", TargetFormat)
@pytest.mark.parametrize("entropy_loss_type", EntropyLossType)
def test_entropy_loss(num_columns, grad_output, logits_scale_factor, loss_masking, target_format, entropy_loss_type):
    if target_format == TargetFormat.labels and entropy_loss_type == EntropyLossType.reverse_kl:
        pytest.skip(reason="Not implemented")
    # TODO: Test tensor-parallel implementation.
    logits, target, loss_mask = _get_cross_entropy_inputs(num_columns, loss_masking, target_format)
    kwargs = {
        "logits": logits,
        "target": target,
        "loss_mask": loss_mask,
        "grad_output": grad_output,
        "logits_scale_factor": logits_scale_factor,
        "target_format": target_format,
        "entropy_loss_type": entropy_loss_type,
    }
    # Torch serves as the reference implementation.
    out_torch, grad_torch = entropy_loss_forward_backward(**kwargs, implementation=EntropyLossImplementation.torch)
    out_fused, grad_fused = entropy_loss_forward_backward(**kwargs, implementation=EntropyLossImplementation.fused)

    # TODO: Why is the error so high with loss masking for reverse KL?
    _compare_entropy_loss_outputs(
        out_fused,
        out_torch,
        grad_output is not None,
        grad_fused,
        grad_torch,
        loss_min_threshold=2e-4 if entropy_loss_type == EntropyLossType.reverse_kl and loss_masking else 5e-6,
    )

    if entropy_loss_type != EntropyLossType.cross_entropy or not torch.cuda.is_available():
        # Triton implementation only supports cross-entropy.
        return
    assert TritonConfig.TRITON_ENABLED
    if num_columns > 65536:
        with pytest.raises(AssertionError):
            entropy_loss_forward_backward(**kwargs, implementation=EntropyLossImplementation.triton)
    else:
        out_triton, grad_triton = entropy_loss_forward_backward(
            **kwargs, implementation=EntropyLossImplementation.triton
        )
        _compare_entropy_loss_outputs(out_triton, out_torch, grad_output is not None, grad_triton, grad_torch)


def _entropy_loss_distributed(
    target_format: TargetFormat,
    entropy_loss_type: EntropyLossType,
    loss_masking: bool,
    group: torch.distributed.ProcessGroup,
):
    # Ensure all workers have the same inputs.
    torch.manual_seed(0)
    rank = group.rank()
    world_size = group.size()
    logits, target, loss_mask = _get_cross_entropy_inputs(1000, loss_masking, target_format)

    kwargs = {
        "loss_mask": loss_mask,
        "grad_output": 1.0,
        "target_format": target_format,
        "implementation": EntropyLossImplementation.fused,
        "entropy_loss_type": entropy_loss_type,
    }
    out_ref, grad_ref = entropy_loss_forward_backward(logits, target, **kwargs)

    out, grad = entropy_loss_forward_backward(
        logits.chunk(world_size, 1)[rank],
        target if target_format == TargetFormat.labels else target.chunk(world_size, 1)[rank],
        group=group,
        **kwargs,
    )
    _compare_entropy_loss_outputs(out, out_ref, True, grad, grad_ref.chunk(world_size, 1)[rank], 1e-4)


def _run_entropy_loss_distributed(test_context: DistributedTestContext, base_path: pathlib.Path):
    for entropy_loss_type in EntropyLossType:
        for target_format in TargetFormat:
            if target_format == TargetFormat.labels and entropy_loss_type == EntropyLossType.reverse_kl:
                continue
            for loss_masking in [False, True]:
                name = f"{entropy_loss_type}_{target_format}_{loss_masking}"
                with test_context.subtest(base_path, name, 2) as subtest:
                    if subtest.do_run:
                        _entropy_loss_distributed(target_format, entropy_loss_type, loss_masking, test_context.group)


@pytest.mark.slow
def test_entropy_loss_distributed_dependency():
    # Mock test so the distributed subtest are placed in the same dependency group.
    pass


@pytest.mark.slow
@pytest.mark.depends_on(on=["test_entropy_loss_distributed_dependency"])
def test_run_entropy_loss_distributed(run_parallel_script, result_path):
    run_parallel_script(
        _run_entropy_loss_distributed,
        (result_path / "test_entropy_loss",),
        world_size=2,
        backend=DistributedBackend.gloo,
        use_cuda=False,  # Disable device count check.
    )


# We don't want to depend on `test_run_entropy_loss_distributed` because we still want to run this in cas of failure.
# This should still run after `test_run_entropy_loss_distributed`
@pytest.mark.slow
@pytest.mark.depends_on(on=["test_entropy_loss_distributed_dependency"])
@pytest.mark.parametrize("target_format", TargetFormat)
@pytest.mark.parametrize("entropy_loss_type", EntropyLossType)
@pytest.mark.parametrize("loss_masking", (False, True))
def test_entropy_loss_distributed(result_path, report_subtest, target_format, entropy_loss_type, loss_masking):
    if target_format == TargetFormat.labels and entropy_loss_type == EntropyLossType.reverse_kl:
        pytest.skip(reason="Not implemented")
    report_subtest(
        result_path / f"test_entropy_loss/{entropy_loss_type}_{target_format}_{loss_masking}", 2, use_cuda=False
    )
