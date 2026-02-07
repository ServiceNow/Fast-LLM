import pathlib
import random

import numpy as np
import pytest
import torch

from fast_llm.core.ops import split_op
from fast_llm.engine.config_utils import data_type
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.engine.distributed.config import DistributedBackend
from fast_llm.functional.config import EntropyLossType, TargetFormat
from fast_llm.functional.entropy_loss import fused_entropy_loss_forward_backward, torch_entropy_loss_forward_backward
from fast_llm.functional.triton import triton_available
from fast_llm.functional.triton.entropy_loss import triton_entropy_loss_forward_backward
from fast_llm.functional.triton.z_loss import triton_z_loss_forward_backward
from fast_llm.layers.language_model.loss.dpo import dpo_loss
from fast_llm.layers.language_model.loss.loss import loss_forward_backward
from fast_llm.layers.language_model.loss.z_loss import fused_z_loss_forward_backward, z_loss
from fast_llm.utils import Assert
from tests.utils.dataset import get_random_spans
from tests.utils.subtest import DistributedTestContext


def _get_lm_loss_inputs(
    num_columns: int, loss_masking: bool, target_format: TargetFormat, batch_shape: tuple[int], dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # We want something moderately close to the target for the test to be meaningful
    logits_var = torch.randn((*batch_shape, num_columns), dtype=dtype.torch, device=device) / 3
    loss_mask = torch.randint(0, 2, batch_shape, dtype=torch.bool, device=device) if loss_masking else None
    if target_format == TargetFormat.labels:
        target = torch.randint(0, num_columns, batch_shape, dtype=torch.int64, device=device)
        logits = torch.nn.functional.one_hot(target, num_columns) + logits_var
        if loss_masking:
            target = torch.where(loss_mask, target, -100)
            loss_mask = None
    else:
        # Target logits are typically in training precision, ex. with distillation model.
        target = torch.randn((*batch_shape, num_columns), dtype=dtype.torch, device=device)
        logits = target + logits_var
        if target_format == TargetFormat.probabilities:
            # Probabilities need to be in full precision for accuracy.
            target = torch.softmax(target, -1, dtype=torch.float32)
    return logits, target, loss_mask


def _compare_losses_and_grads(
    loss: torch.Tensor,
    ref_loss: torch.Tensor,
    has_grad: bool,
    grad: torch.Tensor | None,
    ref_grad: torch.Tensor | None,
    threshold=1e-5,
    group: torch.distributed.ProcessGroup | None = None,
):
    Assert.rms_close_relative(loss, ref_loss, threshold, 1e-6)
    if has_grad:
        Assert.rms_close_relative(
            grad, split_op(ref_grad, group, -1), threshold, 1e-8 if grad.dtype == torch.float32 else 1e-7
        )
    else:
        assert grad is None
        assert ref_grad is None


def reference_dpo_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    reference_model_logits: torch.Tensor,
    chosen_spans: torch.Tensor,
    rejected_spans: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    # TODO: Too similar to the actual implementation.
    policy_log_probs = (
        torch.nn.functional.log_softmax(logits.float(), dim=-1).gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    )
    policy_chosen_logps = sum(
        policy_log_probs[sample_index, begin:end].sum()
        for sample_index, sample_spans in enumerate(chosen_spans)
        for begin, end in sample_spans
    )
    policy_rejected_logps = sum(
        policy_log_probs[sample_index, begin:end].sum()
        for sample_index, sample_spans in enumerate(rejected_spans)
        for begin, end in sample_spans
    )
    reference_log_probs = (
        torch.nn.functional.log_softmax(reference_model_logits.float(), dim=-1)
        .gather(dim=-1, index=labels.unsqueeze(-1))
        .squeeze(-1)
    )
    reference_chosen_logps = sum(
        reference_log_probs[sample_index, begin:end].sum()
        for sample_index, sample_spans in enumerate(chosen_spans)
        for begin, end in sample_spans
    )
    reference_rejected_logps = sum(
        reference_log_probs[sample_index, begin:end].sum()
        for sample_index, sample_spans in enumerate(rejected_spans)
        for begin, end in sample_spans
    )
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    return -torch.nn.functional.logsigmoid(beta * (pi_logratios - ref_logratios)).mean()


# _BATCH_SHAPES = ((64,), (16, 8))
_BATCH_SHAPES = ((1,),)
_LOSS_PARAMETERS = (
    (500, 1.0, 1.0, False, DataType.float32, None, False),  # Simple
    (256, 1.0, 1.0, False, DataType.float32, None, False),  # Power of 2
    (500, None, 1.0, False, DataType.float32, None, False),  # No grad
    (500, 1.0, 1.0, False, DataType.float32, None, True),  # Accumulate
    (500, 1.0, 4.0, False, DataType.float32, None, False),  # Loss scaling
    (500, 4.0, 1.0, False, DataType.float32, None, False),  # Grad scaling
    (500, 1.0, 1.0, True, DataType.float32, None, False),  # Loss masking
    (500, 1.0, 1.0, False, DataType.float16, None, False),  # Fp16
    (500, 1.0, 1.0, False, DataType.float32, 256, False),  # Looped
    (1000, 2.0, 3.0, True, DataType.float32, 256, True),  # Hard
)


def _test_entropy_loss(
    batch_shape,
    num_columns,
    grad_output,
    logits_scale_factor,
    loss_masking,
    target_format,
    entropy_loss_type,
    dtype,
    block_size,
    accumulate,
    group=None,
):
    if target_format == TargetFormat.labels and entropy_loss_type == EntropyLossType.reverse_kl:
        pytest.skip(reason="Reverse KL loss not implemented for target labels")
    # TODO: Test tensor-parallel implementation.
    logits, target, loss_mask = _get_lm_loss_inputs(num_columns, loss_masking, target_format, batch_shape, dtype)
    local_logits = split_op(logits, group, -1).contiguous()
    local_target = target if target_format == TargetFormat.labels else split_op(target, group, -1).contiguous()
    # Torch serves as the reference implementation.
    out_ref, grad_ref = torch_entropy_loss_forward_backward(
        logits=logits,
        target=target,
        loss_mask=loss_mask,
        grad_output=grad_output,
        logits_scale_factor=logits_scale_factor,
        target_format=target_format,
        entropy_loss_type=entropy_loss_type,
    )
    if accumulate:
        previous_grad = torch.randn_like(grad_ref)
        grad_ref = grad_ref + previous_grad
        local_previous_grad = split_op(previous_grad, group, -1).contiguous()
    out_fused, grad_fused = fused_entropy_loss_forward_backward(
        logits=local_logits,
        target=local_target,
        loss_mask=loss_mask,
        grad_logits=local_previous_grad.clone() if accumulate else None,
        grad_output=grad_output,
        group=group,
        logits_scale_factor=logits_scale_factor,
        target_format=target_format,
        entropy_loss_type=entropy_loss_type,
    )
    _compare_losses_and_grads(
        out_fused,
        out_ref,
        grad_output is not None,
        grad_fused,
        grad_ref,
        threshold=1e-5 if data_type == DataType.float32 else 1e-4,
        group=group,
    )

    if not triton_available:
        return
    out_triton, grad_triton = triton_entropy_loss_forward_backward(
        logits=local_logits,
        target=local_target,
        loss_mask=loss_mask,
        grad_logits=local_previous_grad.clone() if accumulate else None,
        grad_output=grad_output,
        logits_scale_factor=logits_scale_factor,
        target_format=target_format,
        entropy_loss_type=entropy_loss_type,
        group=group,
        block_size=block_size,
    )
    _compare_losses_and_grads(
        out_triton,
        out_ref,
        grad_output is not None,
        grad_triton,
        grad_ref,
        threshold=1e-5 if target_format != TargetFormat.probabilities and data_type == DataType.float32 else 1e-4,
        group=group,
    )


def _test_z_loss(
    batch_shape, num_columns, grad_output, logits_scale_factor, loss_masking, dtype, block_size, accumulate, group=None
):
    logits, target, loss_mask = _get_lm_loss_inputs(num_columns, loss_masking, TargetFormat.logits, batch_shape, dtype)
    local_logits = split_op(logits, group, -1).contiguous()
    out_ref, grad_ref = loss_forward_backward(
        grad_output,
        z_loss,
        logits,
        loss_mask=loss_mask,
        logits_scale_factor=logits_scale_factor,
    )
    if accumulate:
        previous_grad = torch.randn_like(grad_ref)
        grad_ref = grad_ref + previous_grad
        local_previous_grad = split_op(previous_grad, group, -1).contiguous()
    out_fused, grad_fused = fused_z_loss_forward_backward(
        logits=local_logits,
        loss_mask=loss_mask,
        grad_logits=local_previous_grad.clone() if accumulate else None,
        grad_output=grad_output,
        group=group,
        logits_scale_factor=logits_scale_factor,
    )
    _compare_losses_and_grads(
        out_fused,
        out_ref,
        grad_output is not None,
        grad_fused,
        grad_ref,
        threshold=1e-5 if data_type == DataType.float32 else 1e-4,
        group=group,
    )
    if not triton_available:
        return
    out_triton, grad_triton = triton_z_loss_forward_backward(
        logits=local_logits,
        loss_mask=loss_mask,
        grad_logits=local_previous_grad.clone() if accumulate else None,
        grad_output=grad_output,
        group=group,
        logits_scale_factor=logits_scale_factor,
        block_size=block_size,
    )
    _compare_losses_and_grads(
        out_triton,
        out_ref,
        grad_output is not None,
        grad_triton,
        grad_ref,
        threshold=1e-5 if data_type == DataType.float32 else 1e-4,
        group=group,
    )


@pytest.mark.slow
@pytest.mark.parametrize("batch_shape", _BATCH_SHAPES)
@pytest.mark.parametrize(
    ("num_columns", "grad_output", "logits_scale_factor", "loss_masking", "dtype", "block_size", "accumulate"),
    _LOSS_PARAMETERS,
)
@pytest.mark.parametrize("target_format", TargetFormat)
@pytest.mark.parametrize("entropy_loss_type", EntropyLossType)
def test_entropy_loss(
    batch_shape,
    num_columns,
    grad_output,
    logits_scale_factor,
    loss_masking,
    target_format,
    entropy_loss_type,
    dtype,
    block_size,
    accumulate,
):
    _test_entropy_loss(
        batch_shape,
        num_columns,
        grad_output,
        logits_scale_factor,
        loss_masking,
        target_format,
        entropy_loss_type,
        dtype,
        block_size,
        accumulate,
    )


@pytest.mark.slow
@pytest.mark.parametrize("batch_shape", _BATCH_SHAPES)
@pytest.mark.parametrize(
    ("num_columns", "grad_output", "logits_scale_factor", "loss_masking", "dtype", "block_size", "accumulate"),
    _LOSS_PARAMETERS,
)
def test_z_loss(
    batch_shape, num_columns, grad_output, logits_scale_factor, loss_masking, dtype, block_size, accumulate
):
    _test_z_loss(
        batch_shape, num_columns, grad_output, logits_scale_factor, loss_masking, dtype, block_size, accumulate
    )


@pytest.mark.skip(reason="DPO loss is broken")
def test_dpo_loss():
    logits = torch.normal(0, 1, (200, 100))
    reference_model_logits = torch.normal(0, 1, (200, 100))
    labels = torch.randint(0, 100, (200,))
    spans = get_random_spans(np.full(10, 50), 0, 10)

    fast_llm_loss = dpo_loss(logits, labels, reference_model_logits, spans[::2], spans[1::2])
    reference_loss = reference_dpo_loss(logits, labels, reference_model_logits, spans[::2], spans[1::2], beta=1)
    Assert.rms_close(fast_llm_loss, reference_loss, 1e-5)


def _run_lm_loss_distributed(test_context: DistributedTestContext, base_path: pathlib.Path, seed: int):
    for batch_shape in _BATCH_SHAPES:
        for (
            num_columns,
            grad_output,
            logits_scale_factor,
            loss_masking,
            dtype,
            block_size,
            accumulate,
        ) in _LOSS_PARAMETERS:
            suffix = f"{num_columns}-{grad_output}-{logits_scale_factor}-{loss_masking}-{dtype}-{block_size}-{accumulate}-{"_".join([str(i) for i in batch_shape])}"
            # Entropy loss
            for entropy_loss_type in EntropyLossType:
                for target_format in TargetFormat:
                    if target_format == TargetFormat.labels and entropy_loss_type == EntropyLossType.reverse_kl:
                        continue
                    with test_context.subtest(
                        base_path, f"{entropy_loss_type}-{target_format}-{suffix}", 2
                    ) as subtest:
                        if subtest.do_run:
                            torch.manual_seed((seed + hash(subtest.name)) % 2**32)
                            _test_entropy_loss(
                                batch_shape,
                                num_columns,
                                grad_output,
                                logits_scale_factor,
                                loss_masking,
                                target_format,
                                entropy_loss_type,
                                dtype,
                                block_size,
                                accumulate,
                                test_context.group,
                            )
            # Z loss
            with test_context.subtest(base_path, f"z_loss-{suffix}", 2) as subtest:
                if subtest.do_run:
                    torch.manual_seed((seed + hash(subtest.name)) % 2**32)
                    _test_z_loss(
                        batch_shape,
                        num_columns,
                        grad_output,
                        logits_scale_factor,
                        loss_masking,
                        dtype,
                        block_size,
                        accumulate,
                        test_context.group,
                    )


@pytest.mark.slow
def test_lm_loss_distributed_dependency():
    # Mock test so the distributed subtest are placed in the same dependency group.
    pass


# We don't want to depend on `test_run_entropy_loss_distributed` because we still want to run this in cas of failure.
# This should still run after `test_run_entropy_loss_distributed`
@pytest.mark.slow
@pytest.mark.depends_on(on=["test_lm_loss_distributed_dependency"])
def test_run_lm_loss_distributed(run_parallel_script, result_path):
    run_parallel_script(
        _run_lm_loss_distributed,
        (result_path / "test_losses", random.randint(0, 2**32 - 1)),
        world_size=2,
        backend=DistributedBackend.nccl if (use_nccl := torch.cuda.device_count() >= 2) else DistributedBackend.gloo,
        use_cuda=use_nccl,  # Disable device count check.
    )


@pytest.mark.slow
@pytest.mark.depends_on(on=["test_lm_loss_distributed_dependency"])
@pytest.mark.parametrize("batch_shape", _BATCH_SHAPES)
@pytest.mark.parametrize(
    ("num_columns", "grad_output", "logits_scale_factor", "loss_masking", "dtype", "block_size", "accumulate"),
    _LOSS_PARAMETERS,
)
@pytest.mark.parametrize(
    "loss_type",
    (
        *(
            f"{entropy_loss_type}-{target_format}"
            for entropy_loss_type in EntropyLossType
            for target_format in TargetFormat
            if target_format != TargetFormat.labels or entropy_loss_type != EntropyLossType.reverse_kl
        ),
        "z_loss",
    ),
)
def test_lm_loss_distributed(
    result_path,
    report_subtest,
    loss_type,
    batch_shape,
    num_columns,
    grad_output,
    logits_scale_factor,
    loss_masking,
    dtype,
    block_size,
    accumulate,
):
    report_subtest(
        result_path
        / f"test_losses/{loss_type}-{num_columns}-{grad_output}-{logits_scale_factor}-{loss_masking}-{dtype}-{block_size}-{accumulate}-{"_".join([str(i) for i in batch_shape])}",
        2,
        use_cuda=False,
    )
