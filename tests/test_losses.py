import pytest
import torch

from fast_llm.functional.config import TargetFormat
from fast_llm.functional.cross_entropy import _torch_reverse_kl_forward_backward


def test_reverse_kl_basic():
    batch_size, vocab_size = 2, 5
    logits = torch.randn(batch_size, vocab_size, requires_grad=True)
    target = torch.randn(batch_size, vocab_size)
    loss_mask = None
    grad_output = 1.0
    logits_scale_factor = 1.0
    teacher_softmax_temperature = 1.0
    target_format = TargetFormat.logits

    loss, grad = _torch_reverse_kl_forward_backward(
        logits,
        target,
        loss_mask,
        grad_output,
        logits_scale_factor,
        target_format,
        teacher_softmax_temperature=teacher_softmax_temperature,
    )
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # scalar
    assert grad is not None
    assert grad.shape == logits.shape
    assert (grad != 0).any()


def test_reverse_kl_with_mask():
    batch_size, vocab_size = 3, 4
    logits = torch.randn(batch_size, vocab_size, requires_grad=True)
    target = torch.randn(batch_size, vocab_size)
    loss_mask = torch.tensor([1.0, 0.0, 1.0])
    grad_output = 0.5
    logits_scale_factor = 1.0
    teacher_softmax_temperature = 1.0
    target_format = TargetFormat.logits

    loss, grad = _torch_reverse_kl_forward_backward(
        logits,
        target,
        loss_mask,
        grad_output,
        logits_scale_factor,
        target_format,
        teacher_softmax_temperature=teacher_softmax_temperature,
    )
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert grad is not None
    assert grad.shape == logits.shape


def test_reverse_kl_no_grad():
    batch_size, vocab_size = 2, 3
    logits = torch.randn(batch_size, vocab_size, requires_grad=True)
    target = torch.randn(batch_size, vocab_size)
    loss_mask = None
    grad_output = None
    logits_scale_factor = 1.0
    teacher_softmax_temperature = 1.0
    target_format = TargetFormat.logits

    loss, grad = _torch_reverse_kl_forward_backward(
        logits,
        target,
        loss_mask,
        grad_output,
        logits_scale_factor,
        target_format,
        teacher_softmax_temperature=teacher_softmax_temperature,
    )
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert grad is None


def test_reverse_kl_invalid_format():
    batch_size, vocab_size = 2, 3
    logits = torch.randn(batch_size, vocab_size, requires_grad=True)
    target = torch.randn(batch_size, vocab_size)
    loss_mask = None
    grad_output = 1.0
    logits_scale_factor = 1.0
    teacher_softmax_temperature = 1.0
    # Use labels format, which should trigger an assertion
    target_format = TargetFormat.labels

    with pytest.raises(AssertionError):
        _torch_reverse_kl_forward_backward(
            logits,
            target,
            loss_mask,
            grad_output,
            logits_scale_factor,
            target_format,
            teacher_softmax_temperature=teacher_softmax_temperature,
        )
