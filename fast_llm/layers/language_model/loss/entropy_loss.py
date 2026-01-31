import typing

import torch

from fast_llm.functional.config import EntropyLossImplementation, EntropyLossType, TargetFormat, TritonConfig
from fast_llm.functional.entropy_loss import fused_entropy_loss_forward_backward, torch_entropy_loss_forward_backward
from fast_llm.functional.triton.cross_entropy import triton_cross_entropy_forward_backward
from fast_llm.layers.language_model.loss.config import (
    LanguageModelDistillationLossConfig,
    LanguageModelLabelEntropyLossConfig,
)
from fast_llm.layers.language_model.loss.loss import LanguageModelLoss
from fast_llm.utils import Assert


def _get_implementation(
    default: EntropyLossImplementation = EntropyLossImplementation.auto,
    loss_type: EntropyLossType = EntropyLossType.cross_entropy,
    vocab_parallel: bool = False,
) -> EntropyLossImplementation:
    # Vocab parallel requires fused.
    if vocab_parallel:
        assert default in (EntropyLossImplementation.auto, EntropyLossImplementation.fused)
        return EntropyLossImplementation.fused

    # Triton only available for cross_entropy
    if TritonConfig.TRITON_ENABLED and torch.cuda.is_available() and loss_type == EntropyLossType.cross_entropy:
        return EntropyLossImplementation.triton if default == EntropyLossImplementation.auto else default
    else:
        assert default != EntropyLossImplementation.triton

    # Otherwise, use fused.
    return EntropyLossImplementation.fused if default == EntropyLossImplementation.auto else default


class LanguageModelLabelEntropyLoss[ConfigType: LanguageModelLabelEntropyLossConfig](LanguageModelLoss[ConfigType]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._implementation = _get_implementation(
            self._config.implementation, self._config.loss_type, self._vocab_parallel
        )

    def forward_backward(
        self,
        logits: "torch.Tensor",
        kwargs: dict[str, typing.Any],
        split_index: int = 0,
    ) -> "tuple[torch.Tensor, torch.Tensor | None]":
        return entropy_loss_forward_backward(
            logits,
            self._get_labels(kwargs, split_index),
            None,  # Labels are already masked
            grad_output=self._get_grad_output(kwargs),
            group=self._parallel_dim.group if self._vocab_parallel else None,
            implementation=self._implementation,
            logits_scale_factor=self._logits_scale_factor,
            target_format=TargetFormat.labels,
            entropy_loss_type=self._config.loss_type,
        )


class LanguageModelDistillationLoss[ConfigType: LanguageModelDistillationLossConfig](LanguageModelLoss[ConfigType]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._prediction_distance > 0:
            raise NotImplementedError()

        self._implementation = _get_implementation(
            self._config.implementation, self._config.loss_type, self._vocab_parallel
        )

    def forward_backward(
        self,
        logits: "torch.Tensor",
        kwargs: dict[str, typing.Any],
        split_index: int = 0,
    ) -> "tuple[torch.Tensor, torch.Tensor | None]":
        return entropy_loss_forward_backward(
            logits,
            self._get_reference_model_logits(self._config.reference_model, kwargs, split_index),
            self._get_loss_mask(kwargs, split_index),
            grad_output=self._get_grad_output(kwargs),
            group=self._parallel_dim.group if self._vocab_parallel else None,
            implementation=self._implementation,
            logits_scale_factor=self._logits_scale_factor,
            target_format=TargetFormat.logits,
            entropy_loss_type=self._config.loss_type,
        )


_ENTROPY_LOSS_IMPLEMENTATIONS = {
    EntropyLossImplementation.torch: torch_entropy_loss_forward_backward,
    EntropyLossImplementation.fused: fused_entropy_loss_forward_backward,
    EntropyLossImplementation.triton: triton_cross_entropy_forward_backward,
}


def entropy_loss_forward_backward(
    logits: torch.Tensor,  # (*batch, vocab)
    target: torch.Tensor,  # (*batch,) or (*batch, vocab)
    loss_mask: torch.Tensor | None,  # (*batch,)
    grad_output: float | None,
    group: torch.distributed.ProcessGroup | None = None,
    implementation: EntropyLossImplementation = EntropyLossImplementation.fused,
    logits_scale_factor: float = 1.0,
    temperature: float = 1.0,
    target_format: TargetFormat = TargetFormat.labels,
    entropy_loss_type: EntropyLossType = EntropyLossType.cross_entropy,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Select the appropriate implementation of cross-entropy.
    The triton implementation from the triton submodule is the fastest and recommended one.
    It doesn't have a tensor-parallel implementation, but can be computed in a sequence-tensor-parallel way,
    which is faster and has a relatively small memory overhead.
    """
    if target_format == TargetFormat.labels:
        Assert.eq(target.shape, logits.shape[:-1])
        Assert.eq(target.dtype, torch.int64)
        assert loss_mask is None
    else:
        Assert.eq(target.shape, logits.shape)
        assert target.dtype.is_floating_point, target.dtype
        if loss_mask is not None:
            Assert.eq(loss_mask.shape, logits.shape[:-1])
    return _ENTROPY_LOSS_IMPLEMENTATIONS[implementation](
        logits,
        target,
        loss_mask,
        grad_output,
        logits_scale_factor,
        target_format,
        entropy_loss_type,
        group,
        temperature=temperature,
    )
