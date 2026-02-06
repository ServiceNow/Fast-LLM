import typing

import torch

from fast_llm.functional.config import EntropyLossImplementation, EntropyLossType, TargetFormat, TritonConfig
from fast_llm.functional.entropy_loss import fused_entropy_loss_forward_backward, torch_entropy_loss_forward_backward
from fast_llm.functional.triton.entropy_loss import triton_entropy_loss_forward_backward
from fast_llm.layers.language_model.loss.config import (
    LanguageModelDistillationLossConfig,
    LanguageModelLabelEntropyLossConfig,
)
from fast_llm.layers.language_model.loss.loss import LanguageModelLoss
from fast_llm.utils import Assert


class LanguageModelLabelEntropyLoss[ConfigType: LanguageModelLabelEntropyLossConfig](LanguageModelLoss[ConfigType]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
            logits_scale_factor=self._logits_scale_factor,
            target_format=TargetFormat.labels,
            entropy_loss_type=self._config.loss_type,
            use_triton=self._config.use_triton,
        )


class LanguageModelDistillationLoss[ConfigType: LanguageModelDistillationLossConfig](LanguageModelLoss[ConfigType]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._prediction_distance > 0:
            raise NotImplementedError()

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
            logits_scale_factor=self._logits_scale_factor,
            target_format=TargetFormat.logits,
            entropy_loss_type=self._config.loss_type,
            use_triton=self._config.use_triton,
        )


_ENTROPY_LOSS_IMPLEMENTATIONS = {
    EntropyLossImplementation.torch: torch_entropy_loss_forward_backward,
    EntropyLossImplementation.fused: fused_entropy_loss_forward_backward,
    EntropyLossImplementation.triton: triton_entropy_loss_forward_backward,
}


def entropy_loss_forward_backward(
    logits: torch.Tensor,  # (*batch, vocab)
    target: torch.Tensor,  # (*batch,) or (*batch, vocab)
    loss_mask: torch.Tensor | None,  # (*batch,)
    grad_output: float | None,
    group: torch.distributed.ProcessGroup | None = None,
    logits_scale_factor: float = 1.0,
    temperature: float = 1.0,
    target_format: TargetFormat = TargetFormat.labels,
    entropy_loss_type: EntropyLossType = EntropyLossType.cross_entropy,
    use_triton: bool | None = None,
    **kwargs,
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
    return (
        triton_entropy_loss_forward_backward
        if TritonConfig.enabled(logits.device, use_triton)
        else fused_entropy_loss_forward_backward
    )(
        logits,
        target,
        loss_mask,
        grad_output,
        logits_scale_factor,
        target_format,
        entropy_loss_type,
        group,
        temperature=temperature,
        **kwargs,
    )
