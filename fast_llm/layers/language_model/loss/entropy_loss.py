import typing

import torch

from fast_llm.functional.config import TargetFormat, TritonConfig
from fast_llm.functional.entropy_loss import fused_entropy_loss_forward_backward
from fast_llm.functional.triton.entropy_loss import triton_entropy_loss_forward_backward
from fast_llm.layers.language_model.loss.config import (
    LanguageModelDistillationLossConfig,
    LanguageModelLabelEntropyLossConfig,
)
from fast_llm.layers.language_model.loss.loss import LanguageModelLoss


class LanguageModelLabelEntropyLoss[ConfigType: LanguageModelLabelEntropyLossConfig](LanguageModelLoss[ConfigType]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward_backward(
        self,
        logits: "torch.Tensor",
        kwargs: dict[str, typing.Any],
        split_index: int = 0,
        grad_logits: torch.Tensor | None = None,
    ) -> "tuple[torch.Tensor, torch.Tensor | None]":
        return (
            triton_entropy_loss_forward_backward
            if TritonConfig.enabled(logits.device, self._config.use_triton)
            else fused_entropy_loss_forward_backward
        )(
            logits,
            self._get_labels(kwargs, split_index),
            None,  # Labels are already masked
            grad_logits=grad_logits,
            grad_output=self._get_grad_output(kwargs),
            group=self._parallel_dim.group if self._vocab_parallel else None,
            logits_scale_factor=self._logits_scale_factor,
            target_format=TargetFormat.labels,
            entropy_loss_type=self._config.loss_type,
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
        grad_logits: torch.Tensor | None = None,
    ) -> "tuple[torch.Tensor, torch.Tensor | None]":
        print("logits", logits.shape)
        return (
            triton_entropy_loss_forward_backward
            if TritonConfig.enabled(logits.device, self._config.use_triton)
            else fused_entropy_loss_forward_backward
        )(
            logits,
            self._get_reference_model_logits(self._config.reference_model, kwargs, split_index),
            self._get_loss_mask(kwargs, split_index),
            grad_output=self._get_grad_output(kwargs),
            grad_logits=grad_logits,
            group=self._parallel_dim.group if self._vocab_parallel else None,
            logits_scale_factor=self._logits_scale_factor,
            target_format=TargetFormat.logits,
            entropy_loss_type=self._config.loss_type,
        )
