import typing

import torch

from fast_llm.functional.config import EntropyLossImplementation, EntropyLossType, TargetFormat, TritonConfig
from fast_llm.functional.entropy_loss import entropy_loss_forward_backward
from fast_llm.layers.language_model.loss.config import (
    LanguageModelDistillationLossConfig,
    LanguageModelLabelEntropyLossConfig,
)
from fast_llm.layers.language_model.loss.loss import LanguageModelLoss


def _get_imlementation(
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
        self._implementation = _get_imlementation(
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

        self._implementation = _get_imlementation(
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
