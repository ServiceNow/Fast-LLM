import typing

import torch

from fast_llm.functional.config import EntropyLossImplementation, EntropyLossType, TargetFormat, TritonConfig
from fast_llm.functional.entropy_loss import entropy_loss_forward_backward
from fast_llm.layers.language_model.loss.config import LanguageModelLabelEntropyLossConfig
from fast_llm.layers.language_model.loss.loss import LanguageModelLoss


class LanguageModelLabelEntropyLoss[ConfigType: LanguageModelLabelEntropyLossConfig](LanguageModelLoss[ConfigType]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._implementation = self._config.implementation

        if self._implementation == EntropyLossImplementation.auto:
            if self._vocab_parallel:
                self._implementation = EntropyLossImplementation.fused
            elif (
                TritonConfig.TRITON_ENABLED
                and torch.cuda.is_available()
                and self._config.loss_type == EntropyLossType.cross_entropy
            ):
                self._implementation = EntropyLossImplementation.triton
            else:
                self._implementation = EntropyLossImplementation.fused
        if (
            self._implementation == EntropyLossImplementation.triton
            and self._config.loss_type != EntropyLossType.cross_entropy
        ):
            raise NotImplementedError(self._config.loss_type)

    def forward_backward(
        self,
        logits: "torch.Tensor",
        kwargs: dict[str, typing.Any],
        grad_output: float | None = None,
        split_index: int = 0,
    ) -> "tuple[torch.Tensor, torch.Tensor | None]":
        return entropy_loss_forward_backward(
            logits,
            self._get_labels(kwargs, split_index),
            None,  # Labels are already masked
            grad_output=grad_output,
            group=self._parallel_dim.group if self._vocab_parallel else None,
            implementation=self._implementation,
            logits_scale_factor=self._logits_scale_factor,
            target_format=TargetFormat.labels,
            entropy_loss_type=self._config.loss_type,
        )


class LanguageModelLabelDistillationLoss[ConfigType: LanguageModelLabelEntropyLossConfig](
    LanguageModelLoss[ConfigType]
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._prediction_distance > 0:
            raise NotImplementedError()

        if self._implementation == EntropyLossImplementation.auto:
            if self._vocab_parallel:
                self._implementation = EntropyLossImplementation.fused
            elif (
                TritonConfig.TRITON_ENABLED
                and torch.cuda.is_available()
                and self._config.loss_type == EntropyLossType.cross_entropy
            ):
                self._implementation = EntropyLossImplementation.triton
            else:
                self._implementation = EntropyLossImplementation.fused
        if (
            self._implementation == EntropyLossImplementation.triton
            and self._config.loss_type != EntropyLossType.cross_entropy
        ):
            raise NotImplementedError(self._config.loss_type)

    def forward_backward(
        self,
        logits: "torch.Tensor",
        kwargs: dict[str, typing.Any],
        grad_output: float | None = None,
        split_index: int = 0,
    ) -> "tuple[torch.Tensor, torch.Tensor | None]":
        return entropy_loss_forward_backward(
            logits,
            self._get_reference_model_logits(self._config.reference_model, kwargs, split_index),
            self._get_loss_mask(kwargs, split_index),
            grad_output=grad_output,
            group=self._parallel_dim.group if self._vocab_parallel else None,
            implementation=self._implementation,
            logits_scale_factor=self._logits_scale_factor,
            target_format=TargetFormat.logits,
            entropy_loss_type=self._config.loss_type,
        )
