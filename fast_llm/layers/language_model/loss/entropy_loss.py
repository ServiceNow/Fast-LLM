import typing

import torch

from fast_llm.functional.config import EntropyLossType, TargetFormat, TritonConfig
from fast_llm.functional.entropy_loss import (
    cross_entropy_from_distribution_core,
    cross_entropy_from_labels_core,
    fused_entropy_loss_forward_backward,
    reverse_kl_from_distribution_core,
)
from fast_llm.functional.triton.entropy_loss import triton_entropy_loss_forward_backward
from fast_llm.layers.language_model.loss.config import (
    LanguageModelDistillationLossConfig,
    LanguageModelLabelEntropyLossConfig,
)
from fast_llm.layers.language_model.loss.loss import LanguageModelLoss


class LanguageModelLabelEntropyLoss[ConfigType: LanguageModelLabelEntropyLossConfig](LanguageModelLoss[ConfigType]):
    def _forward_backward(
        self,
        logits: "torch.Tensor",
        kwargs: dict[str, typing.Any],
        losses: dict | None = None,
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
            divisor=self._get_label_count(kwargs),
        )

    def combinable_extract(self, kwargs: dict[str, typing.Any], split_index: int, register: bool) -> tuple:
        return self._get_labels(kwargs, split_index), self._get_grad_output(kwargs), self._get_label_count(kwargs)

    def combinable_core(
        self,
        logits_norm: torch.Tensor,
        exp_logits: torch.Tensor,
        sum_exp_logits: torch.Tensor,
        logits_max: torch.Tensor,
        group: "torch.distributed.ProcessGroup | None",
        logits_scale_factor: float,
        arguments: tuple,
    ) -> tuple[torch.Tensor, torch.Tensor | None, None]:
        # For labels, forward-KL is identical to cross-entropy (one-hot target entropy is zero).
        target, grad_output, divisor = arguments
        loss, grad = cross_entropy_from_labels_combinable(
            logits_norm, exp_logits, sum_exp_logits, group, target, grad_output, divisor, logits_scale_factor
        )
        return loss, grad, None


class LanguageModelDistillationLoss[ConfigType: LanguageModelDistillationLossConfig](LanguageModelLoss[ConfigType]):
    def _forward_backward(
        self,
        logits: "torch.Tensor",
        kwargs: dict[str, typing.Any],
        losses: dict | None = None,
        split_index: int = 0,
        grad_logits: torch.Tensor | None = None,
    ) -> "tuple[torch.Tensor, torch.Tensor | None]":
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
            temperature=self._config.temperature,
            target_format=TargetFormat.logits,
            entropy_loss_type=self._config.loss_type,
            divisor=self._get_label_count(kwargs),
        )

    def combinable_extract(self, kwargs: dict[str, typing.Any], split_index: int, register: bool) -> tuple:
        return (
            self._get_reference_model_logits(self._config.reference_model, kwargs, split_index),
            self._get_loss_mask(kwargs, split_index),
            self._get_grad_output(kwargs),
            self._get_label_count(kwargs),
            self._config.loss_type,
            self._config.temperature,
        )

    def combinable_core(
        self,
        logits_norm: torch.Tensor,
        exp_logits: torch.Tensor,
        sum_exp_logits: torch.Tensor,
        logits_max: torch.Tensor,
        group: "torch.distributed.ProcessGroup | None",
        logits_scale_factor: float,
        arguments: tuple,
    ) -> tuple[torch.Tensor, torch.Tensor | None, None]:
        target, loss_mask, grad_output, divisor, entropy_loss_type, temperature = arguments
        loss, grad = distillation_combinable(
            logits_norm,
            exp_logits,
            sum_exp_logits,
            group,
            target,
            loss_mask,
            grad_output,
            divisor,
            logits_scale_factor,
            entropy_loss_type,
            temperature,
        )
        return loss, grad, None

    def get_preprocessing_config(self) -> dict[str, typing.Any]:
        return {"return_prediction_mask": True}


def cross_entropy_from_labels_combinable(
    logits_norm: torch.Tensor,  # (*batch, vocab)
    exp_logits: torch.Tensor,  # (*batch, vocab)
    sum_exp_logits: torch.Tensor,  # (*batch,)
    group: "torch.distributed.ProcessGroup | None",
    target: torch.Tensor,  # (*batch,)
    grad_output: float | None,
    divisor: float,
    logits_scale_factor: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Post-softmax cross-entropy-from-labels block over the shared softmax: loss scalar + uncast, masked
    gradient. The caller casts to the logits dtype (the monolithic path defers that to one final cast)."""
    loss_mask = target >= 0
    grad_output = None if grad_output is None else grad_output / divisor * logits_scale_factor
    per_sample_loss, grad = cross_entropy_from_labels_core(
        logits_norm, exp_logits, sum_exp_logits, target, loss_mask, grad_output, group
    )
    loss = (per_sample_loss * loss_mask).sum() / divisor
    if grad is not None:
        grad = grad * loss_mask.unsqueeze(-1)
    return loss, grad


def distillation_combinable(
    logits_norm: torch.Tensor,  # (*batch, vocab)
    exp_logits: torch.Tensor,  # (*batch, vocab)
    sum_exp_logits: torch.Tensor,  # (*batch,)
    group: "torch.distributed.ProcessGroup | None",
    target: torch.Tensor,  # (*batch, vocab) teacher logits
    loss_mask: torch.Tensor | None,  # (*batch,)
    grad_output: float | None,
    divisor: float,
    logits_scale_factor: float,
    entropy_loss_type: EntropyLossType,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Post-softmax distillation block (cross-entropy / forward-KL / reverse-KL from a teacher distribution)
    over the shared student softmax: loss scalar + uncast, masked gradient (the caller casts)."""
    grad_output = None if grad_output is None else grad_output / divisor * logits_scale_factor
    if entropy_loss_type == EntropyLossType.reverse_kl:
        per_sample_loss, grad = reverse_kl_from_distribution_core(
            logits_norm,
            exp_logits,
            sum_exp_logits,
            target,
            grad_output,
            logits_scale_factor,
            TargetFormat.logits,
            group,
            temperature,
        )
    else:
        per_sample_loss, grad = cross_entropy_from_distribution_core(
            logits_norm,
            exp_logits,
            sum_exp_logits,
            target,
            grad_output,
            logits_scale_factor,
            TargetFormat.logits,
            group,
            temperature,
            return_kl_loss=entropy_loss_type == EntropyLossType.forward_kl,
        )
    if loss_mask is not None:
        per_sample_loss = per_sample_loss * loss_mask
    loss = per_sample_loss.sum() / divisor
    if grad is not None and loss_mask is not None:
        grad = grad * loss_mask.unsqueeze(-1)
    return loss, grad
