import typing

import torch

from fast_llm.functional.config import EntropyLossType, TargetFormat, TritonConfig
from fast_llm.functional.entropy_loss import (
    cross_entropy_from_distribution_core,
    cross_entropy_from_labels_core,
    reverse_kl_from_distribution_core,
)
from fast_llm.functional.triton.entropy_loss import triton_entropy_loss_forward_backward
from fast_llm.functional.utils import reduce_losses
from fast_llm.layers.language_model.loss.config import (
    LanguageModelDistillationLossConfig,
    LanguageModelLabelEntropyLossConfig,
)
from fast_llm.layers.language_model.loss.loss import CombinableLoss, SingleLoss

if typing.TYPE_CHECKING:
    from fast_llm.layers.language_model.loss.monolithic import _TritonContext


class LanguageModelLabelEntropyLoss[ConfigType: LanguageModelLabelEntropyLossConfig](
    CombinableLoss, SingleLoss[ConfigType]
):
    def _forward_backward(
        self,
        logits: "torch.Tensor",
        kwargs: dict[str, typing.Any],
        losses: dict | None = None,
        split_index: int = 0,
        grad_logits: torch.Tensor | None = None,
    ) -> "tuple[torch.Tensor, torch.Tensor | None]":
        arguments = self.get_inputs(kwargs, split_index, losses is not None)
        group = self._parallel_dim.group if self._vocab_parallel else None
        if TritonConfig.enabled(logits.device, self._config.use_triton):
            target, grad_output, divisor = arguments
            loss, _, grad_logits = triton_entropy_loss_forward_backward(
                logits,
                target,
                None,  # Labels are already masked
                grad_logits=grad_logits,
                grad_output=grad_output,
                group=group,
                logits_scale_factor=self._logits_scale_factor,
                target_format=TargetFormat.labels,
                entropy_loss_type=self._config.loss_type,
                divisor=divisor,
            )
            return loss, grad_logits
        loss, grad_logits, _ = self.combinable_forward_backward(logits, group, grad_logits, arguments)
        return loss, grad_logits

    def get_inputs(self, kwargs: dict[str, typing.Any], split_index: int, register: bool) -> tuple:
        return self._get_labels(kwargs, split_index), self._get_grad_output(kwargs), self._get_label_count(kwargs)

    @staticmethod
    def fused_core(
        logits_norm: torch.Tensor,
        exp_logits: torch.Tensor,
        sum_exp_logits: torch.Tensor,
        logits_max: torch.Tensor,
        group: "torch.distributed.ProcessGroup | None",
        logits_scale_factor: float,
        arguments: tuple,
    ) -> tuple[torch.Tensor, torch.Tensor | None, None]:
        """Post-softmax cross-entropy-from-labels over the shared softmax. Returns the loss scalar and the
        uncast, masked gradient (the caller casts); no extra outputs. For labels, forward-KL is identical to
        cross-entropy (one-hot target entropy is zero)."""
        target, grad_output, divisor = arguments
        loss_mask = target >= 0
        grad_output = None if grad_output is None else grad_output / divisor * logits_scale_factor
        per_sample_loss, grad = cross_entropy_from_labels_core(
            logits_norm, exp_logits, sum_exp_logits, target, loss_mask, grad_output, group
        )
        loss = reduce_losses(per_sample_loss, divisor, loss_mask)
        if grad is not None:
            grad = grad * loss_mask.unsqueeze(-1)
        return loss, grad, None

    def triton_add_inputs(
        self, context: "_TritonContext", kwargs: dict[str, typing.Any], split_index: int, register: bool
    ) -> None:
        labels, grad_output, divisor = self.get_inputs(kwargs, split_index, register)
        if context.labels is None:
            context.labels = labels
        if context.divisor is None:
            context.divisor = divisor
        context.ce = (grad_output,)

    def triton_finish(
        self, context: "_TritonContext", kwargs: dict[str, typing.Any], split_index: int, register: bool
    ) -> tuple[torch.Tensor, None]:
        return context.ce_loss, None


class LanguageModelDistillationLoss[ConfigType: LanguageModelDistillationLossConfig](
    CombinableLoss, SingleLoss[ConfigType]
):
    def _forward_backward(
        self,
        logits: "torch.Tensor",
        kwargs: dict[str, typing.Any],
        losses: dict | None = None,
        split_index: int = 0,
        grad_logits: torch.Tensor | None = None,
    ) -> "tuple[torch.Tensor, torch.Tensor | None]":
        arguments = self.get_inputs(kwargs, split_index, losses is not None)
        group = self._parallel_dim.group if self._vocab_parallel else None
        if TritonConfig.enabled(logits.device, self._config.use_triton):
            target, loss_mask, grad_output, divisor, entropy_loss_type, temperature = arguments
            loss, _, grad_logits = triton_entropy_loss_forward_backward(
                logits,
                target,
                loss_mask,
                grad_output=grad_output,
                grad_logits=grad_logits,
                group=group,
                logits_scale_factor=self._logits_scale_factor,
                temperature=temperature,
                target_format=TargetFormat.logits,
                entropy_loss_type=entropy_loss_type,
                divisor=divisor,
            )
            return loss, grad_logits
        loss, grad_logits, _ = self.combinable_forward_backward(logits, group, grad_logits, arguments)
        return loss, grad_logits

    def get_inputs(self, kwargs: dict[str, typing.Any], split_index: int, register: bool) -> tuple:
        return (
            self._get_reference_model_logits(self._config.reference_model, kwargs, split_index),
            self._get_loss_mask(kwargs, split_index),
            self._get_grad_output(kwargs),
            self._get_label_count(kwargs),
            self._config.loss_type,
            self._config.temperature,
        )

    @staticmethod
    def fused_core(
        logits_norm: torch.Tensor,
        exp_logits: torch.Tensor,
        sum_exp_logits: torch.Tensor,
        logits_max: torch.Tensor,
        group: "torch.distributed.ProcessGroup | None",
        logits_scale_factor: float,
        arguments: tuple,
    ) -> tuple[torch.Tensor, torch.Tensor | None, None]:
        """Post-softmax distillation over the shared student softmax (cross-entropy / forward-KL / reverse-KL
        from a teacher distribution, adding a teacher softmax at scale `logits_scale_factor / temperature`).
        Returns the loss scalar and the uncast, masked gradient (the caller casts); no extra outputs."""
        target, loss_mask, grad_output, divisor, entropy_loss_type, temperature = arguments
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
        loss = reduce_losses(per_sample_loss, divisor, loss_mask)
        if grad is not None and loss_mask is not None:
            grad = grad * loss_mask.unsqueeze(-1)
        return loss, grad, None

    def triton_add_inputs(
        self, context: "_TritonContext", kwargs: dict[str, typing.Any], split_index: int, register: bool
    ) -> None:
        target, loss_mask, grad_output, divisor, loss_type, temperature = self.get_inputs(
            kwargs, split_index, register
        )
        if context.divisor is None:
            context.divisor = divisor
        context.distillation = (target, loss_mask, grad_output, loss_type, temperature)

    def triton_finish(
        self, context: "_TritonContext", kwargs: dict[str, typing.Any], split_index: int, register: bool
    ) -> tuple[torch.Tensor, None]:
        return context.dist_loss, None

    def get_preprocessing_config(self) -> dict[str, typing.Any]:
        return {"return_prediction_mask": True}
