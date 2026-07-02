import typing

import torch

from fast_llm.functional.config import TritonConfig
from fast_llm.functional.entropy_loss import fused_softmax_base, z_loss_core
from fast_llm.functional.triton.z_loss import triton_z_loss_forward_backward
from fast_llm.functional.utils import reduce_losses
from fast_llm.layers.language_model.loss.config import LanguageModelZLossConfig
from fast_llm.layers.language_model.loss.loss import LanguageModelLoss


class LanguageModelZLoss[ConfigType: LanguageModelZLossConfig](LanguageModelLoss[ConfigType]):
    def _forward_backward(
        self,
        logits: "torch.Tensor",
        kwargs: dict[str, typing.Any],
        losses: dict | None = None,
        split_index: int = 0,
        grad_logits: torch.Tensor | None = None,
    ) -> "tuple[torch.Tensor, torch.Tensor | None]":
        return (
            triton_z_loss_forward_backward
            if TritonConfig.enabled(logits.device, self._config.use_triton)
            else fused_z_loss_forward_backward
        )(
            logits,
            self._get_loss_mask(kwargs, split_index),
            grad_output=self._get_grad_output(kwargs),
            group=self._parallel_dim.group if self._vocab_parallel else None,
            logits_scale_factor=self._logits_scale_factor,
            grad_logits=grad_logits,
            divisor=self._get_label_count(kwargs),
        )

    def combinable_extract(self, kwargs: dict[str, typing.Any], split_index: int, register: bool) -> tuple:
        return self._get_loss_mask(kwargs, split_index), self._get_grad_output(kwargs), self._get_label_count(kwargs)

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
        loss_mask, grad_output, divisor = arguments
        loss, grad = z_loss_combinable(
            exp_logits, sum_exp_logits, logits_max, loss_mask, grad_output, divisor, logits_scale_factor
        )
        return loss, grad, None

    def get_preprocessing_config(self) -> dict[str, typing.Any]:
        return {"return_prediction_mask": True}


@torch.compile
def z_loss(
    logits: torch.Tensor,
    loss_mask: "torch.Tensor | None" = None,
    logits_scale_factor: float = 1.0,
) -> torch.Tensor:
    # TODO: Replace usage in MoE, move to testing.
    logits = logits.float()
    out = torch.logsumexp(logits if logits_scale_factor == 1.0 else logits * logits_scale_factor, dim=-1) ** 2
    if loss_mask is not None:
        out = out * loss_mask
    return torch.mean(out)


def z_loss_combinable(
    exp_logits: torch.Tensor,  # (*batch, vocab)
    sum_exp_logits: torch.Tensor,  # (*batch,)
    logits_max: torch.Tensor,  # (*batch,)
    loss_mask: torch.Tensor | None,  # (*batch,)
    grad_output: float | None,
    divisor: float,
    logits_scale_factor: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    The post-softmax z-loss block shared by `fused_z_loss_forward_backward` (after its softmax) and the
    monolithic head loss (over the shared softmax). Returns the loss scalar and the uncast, masked gradient
    contribution; the caller casts to the logits dtype (the monolithic path defers that to one final cast).
    """
    grad_output = None if grad_output is None else grad_output / divisor * logits_scale_factor
    loss_term, grad = z_loss_core(exp_logits, sum_exp_logits, logits_max, grad_output)
    loss = reduce_losses(loss_term, divisor, loss_mask)
    if grad is not None and loss_mask is not None:
        grad = grad * loss_mask.unsqueeze(-1)
    return loss, grad


@torch.compile
def fused_z_loss_forward_backward(
    logits: torch.Tensor,
    loss_mask: torch.Tensor | None,
    grad_logits: torch.Tensor | None = None,
    grad_output: float | None = None,
    group: torch.distributed.ProcessGroup | None = None,
    logits_scale_factor: float = 1.0,
    divisor: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Z-loss = mean(logsumexp(logits, dim=-1) ** 2)
    Grad = 2 * log_sum_exp_logits * softmax(logits)
    """
    if divisor is None:
        divisor = logits.shape[:-1].numel()
    _, exp_logits, sum_exp_logits, logits_max = fused_softmax_base(logits, logits_scale_factor, group)
    loss, grad = z_loss_combinable(
        exp_logits, sum_exp_logits, logits_max, loss_mask, grad_output, divisor, logits_scale_factor
    )
    if grad is not None:
        grad = grad.to(logits.dtype)
        if grad_logits is None:
            grad_logits = grad
        else:
            grad_logits.add_(grad)

    return loss, grad_logits
