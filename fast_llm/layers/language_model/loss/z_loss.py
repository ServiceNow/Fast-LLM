import typing

import torch

from fast_llm.functional.config import TritonConfig
from fast_llm.functional.entropy_loss import fused_softmax_base, z_loss_core
from fast_llm.functional.triton.z_loss import triton_z_loss_forward_backward
from fast_llm.functional.utils import reduce_losses
from fast_llm.layers.language_model.loss.config import LanguageModelZLossConfig
from fast_llm.layers.language_model.loss.loss import LanguageModelLoss
from fast_llm.layers.language_model.loss.monolithic import MonolithicLossSpec


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

    def get_monolithic_spec(
        self, kwargs: dict[str, typing.Any], split_index: int = 0, losses: dict | None = None
    ) -> MonolithicLossSpec | None:
        return MonolithicLossSpec(
            kind="z_loss",
            name=self.name,
            weight=self._weight,
            logits_scale_factor=self._logits_scale_factor,
            grad_output=self._get_grad_output(kwargs),
            divisor=self._get_label_count(kwargs),
            loss_mask=self._get_loss_mask(kwargs, split_index),
        )

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
    grad_output = None if grad_output is None else grad_output / divisor * logits_scale_factor
    _, exp_logits, sum_exp_logits, logits_max = fused_softmax_base(logits, logits_scale_factor, group)
    loss_term, grad = z_loss_core(exp_logits, sum_exp_logits, logits_max, grad_output)

    loss = reduce_losses(loss_term, divisor, loss_mask)

    if grad is not None:
        if loss_mask is not None:
            grad = grad * loss_mask.unsqueeze(-1)
        grad = grad.to(logits.dtype)
        if grad_logits is None:
            grad_logits = grad
        else:
            grad_logits.add_(grad)

    return loss, grad_logits
