import typing

import torch

from fast_llm.functional.config import TritonConfig
from fast_llm.functional.entropy_loss import fused_softmax_base
from fast_llm.functional.triton.z_loss import triton_z_loss_forward_backward
from fast_llm.layers.language_model.loss.config import LanguageModelZLossConfig
from fast_llm.layers.language_model.loss.loss import LanguageModelLoss


class LanguageModelZLoss[ConfigType: LanguageModelZLossConfig](LanguageModelLoss[ConfigType]):
    def forward_backward(
        self,
        logits: "torch.Tensor",
        kwargs: dict[str, typing.Any],
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
        )


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
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Z-loss = mean(logsumexp(logits, dim=-1) ** 2)
    Grad = 2 * log_sum_exp_logits * softmax(logits)
    """
    grad_output = None if grad_output is None else grad_output / logits.shape[:-1].numel() * logits_scale_factor
    logits_norm, exp_logits, sum_exp_logits, logits_max = fused_softmax_base(logits, logits_scale_factor, group)
    log_sum_exp_logits = sum_exp_logits.log() + logits_max

    per_sample_loss = log_sum_exp_logits**2
    if loss_mask is not None:
        per_sample_loss = per_sample_loss * loss_mask
    loss = per_sample_loss.mean()

    if grad_output is not None:
        grad_base = 2 * grad_output * (log_sum_exp_logits / sum_exp_logits)
        if loss_mask is not None:
            grad_base = grad_base * loss_mask
        grad = (grad_base.unsqueeze(-1) * exp_logits).to(logits.dtype)
        if grad_logits is None:
            grad_logits = grad
        else:
            grad_logits.add_(grad)

    return loss, grad_logits
