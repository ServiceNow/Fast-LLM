import typing

import torch

from fast_llm.layers.language_model.loss.config import LanguageModelZLossConfig
from fast_llm.layers.language_model.loss.loss import LanguageModelLoss, loss_forward_backward


class LanguageModelZLoss[ConfigType: LanguageModelZLossConfig](LanguageModelLoss[ConfigType]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: Support vocab_parallel
        if self._vocab_parallel:
            raise NotImplementedError()

    def forward_backward(
        self,
        logits: "torch.Tensor",
        kwargs: dict[str, typing.Any],
        split_index: int = 0,
    ) -> "tuple[torch.Tensor, torch.Tensor | None]":
        return loss_forward_backward(
            self._get_grad_output(kwargs),
            z_loss,
            logits,
            self._get_loss_mask(kwargs, split_index),
            self._logits_scale_factor,
        )


@torch.compile
def z_loss(
    logits: torch.Tensor,
    loss_mask: "torch.Tensor | None" = None,
    logits_scale_factor: float = 1.0,
) -> torch.Tensor:
    """
    Z-loss = mean(logsumexp(logits, dim=-1) ** 2)
    """
    out = torch.logsumexp(logits if logits_scale_factor == 1.0 else logits * logits_scale_factor, dim=-1) ** 2
    if loss_mask is not None:
        out = out * loss_mask
    return torch.mean(out)
