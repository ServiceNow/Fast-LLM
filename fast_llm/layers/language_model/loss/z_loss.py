import typing

import torch

from fast_llm.functional.config import TritonConfig
from fast_llm.functional.entropy_loss import z_loss_core
from fast_llm.functional.triton.z_loss import triton_z_loss_forward_backward
from fast_llm.functional.utils import reduce_losses
from fast_llm.layers.language_model.loss.config import LanguageModelZLossConfig
from fast_llm.layers.language_model.loss.loss import CombinableLoss, SingleLoss


class LanguageModelZLoss[ConfigType: LanguageModelZLossConfig](CombinableLoss, SingleLoss[ConfigType]):
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
            loss_mask, grad_output, divisor = arguments
            return triton_z_loss_forward_backward(
                logits,
                loss_mask,
                grad_output=grad_output,
                group=group,
                logits_scale_factor=self._logits_scale_factor,
                grad_logits=grad_logits,
                divisor=divisor,
            )
        loss, grad_logits, _ = self.combinable_forward_backward(logits, group, grad_logits, arguments)
        return loss, grad_logits

    def get_inputs(self, kwargs: dict[str, typing.Any], split_index: int, register: bool) -> tuple:
        return self._get_loss_mask(kwargs, split_index), self._get_grad_output(kwargs), self._get_label_count(kwargs)

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
        """Post-softmax z-loss over the shared softmax, called by both the standalone
        `combinable_forward_backward` and the monolithic head loss. Returns the loss scalar and the uncast,
        masked gradient contribution (the caller casts); z-loss emits no extra outputs."""
        loss_mask, grad_output, divisor = arguments
        grad_output = None if grad_output is None else grad_output / divisor * logits_scale_factor
        loss_term, grad = z_loss_core(exp_logits, sum_exp_logits, logits_max, grad_output)
        loss = reduce_losses(loss_term, divisor, loss_mask)
        if grad is not None and loss_mask is not None:
            grad = grad * loss_mask.unsqueeze(-1)
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
