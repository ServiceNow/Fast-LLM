import typing

import torch

from fast_llm.functional.entropy_loss import fused_predicted_logits_from_labels, fused_softmax_base
from fast_llm.layers.language_model.loss.config import LanguageModelGRPOLossConfig, LanguageModelLossKwargs
from fast_llm.layers.language_model.loss.loss import LanguageModelLoss


class LanguageModelGRPOLoss[ConfigType: LanguageModelGRPOLossConfig](LanguageModelLoss[ConfigType]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._prediction_distance > 0:
            raise NotImplementedError()

    def forward_backward(
        self,
        logits: "torch.Tensor",
        kwargs: dict[str, typing.Any],
        split_index: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return grpo_loss_forward_backward(
            logits,
            self._get_labels(kwargs, split_index),
            self._prepare_target(kwargs[LanguageModelLossKwargs.advantages], kwargs, split_index),
            self._prepare_target(kwargs[LanguageModelLossKwargs.old_log_probabilities], kwargs, split_index),
            grad_output=self._get_grad_output(kwargs),
            group=self._parallel_dim.group if self._vocab_parallel else None,
            epsilon_low=self._config.epsilon_low,
            epsilon_high=self._config.epsilon_high,
            logits_scale_factor=self._logits_scale_factor,
        )


@torch.compile
def grpo_loss_forward_backward(
    logits: torch.Tensor,  # (*batch, vocab)
    target: torch.Tensor,  # (*batch,)
    advantages: torch.Tensor,  # (*batch,)
    old_log_probabilities: torch.Tensor,  # (*batch,)
    grad_output: float | None,
    group: torch.distributed.ProcessGroup | None = None,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.2,
    logits_scale_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    grad_output = None if grad_output is None else grad_output / logits.shape[:-1].numel() * logits_scale_factor
    loss_mask = target >= 0

    logits_norm, exp_logits, sum_exp_logits, _ = fused_softmax_base(logits, logits_scale_factor, group)
    predicted_logits, target_masked, target_mask = fused_predicted_logits_from_labels(
        logits_norm, target, loss_mask, group
    )
    probability_ratio = (predicted_logits - sum_exp_logits.log() - old_log_probabilities).exp()

    per_sample_loss = -torch.min(
        probability_ratio * advantages,
        torch.clamp(probability_ratio, 1 - epsilon_low, 1 + epsilon_high) * advantages,
    )
    per_sample_loss = per_sample_loss * loss_mask
    loss = per_sample_loss.mean()

    if grad_output is None:
        grad = None
    else:
        # loss[a>=0] = -a * min(x, 1 + epsilon_high)  =>  grad[a>=0] = -a * (x <= 1 + epsilon_high)
        # loss[a<=0] = a * max(x, 1 - epsilon_low)  =>  grad[a<=0] = a * (x >= 1 - epsilon_low)
        probability_ratio_grad = (
            grad_output
            * (
                torch.clamp_min(advantages, 0) * (probability_ratio <= 1 + epsilon_high)
                + torch.clamp_max(advantages, 0) * (probability_ratio >= 1 - epsilon_low)
            )
            * loss_mask
        )

        # d(probability_ratio)/d(logits) = - probability_ratio * (predicted_probabilities - target_probabilities)
        # (Sign absorbed in probability_ratio_grad)
        predicted_probabilities = exp_logits / sum_exp_logits.unsqueeze_(-1)
        grad = (probability_ratio_grad * probability_ratio).unsqueeze(-1) * predicted_probabilities.scatter_add(
            -1,
            target_masked.unsqueeze(-1),
            -(loss_mask if target_mask is None else target_mask).unsqueeze(-1).to(torch.float32),
        )
        grad = grad.to(logits.dtype)

    return loss, grad
