import typing

import torch

from fast_llm.layers.language_model.loss.config import LanguageModelGRPOLossConfig
from fast_llm.layers.language_model.loss.loss import LanguageModelLoss, loss_forward_backward


class LanguageModelGRPOLoss[ConfigType: LanguageModelGRPOLossConfig](LanguageModelLoss[ConfigType]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: Support vocab_parallel
        if self._num_splits > 1:
            raise NotImplementedError()
        if self._prediction_distance > 0:
            raise NotImplementedError()
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
            grpo_loss,
            logits,
            self._get_loss_mask(kwargs, split_index),
            self._get_labels(kwargs, split_index),
            advantages,
            old_log_probabilities,
            self._config.epsilon_low,
            self._config.epsilon_high,
            self._logits_scale_factor,
        )


def grpo_loss(
    logits: torch.Tensor,
    loss_mask: "torch.Tensor | None",
    labels: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probabilities: torch.Tensor,
    epsilon_low: float,
    epsilon_high: float,
    logits_scale_factor: float = 1.0,
) -> torch.Tensor:
    if logits_scale_factor != 1.0:
        # TODO: Make more efficient.
        logits = logits * logits_scale_factor

    # Log probabilities.
    logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
    target_log_probabilities = torch.gather(logprobs, dim=2, index=labels.unsqueeze(2)).squeeze(2)
    probability_ratio = torch.exp(target_log_probabilities - old_log_probabilities)
    loss = -torch.min(
        probability_ratio * advantages,
        torch.clamp(probability_ratio, 1 - epsilon_low, 1 + epsilon_high) * advantages,
    )
    if loss_mask is not None:
        loss = loss * loss_mask
    return loss.mean()
