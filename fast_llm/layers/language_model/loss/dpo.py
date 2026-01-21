import typing

import torch

from fast_llm.layers.language_model.loss.config import LanguageModelDPOLossConfig, LanguageModelLossKwargs
from fast_llm.layers.language_model.loss.loss import LanguageModelLoss, loss_forward_backward


class LanguageModelDPOLoss[ConfigType: LanguageModelDPOLossConfig](LanguageModelLoss[ConfigType]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._prediction_distance > 0:
            raise NotImplementedError()
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

        if self._get_loss_mask(kwargs, split_index) is not None:
            raise NotImplementedError()

        return loss_forward_backward(
            self._get_grad_output(kwargs),
            dpo_loss,
            logits,
            self._get_labels(kwargs, split_index),
            self._get_reference_model_logits(self._config.reference_model, kwargs, split_index),
            kwargs[LanguageModelLossKwargs.chosen_spans],
            kwargs[LanguageModelLossKwargs.rejected_spans],
            self._config.beta,
        )


def dpo_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    reference_model_logits: torch.Tensor,
    chosen_spans: list[list[tuple[int, int]]],
    rejected_spans: list[list[tuple[int, int]]],
    beta: float = 1.0,
    logits_scale_factor: float = 1.0,
) -> torch.Tensor:

    if logits_scale_factor != 1.0:
        # TODO: Make more efficient.
        logits = logits * logits_scale_factor

    policy_log_probabilities = get_target_log_probabilities(logits, targets)
    policy_log_ratios = _get_target_log_probability_for_spans(
        policy_log_probabilities, chosen_spans
    ) - _get_target_log_probability_for_spans(policy_log_probabilities, rejected_spans)

    reference_log_probabilities = get_target_log_probabilities(reference_model_logits.float().detach(), targets)
    reference_log_ratios = _get_target_log_probability_for_spans(
        reference_log_probabilities, chosen_spans
    ) - _get_target_log_probability_for_spans(reference_log_probabilities, rejected_spans)

    # TODO: ====== Shouldn't the sigmoid be computed independently for each document? =======
    return -torch.nn.functional.logsigmoid(beta * (policy_log_ratios - reference_log_ratios)).mean()


def _get_target_log_probability_for_spans(log_probabilities: torch.Tensor, spans: list[list[tuple[int, int]]]):
    return sum(
        log_probabilities[sample_index, begin:end].sum()
        for sample_index, sample_spans in enumerate(spans)
        for begin, end in sample_spans
    )


@torch.compile
def get_target_log_probabilities(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # Avoid negative (masked) labels.
    targets = targets * (targets >= 0)
    # Gather log probabilities corresponding to the target tokens
    return torch.nn.functional.log_softmax(logits, dim=-1).gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
