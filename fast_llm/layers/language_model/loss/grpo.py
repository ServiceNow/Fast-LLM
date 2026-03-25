import functools
import typing

import torch

from fast_llm.engine.base_model.config import LossDef
from fast_llm.functional.entropy_loss import fused_predicted_logits_from_labels, fused_softmax_base
from fast_llm.functional.utils import reduce_losses
from fast_llm.layers.language_model.loss.config import LanguageModelGRPOLossConfig, LanguageModelLossKwargs
from fast_llm.layers.language_model.loss.loss import LanguageModelLoss


class LanguageModelGRPOLoss[ConfigType: LanguageModelGRPOLossConfig](LanguageModelLoss[ConfigType]):
    def _forward_backward(
        self,
        logits: "torch.Tensor",
        kwargs: dict[str, typing.Any],
        losses: dict | None = None,
        split_index: int = 0,
        grad_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        loss, grad, new_logprobs_mean = fused_grpo_loss_forward_backward(
            logits,
            self._get_labels(kwargs, split_index),
            self._prepare_target(kwargs[LanguageModelLossKwargs.advantages], split_index),
            self._prepare_target(kwargs[LanguageModelLossKwargs.old_log_probabilities], split_index),
            grad_logits=grad_logits,
            grad_output=self._get_grad_output(kwargs),
            group=self._parallel_dim.group if self._vocab_parallel else None,
            epsilon_low=self._config.epsilon_low,
            epsilon_high=self._config.epsilon_high,
            logits_scale_factor=self._logits_scale_factor,
            num_labels_in_seq=(
                None
                if losses is None
                else self._prepare_target(kwargs[LanguageModelLossKwargs.label_counts], split_index)
            ),
            divisor=self._get_label_count(kwargs),
        )

        self._register_loss(
            self._logprob_metric_name, new_logprobs_mean, losses, reduce_op=torch.distributed.ReduceOp.SUM
        )
        return loss, grad

    def get_loss_definitions(self) -> list[LossDef]:
        return super().get_loss_definitions() + [LossDef(self._logprob_metric_name)]

    def get_preprocessing_config(
        self,
    ) -> dict[str, typing.Any]:
        return {"use_grpo_data": True, "return_label_counts": True}

    @functools.cached_property
    def _logprob_metric_name(self) -> str:
        return f"{self._name}_new_logprobs"


@torch.compile
def fused_grpo_loss_forward_backward(
    logits: torch.Tensor,  # (*batch, vocab)
    target: torch.Tensor,  # (*batch,)
    advantages: torch.Tensor,  # (*batch,)
    old_log_probabilities: torch.Tensor,  # (*batch,)
    grad_logits: torch.Tensor | None = None,
    grad_output: float | None = None,
    group: torch.distributed.ProcessGroup | None = None,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.2,
    logits_scale_factor: float = 1.0,
    num_labels_in_seq: (
        torch.Tensor | None
    ) = None,  # (*batch,) — response-span length broadcast per token, 0 for non-response
    divisor: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    if divisor is None:
        divisor = logits.shape[:-1].numel()
    grad_output = None if grad_output is None else grad_output / divisor * logits_scale_factor
    loss_mask = target >= 0

    logits_norm, exp_logits, sum_exp_logits, _ = fused_softmax_base(logits, logits_scale_factor, group)
    predicted_logits, target_masked, target_mask = fused_predicted_logits_from_labels(
        logits_norm, target, loss_mask, group
    )
    new_log_probs = predicted_logits - sum_exp_logits.log()
    probability_ratio = (new_log_probs - old_log_probabilities).exp()

    losses = -torch.min(
        probability_ratio * advantages,
        torch.clamp(probability_ratio, 1 - epsilon_low, 1 + epsilon_high) * advantages,
    )
    loss = reduce_losses(losses, divisor, loss_mask)

    # Sum of per-sequence mean log-probs, matching pipelinerl's new_logprobs metric:
    #   sum_sum(new_logprobs / num_labels_in_seq, masks_shifted, segments)
    # Dividing by num_labels_in_seq (span length broadcast per token) and summing over masked
    # tokens gives mean logprob per sequence; summing those across sequences matches the deepspeed
    # convention exactly (segments are redundant once num_labels_in_seq is correct).
    # Clamp to avoid 0/0=nan when num_labels_in_seq=0 (padded tokens or fully masked documents)
    # — those positions also have loss_mask=0 so they correctly contribute 0 to the sum.
    new_logprobs_mean = (
        None if num_labels_in_seq is None else (new_log_probs * loss_mask / num_labels_in_seq.clamp(min=1)).sum()
    )

    if grad_output is not None:
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

        if grad_logits is None:
            grad_logits = grad
        else:
            grad_logits.add_(grad)

    return loss, grad_logits, new_logprobs_mean
