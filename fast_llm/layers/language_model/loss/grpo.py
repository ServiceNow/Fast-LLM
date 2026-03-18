import typing

import torch

from fast_llm.core.ops import split_op
from fast_llm.functional.entropy_loss import fused_predicted_logits_from_labels, fused_softmax_base
from fast_llm.layers.language_model.loss.config import LanguageModelGRPOLossConfig, LanguageModelLossKwargs
from fast_llm.layers.language_model.loss.loss import LanguageModelLoss


def _compute_num_labels_in_seq(loss_mask: torch.Tensor) -> torch.Tensor:
    """For each response token, compute the total number of response tokens in its contiguous span.

    Non-response tokens get 0. Fully vectorized.

    In a packed sequence of chat conversations, each contiguous block of response tokens
    (loss_mask == 1) corresponds to one sequence's completion. This mirrors pipelinerl's
    ``num_labels_in_seq``: a per-token constant equal to the span length, used to convert
    a sum of logprobs over a span into a per-sequence mean.

    Args:
        loss_mask: 1D bool tensor (True = response token).
    Returns:
        1D float tensor, same length.
    """
    # Assign a unique ID to each contiguous response span; non-response tokens get ID 0.
    prev = torch.cat([loss_mask.new_zeros(1), loss_mask[:-1]])
    span_id = (loss_mask & ~prev).int().cumsum(0) * loss_mask.int()

    # Count tokens per span with a single scatter_add, then broadcast back.
    n = int(span_id.max().item()) + 1
    counts = torch.zeros(n, dtype=torch.float, device=loss_mask.device)
    counts.scatter_add_(0, span_id, torch.ones(len(span_id), dtype=torch.float, device=loss_mask.device))
    counts[0] = 1  # non-response positions get denominator=1 so 0/1=0 (avoids 0/0=nan)
    return counts[span_id]


class LanguageModelGRPOLoss[ConfigType: LanguageModelGRPOLossConfig](LanguageModelLoss[ConfigType]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._prediction_distance > 0:
            raise NotImplementedError()

    @property
    def extra_metric_names(self) -> list[str]:
        return ["new_logprobs"]

    def forward_backward(
        self,
        logits: "torch.Tensor",
        kwargs: dict[str, typing.Any],
        split_index: int = 0,
        grad_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Compute num_labels_in_seq on the full (batch, seq) labels before any parallelism
        # split, then apply the same flatten -> sequence-parallel split -> cross-entropy-split
        # transforms that _prepare_target applies to all other target tensors.
        # This gives the correct span lengths even when sequence parallelism slices the sequence
        # across TP ranks.
        full_loss_mask = kwargs[LanguageModelLossKwargs.labels] >= 0
        num_labels_in_seq = _compute_num_labels_in_seq(full_loss_mask)
        if self._sequence_parallel:
            num_labels_in_seq = split_op(num_labels_in_seq, self._parallel_dim.group, 0)
        if self._num_splits > 1:
            num_labels_in_seq = num_labels_in_seq.chunk(self._num_splits)[split_index]

        loss, grad, new_logprobs_mean = fused_grpo_loss_forward_backward(
            logits,
            self._get_labels(kwargs, split_index),
            self._prepare_target(kwargs[LanguageModelLossKwargs.advantages], kwargs, split_index),
            self._prepare_target(kwargs[LanguageModelLossKwargs.old_log_probabilities], kwargs, split_index),
            grad_logits=grad_logits,
            grad_output=self._get_grad_output(kwargs),
            group=self._parallel_dim.group if self._vocab_parallel else None,
            epsilon_low=self._config.epsilon_low,
            epsilon_high=self._config.epsilon_high,
            logits_scale_factor=self._logits_scale_factor,
            num_labels_in_seq=num_labels_in_seq,
        )
        kwargs[f"_metric_{self._name}_new_logprobs"] = new_logprobs_mean
        return loss, grad

    def get_preprocessing_config(
        self,
    ) -> dict[str, typing.Any]:
        return {"use_grpo_data": True}


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
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    grad_output = None if grad_output is None else grad_output / logits.shape[:-1].numel() * logits_scale_factor
    loss_mask = target >= 0

    logits_norm, exp_logits, sum_exp_logits, _ = fused_softmax_base(logits, logits_scale_factor, group)
    predicted_logits, target_masked, target_mask = fused_predicted_logits_from_labels(
        logits_norm, target, loss_mask, group
    )
    new_log_probs = predicted_logits - sum_exp_logits.log()
    probability_ratio = (new_log_probs - old_log_probabilities).exp()

    per_sample_loss = -torch.min(
        probability_ratio * advantages,
        torch.clamp(probability_ratio, 1 - epsilon_low, 1 + epsilon_high) * advantages,
    )
    per_sample_loss = per_sample_loss * loss_mask
    loss = per_sample_loss.mean()

    # Sum of per-sequence mean log-probs, matching pipelinerl's new_logprobs metric:
    #   sum_sum(new_logprobs / num_labels_in_seq, masks_shifted, segments)
    # Dividing by num_labels_in_seq (span length broadcast per token) and summing over masked
    # tokens gives mean logprob per sequence; summing those across sequences matches the deepspeed
    # convention exactly (segments are redundant once num_labels_in_seq is correct).
    new_logprobs_mean = None if num_labels_in_seq is None else (new_log_probs * loss_mask / num_labels_in_seq).sum()

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
