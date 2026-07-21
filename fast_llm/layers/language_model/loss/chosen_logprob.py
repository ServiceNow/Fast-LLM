import math
import typing

import torch

from fast_llm.layers.language_model.loss.config import LanguageModelChosenLogprobLossConfig
from fast_llm.layers.language_model.loss.loss import LanguageModelLoss
from fast_llm.logging import log_tensor


class LanguageModelChosenLogprobLoss[ConfigType: LanguageModelChosenLogprobLossConfig](LanguageModelLoss[ConfigType]):
    """Logs log π(label) per position via the tensor-log pipeline; contributes nothing to gradients."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Don't surface a "chosen_logprob: 0" line in the training metrics.
        self._do_register_loss = False

    def _forward_backward(
        self,
        logits: "torch.Tensor",
        kwargs: dict[str, typing.Any],
        losses: dict | None = None,
        split_index: int = 0,
        grad_logits: torch.Tensor | None = None,
    ) -> "tuple[torch.Tensor, torch.Tensor | None]":
        if self._vocab_parallel:
            raise NotImplementedError("chosen_logprob loss does not support vocab parallel")
        labels = self._get_labels(kwargs, split_index).reshape(-1).long()
        with torch.no_grad():
            log_probs = torch.log_softmax(logits.float() * self._logits_scale_factor, dim=-1)
            # Mask out-of-range labels (e.g. -100 for prompt tokens in RL data) before gather to
            # avoid CUDA assert. Fast-LLM convention: any label < 0 is masked.
            valid = labels >= 0
            safe_labels = labels.clamp(min=0)
            chosen_logprob = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
            chosen_logprob = chosen_logprob[valid]
        # Capture the full tensor: bias is the mean over all positions, not a sampled subset.
        level = math.ceil(math.log2(max(chosen_logprob.numel(), 1))) + 3
        log_tensor(f"Global : {self._name}", chosen_logprob, level=level)
        return torch.zeros((), dtype=logits.dtype, device=logits.device), grad_logits
