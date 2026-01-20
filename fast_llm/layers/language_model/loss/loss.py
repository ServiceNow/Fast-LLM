import abc
import typing

import torch

from fast_llm.config import Configurable
from fast_llm.core.ops import split_op
from fast_llm.engine.distributed.config import DistributedConfig, DistributedDimNames
from fast_llm.layers.language_model.config import LanguageModelKwargs
from fast_llm.layers.language_model.loss.config import LanguageModelLossConfig, LanguageModelLossKwargs
from fast_llm.utils import Assert


class LanguageModelLoss[ConfigType: LanguageModelLossConfig](Configurable[ConfigType]):
    def __init__(
        self,
        config: ConfigType,
        distributed_config: DistributedConfig,
        *,
        prediction_distance: int = 0,
        prediction_heads: int = 1,
        vocab_parallel: bool = False,
        num_splits: int = 1,
        logits_scale_factor: float = 1.0,
        weight: float = 1.0,
    ):
        super().__init__(config)
        Assert.in_range(prediction_distance, 0, prediction_heads)
        self._prediction_distance = prediction_distance
        self._prediction_heads = prediction_heads
        self._num_splits = num_splits
        self._logits_scale_factor = logits_scale_factor
        self._weight = weight * self._config.weight
        self._vocab_parallel = distributed_config.tensor_parallel > 1 and vocab_parallel
        self._sequence_parallel = distributed_config.sequence_tensor_parallel and not self._vocab_parallel
        self._parallel_dim = distributed_config.get_distributed_dim(DistributedDimNames.tensor)

    @abc.abstractmethod
    def forward_backward(
        self,
        logits: "torch.Tensor",
        kwargs: dict[str, typing.Any],
        grad_output: float | None = None,
        split_index: int = 0,
    ) -> "tuple[torch.Tensor, torch.Tensor | None]":
        pass

    def _prepare_target(
        self,
        target: torch.Tensor | None,
        kwargs: dict[str, typing.Any],
        split_index: int = 0,
        *,
        multi_token_format: bool = False,
    ) -> torch.Tensor | None:
        # MTP shift
        if multi_token_format and self._prediction_heads > 1:
            sequence_first: bool = kwargs[LanguageModelLossKwargs.sequence_first]
            sequence_q_length = target.size(1 - sequence_first) + 1 - self._prediction_heads
            target_slice = slice(self._prediction_distance, self._prediction_distance + sequence_q_length)
            target = target[target_slice] if sequence_first else target[:, target_slice]

        # Flatten the batch and sequence dimensions.
        target = target.flatten(0, -2)

        # Get the local chunk.
        if self._sequence_parallel:
            target = split_op(target, self._parallel_dim.group, 0)

        # Get the chunk for the current split.
        if self._num_splits > 1:
            target = target.chunk(self._num_splits)[split_index]

        return target

    def _get_labels(self, kwargs: dict[str, typing.Any], split_index: int = 0):
        return self._prepare_target(
            kwargs[LanguageModelLossKwargs.labels], kwargs, split_index, multi_token_format=True
        )

    def _get_loss_mask(self, kwargs: dict[str, typing.Any], split_index: int = 0):
        loss_mask = kwargs.get(LanguageModelKwargs.loss_mask)
        return None if loss_mask is None else self._prepare_target(loss_mask, kwargs, split_index)

    def _get_reference_model_logits(self, reference_model: str, kwargs: dict[str, typing.Any], split_index: int = 0):
        return self._prepare_target(kwargs[f"{reference_model}_logits"], kwargs, split_index)


def loss_forward_backward(
    grad_output: float | None, fn: typing.Callable, input_: torch.Tensor, *args, **kwargs
) -> tuple[torch.Tensor, torch.Tensor | None]:
    with torch.set_grad_enabled(grad_output is not None):
        input_ = input_.detach().requires_grad_(grad_output is not None)
        loss = fn(input_, *args, **kwargs)
        if grad_output is None:
            grad = None
        else:
            loss.backward(torch.full_like(loss, grad_output))
            grad = input_.grad.detach().to(input_.dtype)

    return loss, grad
