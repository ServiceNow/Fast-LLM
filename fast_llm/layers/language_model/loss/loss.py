import abc
import typing

import torch

from fast_llm.config import Configurable
from fast_llm.core.ops import split_op
from fast_llm.engine.distributed.config import DistributedConfig, DistributedDimNames
from fast_llm.layers.language_model.config import LanguageModelKwargs
from fast_llm.layers.language_model.loss.config import LanguageModelLossConfig, LanguageModelLossKwargs
from fast_llm.utils import Assert


class LanguageModelLoss[ConfigType: LanguageModelLossConfig](Configurable[ConfigType], torch.nn.Module):
    def __init__(
        self,
        config: ConfigType,
        distributed_config: DistributedConfig,
        *,
        name: str,
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
        self._name = name
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
        split_index: int = 0,
        grad_logits: torch.Tensor | None = None,
    ) -> "tuple[torch.Tensor, torch.Tensor | None]":
        pass

    @property
    def name(self) -> str:
        return self._name

    @property
    def weight(self) -> float:
        return self._weight

    def _prepare_target(
        self,
        target: torch.Tensor | None,
        kwargs: dict[str, typing.Any],
        split_index: int = 0,
        *,
        multi_token_format: bool = False,
        sequence_parallel: bool = True,
    ) -> torch.Tensor | None:
        # MTP shift
        if multi_token_format and self._prediction_heads > 1:
            sequence_q = kwargs[LanguageModelKwargs.sequence_q_dim].size
            target = target.unflatten(
                0, (kwargs[LanguageModelKwargs.batch_dim].size, sequence_q + self._prediction_heads - 1)
            )[:, self._prediction_distance : self._prediction_distance + sequence_q].flatten(0, 1)

        # Get the local chunk.
        if sequence_parallel and self._sequence_parallel:
            target = split_op(target, self._parallel_dim.group, 0)

        # Get the chunk for the current split.
        if self._num_splits > 1:
            target = target.chunk(self._num_splits)[split_index]

        return target

    def _get_grad_output(self, kwargs: dict[str, typing.Any]) -> float | None:
        grad_output = kwargs.get(LanguageModelKwargs.grad_output)
        if grad_output is not None:
            grad_output = (
                grad_output
                * self._weight
                / (self._parallel_dim.size if self._sequence_parallel else 1)
                / self._num_splits
            )
        return grad_output

    def _get_labels(self, kwargs: dict[str, typing.Any], split_index: int = 0):
        return self._prepare_target(
            kwargs[LanguageModelLossKwargs.labels], kwargs, split_index, multi_token_format=True
        )

    def _get_loss_mask(self, kwargs: dict[str, typing.Any], split_index: int = 0):
        loss_mask = kwargs.get(LanguageModelKwargs.loss_mask)
        return None if loss_mask is None else self._prepare_target(loss_mask, kwargs, split_index)

    def _get_reference_model_logits(self, reference_model: str, kwargs: dict[str, typing.Any], split_index: int = 0):
        assert self._prediction_distance == 0
        Assert.incl(
            logits_name := self.module_name.rsplit(".", 2)[0] + f".logits",
            reference_hidden_states := kwargs[f"reference_{reference_model}_hidden_states"],
        )
        # The logits are already sequence-parallel if needed, we don't want to split again.
        return self._prepare_target(reference_hidden_states[logits_name], kwargs, split_index, sequence_parallel=False)


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
            grad = input_.grad.detach()

    return loss, grad
