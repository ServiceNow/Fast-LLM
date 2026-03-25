import abc
import typing

import torch

from fast_llm.config import Configurable
from fast_llm.core.ops import split_op
from fast_llm.engine.base_model.config import LossDef
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
        prediction_distance: int = 1,
        prediction_heads: int = 1,
        vocab_parallel: bool = False,
        num_splits: int = 1,
        logits_scale_factor: float = 1.0,
        weight: float = 1.0,
        register_loss: bool = False,
    ):
        super().__init__(config)
        Assert.in_range_incl(prediction_distance, 1, prediction_heads)
        self._prediction_distance = prediction_distance
        self._prediction_heads = prediction_heads
        self._name = name
        self._num_splits = num_splits
        self._logits_scale_factor = logits_scale_factor
        self._weight = weight * self._config.weight
        self._do_register_loss = register_loss
        self._vocab_parallel = distributed_config.tensor_parallel > 1 and vocab_parallel
        self._sequence_parallel = distributed_config.sequence_tensor_parallel and not self._vocab_parallel
        self._parallel_dim = distributed_config.get_distributed_dim(DistributedDimNames.tensor)

    def forward_backward(
        self,
        logits: "torch.Tensor",
        kwargs: dict[str, typing.Any],
        losses: dict | None = None,
        split_index: int = 0,
        grad_logits: torch.Tensor | None = None,
    ) -> "tuple[torch.Tensor | None, torch.Tensor | None]":
        if losses is None and self._weight is None:
            # Loss computation is not needed, skip.
            return None, None
        loss, grad = self._forward_backward(logits, kwargs, losses, split_index, grad_logits)
        if self._do_register_loss:
            self._register_loss(self.name, loss, losses)
        return loss if self._weight == 1 else loss * self._weight, grad

    @abc.abstractmethod
    def _forward_backward(
        self,
        logits: "torch.Tensor",
        kwargs: dict[str, typing.Any],
        losses: dict | None = None,
        split_index: int = 0,
        grad_logits: torch.Tensor | None = None,
    ) -> "tuple[torch.Tensor, torch.Tensor | None]":
        pass

    def get_loss_definitions(self) -> list[LossDef]:
        return [LossDef(name=self.name)] if self._do_register_loss else []

    def get_preprocessing_config(
        self,
    ) -> dict[str, typing.Any]:
        return {}

    def _register_loss(
        self, name: str, value: torch.Tensor, losses: dict | None, reduce_op=torch.distributed.ReduceOp.AVG
    ):
        if losses is None:
            return
        if self._sequence_parallel:
            # TODO: Async
            torch.distributed.all_reduce(value, op=reduce_op, group=self._parallel_dim.group)
        losses[name].append(value)

    @property
    def name(self) -> str:
        return self._name

    @property
    def weight(self) -> float:
        return self._weight

    def _prepare_target(
        self,
        target: torch.Tensor | None,
        split_index: int = 0,
        *,
        sequence_parallel: bool = True,
        split_by_distance: bool = True,
    ) -> torch.Tensor | None:
        if split_by_distance:
            target = target[self._prediction_distance - 1]
        # Get the local chunk.
        if sequence_parallel and self._sequence_parallel:
            target = split_op(target, self._parallel_dim.group, 0)

        # Get the chunk for the current split.
        if self._num_splits > 1:
            target = target.chunk(self._num_splits)[split_index]

        return target

    def _get_grad_output(self, kwargs: dict[str, typing.Any]) -> float | None:
        grad_output = kwargs.get(LanguageModelKwargs.grad_output)
        return None if grad_output is None else grad_output * self._weight

    def _get_labels(self, kwargs: dict[str, typing.Any], split_index: int = 0):
        return self._prepare_target(kwargs[LanguageModelLossKwargs.labels], split_index)

    def _get_label_count(self, kwargs: dict[str, typing.Any]):
        return kwargs[LanguageModelKwargs.num_labels_in_batch][self._prediction_distance - 1]

    def _get_loss_mask(self, kwargs: dict[str, typing.Any], split_index: int = 0):
        loss_mask = kwargs.get(LanguageModelKwargs.loss_mask)
        return None if loss_mask is None else self._prepare_target(loss_mask, split_index)

    def _get_reference_model_logits(self, reference_model: str, kwargs: dict[str, typing.Any], split_index: int = 0):
        Assert.incl(
            logits_name := self.module_name.rsplit(".", 2)[0] + f".logits",
            reference_hidden_states := kwargs[f"reference_{reference_model}_hidden_states"],
        )
        # The logits are already sequence-parallel if needed, we don't want to split again.
        return self._prepare_target(
            reference_hidden_states[logits_name], split_index, sequence_parallel=False, split_by_distance=False
        )


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
