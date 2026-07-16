import abc
import typing

import torch

from fast_llm.config import Configurable
from fast_llm.core.ops import split_op
from fast_llm.engine.base_model.config import LossDef
from fast_llm.engine.distributed.config import DistributedConfig, DistributedDimNames
from fast_llm.functional.entropy_loss import softmax_base
from fast_llm.layers.language_model.config import LanguageModelKwargs
from fast_llm.layers.language_model.loss.config import LanguageModelLossConfig, LanguageModelLossKwargs
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    from fast_llm.layers.language_model.loss.monolithic import _TritonContext


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
        self._logits_scale_factor = logits_scale_factor * self._config.logits_scale_factor
        self._weight = weight * self._config.weight
        self._do_register_loss = register_loss
        self._vocab_parallel = distributed_config.tensor_parallel > 1 and vocab_parallel
        self._sequence_parallel = distributed_config.sequence_tensor_parallel and not self._vocab_parallel
        self._parallel_dim = distributed_config.get_distributed_dim(DistributedDimNames.tensor)
        self._sequence_data_dim = distributed_config.get_distributed_dim(DistributedDimNames.sequence_data)
        self._sequence_data_active = self._sequence_data_dim.size > 1

    @abc.abstractmethod
    def forward_backward(
        self,
        logits: "torch.Tensor",
        kwargs: dict[str, typing.Any],
        losses: dict | None = None,
        split_index: int = 0,
        grad_logits: torch.Tensor | None = None,
    ) -> "tuple[torch.Tensor | None, torch.Tensor | None]":
        pass

    def get_loss_definitions(self) -> list[LossDef]:
        return [LossDef(name=self.name)] if self._do_register_loss else []

    def get_preprocessing_config(
        self,
    ) -> dict[str, typing.Any]:
        return {}

    def _register_loss(
        self, name: str, value: torch.Tensor, losses: dict | None, reduce_op=torch.distributed.ReduceOp.SUM
    ):
        if losses is None:
            return
        if self._sequence_parallel:
            # TODO: Async
            torch.distributed.all_reduce(value, op=reduce_op, group=self._parallel_dim.group)
        losses[name].append(value.detach())

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
        # A zero-weight loss contributes an all-zero gradient; return `None` so the backward term drops out.
        return None if grad_output is None or self._weight == 0 else grad_output * self._weight

    def _get_labels(self, kwargs: dict[str, typing.Any], split_index: int = 0):
        return self._prepare_target(kwargs[LanguageModelLossKwargs.labels], split_index)

    def _get_label_count(self, kwargs: dict[str, typing.Any]):
        return kwargs[LanguageModelKwargs.num_labels_in_batch][self._prediction_distance - 1]

    def _get_loss_mask(self, kwargs: dict[str, typing.Any], split_index: int = 0):
        loss_mask = kwargs.get(LanguageModelKwargs.loss_mask)
        return None if loss_mask is None else self._prepare_target(loss_mask, split_index)

    def _get_reference_model_logits(self, reference_model: str, kwargs: dict[str, typing.Any], split_index: int = 0):
        # The head owns the `.logits`; split on `.losses.` so the name resolves whether this loss sits
        # directly under the head or nested inside a composite loss (e.g. `MonolithicLoss`).
        Assert.incl(
            logits_name := self.module_name.split(".losses.")[0] + ".logits",
            reference_hidden_states := kwargs[f"reference_{reference_model}_hidden_states"],
        )
        # The logits are already sequence-parallel if needed, we don't want to split again.
        return self._prepare_target(
            reference_hidden_states[logits_name], split_index, sequence_parallel=False, split_by_distance=False
        )


class SingleLoss[ConfigType: LanguageModelLossConfig](LanguageModelLoss[ConfigType]):
    """A loss emitting a single registered, weighted scalar. Subclasses implement `_forward_backward` (the
    loss math and its gradient); this template skips the disabled case, registers the scalar, and applies the
    per-loss weight. Composite losses satisfy `forward_backward` directly rather than through this template."""

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


class CombinableLoss:
    """Mixin for losses that consume the vocabulary softmax, either standalone through
    `combinable_forward_backward` or several sharing one softmax when fused together. Subclasses implement
    `get_inputs` (eager kwargs -> argument tuple, built outside the compiled boundary) and the `fused_core`
    static method (post-softmax math returning `(loss, uncast_grad, extra)`), and override
    `register_combinable_extras` when they emit outputs beyond the loss scalar. A loss whose gradient can't
    be produced inside the compiled boundary (an eager seam between forward and backward) instead has its
    `fused_core` return `(None, None, forward_state)` and completes its loss/gradient in `finish`."""

    _logits_scale_factor: float

    def get_inputs(self, kwargs: dict[str, typing.Any], split_index: int, register: bool) -> tuple:
        raise NotImplementedError()

    @staticmethod
    def fused_core(
        logits_norm: torch.Tensor,
        exp_logits: torch.Tensor,
        sum_exp_logits: torch.Tensor,
        logits_max: torch.Tensor,
        group: "torch.distributed.ProcessGroup | None",
        logits_scale_factor: float,
        arguments: tuple,
    ) -> tuple[torch.Tensor, torch.Tensor | None, typing.Any]:
        raise NotImplementedError()

    @torch.compile
    def combinable_forward_backward(
        self,
        logits: torch.Tensor,
        group: "torch.distributed.ProcessGroup | None",
        grad_logits: torch.Tensor | None,
        arguments: tuple,
    ) -> tuple[torch.Tensor, torch.Tensor | None, typing.Any]:
        """Standalone realization of a single combinable loss: softmax once, this loss's `fused_core`, then
        cast-and-accumulate the gradient. Shares `fused_core` with the fused path, so it is not a second copy
        of the math."""
        logits_norm, exp_logits, sum_exp_logits, logits_max = softmax_base(logits, self._logits_scale_factor, group)
        loss, grad, extra = self.fused_core(
            logits_norm, exp_logits, sum_exp_logits, logits_max, group, self._logits_scale_factor, arguments
        )
        return loss, self._accumulate_grad(grad, logits.dtype, grad_logits), extra

    @staticmethod
    def _accumulate_grad(
        grad: torch.Tensor | None, logits_dtype: torch.dtype, grad_logits: torch.Tensor | None
    ) -> torch.Tensor | None:
        """Cast a computed logits gradient to the logits dtype and accumulate it into `grad_logits` (in place
        when a buffer exists, otherwise adopting it)."""
        if grad is None:
            return grad_logits
        grad = grad.to(logits_dtype)
        if grad_logits is None:
            return grad
        grad_logits.add_(grad)
        return grad_logits

    def register_combinable_extras(
        self, extra: typing.Any, kwargs: dict[str, typing.Any], losses: dict | None, split_index: int
    ) -> None:
        """Register per-loss outputs beyond the loss scalar (produced by `fused_core`). No-op by default."""

    def finish(
        self,
        loss: torch.Tensor | None,
        extra: typing.Any,
        kwargs: dict[str, typing.Any],
        split_index: int,
        grad_logits: torch.Tensor | None,
        logits_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, typing.Any, torch.Tensor | None]:
        """Complete a loss that couldn't finish inside the compiled shared-softmax boundary, accumulating its
        gradient into `grad_logits` and returning `(loss, extra, grad_logits)`. A passthrough by default; a
        loss with an eager seam runs it here from the forward state its `fused_core` returned as `extra`."""
        return loss, extra, grad_logits

    # The triton monolithic kernel has a fixed per-slot signature, so instead of the compiled loop each loss
    # fills its slot in a shared `_TritonContext`: `triton_add_inputs` packs its kernel inputs (reusing the
    # eager `get_inputs`) and `triton_finish` reads its `(loss, extra)` back from the kernel outputs. A loss
    # whose gradient can't be produced in one kernel launch (an eager seam) sets `triton_needs_forward` and
    # runs the seam in `triton_seam` over the pre-computed shared softmax.
    triton_needs_forward: typing.ClassVar[bool] = False

    def triton_add_inputs(
        self, context: "_TritonContext", kwargs: dict[str, typing.Any], split_index: int, register: bool
    ) -> None:
        raise NotImplementedError()

    def triton_seam(
        self,
        context: "_TritonContext",
        softmax: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None],
        kwargs: dict[str, typing.Any],
        split_index: int,
    ) -> None:
        """Run the eager pre-kernel seam over the shared softmax. No-op unless `triton_needs_forward`."""

    def triton_finish(
        self, context: "_TritonContext", kwargs: dict[str, typing.Any], split_index: int, register: bool
    ) -> tuple[torch.Tensor, typing.Any]:
        raise NotImplementedError()

    def triton_metrics_enabled(self, register: bool) -> bool:
        """Whether this loss emits diagnostic metrics from the kernel's shared softmax this step."""
        return False


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
