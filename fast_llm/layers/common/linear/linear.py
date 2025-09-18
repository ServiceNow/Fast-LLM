import logging
import typing

import torch

from fast_llm.engine.base_model.config import ResourceUsageConfig
from fast_llm.engine.distributed.config import DistributedDim
from fast_llm.functional.autograd import wrap_forward_backward
from fast_llm.functional.linear import (
    input_parallel_linear_autograd,
    input_parallel_linear_backward,
    input_parallel_linear_forward,
    linear_backward,
    linear_forward,
    output_parallel_linear_backward,
    output_parallel_linear_forward,
)
from fast_llm.tensor import ParameterMeta, TensorMeta
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)


class LinearLike(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._forward = wrap_forward_backward(self.forward_only, self.backward)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        return self._forward(input_)

    def forward_only(self, input_: torch.Tensor):
        raise NotImplementedError()

    def backward(self, grad_output: torch.Tensor, context: typing.Any) -> torch.Tensor:
        raise NotImplementedError()

    def get_compute_usage(self, input_: TensorMeta, config: ResourceUsageConfig) -> int:
        raise NotImplementedError()


class LinearBase(LinearLike):
    """
    A base module for linear layers holding weights and biases.
    """

    def __init__(
        self,
        weight: ParameterMeta,
        bias: ParameterMeta | None,
        *,
        transposed_weight: bool = False,
    ):
        super().__init__()
        self.weight = weight
        self.bias = bias
        self._transposed_weight = transposed_weight
        if self._transposed_weight:
            self._input_dim, self._output_dim = self.weight.dims
        else:
            self._output_dim, self._input_dim = self.weight.dims

    @property
    def transposed_weight(self) -> bool:
        return self._transposed_weight

    def get_compute_usage(self, input_: TensorMeta, config: ResourceUsageConfig) -> int:
        Assert.eq(input_.size(-1), self._input_dim.size)
        return (
            2
            * (config.forward + 2 * config.backward)
            * (input_.global_shape if config.global_ else input_).numel()
            * (self._output_dim.global_size if config.global_ else self._output_dim.size)
        )


class Linear(LinearBase):
    """
    A basic linear layer without tensor parallelism.
    """

    def forward_only(
        self, input_: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]]:
        return linear_forward(input_, weight=self.weight, bias=self.bias, transposed_weight=self._transposed_weight)

    def backward(self, grad_output: torch.Tensor, context) -> torch.Tensor:  # noqa
        return linear_backward(grad_output, context)


class OutputParallelLinear(LinearBase):
    """
    A linear layer with output (column) tensor parallelism.
    """

    def __init__(
        self,
        weight: ParameterMeta,
        bias: ParameterMeta | None,
        *,
        transposed_weight: bool = False,
        parallel_dim: DistributedDim,
        sequence_parallel: bool = False,
    ):
        super().__init__(weight, bias, transposed_weight=transposed_weight)
        self._parallel_dim = parallel_dim
        self._sequence_parallel = sequence_parallel and self._parallel_dim.size > 1

    def forward_only(self, input_) -> tuple[torch.Tensor, tuple[typing.Any, ...]]:
        return output_parallel_linear_forward(
            input_,
            weight=self.weight,
            bias=self.bias,
            group=self._parallel_dim.group,
            sequence_parallel=self._sequence_parallel,
            transposed_weight=self._transposed_weight,
        )

    def backward(self, grad_output: torch.Tensor, context: tuple[typing.Any, ...]):  # noqa
        return output_parallel_linear_backward(grad_output, context)


class InputParallelLinear(LinearBase):
    """
    A linear layer with input (row) tensor parallelism.
    """

    def __init__(
        self,
        weight: ParameterMeta,
        bias: ParameterMeta | None,
        *,
        transposed_weight: bool = False,
        parallel_dim: DistributedDim,
        sequence_parallel: bool = False,
    ):
        super().__init__(weight, bias, transposed_weight=transposed_weight)
        self._parallel_dim = parallel_dim
        self._sequence_parallel = sequence_parallel and self._parallel_dim.size > 1

    def forward(self, input_: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        # TODO: Use self._forward instead (broken).
        return input_parallel_linear_autograd(
            input_,
            weight=self.weight,
            bias=self.bias,
            group=self._parallel_dim.group,
            sequence_parallel=self._sequence_parallel,
            transposed_weight=self._transposed_weight,
        )

    def forward_only(self, input_: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, tuple[typing.Any, ...]]:
        group = self._parallel_dim.group
        output, context = input_parallel_linear_forward(
            input_,
            weight=self.weight,
            bias=None if group else self.bias,
            group=group,
            sequence_parallel=self._sequence_parallel,
            transposed_weight=self._transposed_weight,
        )
        return output, self.bias if group else None, context

    def backward(self, grad_output: torch.Tensor, context: tuple[typing.Any, ...]) -> torch.Tensor:  # noqa
        # TODO: Needs grad_bias as input too?
        return input_parallel_linear_backward(grad_output, context)
