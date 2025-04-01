import logging
import typing

import torch

from fast_llm.engine.config_utils.tensor_space import TensorDim
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
from fast_llm.tensor import ParameterMeta, init_zeros_

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


class LinearBase(LinearLike):
    """
    A base module for linear layers holding weights and biases.
    """

    def __init__(
        self,
        in_dim: TensorDim,
        out_dim: TensorDim,
        *,
        bias=True,
        weight_init_method,
        bias_init_method=init_zeros_,
        transposed_weight: bool = False,
        auto_bias_grad_accumulation: bool = False,
        lr_scale: float | None | tuple[float | None, ...] = None,
    ):
        super().__init__()
        self._transposed_weight = transposed_weight
        self._in_dim = in_dim
        self._out_dim = out_dim
        self._weight_init_method = weight_init_method
        self.weight = ParameterMeta.from_dims(
            (self._in_dim, self._out_dim) if self._transposed_weight else (self._out_dim, self._in_dim),
            init_method=weight_init_method,
            auto_grad_accumulation=False,
            lr_scale=lr_scale,
        )
        if bias:
            self.bias = ParameterMeta.from_dims(
                (self._out_dim,),
                init_method=bias_init_method,
                weight_decay=False,
                auto_grad_accumulation=auto_bias_grad_accumulation,
                lr_scale=lr_scale,
            )
        else:
            self.bias = None

    @property
    def transposed_weight(self) -> bool:
        return self._transposed_weight


class Linear(LinearBase):
    """
    A basic linear layer without tensor parallelism.
    """

    def __init__(
        self,
        in_dim: TensorDim,
        out_dim: TensorDim,
        *,
        bias=True,
        weight_init_method,
        bias_init_method=init_zeros_,
        transposed_weight: bool = False,
        lr_scale: float | None | tuple[float | None, ...] = None,
    ):
        assert in_dim.parallel_dim is None
        assert out_dim.parallel_dim is None
        super().__init__(
            in_dim,
            out_dim,
            bias=bias,
            weight_init_method=weight_init_method,
            bias_init_method=bias_init_method,
            transposed_weight=transposed_weight,
            lr_scale=lr_scale,
        )

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
        in_dim: TensorDim,
        out_dim: TensorDim,
        *,
        bias=True,
        weight_init_method,
        bias_init_method=init_zeros_,
        transposed_weight: bool = False,
        sequence_parallel: bool = False,
        lr_scale: float | None | tuple[float | None, ...] = None,
    ):
        assert in_dim.parallel_dim is None
        self._group_size = 1 if out_dim.parallel_dim is None else out_dim.parallel_dim.size
        self._sequence_parallel = sequence_parallel and self._group_size > 1
        super().__init__(
            in_dim,
            out_dim,
            bias=bias,
            weight_init_method=weight_init_method,
            bias_init_method=bias_init_method,
            transposed_weight=transposed_weight,
            lr_scale=lr_scale,
        )

    def forward_only(self, input_) -> tuple[torch.Tensor, tuple[typing.Any, ...]]:
        return output_parallel_linear_forward(
            input_,
            weight=self.weight,
            bias=self.bias,
            group=self._out_dim.parallel_group,
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
        in_dim: TensorDim,
        out_dim: TensorDim,
        *,
        bias=True,
        weight_init_method,
        bias_init_method=init_zeros_,
        sequence_parallel: bool = False,
        transposed_weight: bool = False,
        lr_scale: float | None | tuple[float | None, ...] = None,
    ):
        assert out_dim.parallel_dim is None
        self._group_size = 1 if in_dim.parallel_dim is None else in_dim.parallel_dim.size
        self._sequence_parallel = sequence_parallel and self._group_size > 1
        super().__init__(
            in_dim,
            out_dim,
            bias=bias,
            weight_init_method=weight_init_method,
            bias_init_method=bias_init_method,
            transposed_weight=transposed_weight,
            # Tensor-parallel bias is computed in _bias_dropout_grad.
            auto_bias_grad_accumulation=self._group_size > 1,
            lr_scale=lr_scale,
        )

    def forward(self, input_: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        # TODO: Use self._forward instead (broken).
        return input_parallel_linear_autograd(
            input_,
            weight=self.weight,
            bias=self.bias,
            group=self._in_dim.parallel_group,
            sequence_parallel=self._sequence_parallel,
            transposed_weight=self._transposed_weight,
        )

    def forward_only(self, input_: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, tuple[typing.Any, ...]]:
        group = self._in_dim.parallel_group
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
