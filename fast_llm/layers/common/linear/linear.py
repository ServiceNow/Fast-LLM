import abc
import logging
import typing

import torch

from fast_llm.config import Configurable
from fast_llm.engine.config_utils.tensor_dim import TensorDim
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
from fast_llm.layers.common.linear.config import AffineLinearConfig, LinearConfig
from fast_llm.tensor import ParameterMeta

logger = logging.getLogger(__name__)


class LinearLike(torch.nn.Module):
    """
    An interface for linear-like layers.
    """

    def __init__(self):
        super().__init__()
        self._forward = wrap_forward_backward(self.forward_only, self.backward)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        return self._forward(input_)

    def forward_only(self, input_: torch.Tensor):
        raise NotImplementedError()

    def backward(self, grad_output: torch.Tensor, context: typing.Any) -> torch.Tensor:
        raise NotImplementedError()


class LinearBase[ConfigType: LinearConfig](Configurable[ConfigType], LinearLike):
    """
    A base module for linear and affine linear layers that defines weights.
    """

    def __init__(
        self,
        config: ConfigType,
        in_dim: TensorDim,
        out_dim: TensorDim,
        *,
        auto_weight_grad_accumulation: bool = False,
        transposed_weight: bool = False,
        sequence_parallel: bool = False,
        lr_scale: float | None = None,
    ):
        super().__init__(config)
        self._transposed_weight = transposed_weight
        self._sequence_parallel = sequence_parallel
        self._in_dim = in_dim
        self._out_dim = out_dim
        self._lr_scale = lr_scale
        self._init()
        self._config.get_weight()
        self.weight = ParameterMeta.from_dims(
            (self._in_dim, self._out_dim) if self._transposed_weight else (self._out_dim, self._in_dim),
            init_method=self._config.weight_initialization,
            auto_grad_accumulation=auto_weight_grad_accumulation,
            lr_scale=self._lr_scale,
        )
        self.bias = None

    @abc.abstractmethod
    def _init(self):
        # Convenience method to avoid repeating argument lists.
        pass

    @property
    def transposed_weight(self) -> bool:
        return self._transposed_weight

    @property
    def lr_scale(self) -> float | None:
        return self._lr_scale


class Linear[ConfigType: LinearConfig](LinearBase[ConfigType]):
    """
    A (non-affine) linear layer without tensor parallelism.
    """

    def _init(self):
        assert not self._in_dim.is_parallel
        assert not self._out_dim.is_parallel
        assert not self._sequence_parallel

    def forward_only(
        self, input_: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]]:
        return linear_forward(input_, weight=self.weight, bias=self.bias, transposed_weight=self._transposed_weight)

    def backward(self, grad_output: torch.Tensor, context) -> torch.Tensor:  # noqa
        return linear_backward(grad_output, context)


class OutputParallelLinear[ConfigType: LinearConfig](LinearBase[ConfigType]):
    """
    A (non-affine) linear layer with output (column) tensor parallelism.
    """

    _group_size: int

    def _init(self):
        assert not self._in_dim.is_parallel
        self._group_size = 1 if self._out_dim.parallel_dim is None else self._out_dim.parallel_dim.size
        self._sequence_parallel = self._sequence_parallel and self._group_size > 1

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


class InputParallelLinear[ConfigType: LinearConfig](LinearBase[ConfigType]):
    """
    A (non-affine) linear layer with input (row) tensor parallelism.
    """

    _group_size: int

    def _init(self):
        assert not self._out_dim.is_parallel
        self._group_size = 1 if self._in_dim.parallel_dim is None else self._in_dim.parallel_dim.size
        self._sequence_parallel = self._sequence_parallel and self._group_size > 1

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


class AffineLinearBase[ConfigType: AffineLinearConfig](LinearBase[ConfigType]):
    """
    A base module for affine linear layers that defines weights and biases.
    """

    def __init__(
        self,
        config: ConfigType,
        in_dim: TensorDim,
        out_dim: TensorDim,
        *,
        transposed_weight: bool = False,
        sequence_parallel: bool = False,
        auto_weight_grad_accumulation: bool = False,
        auto_bias_grad_accumulation: bool = False,
        lr_scale: float | None = None,
    ):
        super().__init__(
            config,
            in_dim,
            out_dim,
            transposed_weight=transposed_weight,
            sequence_parallel=sequence_parallel,
            auto_weight_grad_accumulation=auto_weight_grad_accumulation,
            lr_scale=lr_scale,
        )
        if self._config.bias:
            self.bias = ParameterMeta.from_dims(
                (self._out_dim,),
                init_method=self._config.bias_initialization,
                weight_decay=False,
                auto_grad_accumulation=auto_bias_grad_accumulation,
                lr_scale=self._lr_scale,
            )


class AffineLinear[ConfigType: AffineLinearConfig](AffineLinearBase, LinearBase[ConfigType]):
    """
    An affine linear layer without tensor parallelism.
    """


class AffineOutputParallelLinear[ConfigType: LinearConfig](AffineLinearBase, OutputParallelLinear[ConfigType]):
    """
    An affine linear layer with output (column) tensor parallelism.
    """


class AffineInputParallelLinear[ConfigType: LinearConfig](AffineLinearBase, InputParallelLinear[ConfigType]):
    """
    An affine linear layer with input (row) tensor parallelism.
    """
