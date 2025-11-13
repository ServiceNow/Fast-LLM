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
from fast_llm.layers.common.linear.config import LinearConfig
from fast_llm.layers.common.peft.config import PeftConfig
from fast_llm.tensor import ConcatenatedParameterMeta, ParameterMeta, TensorMeta
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


def concatenate_linear_layers[
    T: LinearBase
](
    layers: tuple[T, ...],
    configs: tuple[LinearConfig, ...],
    *,
    concatenate_input_dim: bool = False,
    dim_name: str | None = None,
    default_apply_peft: bool | tuple[bool, ...] = False,
    peft: PeftConfig | None,
) -> T:
    # TODO: Simplify.
    # All biases must be either enabled or disabled. TODO: Allow non-constant.
    enable_bias = layers[0].bias is not None
    # Concatenate on `in_dim` (instead of `out_dim`)
    if concatenate_input_dim:
        # TODO: Support this case? (needs one bias instead of a concatenation)
        assert not enable_bias

    cls = type(layers[0])
    # Should not already be wrapped with Peft.
    Assert.incl(cls, (Linear, InputParallelLinear, OutputParallelLinear))
    # The concatenated dimension must be at index zero.
    transposed_weight = concatenate_input_dim
    for layer in layers:
        Assert.eq(layer._transposed_weight, transposed_weight)
        Assert.is_(type(layer), cls)
        Assert.eq(layer.bias is not None, enable_bias)

    if cls in (InputParallelLinear, OutputParallelLinear):
        for layer in layers[1:]:
            Assert.is_(layer._parallel_dim, layers[0]._parallel_dim)
            Assert.eq(layer._sequence_parallel, layers[0]._sequence_parallel)
        args = {"parallel_dim": layers[0]._parallel_dim, "sequence_parallel": layers[0]._sequence_parallel}
    else:
        args = {}

    # TODO: Original parameters won't get names.
    weight = ConcatenatedParameterMeta.from_metas(tuple(layer.weight for layer in layers), dim_name=dim_name)
    bias = (
        ConcatenatedParameterMeta.from_metas(tuple(layer.bias for layer in layers), dim_name=dim_name)
        if enable_bias
        else None
    )

    out = cls(weight, bias, transposed_weight=transposed_weight, **args)
    if peft is not None:
        if isinstance(default_apply_peft, bool):
            default_apply_peft = (default_apply_peft,) * len(layers)
        apply_peft = [
            default if config.apply_peft is None else config.apply_peft
            for config, default in zip(configs, default_apply_peft, strict=True)
        ]
        if len(set(apply_peft)) == 1:
            out_channel_ranges = None
            enabled = apply_peft[0]
        else:
            enabled = True
            out_channel_ranges = tuple(
                split_range
                for split_range, apply_peft_ in zip(weight.dims[0].get_split_ranges(True), apply_peft)
                if apply_peft_
            )
        out = peft.apply_linear(out, enabled, out_channel_ranges=out_channel_ranges)
    return out
