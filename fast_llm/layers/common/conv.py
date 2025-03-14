import logging
import typing

import torch

from fast_llm.engine.config_utils.tensor_space import TensorDim
from fast_llm.tensor import ParameterMeta, init_zeros_

logger = logging.getLogger(__name__)


class Conv1DBase(torch.nn.Module):
    """
    A base module for 1D convolutional layers holding weights and biases.
    """

    def __init__(
        self,
        in_channels: TensorDim,
        out_channels: TensorDim,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        *,
        bias=True,
        weight_init_method,
        bias_init_method=init_zeros_,
        auto_bias_grad_accumulation: bool = False,
        lr_scale: float | None | tuple[float | None, ...] = None,
    ):
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._dilation = dilation
        self._groups = groups
        
        self.weight = ParameterMeta.from_dims(
            (self._out_channels, TensorDim("D_in", self._in_channels.size // groups), TensorDim("D_kernel", self._kernel_size)),
            init_method=weight_init_method,
            auto_grad_accumulation=False,
            lr_scale=lr_scale,
        )
        
        if bias:
            self.bias = ParameterMeta.from_dims(
                (self._out_channels,),
                init_method=bias_init_method,
                weight_decay=False,
                auto_grad_accumulation=auto_bias_grad_accumulation,
                lr_scale=lr_scale,
            )
        else:
            self.bias = None


class Conv1D(Conv1DBase):
    """
    A basic 1D convolutional layer without tensor parallelism.
    """

    def __init__(
        self,
        in_channels: TensorDim,
        out_channels: TensorDim,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        *,
        bias=True,
        weight_init_method,
        bias_init_method=init_zeros_,
        lr_scale: float | None | tuple[float | None, ...] = None,
    ):
        assert in_channels.parallel_dim is None
        assert out_channels.parallel_dim is None
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias=bias,
            weight_init_method=weight_init_method,
            bias_init_method=bias_init_method,
            lr_scale=lr_scale,
        )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.conv1d(
            input_,
            self.weight,
            self.bias,
            stride=self._stride,
            padding=self._padding,
            dilation=self._dilation,
            groups=self._groups,
        )

    def forward_only(
        self, input_: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, dict]]:
        # Store context for backward pass
        context = {
            "input": input_,
            "weight": self.weight,
            "stride": self._stride,
            "padding": self._padding,
            "dilation": self._dilation,
            "groups": self._groups,
        }
        
        output = torch.nn.functional.conv1d(
            input_,
            self.weight,
            self.bias,
            stride=self._stride,
            padding=self._padding,
            dilation=self._dilation,
            groups=self._groups,
        )
        
        return output, (input_, self.weight, context)

    def backward(self, grad_output: torch.Tensor, context: tuple[torch.Tensor, torch.Tensor, dict]) -> torch.Tensor:
        input_, weight, ctx = context
        
        # Calculate gradients
        grad_input = torch.nn.grad.conv1d_input(
            input_.shape,
            weight,
            grad_output,
            stride=ctx["stride"],
            padding=ctx["padding"],
            dilation=ctx["dilation"],
            groups=ctx["groups"],
        )
        
        return grad_input