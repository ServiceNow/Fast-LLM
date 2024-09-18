"""
Wrap the layer implementations of the functional module into autograd-friendly pytorch functions.
"""

import typing

import torch

from fast_llm.utils import Assert


def wrap_forward_backward(forward: typing.Callable, backward: typing.Callable):
    """
    Wrap a (`forward`, `backward`) pair into a differentiable pytorch function.
    Expected format is as follows:
    * `forward(input_, *args, **kwargs)->(*outputs, context)`: Takes any number of arguments.
      The `input_` is a differentiable tensor, while the other ones are arbitrary but may not have gradients.
      It returns a tuple of `outputs` and an arbitrary context for the backward pass.
      TODO: do the outputs need to be a differentiable tensor?
    * `backward(*grad_outputs, context)`: Takes the `outputs` gradients and the context,
      and returns the `input_` gradient.
    """

    class Function(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args):
            outputs = forward(*args)
            Assert.custom(isinstance, outputs, tuple)
            # No need to call `save_for_backward`, we don't want the safety checks anyway.
            ctx.context = outputs[-1]
            ctx.nargs = len(args)
            if len(outputs) == 2:
                return outputs[0]
            else:
                return outputs[:-1]

        @staticmethod
        def backward(ctx, *grad_outputs):
            grad_input = backward(*grad_outputs, ctx.context)
            if not isinstance(grad_input, tuple):
                assert isinstance(grad_input, torch.Tensor)
                grad_input = (grad_input,)
            return *grad_input, *[None for _ in range(ctx.nargs - len(grad_input))]

    def call(*args, **kwargs):
        # TODO: Any way to validate kwargs without making function wrappers?
        return Function.apply(*args, *kwargs.values())

    return call


def grad_is_context(grad_output: torch.Tensor, context: torch.Tensor):  # noqa
    return context
