import typing

import torch

from fast_llm.engine.config_utils.tensor_space import TensorDim
from fast_llm.functional.autograd import wrap_forward_backward
from fast_llm.layers.common.linear import Linear, LinearBase


def lora_linear(
    layer: LinearBase,
    init_method_0,
    init_method_1,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
):
    layer.weight.requires_grad = False
    in_dim = layer._in_dim
    if in_dim.parallel_dim is not None:
        assert in_dim.parallel_dim.size == 1, "LoRA not supported with tensor parallelism."
        in_dim = TensorDim(in_dim.name, in_dim.global_size)
    out_dim = layer._out_dim
    if out_dim.parallel_dim is not None:
        assert out_dim.parallel_dim.size == 1, "LoRA not supported with tensor parallelism."
        out_dim = TensorDim(out_dim.name, out_dim.global_size)

    middle_dim = TensorDim("lora_middle", rank)

    layer.lora_0 = Linear(
        in_dim,
        middle_dim,
        bias=False,
        weight_init_method=init_method_0,
        transposed_weight=layer.transposed_weight,
        lr_scale=layer.weight.lr_scale,
    )
    layer.lora_1 = Linear(
        middle_dim,
        out_dim,
        bias=False,
        weight_init_method=init_method_1,
        transposed_weight=layer.transposed_weight,
        lr_scale=layer.weight.lr_scale,
    )
    # TODO: Implement proper backward pass.
    layer.lora_0.weight.auto_grad_accumulation = True
    layer.lora_1.weight.auto_grad_accumulation = True

    old_forward = layer._forward

    def forward_only(input_: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # TODO: torch compile?
        input_ = input_.detach().requires_grad_()
        with torch.enable_grad():
            output = old_forward(input_) + (alpha / rank) * layer.lora_1(
                layer.lora_0(torch.dropout(input_, dropout, layer.training) if dropout > 0.0 else input_)
            )
        return output.detach(), (input_, output)

    def backward(
        grad_output: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, typing.Callable[[], None] | None]:
        # TODO: Implement proper backward pass.
        input_, output = context
        output.backward(grad_output)
        return input_.grad

    layer._forward = wrap_forward_backward(forward_only, backward)
    layer.forward_only = forward_only
    layer.backward = backward

    return layer
