import typing

import torch

from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.functional.autograd import wrap_forward_backward
from fast_llm.layers.common.linear.linear import Linear, LinearBase


def lora_linear(
    module: LinearBase,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    out_channel_begin: int | None = None,
    out_channel_end: int | None = None,
):
    module.weight.requires_grad = False
    in_dim = module._in_dim
    assert not in_dim.is_parallel, "LoRA not supported with tensor parallelism."
    if in_dim.parallel_dim is not None:
        in_dim = TensorDim(in_dim.name, in_dim.global_size)
    out_dim = module._out_dim
    assert not out_dim.is_parallel, "LoRA not supported with tensor parallelism."
    if out_dim.parallel_dim is not None:
        out_dim = TensorDim(out_dim.name, out_dim.global_size)
    if out_channel_begin is not None or out_channel_end is not None:
        if out_channel_begin is None:
            out_channel_begin = 0
        if out_channel_end is None:
            out_channel_end = out_dim.global_size
        # TODO: This won't work with TP. Use Composite dim structure for proper split?
        out_dim = TensorDim(out_dim.name, out_channel_end - out_channel_begin)

    middle_dim = TensorDim("lora_middle", rank)

    module.lora_0 = Linear(
        in_dim,
        middle_dim,
        bias=False,
        weight_init_method=module.weight.param_init_method,
        transposed_weight=module.transposed_weight,
        lr_scale=module.weight.lr_scale,
    )
    module.lora_1 = Linear(
        middle_dim,
        out_dim,
        bias=False,
        weight_init_method=module.weight.param_init_method,
        transposed_weight=module.transposed_weight,
        lr_scale=module.weight.lr_scale,
    )

    old_forward = module._forward

    def forward_only(input_: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # TODO: torch compile?
        input_ = input_.detach().requires_grad_()
        with torch.enable_grad():
            output = old_forward(input_)
            if isinstance(output, tuple):
                layer_out, tp_bias = output[0]
                assert tp_bias is None
            lora_out = (alpha / rank) * module.lora_1(
                module.lora_0(torch.dropout(input_, dropout, module.training) if dropout > 0.0 else input_)
            )
            if out_channel_begin is None:
                output = output + lora_out
            else:
                output.view(-1, layer_out.size(-1))[:, out_channel_begin:out_channel_end] += lora_out
        return output.detach(), (input_, output)

    def backward(
        grad_output: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, typing.Callable[[], None] | None]:
        # TODO: Implement proper backward pass.
        input_, output = context
        output.backward(grad_output)
        return input_.grad

    module._forward = wrap_forward_backward(forward_only, backward)
    module.forward_only = forward_only
    module.backward = backward

    return module
