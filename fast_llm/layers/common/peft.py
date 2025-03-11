import typing

import torch

from fast_llm.engine.config_utils.tensor_space import TensorDim
from fast_llm.functional.autograd import wrap_forward_backward
from fast_llm.layers.common.linear import Linear, LinearBase, LinearLike


class LoRALinear(LinearLike):
    def __init__(
        self,
        linear: LinearBase,
        init_method_0,
        init_method_1,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.linear = linear
        self.linear.weight.requires_grad = False
        if self.linear._in_dim.parallel_dim is not None or self.linear._out_dim.parallel_dim is not None:
            # TODO: TP support.
            raise ValueError("LoRA not supported with tensor parallelism.")
        self._alpha = alpha
        self._dropout = dropout
        self._transposed_weight = self.linear._transposed_weight
        middle_dim = TensorDim("lora_middle", rank)

        self.layer_0 = Linear(
            self.linear._in_dim,
            middle_dim,
            bias=False,
            weight_init_method=init_method_0,
            transposed_weight=self.linear._transposed_weight,
            lr_scale=self.linear._lr_scale,
        )
        self.layer_1 = Linear(
            middle_dim,
            self.linear._out_dim,
            bias=False,
            weight_init_method=init_method_1,
            transposed_weight=self.linear._transposed_weight,
            lr_scale=self.linear._lr_scale,
        )
        # TODO: Implement proper backward pass.
        self.layer_0.weight.auto_grad_accumulation = True
        self.layer_1.weight.auto_grad_accumulation = True
        self._forward = wrap_forward_backward(self.forward_only, self.backward)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        return self._forward(input_)

    def forward_only(self, input_: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # TODO: torch compile?
        input_ = input_.detach().requires_grad_()
        output = self.linear(input_) + (self._alpha / self._rank) * self.layer_1(
            self.layer_0(torch.dropout(input_, self._dropout, self._training) if self._dropout > 0.0 else input_)
        )
        return output.detach(), (input_, output)

    def backward(
        self, grad_output: torch.Tensor, context: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, typing.Callable[[], None] | None]:
        # TODO: Implement proper backward pass.
        input_, output = context
        output.backward(grad_output)
        return input_.grad, None
