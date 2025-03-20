import typing

import torch

from fast_llm.engine.config_utils.tensor_space import TensorDim
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
        in_dim = self.linear._in_dim
        if in_dim.parallel_dim is not None:
            assert in_dim.parallel_dim.size == 1, "LoRA not supported with tensor parallelism."
            in_dim = TensorDim(in_dim.name, in_dim.global_size)
        out_dim = self.linear._out_dim
        if out_dim.parallel_dim is not None:
            assert out_dim.parallel_dim.size == 1, "LoRA not supported with tensor parallelism."
            out_dim = TensorDim(out_dim.name, out_dim.global_size)

        self._rank = rank
        self._alpha = alpha
        self._dropout = dropout
        self._transposed_weight = self.linear._transposed_weight
        middle_dim = TensorDim("lora_middle", self._rank)

        self.layer_0 = Linear(
            in_dim,
            middle_dim,
            bias=False,
            weight_init_method=init_method_0,
            transposed_weight=self.linear._transposed_weight,
            lr_scale=self.linear.weight.lr_scale,
        )
        self.layer_1 = Linear(
            middle_dim,
            out_dim,
            bias=False,
            weight_init_method=init_method_1,
            transposed_weight=self.linear._transposed_weight,
            lr_scale=self.linear.weight.lr_scale,
        )
        # TODO: Implement proper backward pass.
        self.layer_0.weight.auto_grad_accumulation = True
        self.layer_1.weight.auto_grad_accumulation = True

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
