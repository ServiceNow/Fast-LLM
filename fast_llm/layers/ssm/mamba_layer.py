import logging
import math
import typing

import torch

from fast_llm.engine.config_utils.tensor_space import DefaultDimNames, TensorSpace
from fast_llm.functional.config import ActivationType
from fast_llm.layers.block.block import BlockLayer
from fast_llm.layers.block.config import BlockConfig, BlockKwargs
from fast_llm.layers.common.linear import Linear
from fast_llm.layers.ssm.config import SSMConfig, SSMDimNames
from fast_llm.tensor import LambdaInitializer, ParameterMeta, init_kaiming_, init_ones_
from fast_llm.utils import Assert, get_lr_scale

try:
    from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn as _mamba_inner_fn  # noqa

    _mamba_available = True
except (ImportError, RuntimeError):
    _mamba_available = False

logger = logging.getLogger(__name__)

"""
Note: this is mostly adapted from https://github.com/Zyphra/Zamba2, similar code is also in https://github.com/state-spaces/mamba.
For now it only supports training and not inference.
This works with triton 3.1.0
"""


def init_A(d_state, d_inner) -> LambdaInitializer:
    def init_(meta: ParameterMeta, tensor: torch.Tensor, generator: torch.Generator) -> None:  # noqa
        if tensor.numel() != d_state * d_inner:
            raise ValueError("_init_A requires not supported for tensor slices.")
        torch.log(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=tensor.device)
            .unsqueeze(0)
            .expand(d_inner, d_state),
            out=tensor,
        )

    return LambdaInitializer(init_, requires_global_initialization=True)


def init_dtprojbias(dt_max: float, dt_min: float, dt_init_floor: float) -> LambdaInitializer:
    def init_(meta: ParameterMeta, tensor: torch.Tensor, generator: torch.Generator):  # noqa
        tensor.uniform_(math.log(dt_min), math.log(dt_max), generator=generator).exp_().clamp_min_(dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        tensor.add_(torch.log(-torch.expm1(-tensor)))

    return LambdaInitializer(init_)


class MambaLayer(BlockLayer):
    _mixer_name: typing.ClassVar[str] = "mamba"

    def __init__(
        self,
        config: SSMConfig,
        block_index: int,
        tensor_space: TensorSpace,
        block_config: BlockConfig,
    ):
        super().__init__(
            tensor_space,
            block_index,
            self._mixer_name,
            debug_level=block_config.debug_transformer,
            debug_memory=block_config.debug_transformer_memory,
        )
        assert tensor_space.distributed_config.tensor_parallel == 1, "Tensor-parallel not supported for MambaLayer"
        self._config = config
        # TODO: It's not silu?
        Assert.eq(self._config.activation_type, ActivationType.silu)

        # Tensor dims:
        inner_dim = tensor_space[SSMDimNames.composite_heads_and_head_dim]
        hidden_dim = tensor_space[SSMDimNames.hidden]
        layer_lr_scale = block_config.per_layer_lr_scale[block_index] if block_config.per_layer_lr_scale else None
        lr_scale = get_lr_scale(self._config.mamba_lr_scale, layer_lr_scale)

        # TODO: Backward compatibility?
        # TODO: lr_scale?
        self.in_proj = Linear(
            hidden_dim,
            tensor_space[SSMDimNames.concatenated_inner_projection],
            bias=False,
            weight_init_method=init_kaiming_(hidden_dim.size),
        )

        self.conv1d_weight = ParameterMeta.from_dims(
            (
                inner_dim,
                tensor_space[DefaultDimNames.scalar],
                tensor_space[SSMDimNames.convolution_kernel],
            ),
            init_method=init_kaiming_(inner_dim.size),
            lr_scale=lr_scale,
        )

        self.x_proj = Linear(
            inner_dim,
            tensor_space[SSMDimNames.concatenated_x_projection],
            weight_init_method=init_kaiming_(inner_dim.size),
            bias=False,
            lr_scale=lr_scale,
        )
        self.x_proj.weight.auto_grad_accumulation = True

        # TODO: the weights are initialized a bit differently here https://github.com/state-spaces/mamba/blob/0cce0fa645f100f00620ddf2333c2b7712abfdec/mamba_ssm/modules/mamba_simple.py#L82
        self.dt_proj_weight = ParameterMeta.from_dims(
            (inner_dim, tensor_space[SSMDimNames.dt_rank]),
            init_method=init_kaiming_(self._config.dt_rank),
            lr_scale=lr_scale,
        )

        self.dt_proj_bias = ParameterMeta.from_dims(
            (inner_dim,),
            init_method=init_dtprojbias(self._config.dt_max, self._config.dt_min, self._config.dt_init_floor),
            lr_scale=lr_scale,
        )

        self.A_log = ParameterMeta.from_dims(
            (inner_dim, tensor_space[SSMDimNames.state]),
            weight_decay=False,
            init_method=init_A(self._config.state_size, inner_dim.size),
            lr_scale=lr_scale,
        )

        # D "skip" parameter
        self.D = ParameterMeta.from_dims(
            (inner_dim,),
            weight_decay=False,
            init_method=init_ones_,
            lr_scale=lr_scale,
        )

        self.out_proj = Linear(
            inner_dim,
            hidden_dim,
            bias=False,  # TODO: note, if bias is used there is a problem in the MambaInnerFn.backward for the bias grads. I think this bias is not used in other mamba repos.
            weight_init_method=init_kaiming_(hidden_dim.size),
            lr_scale=lr_scale,
        )
        self.out_proj.weight.auto_grad_accumulation = True

    def forward(self, input_: torch.Tensor, kwargs: dict[str, typing.Any]) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert _mamba_available
        in_proj = self.in_proj(input_).permute((1, 2, 0) if kwargs[BlockKwargs.sequence_first] else (0, 2, 1))

        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        # not, if we wanbt to support inference, we would need to imp.lement slow path here, see https://github.com/Zyphra/Zamba2/blob/1b182f40f2257f822cc06dd785df53d67d691a15/mamba_layer.py#L172s
        out = _mamba_inner_fn(
            in_proj,
            self.conv1d_weight,
            None,
            self.x_proj.weight,
            self.dt_proj_weight,
            self.out_proj.weight,
            self.out_proj.bias,  # is None here
            -torch.exp(self.A_log.float()),
            None,  # input-dependent B
            None,  # input-dependent C
            self.D.float(),
            delta_bias=self.dt_proj_bias.float(),
            delta_softplus=True,
        )
        if kwargs[BlockKwargs.sequence_first]:
            out = out.transpose(0, 1)
        return out, None
