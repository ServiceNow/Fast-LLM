import math
import typing
from typing import Callable

import einops
import torch

from fast_llm.engine.config_utils.tensor_space import DefaultDimNames, TensorSpace
from fast_llm.layers.common.linear import Linear
from fast_llm.layers.ssm.config import SSMConfig, SSMDimNames
from fast_llm.layers.transformer.config import TransformerConfig
from fast_llm.layers.transformer.transformer import Mixer
from fast_llm.tensor import ParameterMeta, init_kaiming_, init_ones_
from fast_llm.utils import get_lr_scale

try:
    from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn as _mamba_inner_fn  # noqa

    _mamba_available = True
except (ImportError, RuntimeError):
    _mamba_available = False

"""
Note: this is mostly adapted from https://github.com/Zyphra/Zamba2, similar code is also in https://github.com/state-spaces/mamba.
For now it only supports training and not inference.
This works with triton 3.1.0
"""


def init_A(d_state, d_inner) -> Callable[[ParameterMeta, torch.Tensor, torch.Generator], torch.Tensor]:
    def init_(meta: ParameterMeta, tensor: torch.Tensor, generator: torch.Generator):  # noqa
        # S4D real initialization
        # TODO: adopt this initialization to work for tensor parallel setting!
        A = einops.repeat(torch.arange(1, d_state + 1, dtype=torch.float32), "n -> d n", d=d_inner).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if tensor.shape != A_log.shape:
            if tensor.numel() == A_log.numel():
                tensor_view = tensor.view(d_inner, d_state)
                tensor_view.copy_(A_log)
            else:
                raise ValueError(f"Tensor size {tensor.numel()} doesn't match expected size {A_log.numel()}")
        else:
            tensor.copy_(A_log)
        return tensor

    return init_


def init_dtprojbias(
    d_inner: int, dt_max: float, dt_min: float, dt_init_floor: float
) -> Callable[[ParameterMeta, torch.Tensor, torch.Generator], torch.Tensor]:
    def init_(meta: ParameterMeta, tensor: torch.Tensor, generator: torch.Generator):  # noqa
        dt = torch.exp(torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(
            min=dt_init_floor
        )
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        tensor.copy_(inv_dt)
        return tensor

    return init_


class MambaLayer(Mixer):
    _mixer_name: typing.ClassVar[str] = "mamba"

    def __init__(
        self,
        config: SSMConfig,
        block_index: int,
        tensor_space: TensorSpace,
        transformer_config: TransformerConfig,
    ):
        super().__init__(tensor_space, block_index, debug_level=transformer_config.debug_transformer)
        self.config: SSMConfig = config

        # Tensor dims:
        td_inner = tensor_space[SSMDimNames.inner_dim]
        td_inner_proj = tensor_space[SSMDimNames.inner_proj_mamba]  # TensorDim("D_inner_2", self.d_inner * 2)
        tdt_rank = tensor_space[SSMDimNames.dt_rank]
        td_x_proj = tensor_space[SSMDimNames.x_proj_dim]
        td_state = tensor_space[SSMDimNames.state_dim]
        td_model = tensor_space[SSMDimNames.model_dim]
        td_conv_kernel = tensor_space[SSMDimNames.conv_kernel_size]
        self.d_conv = td_conv_kernel.size
        self.d_inner = td_inner.size
        self.d_state = td_state.size
        self.d_model = td_model.size
        self.dt_rank = tdt_rank.size
        layer_lr_scale = config.per_layer_lr_scale[block_index] if config.per_layer_lr_scale else None
        mamba_layer_lr_scale = get_lr_scale(self.config.mamba_lr_scale, layer_lr_scale)

        self.in_proj_weight = ParameterMeta.from_dims(
            (td_inner_proj, td_model),
            init_method=init_kaiming_(td_model.size),
        )

        self.conv1d_weight = ParameterMeta.from_dims(
            (td_inner, tensor_space[DefaultDimNames.scalar], td_conv_kernel),
            init_method=init_kaiming_(td_inner.size),
            lr_scale=mamba_layer_lr_scale,
        )

        self.conv1d_bias = None

        self.activation = "silu"
        self.act = torch.nn.SiLU()

        self.x_proj = Linear(
            td_inner,
            td_x_proj,
            weight_init_method=init_kaiming_(td_inner.size),
            bias=False,
            lr_scale=mamba_layer_lr_scale,
        )
        self.x_proj.weight.auto_grad_accumulation = True

        # TODO: the weights are initialized a bit differently here https://github.com/state-spaces/mamba/blob/0cce0fa645f100f00620ddf2333c2b7712abfdec/mamba_ssm/modules/mamba_simple.py#L82
        self.dt_proj_weight = ParameterMeta.from_dims(
            (td_inner, tdt_rank),
            init_method=init_kaiming_(tdt_rank.size),
            lr_scale=mamba_layer_lr_scale,
        )

        self.dt_proj_bias = ParameterMeta.from_dims(
            (td_inner,),
            init_method=init_dtprojbias(
                self.d_inner, self.config.dt_max, self.config.dt_min, self.config.dt_init_floor
            ),
            lr_scale=mamba_layer_lr_scale,
        )

        self.A_log = ParameterMeta.from_dims(
            (td_inner, td_state),
            weight_decay=False,
            init_method=init_A(self.d_state, self.d_inner),
            lr_scale=mamba_layer_lr_scale,
        )

        # D "skip" parameter
        self.D = ParameterMeta.from_dims(
            (td_inner,),
            weight_decay=False,
            init_method=init_ones_,
            lr_scale=mamba_layer_lr_scale,
        )

        self.out_proj = Linear(
            td_inner,
            td_model,
            bias=False,  # TODO: note, if bias is used there is a problem in the MambaInnerFn.backward for the bias grads. I think this bias is not used in other mamba repos.
            weight_init_method=init_kaiming_(td_model.size),
            lr_scale=mamba_layer_lr_scale,
        )
        self.out_proj.weight.auto_grad_accumulation = True

    def forward(self, hidden_states, kwargs):
        assert _mamba_available
        batch, seqlen, dim = hidden_states.shape

        # We do matmul and transpose BLH -> HBL at the same time
        xz = einops.rearrange(
            self.in_proj_weight @ einops.rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        # not, if we wanbt to support inference, we would need to imp.lement slow path here, see https://github.com/Zyphra/Zamba2/blob/1b182f40f2257f822cc06dd785df53d67d691a15/mamba_layer.py#L172s
        out = _mamba_inner_fn(
            xz,
            self.conv1d_weight,
            self.conv1d_bias,
            self.x_proj.weight,
            self.dt_proj_weight,
            self.out_proj.weight,
            self.out_proj.bias,  # is None here
            A,
            None,  # input-dependent B
            None,  # input-dependent C
            self.D.float(),
            delta_bias=self.dt_proj_bias.float(),
            delta_softplus=True,
        )
        return out, None
