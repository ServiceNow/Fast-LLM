import math
from typing import Callable

import torch
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn

from fast_llm.engine.config_utils.tensor_space import TensorDim, TensorSpace
from fast_llm.layers.common.linear import Linear
from fast_llm.layers.ssm.config import SSMDimNames, SSMLayerConfig
from fast_llm.tensor import ParameterMeta, init_ones_, kaiming_init_

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    # causal_conv1d_fn = None
    raise ImportError("Causal conv1d not installed")


"""
Note: this is mostly addapted from https://github.com/Zyphra/Zamba2, similar code is aslo in https://github.com/state-spaces/mamba.
For now it only supports training and not inference.
This works with triton 3.1.0
"""


def init_A(d_state, d_inner) -> Callable[[ParameterMeta, torch.Tensor, torch.Generator], torch.Tensor]:
    def init_(meta: ParameterMeta, tensor: torch.Tensor, generator: torch.Generator):  # noqa
        # S4D real initialization
        # TODO: adopt this innitialization to work for tensor parallel setting!
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32), "n -> d n", d=d_inner).contiguous()
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
    d_inner: int, dt_max: float, dt_min: float, dt_init_floor: float, factory_kwargs: dict
) -> Callable[[ParameterMeta, torch.Tensor, torch.Generator], torch.Tensor]:
    def init_(meta: ParameterMeta, tensor: torch.Tensor, generator: torch.Generator):  # noqa
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        tensor.copy_(inv_dt)
        return tensor

    return init_


class MambaLayer(torch.nn.Module):
    def __init__(
        self,
        config: SSMLayerConfig,
        layer_idx: int,
        tensor_space: TensorSpace,
    ):
        factory_kwargs = {}
        super().__init__()
        self.config: SSMLayerConfig = config
        self.use_fast_path = config.use_fast_path if mamba_inner_fn is not None else False
        self.layer_idx = layer_idx

        self._debug_mode = config.debug_ssm

        # Tensor dims:
        td_inner = tensor_space.get_tensor_dim(SSMDimNames.d_inner)
        td_inner_proj = tensor_space.get_tensor_dim(
            SSMDimNames.d_inner_proj_m
        )  # TensorDim("D_inner_2", self.d_inner * 2)
        tdt_rank = tensor_space.get_tensor_dim(SSMDimNames.dt_rank)
        td_x_proj = tensor_space.get_tensor_dim(SSMDimNames.d_x_proj)
        td_state = tensor_space.get_tensor_dim(SSMDimNames.d_state)
        td_model = tensor_space.get_tensor_dim(SSMDimNames.d_model)
        td_conv_kernel = tensor_space.get_tensor_dim(SSMDimNames.d_conv_kernel)
        self.d_conv = td_conv_kernel.size
        self.d_inner = td_inner.size
        self.d_state = td_state.size
        self.d_model = td_model.size
        self.dt_rank = tdt_rank.size

        self.in_proj_weight = ParameterMeta.from_dims(
            (td_inner_proj, td_model),
            init_method=kaiming_init_(td_model.size),
        )

        self.conv1d_weight = ParameterMeta.from_dims(
            (td_inner, TensorDim("D_inner_2", self.d_inner // self.d_inner), td_conv_kernel),
            init_method=kaiming_init_(td_inner.size),
        )

        self.conv1d_bias = None

        self.activation = "silu"
        self.act = torch.nn.SiLU()

        self.x_proj = Linear(
            td_inner,
            td_x_proj,
            weight_init_method=kaiming_init_(td_inner.size),
            bias=False,
            **factory_kwargs,
        )
        if self.use_fast_path:
            self.x_proj.weight.auto_grad_accumulation = True

        # TODO: the weights are innitialized a bit differently here https://github.com/state-spaces/mamba/blob/0cce0fa645f100f00620ddf2333c2b7712abfdec/mamba_ssm/modules/mamba_simple.py#L82
        self.dt_proj_weight = ParameterMeta.from_dims(
            (td_inner, tdt_rank),
            init_method=kaiming_init_(tdt_rank.size),
        )

        self.dt_proj_bias = ParameterMeta.from_dims(
            (td_inner,),
            init_method=init_dtprojbias(
                self.d_inner, self.config.dt_max, self.config.dt_min, self.config.dt_init_floor, factory_kwargs
            ),
        )

        self.A_log = ParameterMeta.from_dims(
            (td_inner, td_state),
            weight_decay=False,
            init_method=init_A(self.d_state, self.d_inner),
        )

        # D "skip" parameter
        self.D = ParameterMeta.from_dims(
            (td_inner,),
            weight_decay=False,
            init_method=init_ones_,
        )

        self.out_proj = Linear(
            td_inner,
            td_model,
            bias=False,  # TODO: note, if bias is used there is a problem in the MambaInnerFn.backward for the bias grads. I think this bias is not used in other mamba repos.
            weight_init_method=kaiming_init_(td_model.size),
            **factory_kwargs,
        )
        if self.use_fast_path:
            self.out_proj.weight.auto_grad_accumulation = True

    def forward(self, hidden_states, **kwargs):
        batch, seqlen, dim = hidden_states.shape
        # print("IN MAMBA LAYER FORWARD: ", hidden_states.shape)
        conv_state, ssm_state = None, None

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj_weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self._debug_mode:
            print("XZ: ", xz.shape)

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path:  # Doesn't support outputting the states, only for training
            out = mamba_inner_fn(
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
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                conv_state.copy_(x[:, :, -self.d_conv :])  # Update state (B D W)
            if causal_conv1d_fn is None:
                raise NotImplementedError("Causal conv1d not implemented")
                # x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x,
                    rearrange(self.conv1d_weight, "d 1 w -> d w"),
                    self.conv1d_bias,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj_weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj_bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out
