import math
from typing import Optional, Union, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from fast_llm.engine.config_utils.tensor_space import TensorDim

from einops import rearrange, repeat
from fast_llm.layers.ssm.config import MambaConfig
from fast_llm.layers.common.linear import Linear
from fast_llm.layers.common.conv import Conv1D
from fast_llm.tensor import ParameterMeta, init_zeros_, init_ones_

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn = None
    causal_conv1d_update = None
    

from ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

try:
    from ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None
    

'''
Note: this is mostly copied from https://github.com/Zyphra/Zamba2, similar code is aslo in https://github.com/state-spaces/mamba
'''

def kaiming_init(a=math.sqrt(5)):
    # same as torch linear layer init https://github.com/pytorch/pytorch/blob/b248edd7ccae15cf3a2dd86442149e4bc029846a/torch/nn/modules/linear.py#L114
    def init_(meta: ParameterMeta, tensor: torch.Tensor, generator: torch.Generator):  # noqa
        nn.init.kaiming_normal_(tensor, a=a)
        return tensor
    return init_

def init_A(d_state, d_inner) -> Callable[[ParameterMeta, torch.Tensor, torch.Generator], torch.Tensor]:
    def init_(meta: ParameterMeta, tensor: torch.Tensor, generator: torch.Generator):  # noqa        
        # S4D real initialization
        # TODO: make sure the innitialization works
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=d_inner
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        tensor.copy_(A_log)
        return tensor

    return init_

def init_dtprojbias(d_inner: int, dt_max: float, dt_min: float, dt_init_floor: float, factory_kwargs: dict) -> Callable[[ParameterMeta, torch.Tensor, torch.Generator], torch.Tensor]:
    def init_(meta: ParameterMeta, tensor: torch.Tensor, generator: torch.Generator):  # noqa        
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        tensor.copy_(inv_dt)
        return tensor
    return init_

class MambaLayer(nn.Module):
    def __init__(
        self,
        config: MambaConfig,
        layer_idx = 0
    ):
        factory_kwargs = {}
        super().__init__()
        self.config = config
        self.d_model = config.hidden_size
        self.d_state = config.state_size
        self.d_conv = config.conv_dimension
        self.expand = config.expansion_factor
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if config.dt_rank == "auto" else config.dt_rank
        self.use_fast_path = config.use_fast_path if mamba_inner_fn is not None else False
        self.layer_idx = layer_idx
        self.device = config.device


        # Tensor dims:
        # TODO: how to correctly set parallel dims for distributed training?
        td_inner = TensorDim("D_inner", self.d_inner)
        td_inner_times2 = TensorDim("D_inner_2", self.d_inner * 2)
        tdt_rank = TensorDim("D_rank", self.dt_rank)
        td_x_proj = TensorDim("D_x_proj", self.dt_rank + self.d_state * 2)
        td_state = TensorDim("D_state", self.d_state)
        td_model = TensorDim("D_model", self.d_model)
        
        self.in_proj = Linear(td_model, td_inner_times2,
                              weight_init_method=kaiming_init(),
                              bias_init_method=init_zeros_, bias=config.add_bias_linear, **factory_kwargs) # for upscaled x and residual

        self.conv1d = Conv1D(
            in_channels=td_inner,
            out_channels=td_inner,
            bias=config.conv_bias,
            kernel_size=self.d_conv,
            groups=self.d_inner, # independent kernel per d_inned
            padding=self.d_conv - 1,
            weight_init_method=kaiming_init(),
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()
        

        self.x_proj = Linear(
            td_inner, td_x_proj, weight_init_method=kaiming_init(), bias=False, **factory_kwargs
        )

        self.dt_proj = Linear(tdt_rank, td_inner, bias=True, weight_init_method=kaiming_init(),
                              bias_init_method=init_dtprojbias(self.d_inner, config.dt_max, config.dt_min, config.dt_init_floor, factory_kwargs), 
                              **factory_kwargs)# TODO: check init method

        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True


        self.A_log = ParameterMeta(torch.empty(self.d_inner, self.d_state, device="meta"), 
                                    dims=(td_inner, td_state), 
                                    weight_decay=False, 
                                    init_method=init_A(self.d_state, self.d_inner))

        # D "skip" parameter
        self.D = ParameterMeta(torch.empty(self.d_inner, device="meta"), 
                                dims=(td_inner,), 
                                weight_decay=False, 
                                init_method=init_ones_)

        self.out_proj = Linear(td_inner, td_model, 
                               bias=False, # TODO: note, if bias is used there is a problem in the MambaInnerFn.backward for the bias grads. I think this bias is not used in other mamba repos.
                               weight_init_method=kaiming_init(),
                               **factory_kwargs)

    def forward(self, hidden_states, from_shared_proj = None, inference_params=None):
        batch, seqlen, dim = hidden_states.shape
        # print("IN MAMBA LAYER FORWARD: ", hidden_states.shape)
        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.sequence_len_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time    

        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        print("XZ: ", xz.shape)

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and inference_params is None:  # Doesn't support outputting the states
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                conv_state.copy_(x[:, :, -self.d_conv :])  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x,
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    self.conv1d.bias,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
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
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out
    
    # INFERENCE-SPECIFIC CODE, CURRENTLY UNUSED
    def step(self, hidden_states, conv_state, ssm_state):
        # TODO: OO, remove this and the following, as this is specific to inference?
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state