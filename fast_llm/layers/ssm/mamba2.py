"""
This code is adapted from https://github.com/jxiw/MambaInLlama/blob/main/mamba2/hybrid_mamba_layer.py
"""

import math
import typing

import einops
import torch
from causal_conv1d import causal_conv1d_fn as _causal_conv1d_fn
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

from fast_llm.engine.config_utils.tensor_space import TensorDim, TensorSpace
from fast_llm.layers.common.linear import Linear
from fast_llm.layers.ssm.config import SSMConfig, SSMDimNames
from fast_llm.tensor import ParameterMeta, init_fill_, init_ones_, init_uniform_, kaiming_init_
from fast_llm.utils import get_lr_scale


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def bias_init_method(conv_weight):
    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(conv_weight)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    return init_uniform_(-bound, bound)


class Mamba2(torch.nn.Module):
    def __init__(
        self,
        config: SSMConfig,
        layer_idx: int,
        tensor_space: TensorSpace,
        return_input: bool = False,
    ):
        # factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.config: SSMConfig = config
        bias: bool = config.add_bias_linear
        self.layer_idx = layer_idx
        self._return_input = return_input
        layer_lr_scale: float | None = config.per_layer_lr_scale[layer_idx] if config.per_layer_lr_scale else None
        mamba_layer_lr_scale: float | tuple[float | None, ...] | None = get_lr_scale(
            self.config.mamba_lr_scale, layer_lr_scale
        )

        td_inner: TensorDim = tensor_space.get_tensor_dim(name=SSMDimNames.inner_dim)
        td_state: TensorDim = tensor_space.get_tensor_dim(name=SSMDimNames.state_dim)
        td_model: TensorDim = tensor_space.get_tensor_dim(name=SSMDimNames.model_dim)
        tdt_rank: TensorDim = tensor_space.get_tensor_dim(name=SSMDimNames.dt_rank)
        # tensor_space.get_tensor_dim(SSMDimNames.conv_dim)
        # tensor_space.get_tensor_dim(SSMDimNames.qk_heads)
        # tensor_space.get_tensor_dim(SSMDimNames.v_heads)
        td_xb: TensorDim = tensor_space.get_tensor_dim(name=SSMDimNames.d_xb)
        td_inner_proj: TensorDim = tensor_space.get_tensor_dim(name=SSMDimNames.inner_proj_mamba2)
        td_conv_kernel: TensorDim = tensor_space.get_tensor_dim(name=SSMDimNames.conv_kernel_size)

        self.repeat_kv_before_conv = config.repeat_kv_before_conv

        # self.d_model = d_model
        # self.d_state = d_state
        # self.d_conv = d_conv
        # self.conv_init = conv_init
        # self.expand = expand
        # self.process_group = process_group
        # self.sequence_parallel = sequence_parallel
        # self.world_size = 1 if process_group is None else process_group.size()
        # self.local_rank = 0 if process_group is None else process_group.rank()
        # self.d_inner = d_inner if d_inner is not None else (self.expand * self.d_model) // self.world_size
        # # assert self.d_inner * self.world_size == self.expand * self.d_model
        # self.headdim = headdim
        # self.d_ssm = self.d_inner if d_ssm is None else d_ssm // self.world_size
        # assert ngroups % self.world_size == 0
        # self.ngroups = ngroups // self.world_size
        # assert self.d_ssm % self.headdim == 0
        # self.nheads = self.d_ssm // self.headdim
        # self.D_has_hdim = D_has_hdim
        # self.rmsnorm = rmsnorm
        # self.norm_before_gate = norm_before_gate
        # self.dt_limit = dt_limit
        # self.activation = "silu"
        # self.chunk_size = chunk_size
        # self.use_mem_eff_path = use_mem_eff_path
        # self.layer_idx = layer_idx
        # self.d_xb = d_xb
        # self.repeat_group = self.d_inner // self.d_xb
        # self.repeat_kv_before_conv = repeat_kv_before_conv

        self.d_state = td_state.size
        self.d_model = td_model.size
        self.d_xb = td_xb.size
        self.d_inner = td_inner.size
        self.dt_rank = tdt_rank.size
        # conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state

        if self.repeat_kv_before_conv:
            self.conv1d_weight = ParameterMeta.from_dims(
                (td_inner, TensorDim("1", 1), td_conv_kernel),
                init_method=init_uniform_(
                    1 / math.sqrt(td_inner.size * td_conv_kernel.size),
                    1 / math.sqrt(td_inner.size * td_conv_kernel.size),
                ),  # see https://github.com/pytorch/pytorch/blob/1eba9b3aa3c43f86f4a2c807ac8e12c4a7767340/torch/nn/modules/conv.py#L180C53-L180C67
                lr_scale=mamba_layer_lr_scale,
            )
            self.conv1d_bias = ParameterMeta.from_dims(
                (td_inner,), init_method=bias_init_method(self.conv1d_weight), lr_scale=mamba_layer_lr_scale
            )
            # self.conv1d = nn.Conv1d(
            #     in_channels=td_inner.size,
            #     out_channels=td_inner.size,
            #     bias=True,
            #     kernel_size=td_conv_kernel.size,
            #     groups=td_inner.size,
            #     padding=td_conv_kernel.size - 1,
            # )
        else:
            self.conv1d_weight = ParameterMeta.from_dims(
                (td_xb, TensorDim("1", 1), td_conv_kernel),
                init_method=init_uniform_(
                    1 / math.sqrt(td_xb.size * td_conv_kernel.size),
                    1 / math.sqrt(td_xb.size * td_conv_kernel.size),
                ),
            )
            self.conv1d_bias = ParameterMeta.from_dims(
                (td_xb,), init_method=bias_init_method(self.conv1d_weight), lr_scale=mamba_layer_lr_scale
            )
            # self.conv1d = nn.Conv1d(
            #     in_channels=td_xb.size,
            #     out_channels=td_xb.size,
            #     bias=True,
            #     kernel_size=td_conv_kernel.size,
            #     groups=td_xb.size,
            #     padding=td_conv_kernel.size - 1,
            # )

        self.activation = "silu"
        # self.act = nn.SiLU() # not used

        self.num_xb_head = td_xb.size // td_state.size
        self.num_C_head = td_inner.size // td_state.size
        self.repeat_group = self.num_C_head // self.num_xb_head

        self.in_proj = Linear(
            td_model,
            td_inner_proj,
            bias=bias,
            weight_init_method=kaiming_init_(td_model.size),
            lr_scale=mamba_layer_lr_scale,
        )

        # Initialize special dt projection to preserve variance at initialization
        dt_scale = 1.0
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if config.dt_init == "constant":
            dt_init = init_fill_(dt_init_std)
            # nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            dt_init = init_uniform_(-dt_init_std, dt_init_std)
            # nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt_max = 0.1
        dt_min = 0.001
        dt_init_floor = 1e-4
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(
            min=dt_init_floor
        )
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))

        def init_from_tensor_(
            value: torch.Tensor,
        ) -> typing.Callable[[ParameterMeta, torch.Tensor, torch.Generator], torch.Tensor]:
            def init_(meta: ParameterMeta, tensor: torch.Tensor, generator: torch.Generator):  # noqa
                return tensor.copy_(value)

            return init_

        self.dt_proj = Linear(
            tdt_rank,
            td_inner,
            bias=False,
            weight_init_method=dt_init,
            # bias_init_method=init_ones_,  # init_from_tensor_(inv_dt),
            lr_scale=mamba_layer_lr_scale,
        )
        # define bias outside the linear layer since its also used in the selective_scan_fn
        self.dt_proj_bias = ParameterMeta.from_dims(
            (td_inner,), init_method=init_from_tensor_(inv_dt), lr_scale=mamba_layer_lr_scale
        )

        A = einops.repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A).flatten()  # Keep A_log in fp32
        self.A_log = ParameterMeta.from_dims(
            (td_inner, td_state),
            init_method=init_from_tensor_(A_log),
            lr_scale=mamba_layer_lr_scale,
            weight_decay=False,
        )

        self.D = ParameterMeta.from_dims(
            (td_inner,),
            weight_decay=False,
            init_method=init_ones_,
            lr_scale=mamba_layer_lr_scale,
        )

        self.out_proj = Linear(
            td_inner,
            td_model,
            bias=bias,
            weight_init_method=kaiming_init_(td_inner.size),
        )

    def forward(self, hidden_states, kwargs):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape
        outputs = {}

        conv_state, ssm_state = None, None

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        zxbcdt = self.in_proj(hidden_states)
        z, x, B, C, dt = torch.split(zxbcdt, [self.d_inner, self.d_xb, self.d_xb, self.d_inner, self.dt_rank], dim=-1)

        x = einops.rearrange(x, "b l d -> b d l")
        z = einops.rearrange(z, "b l d -> b d l")

        B = einops.rearrange(B, "b l (n_group dstate) -> b n_group l dstate", dstate=self.d_state)
        B = repeat_kv(B, self.repeat_group)  # B, n_group, L, H
        B = einops.rearrange(B, "b n_group l dstate -> b n_group dstate l").contiguous()
        C = einops.rearrange(C, "b l (n_group dstate) -> b n_group dstate l", dstate=self.d_state).contiguous()

        dt = self.dt_proj(dt) + self.dt_proj_bias  # B, L, d_inner
        dt = einops.rearrange(dt, "b l d -> b d l")  # B, d_inner, L

        if self.repeat_kv_before_conv:
            assert self.repeat_group > 0
            x = einops.rearrange(x, "b (n_group dstate) l -> b n_group l dstate", dstate=self.d_state)
            x = repeat_kv(x, self.repeat_group)
            x = einops.rearrange(x, "b n_group l dstate -> b (n_group dstate) l")

        # Compute short convolution
        # if conv_state is not None:
        #     # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
        #     # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
        #     # Update state (B D W)
        #     conv_state.copy_(torch.nn.functional.pad(x, (self.d_conv - x.shape[-1], 0)))

        assert self.activation in ["silu", "swish"]
        x = _causal_conv1d_fn(
            x=x,
            weight=einops.rearrange(self.conv1d_weight, "d 1 w -> d w"),
            bias=self.conv1d_bias,
            activation=self.activation,
        )  # B, L, D

        if not self.repeat_kv_before_conv:
            x = einops.rearrange(x, "b (n_group dstate) l -> b n_group l dstate", dstate=self.d_state)
            x = repeat_kv(x, self.repeat_group)
            x = einops.rearrange(x, "b n_group l dstate -> b (n_group dstate) l")

        y = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=z,
            delta_bias=self.dt_proj_bias.float(),  # self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=False,
        )

        if ssm_state is not None:
            y, last_state = y
            ssm_state.copy_(einops.rearrange(last_state, "b (h d) n -> b h d n", h=self.num_C_head))

        y = einops.rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        outputs["hidden_states"] = out[:, :seqlen, :].contiguous()
        return outputs["hidden_states"], None
