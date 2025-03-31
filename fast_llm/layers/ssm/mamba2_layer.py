import math
from typing import Callable
import torch
import torch.nn as nn
from fast_llm.engine.config_utils.tensor_space import TensorDim, TensorSpace

from einops import rearrange, repeat
from fast_llm.layers.ssm.config import MambaConfig, SSMDimNames
from fast_llm.layers.common.linear import Linear
from fast_llm.tensor import ParameterMeta, init_ones_, init_uniform_, init_zeros_
import torch.nn.functional as F

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None

try:
    from ops.triton.layernorm_gated import RMSNorm as RMSNormGated, LayerNorm
except ImportError:
    RMSNormGated, LayerNorm = None, None

from ops.triton.ssd_combined import mamba_chunk_scan_combined
from ops.triton.ssd_combined import mamba_split_conv1d_scan_combined


"""
Note: this is mostly addapted from https://github.com/Zyphra/Zamba2, similar code is aslo in https://github.com/state-spaces/mamba.
For now it only supports training and not inference.
Triton: 2.3.1
"""


def kaiming_init(d_in, d_out, a=math.sqrt(5)):
    # same as torch linear layer init https://github.com/pytorch/pytorch/blob/b248edd7ccae15cf3a2dd86442149e4bc029846a/torch/nn/modules/linear.py#L114
    def init_(meta: ParameterMeta, tensor: torch.Tensor, generator: torch.Generator):  # noqa
        tensor_view = tensor.view(d_out, d_in)
        nn.init.kaiming_normal_(tensor_view, a=a)
        return tensor

    return init_


def init_A(_min, _max, nheads, dtype) -> Callable[[ParameterMeta, torch.Tensor, torch.Generator], torch.Tensor]:
    def init_(meta: ParameterMeta, tensor: torch.Tensor, generator: torch.Generator):  # noqa
        A = torch.empty(nheads, dtype=torch.float32).uniform_(_min, _max)
        A_log = torch.log(A).to(dtype=dtype)
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


class Mamba2Layer(nn.Module):
    def __init__(
        self,
        config: MambaConfig,
        layer_idx: int,
        tensor_space: TensorSpace,
    ):
        factory_kwargs = {"device": "meta"}  # {"dtype": dtype}
        super().__init__()
        self.config: MambaConfig = config
        bias = False  # TODO: hard coded
        self.layer_idx = layer_idx

        td_inner = tensor_space.get_tensor_dim(SSMDimNames.d_inner)
        tdt_rank = tensor_space.get_tensor_dim(SSMDimNames.dt_rank)
        td_state = tensor_space.get_tensor_dim(SSMDimNames.d_state)
        td_model = tensor_space.get_tensor_dim(SSMDimNames.d_model)
        td_conv = tensor_space.get_tensor_dim(SSMDimNames.d_conv)
        td_nheads = tensor_space.get_tensor_dim(SSMDimNames.d_nheads)
        td_headdim = tensor_space.get_tensor_dim(SSMDimNames.d_headdim)

        td_inner_proj = tensor_space.get_tensor_dim(SSMDimNames.d_inner_proj)
        self.d_conv = td_conv.size
        self.d_inner = td_inner.size
        self.d_state = td_state.size
        self.d_model = td_model.size
        self.dt_rank = tdt_rank.size
        self.nheads = td_nheads.size
        self.headdim = config.headdim
        self.ngroups = config.ngroups

        # Order: [z, x, B, C, dt]
        # d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = Linear(
            td_model, td_inner_proj, bias=bias, weight_init_method=kaiming_init(td_model.size, td_inner_proj.size)
        )

        self.conv1d_weight = ParameterMeta(
            torch.empty(
                self.d_inner, self.d_inner // self.d_inner, self.d_conv, dtype=torch.float32, **factory_kwargs
            ),
            dims=(td_inner, TensorDim("D_inner_2", self.d_inner // self.d_inner), td_conv),
            init_method=(
                kaiming_init(td_inner.size, self.d_conv)
                if self.config.conv_init is None
                else init_uniform_(-self.config.conv_init, self.config.conv_init)
            ),
        )

        self.conv1d_bias = None

        self.learnable_init_states = False
        if self.learnable_init_states:
            self.init_states = ParameterMeta(
                torch.empty(self.nheads, self.headdim, self.d_state, **factory_kwargs),
                dims=(td_nheads, td_headdim, td_state),
                init_method=init_zeros_,
                weight_decay=False,
            )

        self.activation = "silu"
        self.act = nn.SiLU()

        # # Initialize log dt bias
        # dt = torch.exp(
        #     torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
        #     + math.log(dt_min)
        # )
        # dt = torch.clamp(dt, min=dt_init_floor)
        # # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        # inv_dt = dt + torch.log(-torch.expm1(-dt))
        # self.dt_bias = nn.Parameter(inv_dt)

        self.dt_bias = ParameterMeta(
            torch.empty(self.nheads, **factory_kwargs), dims=(td_nheads,), init_method=init_zeros_, weight_decay=False
        )

        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        # self.dt_bias._no_weight_decay = True

        self.dt_limit = (0.0, float("inf"))
        A_init_range = (1, 16)
        # A parameter
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        self.A_log = ParameterMeta(
            torch.empty(self.nheads, dtype=torch.float32, **factory_kwargs),
            dims=(td_nheads,),
            weight_decay=False,
            init_method=init_A(A_init_range[0], A_init_range[1], self.nheads, torch.float32),
        )

        # D "skip" parameter
        self.D = ParameterMeta(
            torch.empty(self.nheads, **factory_kwargs), dims=(td_nheads,), weight_decay=False, init_method=init_ones_
        )

        # Extra normalization layer right before output projection
        assert RMSNormGated is not None
        if self.config.use_mem_eff_path:
            self.norm_weight = ParameterMeta(
                torch.empty(self.d_inner, **factory_kwargs),
                dims=(td_inner,),
                weight_decay=False,
                init_method=init_ones_
            )
            self.norm_eps = 1e-5
        else:
            self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False)

        if self.config.use_mem_eff_path:
            self.out_proj_weight = ParameterMeta(
                torch.empty(td_model.size, td_inner.size, dtype=torch.float32, **factory_kwargs),
                dims=(td_model, td_inner),
                init_method=kaiming_init(td_inner.size, td_model.size),
            )

            self.out_proj_bias = None
        else:
            self.out_proj = Linear(
                td_inner,
                td_model,
                bias=False,
                weight_init_method=kaiming_init(td_inner.size, td_model.size),
            )

    def forward(self, u, seq_idx=None, **kwargs):
        """
        u: (B, L, D)
        Returns: same shape as u
        """
        batch, seqlen, dim = u.shape

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        A = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)
        initial_states = repeat(self.init_states, "... -> b ...", b=batch) if self.learnable_init_states else None
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        if self.config.use_mem_eff_path:
            # Fully fused path
            out = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d_weight, "d 1 w -> d w"),
                self.conv1d_bias,
                self.dt_bias,
                A,
                D=self.D,
                chunk_size=self.config.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm_weight,
                rmsnorm_eps=self.norm_eps,
                outproj_weight=self.out_proj_weight,
                outproj_bias=self.out_proj_bias,
                headdim=self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=False,
                initial_states=initial_states,
                **dt_limit_kwargs,
            )
        else:
            z, xBC, dt = torch.split(
                zxbcdt, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1
            )
            dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)
            assert self.activation in ["silu", "swish"]

            # 1D Convolution
            if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                raise NotImplementedError("Causal conv1d not implemented")
                # xBC = self.act(
                #     self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)
                # )  # (B, L, self.d_inner + 2 * ngroups * d_state)
                # xBC = xBC[:, :seqlen, :]
            else:
                xBC = causal_conv1d_fn(
                    x=xBC.transpose(1, 2),
                    weight=rearrange(self.conv1d_weight, "d 1 w -> d w"),
                    bias=self.conv1d_bias,
                    activation=self.activation,
                ).transpose(1, 2)

            # Split into 3 main branches: X, B, C
            # These correspond to V, K, Q respectively in the SSM/attention duality
            x, B, C = torch.split(
                xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1
            )
            y = mamba_chunk_scan_combined(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                dt,
                A,
                rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.config.chunk_size,
                D=self.D,
                z=None,
                seq_idx=seq_idx,
                initial_states=initial_states,
                **dt_limit_kwargs,
            )
            y = rearrange(y, "b l h p -> b l (h p)")

            # Multiply "gate" branch and apply extra normalization layer
            y = self.norm(y, z)
            out = self.out_proj(y)
        return out
