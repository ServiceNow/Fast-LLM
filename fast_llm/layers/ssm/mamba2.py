"""
This code is adapted from https://github.com/jxiw/MambaInLlama/blob/main/mamba2/hybrid_mamba_layer.py
"""

import math

import causal_conv1d
import einops
import mamba_ssm.ops.triton.ssd_combined
import torch
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated

from fast_llm.engine.config_utils.tensor_space import TensorSpace
from fast_llm.layers.common.linear import Linear
from fast_llm.layers.ssm.config import SSMConfig, SSMDimNames
from fast_llm.tensor import kaiming_init_


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


class Mamba2(torch.nn.Module):
    def __init__(
        self,
        config: SSMConfig,
        layer_idx: int,
        tensor_space: TensorSpace,
    ):
        # factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.config: SSMConfig = config
        bias = config.add_bias_linear
        self.layer_idx = layer_idx

        td_inner = tensor_space.get_tensor_dim(SSMDimNames.inner_dim)
        tensor_space.get_tensor_dim(SSMDimNames.state_dim)
        td_model = tensor_space.get_tensor_dim(SSMDimNames.model_dim)
        tensor_space.get_tensor_dim(SSMDimNames.conv_dim)
        tensor_space.get_tensor_dim(SSMDimNames.qk_heads)
        tensor_space.get_tensor_dim(SSMDimNames.v_heads)
        tensor_space.get_tensor_dim(SSMDimNames.conv_kernel_size)
        tensor_space.get_tensor_dim(SSMDimNames.inner_proj_mamba2)

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

        assert self.d_inner == self.ngroups * self.d_state
        assert self.d_inner == self.d_ssm

        self.nheads = self.ngroups
        self.headdim = self.d_state

        # Order: [z, x, B, C, dt]
        # [hidden_dim, hidden_dim, d_state]
        d_in_proj = self.d_inner + self.d_xb + self.d_xb + self.d_inner + self.nheads
        # d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        if self.process_group is None:
            self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        else:
            self.in_proj = ColumnParallelLinear(
                self.d_model,
                d_in_proj * self.world_size,
                bias=bias,
                process_group=self.process_group,
                sequence_parallel=self.sequence_parallel,
                **factory_kwargs,
            )

        # conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state

        if self.repeat_kv_before_conv:
            conv_dim = self.d_inner + self.d_inner + self.d_inner
            self.conv1d = nn.Conv1d(
                in_channels=conv_dim,
                out_channels=conv_dim,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=conv_dim,
                padding=d_conv - 1,
                **factory_kwargs,
            )
        else:
            conv_dim = self.d_inner + self.d_xb + self.d_xb
            self.conv1d = nn.Conv1d(
                in_channels=conv_dim,
                out_channels=conv_dim,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=conv_dim,
                padding=d_conv - 1,
                **factory_kwargs,
            )

        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device))
        self.D._no_weight_decay = True

        if self.rmsnorm:
            assert RMSNormGated is not None
            self.norm = RMSNormGated(
                self.d_ssm,
                eps=1e-5,
                norm_before_gate=self.norm_before_gate,
                group_size=self.d_ssm // ngroups,
                **factory_kwargs,
            )

        # self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_proj = Linear(
            td_inner,
            td_model,
            bias=bias,
            weight_init_method=kaiming_init_(td_inner.size),
        )

    def forward(self, u, seqlen=None, seq_idx=None, cu_seqlens=None, inference_params=None):
        """
        u: (batch, seqlen, hidden_dim) if seqlen=None.
            If seqlen is not None, u is (batch * seqlen, hidden_dim). This is so that when we
            split u during sequence parallel, we split the batch * seqlen dimension
            (in case batch is small).
        Returns: same shape as u
        """
        seqlen_og = seqlen
        if seqlen is None:
            batch, seqlen, dim = u.shape
        else:
            batch_seqlen, dim = u.shape
            batch = batch_seqlen // seqlen

        conv_state, ssm_state = None, None
        if inference_params is not None:
            inference_batch = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else batch
            conv_state, ssm_state = self._get_states_from_cache(inference_params, inference_batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(u, conv_state, ssm_state)
                return out

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj) or (B * L, d_in_proj)
        if seqlen_og is not None:
            zxbcdt = einops.rearrange(zxbcdt, "(b l) d -> b l d", l=seqlen)
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        A = -torch.exp(self.A_log.float())  # (nheads) or (d_inner, d_state)
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        # [z, x, B, C, dt]
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_inner - 2 * self.d_xb - self.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt, [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.d_xb, self.nheads], dim=-1
        )

        if self.repeat_kv_before_conv:
            x, B, C = torch.split(xBC, [self.d_xb, self.d_xb, self.ngroups * self.d_state], dim=-1)
            # minic the GQA
            x = einops.rearrange(x, "b l (xb_group dstate) -> b xb_group l dstate", dstate=self.d_state)
            x = repeat_kv(x, self.repeat_group)
            # x shape: (bsz, n_group, l, dim)
            B = einops.rearrange(B, "b l (xb_group dstate) -> b xb_group l dstate", dstate=self.d_state)
            B = repeat_kv(B, self.repeat_group)
            # combine x, B, C
            x = einops.rearrange(x, "b g l p -> b l (g p)")
            B = einops.rearrange(B, "b g l p -> b l (g p)")
            xBC = torch.cat((x, B, C), dim=-1)

        if conv_state is not None:
            if cu_seqlens is None:
                # If we just take xBC[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                xBC_t = einops.rearrange(xBC, "b l d -> b d l")
                conv_state.copy_(
                    torch.nn.functional.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0))
                )  # Update state (B D W)
            else:
                assert (
                    causal_conv1d.causal_conv1d_varlen_states is not None
                ), "varlen inference requires causal_conv1d package"
                assert batch == 1, "varlen inference only supports batch dimension 1"
                conv_varlen_states = causal_conv1d.causal_conv1d_varlen_states(
                    xBC.squeeze(0), cu_seqlens, state_len=conv_state.shape[-1]
                )
                conv_state.copy_(conv_varlen_states)
        assert self.activation in ["silu", "swish"]

        if causal_conv1d.causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
            assert seq_idx is None, "varlen conv1d requires the causal_conv1d package"
            xBC = self.act(
                self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, -(self.d_conv - 1) :]
            )  # (B, L, self.d_ssm + 2 * ngroups * d_state)
        else:
            xBC = causal_conv1d.causal_conv1d_fn(
                xBC.transpose(1, 2),
                einops.rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
                seq_idx=seq_idx,
            ).transpose(1, 2)

        if self.repeat_kv_before_conv:
            x, B, C = torch.split(
                xBC, [self.ngroups * self.d_state, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1
            )

            y = mamba_ssm.ops.triton.ssd_combined.mamba_chunk_scan_combined(
                einops.rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                dt,
                A,
                einops.rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                einops.rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=einops.rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                z=einops.rearrange(z, "b l (h p) -> b l h p", p=self.headdim) if not self.rmsnorm else None,
                dt_bias=self.dt_bias,
                dt_softplus=True,
                seq_idx=seq_idx,
                cu_seqlens=cu_seqlens,
                **dt_limit_kwargs,
                return_final_states=ssm_state is not None,
                return_varlen_states=cu_seqlens is not None and inference_params is not None,
            )

        else:
            # self.d_xb + self.d_xb + self.d_inner
            x, B, C = torch.split(xBC, [self.d_xb, self.d_xb, self.ngroups * self.d_state], dim=-1)

            # minic the GQA
            x = einops.rearrange(x, "b l (xb_group dstate) -> b xb_group l dstate", dstate=self.d_state)
            x = repeat_kv(x, self.repeat_group)
            # x shape: (bsz, n_group, l, dim)

            B = einops.rearrange(B, "b l (xb_group dstate) -> b xb_group l dstate", dstate=self.d_state)
            B = repeat_kv(B, self.repeat_group)

            y = mamba_ssm.ops.triton.ssd_combined.mamba_chunk_scan_combined(
                # einops.rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                einops.rearrange(x, "b g l p -> b l g p"),
                dt,
                A,
                # einops.rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                einops.rearrange(B, "b g l n -> b l g n"),
                einops.rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=einops.rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                z=einops.rearrange(z, "b l (h p) -> b l h p", p=self.headdim) if not self.rmsnorm else None,
                dt_bias=self.dt_bias,
                dt_softplus=True,
                seq_idx=seq_idx,
                cu_seqlens=cu_seqlens,
                **dt_limit_kwargs,
                return_final_states=ssm_state is not None,
                return_varlen_states=cu_seqlens is not None and inference_params is not None,
            )

        if ssm_state is not None:
            y, last_state, *rest = y
            if cu_seqlens is None:
                ssm_state.copy_(last_state)
            else:
                varlen_states = rest[0]
                ssm_state.copy_(varlen_states)
        y = einops.rearrange(y, "b l h p -> b l (h p)")
        if self.rmsnorm:
            y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([torch.nn.functional.silu(z0) * x0, y], dim=-1)
        if seqlen_og is not None:
            y = einops.rearrange(y, "b l d -> (b l) d")
        out = self.out_proj(y)
        return out

        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_conv,
                self.conv1d.weight.shape[0],
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            ).transpose(1, 2)
            ssm_state = torch.zeros(
                batch_size,
                self.nheads,
                self.headdim,
                self.d_state,
                device=self.in_proj.weight.device,
                dtype=self.in_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state
