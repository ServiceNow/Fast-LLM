"""
Apriel2 modeling - HuggingFace format that mirrors Fast-LLM's architecture.
"""

import math
import random
from typing import Any, Optional, Union
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from transformers import GenerationMixin, PreTrainedModel
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import logging

from fast_llm_external_models.apriel2.configuration_apriel2 import Apriel2Config
from fast_llm_external_models.apriel2.cache import Apriel2Cache
from transformers.models.mistral.modeling_mistral import (
    MistralAttention,
    MistralMLP,
    MistralRMSNorm,
)
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextGatedDeltaNet

from transformers.utils.import_utils import is_torch_flex_attn_available
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

# Try to import optimized kernels from mamba_ssm
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
    _mamba_ssm_available = True
except ImportError:
    causal_conv1d_fn = None
    causal_conv1d_update = None
    selective_scan_fn = None
    selective_state_update = None
    _mamba_ssm_available = False

# Try to import FLA (Fused Linear Attention) library for optimizations
try:
    from fla.modules import FusedRMSNormGated
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    _fla_available = True
except ImportError:
    FusedRMSNormGated = None
    chunk_gated_delta_rule = None
    fused_recurrent_gated_delta_rule = None
    _fla_available = False

if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask
else:
    BlockMask = torch.Tensor

logger = logging.get_logger(__name__)

is_fast_path_available = _mamba_ssm_available and all((selective_state_update, causal_conv1d_fn, causal_conv1d_update))

# Log availability of optimized kernels
if not _mamba_ssm_available:
    logger.warning("mamba_ssm library not available. Mamba layers will not work without it.")
if not _fla_available:
    logger.info("FLA (Fused Linear Attention) library not available. Using fallback implementations for GatedDeltaNet.")
if not is_fast_path_available:
    logger.warning("Fast path for Mamba is not available. Some kernels are missing.")


# Helper functions for Mamba
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


@torch.compile
def segsum(x):
    """More stable segment sum calculation."""
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


@torch.compile
def materialize_mixer(A_log, B, C, D):
    """
    Since the transfer matrix will be equated to the attention matrix,
    we need to support the form: torch.matmul(attn_weights, value_states).
    Thus, y = torch.matmul(T, X)
    """
    batch_size, length, n_heads, d_state = B.shape
    assert A_log.shape == (batch_size, length, n_heads)
    assert B.shape == C.shape == (batch_size, length, n_heads, d_state)

    A_log = rearrange(-F.softplus(A_log), "b l h -> b h l")
    powers = torch.exp(segsum(A_log))
    T = torch.einsum("blhn,bshn,bhls->bhsl", C, B, powers)

    if D is not None:
        T[:, :, torch.arange(length), torch.arange(length)] += D.view(1, n_heads, 1)

    T = rearrange(T, "b h z l -> b h l z")
    return T


def apply_mask_to_padding_states(hidden_states, attention_mask):
    """Tunes out the hidden states for padding tokens."""
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
    return hidden_states


class Apriel2Attention(nn.Module):
    """
    Attention wrapper that handles rotary embeddings internally.
    Contains self.self_attn and self.rotary_emb as sub-modules.
    Mirrors Fast-LLM's architecture where each Attention has its own rotary.
    """

    def __init__(self, d_model: int, mixer_config: dict, layer_idx: int, config):
        super().__init__()

        # Extract attention parameters from mixer_config
        num_heads = mixer_config.get("heads", 32)
        num_key_value_heads = mixer_config.get("head_groups", num_heads)
        head_dim = mixer_config.get("head_size", d_model // num_heads)
        rope_theta = (
            mixer_config.get("rotary", {}).get("theta", 10000.0)
            if isinstance(mixer_config.get("rotary"), dict)
            else 10000.0
        )

        # Create attention config
        attn_config = SimpleNamespace(
            hidden_size=d_model,
            num_attention_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=rope_theta,
            attention_dropout=0.0,
            sliding_window=mixer_config.get("sliding_window", None),
            _attn_implementation=config._attn_implementation,
        )

        # Create attention sub-module
        self.self_attn = MistralAttention(attn_config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple] = None,
        **kwargs,
    ):
        return self.self_attn(hidden_states, position_embeddings, attention_mask, **kwargs)


def create_mixer(mixer_config: dict, hidden_size: int, layer_idx: int, config, allow_stochastic: bool = True):
    mixer_type = mixer_config.get("type", "attention")

    if mixer_type == "attention":
        return Apriel2Attention(hidden_size, mixer_config, layer_idx, config)
    elif mixer_type == "mamba":
        return Apriel2Mamba(hidden_size, mixer_config, layer_idx=layer_idx)
    elif mixer_type == "gated_delta_net":
        return Apriel2GatedDeltaNet(hidden_size, mixer_config, layer_idx=layer_idx)
    elif mixer_type == "kimi_linear_attention":
        return KimiLinearAttention(hidden_size, mixer_config, layer_idx=layer_idx)
    elif mixer_type == "stochastic":
        if not allow_stochastic:
            raise ValueError("Stochastic mixers cannot contain nested stochastic mixers")
        return Apriel2StochasticMixer(mixer_config, config, layer_idx)
    else:
        raise ValueError(f"Unknown mixer type: {mixer_type}")


class Apriel2Mamba(nn.Module):
    """Mamba mixer."""

    def __init__(
        self,
        d_model,
        config_dict: dict,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        """Initialize Mamba from a config dictionary."""
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        # Extract parameters from config dict
        d_state = config_dict.get("state_size", 16)
        d_inner = config_dict.get("d_inner")
        d_xb = config_dict.get("d_xb", None)
        d_conv = config_dict.get("d_conv", 4)
        expand = config_dict.get("expand", 2)
        dt_rank = config_dict.get("dt_rank", "auto")
        dt_min = config_dict.get("dt_min", 0.001)
        dt_max = config_dict.get("dt_max", 0.1)
        dt_init = config_dict.get("dt_init", "random")
        dt_scale = config_dict.get("dt_scale", 1.0)
        dt_init_floor = config_dict.get("dt_init_floor", 1e-4)
        repeat_kv_before_conv = config_dict.get("repeat_kv_before_conv", True)
        conv_bias = config_dict["conv_bias"]
        bias = config_dict.get("add_linear_biases", False)
        dt_proj_bias = config_dict["dt_proj_bias"]

        self.d_model = d_model
        self.d_xb = d_xb if d_xb is not None else d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_inner if d_inner is not None else int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = True
        self.layer_idx = layer_idx
        self.repeat_kv_before_conv = repeat_kv_before_conv

        if self.repeat_kv_before_conv:
            self.conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )
        else:
            self.conv1d = nn.Conv1d(
                in_channels=self.d_xb,
                out_channels=self.d_xb,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_xb,
                padding=d_conv - 1,
                **factory_kwargs,
            )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.num_xb_head = self.d_xb // self.d_state
        self.num_C_head = self.d_inner // self.d_state
        self.repeat_group = self.num_C_head // self.num_xb_head

        self.in_proj = nn.Linear(self.d_model, 2 * self.d_xb + 2 * self.d_inner, bias=bias, **factory_kwargs)
        self.dt_in_proj = nn.Linear(self.d_model, self.dt_rank, bias=bias, **factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=dt_proj_bias, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        if self.dt_proj.bias is not None:
            dt = torch.exp(
                torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                self.dt_proj.bias.copy_(inv_dt)
            self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value=None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Forward pass for Mamba."""
        if not is_fast_path_available:
            raise RuntimeError(
                "Mamba requires mamba_ssm library with causal_conv1d and selective_scan kernels. "
                "Install with: pip install mamba-ssm causal-conv1d"
            )
        if "cuda" not in self.in_proj.weight.device.type:
            raise RuntimeError("Mamba only supports CUDA devices. Current device: " + str(self.in_proj.weight.device))

        cache_position = kwargs.get("cache_position", None)
        batch, seqlen, dim = hidden_states.shape

        ssm_state, conv_state = None, None
        use_precomputed_states = False

        seqlen_offset = kwargs.get("seqlen_offset", cache_position[0]) if cache_position is not None else 0
        use_precomputed_states = (
            past_key_value is not None
            and isinstance(past_key_value, Apriel2Cache)
            and past_key_value.conv_states[self.layer_idx] is not None
            and seqlen == 1
            and past_key_value.conv_states[self.layer_idx].shape[0]
            == past_key_value.recurrent_states[self.layer_idx].shape[0]
            == batch
            and cache_position is not None
            and seqlen_offset > 0
        )

        ssm_state, conv_state = self._get_states_from_cache(past_key_value, batch)
        # Adaptive mode selection: use step() for single-token generation
        # This provides significant speedup during autoregressive decoding
        if use_precomputed_states:
            out, _, _ = self.step(hidden_states, conv_state, ssm_state)
            return (out,)

        A = -torch.exp(self.A_log.float())

        zxbc = self.in_proj(hidden_states)
        z, x, B, C = torch.split(
            zxbc,
            [self.d_inner, self.d_xb, self.d_xb, self.d_inner],
            dim=-1,
        )

        x = rearrange(x, "b l d -> b d l")
        z = rearrange(z, "b l d -> b d l")

        B = rearrange(B, "b l (n_group dstate) -> b n_group l dstate", dstate=self.d_state)
        B = repeat_kv(B, self.repeat_group)
        B = rearrange(B, "b n_group l dstate -> b n_group dstate l").contiguous()
        C = rearrange(C, "b l (n_group dstate) -> b n_group dstate l", dstate=self.d_state).contiguous()

        dt = self.dt_proj(self.dt_in_proj(hidden_states))
        dt = rearrange(dt, "b l d -> b d l")

        if self.repeat_kv_before_conv:
            x = rearrange(x, "b (n_group dstate) l -> b n_group l dstate", dstate=self.d_state)
            x = repeat_kv(x, self.repeat_group)
            x = rearrange(x, "b n_group l dstate -> b (n_group dstate) l")

        if conv_state is not None:
            conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))

        # Compute short convolution
        if causal_conv1d_fn is None:
            x = self.act(self.conv1d(x)[..., :seqlen])
        else:
            assert self.activation in ["silu", "swish"]
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
            )

        if not self.repeat_kv_before_conv:
            x = rearrange(x, "b (n_group dstate) l -> b n_group l dstate", dstate=self.d_state)
            x = repeat_kv(x, self.repeat_group)
            x = rearrange(x, "b n_group l dstate -> b (n_group dstate) l")

        y = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=z,
            delta_bias=self.dt_proj.bias.float() if self.dt_proj.bias is not None else None,
            delta_softplus=True,
            return_last_state=(ssm_state is not None),
        )

        if ssm_state is not None:
            y, last_state = y
            ssm_state.copy_(rearrange(last_state, "b (h d) n -> b h d n", h=self.num_C_head))

        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)

        return (out[:, :seqlen, :],)

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"

        hidden_states_input = hidden_states.squeeze(1)

        A = -torch.exp(self.A_log.float())

        zxbc = self.in_proj(hidden_states_input)
        z, x, B, C = torch.split(
            zxbc,
            [self.d_inner, self.d_xb, self.d_xb, self.d_inner],
            dim=-1,
        )

        B = rearrange(B, "b (n_group dstate) -> b n_group dstate", dstate=self.d_state)
        B = torch.repeat_interleave(B, dim=1, repeats=self.repeat_group)
        C = rearrange(C, "b (n_group dstate) -> b n_group dstate", dstate=self.d_state).contiguous()

        dt = self.dt_proj(self.dt_in_proj(hidden_states_input))

        if self.repeat_kv_before_conv:
            x = rearrange(x, "b (n_group dstate) -> b n_group dstate", dstate=self.d_state)
            x = torch.repeat_interleave(x, dim=1, repeats=self.repeat_group)
            x = rearrange(x, "b n_group dstate -> b (n_group dstate)")

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)
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

        if not self.repeat_kv_before_conv:
            x = rearrange(x, "b (n_group dstate) -> b n_group dstate", dstate=self.d_state)
            x = torch.repeat_interleave(x, dim=1, repeats=self.repeat_group)
            x = rearrange(x, "b n_group dstate -> b (n_group dstate)")

        x = rearrange(x, "b (h d) -> b h d", h=self.num_C_head)
        dt = rearrange(dt, "b (h d) -> b h d", h=self.num_C_head)
        A = rearrange(A, "(h d) n -> h d n", h=self.num_C_head)
        D = rearrange(self.D, "(h d) -> h d", h=self.num_C_head)
        z = rearrange(z, "b (h d) -> b h d", h=self.num_C_head)
        dt_bias = rearrange(self.dt_proj.bias, "(h d) -> h d", h=self.num_C_head) if self.dt_proj.bias is not None else None

        # SSM step
        assert selective_state_update is not None
        y = selective_state_update(ssm_state, x, dt, A, B, C, D, z=z, dt_bias=dt_bias, dt_softplus=True)
        y = rearrange(y, "b h d -> b (h d)")
        out = self.out_proj(y)

        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        if self.repeat_kv_before_conv:
            conv_state = torch.zeros(batch_size, self.d_inner, self.d_conv, device=device, dtype=conv_dtype)
        else:
            conv_state = torch.zeros(batch_size, self.d_xb, self.d_conv, device=device, dtype=conv_dtype)
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.num_C_head, self.d_inner // self.num_C_head, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if inference_params is None or not isinstance(inference_params, Apriel2Cache):
            return None, None

        if inference_params.conv_states[self.layer_idx] is None:
            conv_state, ssm_state = self.allocate_inference_cache(batch_size, max_seqlen=0)
            inference_params.conv_states[self.layer_idx] = conv_state
            inference_params.recurrent_states[self.layer_idx] = ssm_state

        ssm_state = inference_params.recurrent_states[self.layer_idx]
        conv_state = inference_params.conv_states[self.layer_idx]

        if initialize_states:
            ssm_state.zero_()
            conv_state.zero_()

        return ssm_state, conv_state


class Apriel2GatedDeltaNet(nn.Module):
    """Wrapper around Qwen3NextGatedDeltaNet to match apriel2 interface."""

    def __init__(
        self,
        d_model,
        config_dict: dict,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        super().__init__()

        # Map config_dict to Qwen3NextConfig format
        config = SimpleNamespace(
            hidden_size=d_model,
            linear_num_value_heads=config_dict.get("num_value_heads", 32),
            linear_num_key_heads=config_dict.get("num_key_heads", 8),
            linear_key_head_dim=config_dict.get("key_head_dim", 64),
            linear_value_head_dim=config_dict.get("value_head_dim", 64),
            linear_conv_kernel_dim=config_dict.get("conv_kernel_size", 4),
            hidden_act=config_dict.get("activation", "silu"),
            rms_norm_eps=config_dict.get("norm_eps", 1e-5),
            dtype=dtype,
        )

        self.gdn = Qwen3NextGatedDeltaNet(config, layer_idx)

    def forward(self, hidden_states: torch.Tensor, past_key_value=None, attention_mask=None, **kwargs):
        cache_position = kwargs.get("cache_position", None)
        output = self.gdn(
            hidden_states, cache_params=past_key_value, cache_position=cache_position, attention_mask=attention_mask
        )
        return (output,)


class KimiLinearAttention(nn.Module):
    """KimiLinearAttention mixer - stub for future implementation."""

    def __init__(
        self,
        d_model,
        config_dict: dict,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        raise NotImplementedError("KimiLinearAttention not yet implemented in apriel2")

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        raise NotImplementedError("KimiLinearAttention not yet implemented in apriel2")


class Apriel2DecoderBlock(nn.Module):
    def __init__(self, config: Apriel2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Get block name and config for this layer
        self.block_name = config.get_block_name(layer_idx)
        block_config = config.get_block_config(layer_idx)

        # Create mixer based on type
        mixer_config = block_config.get("mixer", {"type": "attention"})
        self.mixer = create_mixer(mixer_config, config.hidden_size, layer_idx, config, allow_stochastic=True)

        # Create MLP
        mlp_config = block_config.get("mlp", {"type": "mlp"})
        self.mlp = self._create_mlp(mlp_config, config)

        # Create normalization layers
        norm_config = block_config.get("normalization", {"type": "rms_norm"})
        self.input_layernorm = self._create_norm(norm_config, config)
        self.post_attention_layernorm = self._create_norm(norm_config, config)

    def _create_mlp(self, mlp_config: dict, config: Apriel2Config):
        """Create MLP based on config."""
        mlp_type = mlp_config.get("type", "mlp")

        if mlp_type == "mlp":
            intermediate_size = mlp_config.get("intermediate_size", config.hidden_size * 4)
            mlp_cfg = SimpleNamespace(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=mlp_config.get("activation", "silu"),
            )
            return MistralMLP(mlp_cfg)
        else:
            raise ValueError(f"Unknown MLP type: {mlp_type}")

    def _create_norm(self, norm_config: dict, config: Apriel2Config):
        """Create normalization layer based on config."""
        norm_type = norm_config.get("type", "rms_norm")
        if norm_type == "rms_norm":
            return MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        elif norm_type == "layer_norm":
            return nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            raise ValueError(f"Unknown normalization type: {norm_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Apriel2Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings=None,
        **kwargs,
    ) -> tuple:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        mixer_outputs = self.mixer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = mixer_outputs[0]
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (mixer_outputs[1],) if len(mixer_outputs) > 1 else (None,)
        if use_cache:
            outputs += (mixer_outputs[2] if len(mixer_outputs) > 2 else None,)

        return outputs


class Apriel2StochasticMixer(nn.Module):
    """
    Stochastic mixer that contains multiple mixer options.

    During training: randomly samples one mixer per forward pass
    During inference: uses the main_mixer
    """

    def __init__(self, mixer_config: dict, config: Apriel2Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        # Get sub-mixer configs
        mixers_config = mixer_config.get("mixers", {})
        self.main_mixer_name = mixer_config.get("main_mixer_name", list(mixers_config.keys())[0])

        # Sampling strategy
        self.sampling_strategy = mixer_config.get("sampling_strategy", "uniform")
        sampling_weights = mixer_config.get("sampling_weights", None)

        # Create each sub-mixer
        self.mixers = nn.ModuleDict()
        for name, sub_mixer_config in mixers_config.items():
            self.mixers[name] = create_mixer(
                sub_mixer_config, config.hidden_size, layer_idx, config, allow_stochastic=False
            )

        # Set up sampling probabilities
        mixer_names = list(self.mixers.keys())
        if self.sampling_strategy == "uniform":
            self._sampling_probs = [1.0 / len(self.mixers)] * len(self.mixers)
        elif self.sampling_strategy == "weighted":
            if sampling_weights is None:
                raise ValueError("sampling_weights must be provided when using weighted sampling strategy")
            # Normalize weights to sum to 1.0
            total = sum(sampling_weights.get(name, 1.0) for name in mixer_names)
            self._sampling_probs = [sampling_weights.get(name, 1.0) / total for name in mixer_names]
        else:
            raise ValueError(f"Unknown sampling_strategy: {self.sampling_strategy}")

        self._mixer_names = mixer_names
        logger.info(
            f"Initialized Apriel2StochasticMixer at layer {layer_idx} with {len(self.mixers)} mixers: "
            f"{', '.join(mixer_names)} (main={self.main_mixer_name}, strategy={self.sampling_strategy})"
        )

    def forward(
        self, hidden_states: torch.Tensor, attention_mask=None, position_embeddings: Optional[dict] = None, **kwargs
    ):
        # Sample mixer during training, use main_mixer during inference
        if self.training:
            mixer_name = random.choices(self._mixer_names, weights=self._sampling_probs)[0]
        else:
            mixer_name = self.main_mixer_name

        # Set active mixer in cache for proper state routing
        past_key_value = kwargs.get("past_key_value")
        if past_key_value is not None and hasattr(past_key_value, "set_active_mixer"):
            past_key_value.set_active_mixer(self.layer_idx, mixer_name)

        mixer = self.mixers[mixer_name]
        mixer_position_embeddings = position_embeddings.get(mixer_name) if position_embeddings else None
        mixer_attention_mask = (
            attention_mask.get(mixer_name) if isinstance(attention_mask, dict) else attention_mask
        )
        return mixer(
            hidden_states, attention_mask=mixer_attention_mask, position_embeddings=mixer_position_embeddings, **kwargs
        )


class Apriel2PreTrainedModel(PreTrainedModel):
    config_class = Apriel2Config
    base_model_prefix = "model"
    _no_split_modules = ["Apriel2DecoderBlock"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = False
    _supports_static_cache = False
    _supports_attention_backend = True

    def _prepare_cache_for_generation(
        self, generation_config, model_kwargs, assistant_model, batch_size, max_cache_length, *args
    ):
        if generation_config.use_cache is False:
            return
        model_kwargs["past_key_values"] = Apriel2Cache(config=self.config)

    def _init_weights(self, module):
        std = self.config.initializer_range if hasattr(self.config, "initializer_range") else 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, MistralRMSNorm):
            module.weight.data.fill_(1.0)


class Apriel2Model(Apriel2PreTrainedModel):
    def __init__(self, config: Apriel2Config):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        # Build shared rotary embeddings (one per unique block type)
        self.rotary_embs = nn.ModuleDict()
        self._build_rotary_embs()

        # Decoder blocks
        self.layers = nn.ModuleList(
            [Apriel2DecoderBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        # Final norm
        self.norm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        self.post_init()

    def _create_rotary_emb_for_attention(self, mixer_config: dict):
        from transformers.models.mistral.modeling_mistral import MistralRotaryEmbedding

        head_dim = mixer_config.get("head_size", self.config.hidden_size // mixer_config.get("heads", 32))
        rope_theta = (
            mixer_config.get("rotary", {}).get("theta", 10000.0)
            if isinstance(mixer_config.get("rotary"), dict)
            else 10000.0
        )

        rotary_config = SimpleNamespace(
            max_position_embeddings=self.config.max_position_embeddings,
            rope_theta=rope_theta,
            head_dim=head_dim,
            hidden_size=self.config.hidden_size,
            num_attention_heads=mixer_config.get("heads", 32),
            partial_rotary_factor=1.0,
        )
        return MistralRotaryEmbedding(config=rotary_config)

    def _build_attn_config_for_mask(self, mixer_config: dict):
        """Build attention config for causal mask creation."""
        num_heads = mixer_config.get("heads", 32)
        num_key_value_heads = mixer_config.get("head_groups", num_heads)
        head_dim = mixer_config.get("head_size", self.config.hidden_size // num_heads)

        return SimpleNamespace(
            hidden_size=self.config.hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            max_position_embeddings=self.config.max_position_embeddings,
            sliding_window=mixer_config.get("sliding_window", None),
            _attn_implementation=self.config._attn_implementation,
        )

    def _build_rotary_embs(self):
        """Build rotary embedding instances for all unique attention blocks."""
        decoder_type = self.config.decoder.get("type", "fixed")

        if decoder_type == "fixed":
            block_config = self.config.decoder.get("block", {})
            self._build_rotary_embs_for_block("block", block_config)
        elif decoder_type == "pattern":
            blocks = self.config.decoder.get("blocks", {})
            for block_name, block_config in blocks.items():
                self._build_rotary_embs_for_block(block_name, block_config)
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")

    def _build_rotary_embs_for_block(self, block_name: str, block_config: dict):
        """Build rotary embeddings for a single block and its mixers."""
        mixer_config = block_config.get("mixer", {})
        mixer_type = mixer_config.get("type")

        if mixer_type == "attention":
            self.rotary_embs[block_name] = self._create_rotary_emb_for_attention(mixer_config)
        elif mixer_type == "stochastic":
            mixers = mixer_config.get("mixers", {})
            nested_dict = nn.ModuleDict()
            for mixer_name, sub_mixer_config in mixers.items():
                if sub_mixer_config.get("type") == "attention":
                    nested_dict[mixer_name] = self._create_rotary_emb_for_attention(sub_mixer_config)
            if len(nested_dict) > 0:
                self.rotary_embs[block_name] = nested_dict

    def _create_causal_mask(
        self,
        attn_config,
        input_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.LongTensor,
        past_key_values: Optional[Apriel2Cache],
        cache_position: torch.Tensor,
    ) -> Optional[Union[torch.Tensor, BlockMask]]:
        """Create causal mask for an attention config."""

        mask_function = create_causal_mask if attn_config.sliding_window is None else create_sliding_window_causal_mask
        return mask_function(
            config=attn_config,
            input_embeds=input_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

    def _compute_position_embeddings_and_masks(
        self,
        input_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.LongTensor,
        past_key_values: Optional[Apriel2Cache],
        cache_position: torch.Tensor,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Compute position embeddings and attention masks for all unique attention blocks."""
        position_embeddings = {}
        attention_masks = {}
        decoder_type = self.config.decoder.get("type", "fixed")

        if decoder_type == "fixed":
            block_config = self.config.decoder.get("block", {})
            self._compute_for_block(
                "block",
                block_config,
                input_embeds,
                attention_mask,
                position_ids,
                past_key_values,
                cache_position,
                position_embeddings,
                attention_masks,
            )
        elif decoder_type == "pattern":
            blocks = self.config.decoder.get("blocks", {})
            for block_name, block_config in blocks.items():
                self._compute_for_block(
                    block_name,
                    block_config,
                    input_embeds,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    cache_position,
                    position_embeddings,
                    attention_masks,
                )
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")

        return position_embeddings, attention_masks

    def _compute_for_block(
        self,
        block_name: str,
        block_config: dict,
        input_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.LongTensor,
        past_key_values: Optional[Apriel2Cache],
        cache_position: torch.Tensor,
        position_embeddings: dict[str, Any],
        attention_masks: dict[str, Any],
    ) -> None:
        """Compute position embeddings and attention masks for a block."""
        mixer_config = block_config.get("mixer", {})
        mixer_type = mixer_config.get("type")

        if mixer_type == "attention":
            rotary_emb = self.rotary_embs[block_name]
            cos, sin = rotary_emb(input_embeds, position_ids)
            attn_config = self._build_attn_config_for_mask(mixer_config)
            causal_mask = self._create_causal_mask(
                attn_config, input_embeds, attention_mask, position_ids, past_key_values, cache_position
            )

            position_embeddings[block_name] = (cos, sin)
            attention_masks[block_name] = causal_mask

        elif mixer_type == "stochastic":
            mixers = mixer_config.get("mixers", {})
            nested_pos_embs = {}
            nested_masks = {}

            for mixer_name, sub_mixer_config in mixers.items():
                if sub_mixer_config.get("type") == "attention":
                    rotary_emb = self.rotary_embs[block_name][mixer_name]
                    cos, sin = rotary_emb(input_embeds, position_ids)
                    attn_config = self._build_attn_config_for_mask(sub_mixer_config)
                    causal_mask = self._create_causal_mask(
                        attn_config, input_embeds, attention_mask, position_ids, past_key_values, cache_position
                    )

                    nested_pos_embs[mixer_name] = (cos, sin)
                    nested_masks[mixer_name] = causal_mask

            if nested_pos_embs:
                position_embeddings[block_name] = nested_pos_embs
                attention_masks[block_name] = nested_masks

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Apriel2Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = Apriel2Cache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        position_embeddings, causal_masks = self._compute_position_embeddings_and_masks(
            inputs_embeds, attention_mask, position_ids, past_key_values, cache_position
        )

        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            block_name = self.config.get_block_name(layer_idx)
            layer_position_embeddings = position_embeddings.get(block_name)
            layer_attention_mask = causal_masks.get(block_name)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=layer_attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                position_embeddings=layer_position_embeddings,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if use_cache:
            next_decoder_cache = past_key_values

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attns] if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Apriel2ForCausalLM(Apriel2PreTrainedModel, GenerationMixin):
    """Apriel2 model with a language modeling head."""

    def __init__(self, config: Apriel2Config):
        super().__init__(config)
        self.model = Apriel2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Apriel2Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Forward through model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
