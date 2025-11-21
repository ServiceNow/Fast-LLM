"""
Apriel2 modeling - HuggingFace format that mirrors Fast-LLM's architecture.

This implementation:
- Uses declarative mixer/block hierarchy
- Each mixer type instantiated with its own config
- Supports stochastic mixers natively
- Can represent different attention configs in same stochastic mixer
"""

import math
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
import torch.nn.functional as F
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from mamba_ssm.ops.triton.selective_state_update import selective_state_update
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from torch import nn
from transformers import GenerationMixin, PreTrainedModel
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import logging

from fast_llm_external_models.apriel2.configuration_apriel2 import Apriel2Config

# Import existing components we can reuse
from transformers.models.mistral.modeling_mistral import (
    MistralAttention,
    MistralMLP,
    MistralRMSNorm,
)

logger = logging.get_logger(__name__)

is_fast_path_available = all((selective_state_update, causal_conv1d_fn, causal_conv1d_update))


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
        from types import SimpleNamespace
        from transformers.models.mistral.modeling_mistral import MistralRotaryEmbedding
        import transformers.models.mistral.modeling_mistral as mistral_module

        # Monkey-patch eager_attention_forward to add debug prints (ONCE)
        if not hasattr(mistral_module.eager_attention_forward, '_debug_patched'):
            original_eager_attention = mistral_module.eager_attention_forward
            def debug_eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
                print(f"[ACTUAL eager_attention] query: shape={query.shape}, mean={query.mean().item():.6f}")
                print(f"[ACTUAL eager_attention] key: shape={key.shape}, mean={key.mean().item():.6f}")
                print(f"[ACTUAL eager_attention] value: shape={value.shape}, mean={value.mean().item():.6f}")
                print(f"[ACTUAL eager_attention] attention_mask is not None: {attention_mask is not None}")
                if attention_mask is not None and hasattr(attention_mask, 'shape'):
                    print(f"[ACTUAL eager_attention] attention_mask: shape={attention_mask.shape}, dtype={attention_mask.dtype}")
                    if attention_mask.numel() > 0:
                        print(f"[ACTUAL eager_attention] attention_mask stats: min={attention_mask.min().item()}, max={attention_mask.max().item()}, has large negatives: {(attention_mask < -1e10).any().item()}")
                print(f"[ACTUAL eager_attention] scaling: {scaling}")

                result = original_eager_attention(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
                attn_output, attn_weights = result
                print(f"[ACTUAL eager_attention] attn_output: shape={attn_output.shape}, mean={attn_output.mean().item():.6f}")
                if attn_weights is not None:
                    print(f"[ACTUAL eager_attention] attn_weights: shape={attn_weights.shape}, mean={attn_weights.mean().item():.6f}, max={attn_weights.max().item():.6f}")
                    print(f"[ACTUAL eager_attention] attn_weights sample [0,0,0,:5]: {attn_weights[0,0,0,:5].tolist()}")
                return result

            debug_eager_attention_forward._debug_patched = True
            mistral_module.eager_attention_forward = debug_eager_attention_forward

        # Extract attention parameters from mixer_config
        num_heads = mixer_config.get("heads", 32)
        num_key_value_heads = mixer_config.get("head_groups", num_heads)
        head_dim = mixer_config.get("head_size", d_model // num_heads)
        rope_theta = mixer_config.get("rotary", {}).get("theta", 10000.0) if isinstance(mixer_config.get("rotary"), dict) else 10000.0

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
            _attn_implementation="eager",
        )

        # Create attention sub-module
        self.self_attn = MistralAttention(attn_config, layer_idx)

        # Create rotary embeddings for this attention layer
        # We need to use per-block head_dim, not global config.head_dim
        # Create a config-like object that MistralRotaryEmbedding can use
        rotary_config = SimpleNamespace(
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=rope_theta,
            head_dim=head_dim,
            hidden_size=d_model,
            num_attention_heads=num_heads,
            partial_rotary_factor=1.0,  # Use full rotary, not partial
        )
        self.rotary_emb = MistralRotaryEmbedding(config=rotary_config)
        # Debug: print what inv_freq was computed
        print(f"[Apriel2Attention Init] Created rotary_emb with head_dim={head_dim}, theta={rope_theta}")
        print(f"[Apriel2Attention Init] inv_freq: shape={self.rotary_emb.inv_freq.shape}, mean={self.rotary_emb.inv_freq.mean().item():.6f}")

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.LongTensor] = None, **kwargs):
        print(f"[HF Apriel2Attention.forward] Input: shape={hidden_states.shape}, mean={hidden_states.mean().item():.6f}")

        # Get cache-related parameters
        past_key_values = kwargs.get('past_key_value', None)
        cache_position = kwargs.get('cache_position', None)

        # Compute cache_position if not provided
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + hidden_states.shape[1], device=hidden_states.device
            )

        # Create causal mask (per-block, since sliding_window can differ)
        from transformers.models.mistral.modeling_mistral import create_causal_mask, create_sliding_window_causal_mask
        mask_function = create_causal_mask if self.self_attn.config.sliding_window is None else create_sliding_window_causal_mask
        causal_mask = mask_function(
            config=self.self_attn.config,
            input_embeds=hidden_states,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        print(f"[HF Apriel2Attention.forward] Created causal_mask: {causal_mask is not None}")
        if causal_mask is not None and hasattr(causal_mask, 'shape'):
            print(f"[HF Apriel2Attention.forward] causal_mask: shape={causal_mask.shape}, has large negatives: {(causal_mask < -1e10).any().item() if causal_mask.numel() > 0 else 'N/A'}")

        # Use the causal mask for attention
        attention_mask = causal_mask

        # Compute position_embeddings for this attention layer
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Call self.self_attn - the REAL attention implementation
        print(f"[HF Apriel2Attention.forward] Calling self.self_attn...")
        output = self.self_attn(hidden_states, position_embeddings, attention_mask, **kwargs)
        result = output[0] if isinstance(output, tuple) else output
        print(f"[HF Apriel2Attention.forward] Output: shape={result.shape}, mean={result.mean().item():.6f}, std={result.std().item():.6f}")
        return output


def create_attention_from_config(d_model: int, mixer_config: dict, layer_idx: int, config):
    """
    Smart constructor for attention that respects per-mixer configs.

    Creates an Apriel2Attention instance with parameters from mixer_config.
    """
    return Apriel2Attention(d_model, mixer_config, layer_idx, config)


def create_mixer(mixer_config: dict, hidden_size: int, layer_idx: int, config, allow_stochastic: bool = True):
    """
    Create a mixer from config.

    Args:
        mixer_config: Mixer configuration dict
        hidden_size: Model hidden size
        layer_idx: Layer index
        config: Full model config
        allow_stochastic: Whether to allow stochastic mixers (False for sub-mixers)

    Returns:
        Mixer module instance
    """
    mixer_type = mixer_config.get("type", "attention")

    if mixer_type == "attention":
        return create_attention_from_config(hidden_size, mixer_config, layer_idx, config)
    elif mixer_type == "mamba":
        return Mamba(hidden_size, mixer_config, layer_idx=layer_idx)
    elif mixer_type == "gated_delta_net":
        return GatedDeltaNet(hidden_size, mixer_config, layer_idx=layer_idx)
    elif mixer_type == "kimi_linear_attention":
        return KimiLinearAttention(hidden_size, mixer_config, layer_idx=layer_idx)
    elif mixer_type == "stochastic":
        if not allow_stochastic:
            raise ValueError("Stochastic mixers cannot contain nested stochastic mixers")
        # Import here to avoid circular dependency
        return Apriel2StochasticMixer(mixer_config, config, layer_idx)
    else:
        raise ValueError(f"Unknown mixer type: {mixer_type}")



class Mamba(nn.Module):
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
        assert is_fast_path_available and "cuda" in self.in_proj.weight.device.type, "Only support fast path on cuda"

        batch, seqlen, dim = hidden_states.shape

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
            return_last_state=False,
        )

        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)

        return (out[:, :seqlen, :],)


class GatedDeltaNet(nn.Module):
    """GatedDeltaNet mixer - stub for future implementation."""

    def __init__(
        self,
        d_model,
        config_dict: dict,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        raise NotImplementedError("GatedDeltaNet not yet implemented in apriel2")

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        raise NotImplementedError("GatedDeltaNet not yet implemented in apriel2")


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
    """
    A single decoder block with mixer + MLP + normalization.

    The mixer can be:
    - Attention (various configs)
    - Mamba
    - GatedDeltaNet
    - KimiLinearAttention
    - Stochastic (containing multiple mixers)
    """

    def __init__(self, config: Apriel2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Get block config for this layer
        block_config = config.get_block_config(layer_idx)

        # Create mixer based on type
        mixer_config = block_config.get("mixer", {"type": "attention"})
        self.mixer = self._create_mixer(mixer_config, config, layer_idx)

        # Create MLP
        mlp_config = block_config.get("mlp", {"type": "mlp"})
        self.mlp = self._create_mlp(mlp_config, config)

        # Create normalization layers
        norm_config = block_config.get("normalization", {"type": "rms_norm"})
        self.input_layernorm = self._create_norm(norm_config, config)
        self.post_attention_layernorm = self._create_norm(norm_config, config)

    def _create_mixer(self, mixer_config: dict, config: Apriel2Config, layer_idx: int):
        """Create mixer based on config type."""
        return create_mixer(mixer_config, config.hidden_size, layer_idx, config, allow_stochastic=True)

    def _create_mlp(self, mlp_config: dict, config: Apriel2Config):
        """Create MLP based on config."""
        from types import SimpleNamespace

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
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple:
        print(f"[DecoderBlock {self.layer_idx}] Input: mean={hidden_states.mean().item():.6f}, std={hidden_states.std().item():.6f}")

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        print(f"[DecoderBlock {self.layer_idx}] After input_layernorm: mean={hidden_states.mean().item():.6f}, std={hidden_states.std().item():.6f}")

        # Mixer forward (rotary embeddings handled internally by Apriel2Attention)
        mixer_outputs = self.mixer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = mixer_outputs[0]
        print(f"[DecoderBlock {self.layer_idx}] After mixer: mean={hidden_states.mean().item():.6f}, std={hidden_states.std().item():.6f}")
        hidden_states = residual + hidden_states
        print(f"[DecoderBlock {self.layer_idx}] After mixer residual: mean={hidden_states.mean().item():.6f}, std={hidden_states.std().item():.6f}")

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        print(f"[DecoderBlock {self.layer_idx}] After post_attention_layernorm: mean={hidden_states.mean().item():.6f}, std={hidden_states.std().item():.6f}")
        hidden_states = self.mlp(hidden_states)
        print(f"[DecoderBlock {self.layer_idx}] After MLP: mean={hidden_states.mean().item():.6f}, std={hidden_states.std().item():.6f}")
        hidden_states = residual + hidden_states
        print(f"[DecoderBlock {self.layer_idx}] Block output: mean={hidden_states.mean().item():.6f}, std={hidden_states.std().item():.6f}")

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

        # Create each sub-mixer
        self.mixers = nn.ModuleDict()
        for name, sub_mixer_config in mixers_config.items():
            self.mixers[name] = self._create_sub_mixer(sub_mixer_config, config, layer_idx)

    def _create_sub_mixer(self, sub_mixer_config: dict, config: Apriel2Config, layer_idx: int):
        """Create a sub-mixer for the stochastic mixer."""
        return create_mixer(sub_mixer_config, config.hidden_size, layer_idx, config, allow_stochastic=False)

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        """Forward pass - use main mixer for inference, random for training."""
        # For now, always use main mixer
        # TODO: Add training-time sampling
        mixer = self.mixers[self.main_mixer_name]
        return mixer(hidden_states, **kwargs)


class Apriel2Model(PreTrainedModel):
    """The Apriel2 model - embeddings + decoder blocks + final norm."""

    config_class = Apriel2Config

    def __init__(self, config: Apriel2Config):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        # Decoder blocks
        self.layers = nn.ModuleList(
            [Apriel2DecoderBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        # Final norm
        self.norm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
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

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
            position_ids = cache_position.unsqueeze(0)

        hidden_states = inputs_embeds

        # Decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        print(f"[Apriel2Model] Before final norm: mean={hidden_states.mean().item():.6f}, std={hidden_states.std().item():.6f}")
        hidden_states = self.norm(hidden_states)
        print(f"[Apriel2Model] After final norm: mean={hidden_states.mean().item():.6f}, std={hidden_states.std().item():.6f}")

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Apriel2ForCausalLM(PreTrainedModel):
    """Apriel2 model with a language modeling head."""

    config_class = Apriel2Config

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
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
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
            **kwargs,
        )

        hidden_states = outputs[0]
        print(f"[Apriel2ForCausalLM] Before lm_head: mean={hidden_states.mean().item():.6f}, std={hidden_states.std().item():.6f}")
        print(f"[Apriel2ForCausalLM] lm_head.weight: shape={self.lm_head.weight.shape}, mean={self.lm_head.weight.mean().item():.6f}, std={self.lm_head.weight.std().item():.6f}")
        print(f"[Apriel2ForCausalLM] embed_tokens.weight: shape={self.model.embed_tokens.weight.shape}, mean={self.model.embed_tokens.weight.mean().item():.6f}, std={self.model.embed_tokens.weight.std().item():.6f}")
        print(f"[Apriel2ForCausalLM] lm_head and embed_tokens are same object: {self.lm_head.weight is self.model.embed_tokens.weight}")
        logits = self.lm_head(hidden_states)
        print(f"[Apriel2ForCausalLM] After lm_head (before float()): mean={logits.mean().item():.6f}, std={logits.std().item():.6f}")
        logits = logits.float()
        print(f"[Apriel2ForCausalLM] After float(): mean={logits.mean().item():.6f}, std={logits.std().item():.6f}")

        loss = None
        if labels is not None:
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
