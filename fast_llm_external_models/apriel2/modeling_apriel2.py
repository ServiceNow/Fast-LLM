"""Apriel2 HuggingFace model implementation."""

import math
import random
from types import SimpleNamespace
from typing import Any, Optional, TypedDict, Union

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from transformers import GenerationMixin, PreTrainedModel
from transformers.cache_utils import Cache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import eager_attention_forward
from transformers.models.mistral.modeling_mistral import MistralMLP, MistralRMSNorm, apply_rotary_pos_emb
from transformers.processing_utils import Unpack
from transformers.utils import logging

from .configuration_apriel2 import Apriel2Config, Apriel2TextConfig

# =============================================================================
# Kernel implementation flags (for debugging vLLM vs FLA/mamba_ssm differences)
# =============================================================================
USE_VLLM_CONV = False
USE_VLLM_GDN_OPS = False
USE_VLLM_GATED_NORM = False
USE_VLLM_MAMBA_OPS = False  # Not yet implemented in vLLM wrapper

# Causal conv1d
try:
    if USE_VLLM_CONV:
        from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
    else:
        from causal_conv1d import causal_conv1d_fn
    # causal_conv1d_update always from causal_conv1d (vLLM's has different signature)
    from causal_conv1d import causal_conv1d_update
except ImportError:
    causal_conv1d_fn = None
    causal_conv1d_update = None

# GDN ops (chunk_gated_delta_rule, fused_recurrent_gated_delta_rule)
try:
    if USE_VLLM_GDN_OPS:
        from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    else:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
except ImportError:
    chunk_gated_delta_rule = None
    fused_recurrent_gated_delta_rule = None

# Gated RMSNorm
try:
    if USE_VLLM_GATED_NORM:
        from vllm.model_executor.layers.fla.ops.layernorm_guard import rmsnorm_fn as rms_norm_gated
    else:
        from fla.modules.fused_norm_gate import rms_norm_gated
except ImportError:
    rms_norm_gated = None

# KDA ops
try:
    from fla.ops.kda import chunk_kda, fused_recurrent_kda
    from fla.ops.kda.gate import fused_kda_gate
except ImportError:
    chunk_kda = None
    fused_recurrent_kda = None
    fused_kda_gate = None

# Mamba/SSM ops
try:
    if USE_VLLM_MAMBA_OPS:
        raise ImportError("vLLM mamba ops not yet wrapped")
    else:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_scan_fn = None
    selective_state_update = None

logger = logging.get_logger(__name__)


# =============================================================================
# Cache Classes
# =============================================================================


class _AttentionCache:
    __slots__ = ["key", "value", "window", "cumulative_length"]

    def __init__(self, window=None):
        self.key = None
        self.value = None
        self.window = window
        self.cumulative_length = 0

    def update(self, key, value):
        new_tokens = key.shape[-2]
        self.cumulative_length += new_tokens

        if self.key is None:
            if self.window and key.shape[-2] > self.window:
                self.key = key[..., -self.window :, :].contiguous()
                self.value = value[..., -self.window :, :].contiguous()
            else:
                self.key = key.contiguous()
                self.value = value.contiguous()
        else:
            if self.window:
                self.key = self._window(self.key, key)
                self.value = self._window(self.value, value)
            else:
                self.key = torch.cat([self.key, key], -2)
                self.value = torch.cat([self.value, value], -2)
        return self.key, self.value

    def _window(self, cache, new):
        if cache.shape[-2] == self.window and new.shape[-2] == 1:
            cache = cache.roll(-1, -2)
            cache[..., -1:, :] = new
            return cache
        return torch.cat([cache, new], -2)[..., -self.window :, :].contiguous()

    def reset(self):
        self.key = None
        self.value = None
        self.cumulative_length = 0

    def reorder(self, beam_idx):
        if self.key is not None:
            self.key = self.key.index_select(0, beam_idx.to(self.key.device))
            self.value = self.value.index_select(0, beam_idx.to(self.value.device))

    def crop(self, max_length):
        if self.key is not None:
            self.key = self.key[..., :max_length, :]
            self.value = self.value[..., :max_length, :]
            self.cumulative_length = self.key.shape[-2]

    def batch_repeat(self, repeats):
        if self.key is not None:
            self.key = self.key.repeat_interleave(repeats, dim=0)
            self.value = self.value.repeat_interleave(repeats, dim=0)

    def batch_select(self, indices):
        if self.key is not None:
            self.key = self.key.index_select(0, indices.to(self.key.device))
            self.value = self.value.index_select(0, indices.to(self.value.device))

    @property
    def is_initialized(self):
        return self.key is not None

    @property
    def batch_size(self):
        return self.key.shape[0] if self.key is not None else None


class _SSMCache:
    __slots__ = ["conv", "recurrent"]

    def __init__(self):
        self.conv = None
        self.recurrent = None

    def reset(self):
        self.conv = None
        self.recurrent = None

    def reorder(self, beam_idx):
        if self.conv is not None:
            if isinstance(self.conv, tuple):
                self.conv = tuple(c.index_select(0, beam_idx.to(c.device)) for c in self.conv)
            else:
                self.conv = self.conv.index_select(0, beam_idx.to(self.conv.device))
        if self.recurrent is not None:
            self.recurrent = self.recurrent.index_select(0, beam_idx.to(self.recurrent.device))

    def crop(self, max_length):
        pass  # SSM caches don't have sequence dimension to crop

    def batch_repeat(self, repeats):
        if self.conv is not None:
            if isinstance(self.conv, tuple):
                self.conv = tuple(c.repeat_interleave(repeats, dim=0) for c in self.conv)
            else:
                self.conv = self.conv.repeat_interleave(repeats, dim=0)
        if self.recurrent is not None:
            self.recurrent = self.recurrent.repeat_interleave(repeats, dim=0)

    def batch_select(self, indices):
        if self.conv is not None:
            if isinstance(self.conv, tuple):
                self.conv = tuple(c.index_select(0, indices.to(c.device)) for c in self.conv)
            else:
                self.conv = self.conv.index_select(0, indices.to(self.conv.device))
        if self.recurrent is not None:
            self.recurrent = self.recurrent.index_select(0, indices.to(self.recurrent.device))

    @property
    def is_initialized(self):
        return self.conv is not None

    @property
    def batch_size(self):
        if self.conv is None:
            return None
        if isinstance(self.conv, tuple):
            return self.conv[0].shape[0]
        return self.conv.shape[0]


class _DummyCacheLayer:
    pass


class Apriel2Cache(Cache):

    def __init__(self, config):
        super().__init__(layer_class_to_replicate=_DummyCacheLayer)
        self.config = config
        n = config.decoder["num_blocks"]
        self.layers = []
        self.mixer_types = []
        self.active_mixers = [None] * n

        for i in range(n):
            block = config.get_block_config(i)
            mixer = block.get("mixer", {})
            mtype = mixer.get("type", "attention")

            if mtype == "stochastic":
                sub = {}
                main = mixer.get("main_mixer_name")
                for name, cfg in mixer.get("mixers", {}).items():
                    if cfg.get("type") == "attention":
                        sub[name] = _AttentionCache(cfg.get("window_size"))
                    else:
                        sub[name] = _SSMCache()
                self.layers.append(sub)
                self.mixer_types.append(mixer["mixers"][main].get("type") if main else "attention")
            elif mtype == "attention":
                self.layers.append(_AttentionCache(mixer.get("window_size")))
                self.mixer_types.append("attention")
            else:
                self.layers.append(_SSMCache())
                self.mixer_types.append(mtype)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        layer = self.layers[layer_idx]
        if isinstance(layer, dict):
            mixer = self.active_mixers[layer_idx]
            if mixer is None:
                raise RuntimeError(f"Stochastic layer {layer_idx} needs active_mixer set")
            return layer[mixer].update(key_states, value_states)
        return layer.update(key_states, value_states)

    def set_active_mixer(self, layer_idx, mixer_name):
        self.active_mixers[layer_idx] = mixer_name

    def get_seq_length(self, layer_idx=0):
        """Returns the cumulative sequence length of tokens seen by the cache.

        For sliding window caches, this returns the total tokens seen (not just cached).
        This matches HuggingFace's DynamicSlidingWindowLayer behavior.
        """
        layer = self.layers[layer_idx]
        if isinstance(layer, dict):
            mixer = self.active_mixers[layer_idx]
            if mixer and isinstance(layer[mixer], _AttentionCache):
                return layer[mixer].cumulative_length
            return 0
        if isinstance(layer, _AttentionCache):
            return layer.cumulative_length
        return 0

    def get_max_cache_shape(self, layer_idx=0):
        layer = self.layers[layer_idx]
        if isinstance(layer, dict):
            mixer = self.active_mixers[layer_idx]
            if mixer and isinstance(layer[mixer], _AttentionCache):
                return layer[mixer].window
        elif isinstance(layer, _AttentionCache):
            return layer.window
        return None

    def get_mask_sizes(self, cache_position, layer_idx):
        """Return the length and offset of the cache, used to generate the attention mask.

        For standard (non-sliding) attention:
            kv_offset = 0 (KV[0] corresponds to sequence position 0)
            kv_length = cumulative_length + query_length

        For sliding window attention:
            kv_offset = max(cumulative_length - window + 1, 0)
            kv_length = min(cumulative_length, window - 1) + query_length

        For SSM/linear layers:
            kv_offset = 0, kv_length = query_length (no KV cache to attend to)
        """
        query_length = cache_position.shape[0]
        layer = self.layers[layer_idx]

        # Handle stochastic layers by getting the active mixer's cache
        if isinstance(layer, dict):
            mixer = self.active_mixers[layer_idx]
            if mixer is None:
                # No active mixer set, return defaults
                return query_length, 0
            cache = layer[mixer]
        else:
            cache = layer

        # SSM layers don't have KV cache for attention mask purposes
        if isinstance(cache, _SSMCache):
            return query_length, 0

        # Attention cache - check if sliding window
        if isinstance(cache, _AttentionCache):
            cumulative = cache.cumulative_length
            window = cache.window

            if window is not None:
                # Sliding window attention
                kv_offset = max(cumulative - window + 1, 0)
                if cumulative >= window:
                    kv_length = window - 1 + query_length
                else:
                    kv_length = cumulative + query_length
            else:
                # Full attention
                kv_offset = 0
                kv_length = cumulative + query_length

            return kv_length, kv_offset

        # Fallback
        return query_length, 0

    @property
    def has_previous_state(self):
        return any(isinstance(cache, _SSMCache) and cache.conv is not None for cache in self._iter_caches())

    @property
    def key_cache(self):
        return _LayerListAccessor(self, "key")

    @property
    def value_cache(self):
        return _LayerListAccessor(self, "value")

    @property
    def conv_states(self):
        return _LayerListAccessor(self, "conv")

    @property
    def recurrent_states(self):
        return _LayerListAccessor(self, "recurrent")

    def _iter_caches(self):
        """Iterate over all leaf cache objects (flattening stochastic layer dicts)."""
        for layer in self.layers:
            if isinstance(layer, dict):
                yield from layer.values()
            else:
                yield layer

    def reorder_cache(self, beam_idx):
        for cache in self._iter_caches():
            cache.reorder(beam_idx)

    def reset(self):
        for cache in self._iter_caches():
            cache.reset()

    def crop(self, max_length):
        for cache in self._iter_caches():
            cache.crop(max_length)

    def batch_repeat_interleave(self, repeats):
        for cache in self._iter_caches():
            cache.batch_repeat(repeats)

    def batch_select_indices(self, indices):
        for cache in self._iter_caches():
            cache.batch_select(indices)

    @property
    def is_compileable(self):
        return False

    @property
    def is_initialized(self):
        return any(cache.is_initialized for cache in self._iter_caches())

    @property
    def is_sliding(self):
        result = []
        for layer in self.layers:
            if isinstance(layer, dict):
                has_sliding = any(
                    isinstance(cache, _AttentionCache) and cache.window is not None for cache in layer.values()
                )
                result.append(has_sliding)
            elif isinstance(layer, _AttentionCache):
                result.append(layer.window is not None)
            else:
                result.append(False)
        return result

    @property
    def max_batch_size(self):
        for cache in self._iter_caches():
            bs = cache.batch_size
            if bs is not None:
                return bs
        return None

    @property
    def max_cache_len(self):
        windows = [
            cache.window
            for cache in self._iter_caches()
            if isinstance(cache, _AttentionCache) and cache.window is not None
        ]
        return min(windows) if windows else None

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, idx):
        layer = self.layers[idx]
        if isinstance(layer, dict):
            mixer = self.active_mixers[idx]
            if mixer and isinstance(layer[mixer], _AttentionCache):
                c = layer[mixer]
                if c.key is not None:
                    return c.key, c.value
        elif isinstance(layer, _AttentionCache):
            if layer.key is not None:
                return layer.key, layer.value

        for i, l in enumerate(self.layers):
            if isinstance(l, _AttentionCache) and l.key is not None:
                return torch.empty((0,), device=l.key.device, dtype=l.key.dtype), torch.empty(
                    (0,), device=l.key.device, dtype=l.key.dtype
                )
            elif isinstance(l, dict):
                for c in l.values():
                    if isinstance(c, _AttentionCache) and c.key is not None:
                        return torch.empty((0,), device=c.key.device, dtype=c.key.dtype), torch.empty(
                            (0,), device=c.key.device, dtype=c.key.dtype
                        )
        return torch.empty((0,)), torch.empty((0,))


class _LayerListAccessor:
    __slots__ = ["cache", "attr"]

    def __init__(self, cache, attr):
        self.cache = cache
        self.attr = attr

    def __getitem__(self, idx):
        layer = self.cache.layers[idx]
        if isinstance(layer, dict):
            mixer = self.cache.active_mixers[idx]
            if mixer is None:
                raise RuntimeError(
                    f"Stochastic layer {idx} requires set_active_mixer() to be called before accessing cache. "
                    f"Available mixers: {list(layer.keys())}"
                )
            return getattr(layer[mixer], self.attr)
        return getattr(layer, self.attr, None)

    def __setitem__(self, idx, value):
        layer = self.cache.layers[idx]
        if isinstance(layer, dict):
            mixer = self.cache.active_mixers[idx]
            if mixer is None:
                raise RuntimeError(
                    f"Stochastic layer {idx} requires set_active_mixer() to be called before accessing cache. "
                    f"Available mixers: {list(layer.keys())}"
                )
            setattr(layer[mixer], self.attr, value)
        elif hasattr(layer, self.attr):
            setattr(layer, self.attr, value)


# =============================================================================
# TypedDict Classes
# =============================================================================


class BlockSequenceKwargs(TypedDict, total=False):
    attention_mask: Optional[torch.Tensor]
    position_ids: Optional[torch.LongTensor]
    cache_position: Optional[torch.LongTensor]
    past_key_values: Optional[Apriel2Cache]
    output_attentions: bool
    output_hidden_states: bool
    use_cache: bool


class PreprocessingOutput(TypedDict, total=False):
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]]
    attention_mask: Optional[torch.Tensor]


class CausalConv1d(nn.Conv1d):
    """
    Causal 1D convolution that pads only on the left side.

    Subclasses nn.Conv1d for weight storage/checkpoint compatibility, but overrides
    forward to use proper causal (left-only) padding instead of nn.Conv1d's symmetric padding.

    Supports:
    - Prefill mode: process full sequence, optionally return final state for caching
    - Decode mode: single-token update using cached conv state

    Requires causal_conv1d library for CUDA kernels (no PyTorch fallback).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        activation: str = "silu",
        **kwargs,
    ):
        # Remove padding from kwargs since we handle it ourselves
        kwargs.pop("padding", None)
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=0,  # No built-in padding; we handle it in forward
            **kwargs,
        )
        self._activation = activation

    @property
    def _weight(self) -> torch.Tensor:
        """Weight in [dim, kernel_size] format for causal_conv1d functions."""
        return self.weight.squeeze(1)

    def forward(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor | None = None,
        return_final_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Apply causal convolution.

        Args:
            x: Input tensor [batch, dim, seq_len]
            conv_state: Previous conv state [batch, dim, kernel_size-1] for continuing
                        from cached state. If None, starts fresh.
            return_final_state: If True, return (output, final_state) tuple where
                                final_state can be used for subsequent decode steps.

        Returns:
            If return_final_state is False: output tensor [batch, dim, seq_len]
            If return_final_state is True: (output, final_state) tuple
        """
        batch_size, dim, seq_len = x.shape
        state_len = self.kernel_size[0] - 1

        if USE_VLLM_CONV:
            # vLLM expects x as [dim, total_tokens]
            # x shape: [batch, dim, seq]
            # x_flat[:, t] should equal x[batch_for_t, :, seq_for_t]
            # permute to [dim, batch, seq], then reshape to [dim, batch*seq]
            x_flat = x.permute(1, 0, 2).reshape(dim, batch_size * seq_len).contiguous()

            # Create conv_states buffer: [batch, dim, state_len]
            # vLLM requires stride(1) == 1 (dim dimension contiguous)
            # Create as [batch, state_len, dim] contiguous, then transpose to get right strides
            conv_states = x.new_zeros(batch_size, state_len, dim).transpose(1, 2)

            # Create query_start_loc: cumulative sequence lengths
            # For batch_size sequences each of length seq_len
            query_start_loc = torch.arange(
                0, batch_size * seq_len + 1, seq_len,
                dtype=torch.int32, device=x.device
            )

            # has_initial_state: all False (no prior state)
            has_initial_state = torch.zeros(batch_size, dtype=torch.bool, device=x.device)

            # cache_indices: identity mapping
            cache_indices = torch.arange(batch_size, dtype=torch.int32, device=x.device)

            # Call vLLM's causal_conv1d_fn
            out_flat = causal_conv1d_fn(
                x_flat,
                self._weight,
                self.bias,
                conv_states,
                query_start_loc,
                cache_indices=cache_indices,
                has_initial_state=has_initial_state,
                activation=self._activation,
            )

            # Convert back: [dim, total_tokens] -> [batch, dim, seq]
            # out_flat shape: [dim, batch*seq]
            # reshape to [dim, batch, seq], then permute to [batch, dim, seq]
            out = out_flat.reshape(dim, batch_size, seq_len).permute(1, 0, 2)

            if return_final_state:
                # conv_states was updated in-place by vLLM's implementation
                # Return it in the expected format: [batch, dim, state_len]
                return out, conv_states
            return out

        # FLA/causal_conv1d path below
        # Edge case: seq_len==1 with return_final_state
        # CUDA kernel limitation: return_final_states requires channel-last layout,
        # which is impossible when seq_len==1. Handle via update() with zero-init state.
        if return_final_state and seq_len == 1:
            # Initialize zero state if none provided, with channel-last layout for CUDA kernel
            if conv_state is None:
                # Create channel-last state: stride(1) == 1
                conv_state = x.new_zeros(batch_size, state_len, dim).transpose(1, 2)
            # Use update() which handles single tokens efficiently
            out = causal_conv1d_update(
                x.squeeze(2),  # [batch, dim, 1] -> [batch, dim]
                conv_state,
                self._weight,
                bias=self.bias,
                activation=self._activation,
            )
            return out.unsqueeze(2), conv_state  # [batch, dim, 1], updated state

        # Standard CUDA path
        if return_final_state:
            # causal_conv1d requires channel-last layout for returning final states.
            # Channel-last means: stride(1)==1 AND stride(2)==dim (channels are contiguous).
            # For shape [batch, dim, seq], standard contiguous is (dim*seq, seq, 1).
            # Channel-last is (dim*seq, 1, dim) - achieved via transpose+contiguous+transpose.
            if x.stride(1) != 1 or x.stride(2) < dim:
                x = x.transpose(1, 2).contiguous().transpose(1, 2)
            # Allocate final state buffer with correct memory layout
            # causal_conv1d requires final_states.stride(1) == 1
            final_state = x.new_zeros(batch_size, state_len, dim).transpose(1, 2)
        else:
            final_state = None

        out = causal_conv1d_fn(
            x,
            self._weight,
            bias=self.bias,
            initial_states=conv_state,
            return_final_states=return_final_state,
            final_states_out=final_state,
            activation=self._activation,
        )

        if return_final_state:
            if isinstance(out, tuple):
                out, final_state = out
            # final_state has shape [batch, dim, state_len] with channel-last strides
            # Ensure it's safe for in-place updates by subsequent CUDA kernel calls
            assert final_state is not None
            if final_state.stride(1) == 1:
                # Make a copy that's safe to modify in-place
                final_state = final_state.clone()
            return out, final_state
        return out

    def update(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single-token decode step using cached conv state.

        Args:
            x: Input tensor [batch, dim] (single token)
            conv_state: Conv state [batch, dim, kernel_size-1], will be updated in-place

        Returns:
            Output tensor [batch, dim]
        """
        return causal_conv1d_update(
            x,
            conv_state,
            self._weight,
            bias=self.bias,
            activation=self._activation,
        )


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
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
    return hidden_states


class Apriel2Attention(nn.Module):
    """Multi-headed attention with support for GQA and configurable causality.

    Config options (Fast-LLM naming):
        heads: Number of query heads
        head_groups: Number of key/value heads (for GQA)
        head_size: Dimension per head
        add_linear_biases: Whether to use biases in projections
        causal: Whether to use causal masking
        window_size: Optional sliding window size
        rotary: Rotary embedding config dict
    """

    def __init__(self, d_model: int, mixer_config: dict, layer_idx: int, config):
        super().__init__()
        self.config = config
        self.mixer_config = mixer_config
        self.layer_idx = layer_idx

        # Extract config using Fast-LLM naming
        self.num_heads = mixer_config["heads"]
        self.num_key_value_heads = mixer_config.get("head_groups", self.num_heads)
        self.head_dim = mixer_config["head_size"]
        self.hidden_size = d_model

        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = mixer_config.get("causal", True)
        self.window_size = mixer_config.get("window_size")

        # cross_document_attention: if False, use cu_seqlens to isolate sequences (e.g., images)
        self.cross_document_attention = mixer_config.get("cross_document_attention", True)

        # Bias configuration mirroring Fast-LLM's structure:
        # - add_linear_biases: bool (default for all projections)
        # - query_layer: {"bias": {"enabled": bool}} (per-layer override)
        # - key_layer: {"bias": {"enabled": bool}}
        # - value_layer: {"bias": {"enabled": bool}}
        # - dense_layer: {"bias": {"enabled": bool}}
        default_bias = mixer_config.get("add_linear_biases", False)

        def get_layer_bias(layer_name: str) -> bool:
            layer_cfg = mixer_config.get(layer_name, {})
            bias_cfg = layer_cfg.get("bias", {})
            enabled = bias_cfg.get("enabled")
            return default_bias if enabled is None else enabled

        q_bias = get_layer_bias("query_layer")
        k_bias = get_layer_bias("key_layer")
        v_bias = get_layer_bias("value_layer")
        o_bias = get_layer_bias("dense_layer")

        # Projections
        self.q_proj = nn.Linear(d_model, self.num_heads * self.head_dim, bias=q_bias)
        self.k_proj = nn.Linear(d_model, self.num_key_value_heads * self.head_dim, bias=k_bias)
        self.v_proj = nn.Linear(d_model, self.num_key_value_heads * self.head_dim, bias=v_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, d_model, bias=o_bias)

    @classmethod
    def setup(
        cls,
        mixer_config: dict,
        hidden_size: int,
        max_position_embeddings: int,
    ) -> nn.ModuleDict:
        """
        Setup resources needed by this mixer (rotary embeddings).
        Called once per block type, before instances are created.

        Args:
            mixer_config: Mixer configuration dict
            hidden_size: Model hidden size
            max_position_embeddings: Maximum sequence length

        Returns:
            ModuleDict containing 'rotary_emb'
        """
        rotary_config_dict = mixer_config["rotary"]
        rotary_type = rotary_config_dict["type"]
        rope_theta = rotary_config_dict["theta"]
        num_heads = mixer_config["heads"]
        head_dim = mixer_config["head_size"]

        if rotary_type == "pixtral_2d":
            from transformers.models.pixtral.modeling_pixtral import PixtralRotaryEmbedding

            rotary_config = SimpleNamespace(
                head_dim=head_dim,
                rope_theta=rope_theta,
                image_size=rotary_config_dict["max_image_size"],
                patch_size=rotary_config_dict["patch_size"],
            )
            return nn.ModuleDict({"rotary_emb": PixtralRotaryEmbedding(config=rotary_config)})

        elif rotary_type == "mistral_1d":
            from transformers.models.mistral.modeling_mistral import MistralRotaryEmbedding

            rotary_config = SimpleNamespace(
                max_position_embeddings=max_position_embeddings,
                rope_theta=rope_theta,
                head_dim=head_dim,
                hidden_size=hidden_size,
                num_attention_heads=num_heads,
                partial_rotary_factor=1.0,
            )
            return nn.ModuleDict({"rotary_emb": MistralRotaryEmbedding(config=rotary_config)})

        else:
            raise ValueError(f"Unknown rotary type: {rotary_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple] = None,
        past_key_values: Optional[Any] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Select attention implementation
        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            sliding_window=self.window_size,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    def preprocess(
        self,
        hidden_states: torch.Tensor,
        resources: Optional[nn.ModuleDict],
        **kwargs: Unpack[BlockSequenceKwargs],
    ) -> PreprocessingOutput:
        """
        Compute attention preprocessing: position embeddings and masks.

        Args:
            hidden_states: Current hidden states (for shape/device)
            resources: ModuleDict of resources from setup() (contains 'rotary_emb')
            **kwargs: Metadata including:
                - position_ids: Position IDs for rotary embedding
                - sequence_lengths: [n1, n2, ...] for sequence isolation
                - attention_mask, cache_position, past_key_values, etc.

        Returns:
            PreprocessingOutput with position_embeddings, attention_mask, and flash_attn_kwargs
        """
        position_ids = kwargs.get("position_ids")

        # Compute position embeddings using rotary_emb from resources
        position_embeddings = None
        if resources is not None and "rotary_emb" in resources and position_ids is not None:
            rotary_emb = resources["rotary_emb"]
            cos, sin = rotary_emb(hidden_states, position_ids)
            position_embeddings = (cos, sin)

        # Handle sequence isolation (cross_document_attention=False)
        sequence_lengths = kwargs.get("sequence_lengths")
        flash_attn_kwargs = {}
        mask = kwargs.get("attention_mask")

        if not self.cross_document_attention and sequence_lengths is not None:
            # Compute cu_seqlens for flash attention or block diagonal mask for others
            attn_impl = getattr(self.config, "_attn_implementation", "eager")

            if attn_impl == "flash_attention_2":
                # Flash attention: use cu_seqlens for varlen attention
                cu_seqlens = torch.tensor(
                    [0] + list(torch.cumsum(torch.tensor(sequence_lengths), dim=0).tolist()),
                    dtype=torch.int32,
                    device=hidden_states.device,
                )
                max_seqlen = max(sequence_lengths)
                flash_attn_kwargs = {
                    "cu_seq_lens_q": cu_seqlens,
                    "cu_seq_lens_k": cu_seqlens,
                    "max_length_q": max_seqlen,
                    "max_length_k": max_seqlen,
                }
                mask = None  # Flash varlen doesn't use attention_mask
            else:
                # Non-flash: use block diagonal mask
                mask = _generate_block_attention_mask(sequence_lengths, hidden_states)

        elif self.is_causal and kwargs.get("cache_position") is not None:
            # Causal attention - compute causal mask
            mask_function = create_causal_mask if self.window_size is None else create_sliding_window_causal_mask

            # Build config for mask creation
            mask_config = SimpleNamespace(
                hidden_size=self.config.hidden_size,
                num_attention_heads=self.num_heads,
                num_key_value_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
                max_position_embeddings=self.config.embeddings["max_position_embeddings"],
                sliding_window=self.window_size,
                _attn_implementation=getattr(self.config, "_attn_implementation", "eager"),
            )

            mask = mask_function(
                config=mask_config,
                input_embeds=hidden_states,
                attention_mask=kwargs.get("attention_mask"),
                cache_position=kwargs["cache_position"],
                past_key_values=kwargs.get("past_key_values"),
                position_ids=position_ids,
            )

        # Return computed tensors
        return {
            "position_embeddings": position_embeddings,
            "attention_mask": mask,
            **flash_attn_kwargs,
        }


# Shared helper functions for both text and vision models


def get_mixer_class(mixer_type: str) -> type:
    """Map mixer type string to mixer class."""
    if mixer_type == "attention":
        return Apriel2Attention
    elif mixer_type == "mamba":
        return Apriel2Mamba
    elif mixer_type == "gdn":
        return Apriel2GatedDeltaNet
    elif mixer_type == "kda":
        return KimiDeltaAttention
    elif mixer_type == "stochastic":
        return Apriel2StochasticMixer
    else:
        raise ValueError(f"Unknown mixer type: {mixer_type}")


def create_mixer(mixer_config: dict, hidden_size: int, layer_idx: int, config, allow_stochastic: bool = True):
    """Create a mixer instance from config. Uses get_mixer_class() for type→class mapping."""
    # TODO: make constructor signatures uniform across mixer types and remove this function
    mixer_type = mixer_config.get("type", "attention")
    mixer_class = get_mixer_class(mixer_type)  # Handles unknown types

    # Different mixer types have different constructor signatures
    if mixer_type == "attention":
        return mixer_class(hidden_size, mixer_config, layer_idx, config)
    elif mixer_type == "stochastic":
        if not allow_stochastic:
            raise ValueError("Stochastic mixers cannot contain nested stochastic mixers")
        return mixer_class(mixer_config, config, layer_idx)
    else:
        # mamba, gdn, kda all have same signature
        return mixer_class(hidden_size, mixer_config, layer_idx=layer_idx)


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

        self.activation = "silu"  # Hardcoded for Mamba

        if self.repeat_kv_before_conv:
            self.conv1d = CausalConv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                activation=self.activation,
                **factory_kwargs,
            )
        else:
            self.conv1d = CausalConv1d(
                in_channels=self.d_xb,
                out_channels=self.d_xb,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_xb,
                activation=self.activation,
                **factory_kwargs,
            )

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
        past_key_values=None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Forward pass for Mamba."""
        cache_position = kwargs.get("cache_position", None)
        batch, seqlen, dim = hidden_states.shape

        ssm_state, conv_state = None, None
        use_precomputed_states = False

        seqlen_offset = kwargs.get("seqlen_offset", cache_position[0]) if cache_position is not None else 0
        use_precomputed_states = (
            past_key_values is not None
            and isinstance(past_key_values, Apriel2Cache)
            and past_key_values.conv_states[self.layer_idx] is not None
            and seqlen == 1
            and past_key_values.conv_states[self.layer_idx].shape[0]
            == past_key_values.recurrent_states[self.layer_idx].shape[0]
            == batch
            and cache_position is not None
            and seqlen_offset > 0
        )

        ssm_state, conv_state = self._get_states_from_cache(past_key_values, batch)
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

        # Compute short convolution
        if conv_state is not None:
            # Store padded input for future decode steps (convention: state size = d_conv)
            conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))
        x = self.conv1d(x)

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

    @classmethod
    def setup(
        cls,
        mixer_config: dict,
        hidden_size: int,
        max_position_embeddings: int,
    ) -> nn.ModuleDict:
        """Mamba has no setup resources - returns empty ModuleDict."""
        return nn.ModuleDict()

    def preprocess(
        self,
        hidden_states: torch.Tensor,
        resources: Optional[nn.ModuleDict],
        **kwargs: Unpack[BlockSequenceKwargs],
    ) -> PreprocessingOutput:
        """Mamba has no preprocessing - returns empty dict."""
        return {}

    def step(self, hidden_states, conv_state, ssm_state):
        hidden_states.dtype
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
        x = self.conv1d.update(x, conv_state)

        if not self.repeat_kv_before_conv:
            x = rearrange(x, "b (n_group dstate) -> b n_group dstate", dstate=self.d_state)
            x = torch.repeat_interleave(x, dim=1, repeats=self.repeat_group)
            x = rearrange(x, "b n_group dstate -> b (n_group dstate)")

        x = rearrange(x, "b (h d) -> b h d", h=self.num_C_head)
        dt = rearrange(dt, "b (h d) -> b h d", h=self.num_C_head)
        A = rearrange(A, "(h d) n -> h d n", h=self.num_C_head)
        D = rearrange(self.D, "(h d) -> h d", h=self.num_C_head)
        z = rearrange(z, "b (h d) -> b h d", h=self.num_C_head)
        dt_bias = (
            rearrange(self.dt_proj.bias, "(h d) -> h d", h=self.num_C_head) if self.dt_proj.bias is not None else None
        )

        # SSM step
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


class GatedRMSNormalization(nn.Module):
    """
    Gated RMS normalization layer matching Fast-LLM's implementation.
    Uses fla.modules.fused_norm_gate.rms_norm_gated or vLLM's rmsnorm_fn.

    Args:
        hidden_size: Size of the hidden dimension
        eps: Epsilon for numerical stability
        activation: Gating activation function ("silu" or "sigmoid")
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5, activation: str = "silu"):
        super().__init__()
        if rms_norm_gated is None:
            raise ImportError(
                "GatedRMSNormalization requires rms_norm_gated. "
                "Install fla-core or ensure vLLM is available."
            )
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.activation = activation

    def forward(self, input_: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if USE_VLLM_GATED_NORM:
            # vLLM's rmsnorm_fn signature: (x, weight, bias, z, eps, group_size, norm_before_gate)
            return rms_norm_gated(
                input_,
                self.weight,
                None,  # bias
                z=gate,
                eps=self.eps,
                group_size=None,
                norm_before_gate=True,
            )
        else:
            # FLA's rms_norm_gated signature
            return rms_norm_gated(
                input_,
                gate,
                self.weight,
                None,
                activation=self.activation,
                eps=self.eps,
                residual=None,
                prenorm=False,
                residual_in_fp32=False,
            )


class Apriel2GatedDeltaNet(nn.Module):
    """
    Gated Delta Net implementation matching Fast-LLM's gdn.py exactly.

    Weight names and config parameters match Fast-LLM:
    - in_proj_qkvz, in_proj_ba, convolution, out_proj, dt_bias, A_log, norm
    - value_heads, key_heads, key_head_dim, value_head_dim

    Uses Fast-LLM's flat QKVZ layout: [Q_all | K_all | V_all | Z_all]
    Uses fla.ops.gated_delta_rule.chunk_gated_delta_rule when available.
    """

    def __init__(
        self,
        d_model,
        config_dict: dict,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = d_model

        # Config params - match Fast-LLM naming (value_heads, key_heads, etc.)
        self.activation = config_dict["convolution_layer"].get("activation", "silu")
        self.value_heads = config_dict.get("value_heads", 32)
        self.key_heads = config_dict.get("key_heads", 8)
        self.key_head_dim = config_dict.get("key_head_dim", 64)
        self.value_head_dim = config_dict.get("value_head_dim", 64)
        self.conv_kernel_size = config_dict["convolution_layer"]["kernel_size"]
        self.norm_eps = config_dict.get("norm_eps", 1e-5)

        # Derived dimensions
        self.key_dim = self.key_head_dim * self.key_heads
        self.value_dim = self.value_head_dim * self.value_heads
        self.conv_dim = self.key_dim * 2 + self.value_dim  # Q, K, V (no Z in conv)
        self.qkvz_dim = self.key_dim * 2 + self.value_dim * 2  # Q, K, V, Z
        self.value_heads_per_key = self.value_heads // self.key_heads

        # Projection layers - names match Fast-LLM exactly
        self.in_proj_qkvz = nn.Linear(d_model, self.qkvz_dim, bias=False, device=device, dtype=dtype)
        self.in_proj_ba = nn.Linear(d_model, self.value_heads * 2, bias=False, device=device, dtype=dtype)
        self.out_proj = nn.Linear(self.value_dim, d_model, bias=False, device=device, dtype=dtype)

        # Convolution - named 'convolution' to match Fast-LLM
        self.convolution = CausalConv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            activation=self.activation,
            device=device,
            dtype=dtype,
        )

        # Learnable parameters - match Fast-LLM initialization
        self.dt_bias = nn.Parameter(torch.ones(self.value_heads, device=device, dtype=dtype))
        self.A_log = nn.Parameter(torch.zeros(self.value_heads, device=device, dtype=dtype).uniform_(0, 16).log())

        # Normalization layer - named 'norm' with 'weight' param to match Fast-LLM
        self.norm = GatedRMSNormalization(self.value_head_dim, eps=self.norm_eps)

        # Require FLA kernels - no silent fallback to unoptimized code paths
        if chunk_gated_delta_rule is None or fused_recurrent_gated_delta_rule is None:
            raise ImportError(
                "GatedDeltaNet requires the fla library for optimized kernels. " "Install with: pip install fla-core"
            )

    _debug_enabled = False  # Set to True for debugging
    _debug_layer = False  # num_tokens <= 10
    _debug_state = False  # Debug recurrent state
    _debug_output = False  # Debug output hidden states during decode

    def _debug_tensor(self, name: str, t: torch.Tensor):
        if not self._debug_enabled:
            return
        if t is None:
            print(f"[TF-GDN layer={self.layer_idx}] {name}: None")
            return
        try:
            flat = t.flatten()[:8]
            vals = ", ".join(f"{v:.6f}" for v in flat.float().tolist())
            print(f"[TF-GDN layer={self.layer_idx}] {name}: shape={t.shape}, dtype={t.dtype}, "
                  f"mean={t.float().mean().item():.6f}, std={t.float().std().item():.6f}, "
                  f"first8=[{vals}]")
        except Exception as e:
            print(f"[TF-GDN layer={self.layer_idx}] {name}: ERROR accessing tensor: {e}")

    def _debug_print(self, msg: str):
        if not self._debug_enabled:
            return
        print(f"[TF-GDN layer={self.layer_idx}] {msg}")

    def _debug_state_stats(self, name: str, state: torch.Tensor, seq_len: int):
        """Debug recurrent state with statistics."""
        if not self._debug_state or state is None:
            return
        try:
            flat = state.flatten()
            first8 = ", ".join(f"{v:.6f}" for v in flat[:8].float().tolist())
            print(f"[TF-GDN L{self.layer_idx}] {name} (seq_len={seq_len}): shape={state.shape}, "
                  f"mean={state.float().mean().item():.6f}, std={state.float().std().item():.6f}, "
                  f"min={state.float().min().item():.6f}, max={state.float().max().item():.6f}, "
                  f"first8=[{first8}]")
        except Exception as e:
            print(f"[TF-GDN L{self.layer_idx}] {name}: ERROR accessing state: {e}")

    def _fix_query_key_value_ordering(self, mixed_qkvz: torch.Tensor, mixed_ba: torch.Tensor):
        """
        Split QKVZ and BA tensors using Fast-LLM's flat layout.

        Fast-LLM layout: [Q_all_heads | K_all_heads | V_all_heads | Z_all_heads]
        """
        # Split QKVZ - flat layout matching Fast-LLM
        qkv_sizes = (
            self.key_dim,  # Q: key_heads * key_head_dim
            self.key_dim,  # K: key_heads * key_head_dim
            self.value_dim,  # V: value_heads * value_head_dim
            self.value_dim,  # Z: value_heads * value_head_dim
        )
        query, key, value, z = torch.split(mixed_qkvz, qkv_sizes, dim=-1)

        # Reshape to head format: [batch, seq, heads, head_dim]
        query = query.reshape(*query.shape[:-1], self.key_heads, self.key_head_dim)
        key = key.reshape(*key.shape[:-1], self.key_heads, self.key_head_dim)
        value = value.reshape(*value.shape[:-1], self.value_heads, self.value_head_dim)
        z = z.reshape(*z.shape[:-1], self.value_heads, self.value_head_dim)

        # Split BA - flat layout: [beta_all | alpha_all]
        beta, alpha = torch.split(mixed_ba, (self.value_heads, self.value_heads), dim=-1)

        return query, key, value, z, beta, alpha

    def _ensure_cache_initialized(self, past_key_values, batch_size, device, dtype):
        """Initialize cache if it doesn't exist for this layer."""
        if past_key_values is None:
            return

        if past_key_values.conv_states[self.layer_idx] is None:
            conv_state = torch.zeros(batch_size, self.conv_dim, self.conv_kernel_size, device=device, dtype=dtype)
            past_key_values.conv_states[self.layer_idx] = conv_state

        if past_key_values.recurrent_states[self.layer_idx] is None:
            recurrent_state = torch.zeros(
                batch_size, self.value_heads, self.key_head_dim, self.value_head_dim, device=device, dtype=dtype
            )
            past_key_values.recurrent_states[self.layer_idx] = recurrent_state

    def forward(self, hidden_states: torch.Tensor, past_key_values=None, attention_mask=None, **kwargs):
        cache_position = kwargs.get("cache_position", None)
        batch_size, seq_len, _ = hidden_states.shape

        self._debug_print(f"===== FORWARD START (batch={batch_size}, seq={seq_len}) =====")
        self._debug_tensor("hidden_states", hidden_states)

        # Get conv and recurrent state from cache if available
        conv_state = None
        recurrent_state = None
        if past_key_values is not None:
            conv_state = past_key_values.conv_states[self.layer_idx]
            recurrent_state = past_key_values.recurrent_states[self.layer_idx]

        # Check if using precomputed states (single token decode with cache)
        # Must check that conv_state exists for THIS layer (not just overall has_previous_state)
        use_precomputed_states = (
            past_key_values is not None and conv_state is not None and seq_len == 1 and cache_position is not None
        )
        self._debug_print(f"use_precomputed_states={use_precomputed_states}")

        # Project to QKVZ and BA
        mixed_qkvz = self.in_proj_qkvz(hidden_states)
        mixed_ba = self.in_proj_ba(hidden_states)
        self._debug_tensor("mixed_qkvz", mixed_qkvz)
        self._debug_tensor("mixed_ba", mixed_ba)

        # Split into components using Fast-LLM's flat layout
        query, key, value, z, beta, alpha = self._fix_query_key_value_ordering(mixed_qkvz, mixed_ba)
        self._debug_tensor("query (after split)", query)
        self._debug_tensor("key (after split)", key)
        self._debug_tensor("value (after split)", value)
        self._debug_tensor("z (after split)", z)
        self._debug_tensor("beta (after split)", beta)
        self._debug_tensor("alpha (after split)", alpha)

        # Flatten QKV for convolution (no Z in conv)
        query_flat = query.reshape(batch_size, seq_len, -1)
        key_flat = key.reshape(batch_size, seq_len, -1)
        value_flat = value.reshape(batch_size, seq_len, -1)
        mixed_qkv = torch.cat([query_flat, key_flat, value_flat], dim=-1)
        mixed_qkv = mixed_qkv.transpose(1, 2)  # [batch, conv_dim, seq]
        mixed_qkv_before_conv = mixed_qkv  # Save for debug
        self._debug_tensor("mixed_qkv (before conv)", mixed_qkv)
        self._debug_tensor("conv_weight", self.convolution.weight)
        self._debug_tensor("conv_bias", self.convolution.bias)

        # Apply causal convolution
        if use_precomputed_states:
            # Single token decode - use cached conv state
            self._debug_print("Using conv.update (decode path)")
            mixed_qkv = self.convolution.update(
                mixed_qkv.squeeze(2),  # [batch, conv_dim, 1] -> [batch, conv_dim]
                conv_state,
            ).unsqueeze(
                2
            )  # [batch, conv_dim] -> [batch, conv_dim, 1]
        else:
            # Prefill mode
            self._debug_print("Using conv.forward (prefill path)")
            use_cache = past_key_values is not None
            if use_cache:
                mixed_qkv, final_state = self.convolution(mixed_qkv, return_final_state=True)
                past_key_values.conv_states[self.layer_idx] = final_state
            else:
                mixed_qkv = self.convolution(mixed_qkv)

        mixed_qkv = mixed_qkv.transpose(1, 2)  # [batch, seq, conv_dim]
        self._debug_tensor("mixed_qkv (after conv)", mixed_qkv)

        # Split back after convolution
        query_flat, key_flat, value_flat = torch.split(mixed_qkv, (self.key_dim, self.key_dim, self.value_dim), dim=-1)
        query = query_flat.reshape(batch_size, seq_len, self.key_heads, self.key_head_dim)
        key = key_flat.reshape(batch_size, seq_len, self.key_heads, self.key_head_dim)
        value = value_flat.reshape(batch_size, seq_len, self.value_heads, self.value_head_dim)
        self._debug_tensor("query (after conv)", query)
        self._debug_tensor("key (after conv)", key)
        self._debug_tensor("value (after conv)", value)

        # Compute gating - match Fast-LLM exactly
        beta_gate = beta.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(alpha.float() + self.dt_bias)
        self._debug_tensor("beta_gate", beta_gate)
        self._debug_tensor("g", g)
        self._debug_tensor("A_log", self.A_log)
        self._debug_tensor("dt_bias", self.dt_bias)

        # Expand K heads to V heads if grouped query attention
        if self.value_heads_per_key > 1:
            query = query.repeat_interleave(self.value_heads_per_key, dim=2)
            key = key.repeat_interleave(self.value_heads_per_key, dim=2)
            self._debug_print(f"Expanded q/k heads: {self.key_heads} -> {self.value_heads}")
            self._debug_tensor("query (after expand)", query)
            self._debug_tensor("key (after expand)", key)

        # Run gated delta rule (FLA kernels required)
        self._debug_tensor("recurrent_state (initial)", recurrent_state)
        if not use_precomputed_states:
            # Chunked mode for prefill
            self._debug_print("Using chunk_gated_delta_rule (prefill)")
            # Debug PREFILL INPUTS before kernel call
            if self._debug_state:
                print(f"[TF-GDN L{self.layer_idx}] PREFILL INPUTS:")
                print(f"  hidden_states: shape={hidden_states.shape}, first8={hidden_states.flatten()[:8].tolist()}")
                print(f"  mixed_qkv_before_conv: shape={mixed_qkv_before_conv.shape}, first8={mixed_qkv_before_conv.flatten()[:8].tolist()}")
                print(f"  q: shape={query.shape}, first8={query.flatten()[:8].tolist()}")
                print(f"  k: shape={key.shape}, first8={key.flatten()[:8].tolist()}")
                print(f"  v: shape={value.shape}, first8={value.flatten()[:8].tolist()}")
                print(f"  g: shape={g.shape}, first8={g.flatten()[:8].tolist()}")
                print(f"  beta: shape={beta_gate.shape}, first8={beta_gate.flatten()[:8].tolist()}")
                print(f"  initial_state: {recurrent_state}")
            output, last_recurrent_state = chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta_gate,
                initial_state=recurrent_state,
                output_final_state=past_key_values is not None,
                use_qk_l2norm_in_kernel=True,
            )
            # Ensure state is in same dtype as hidden_states (fla kernel may return float32)
            if last_recurrent_state is not None:
                last_recurrent_state = last_recurrent_state.to(hidden_states.dtype)
            self._debug_state_stats("PREFILL out_state", last_recurrent_state, seq_len)
        else:
            # Recurrent mode for single token decode
            self._debug_print("Using fused_recurrent_gated_delta_rule (decode)")
            self._debug_state_stats("DECODE in_state", recurrent_state, seq_len)
            # Debug decode inputs
            if self._debug_state:
                print(f"[TF-GDN L{self.layer_idx}] DECODE inputs: q={query.flatten()[:4].tolist()}, k={key.flatten()[:4].tolist()}, v={value.flatten()[:4].tolist()}, g={g.flatten()[:4].tolist()}, beta={beta_gate.flatten()[:4].tolist()}")
            # vLLM and FLA have different signatures:
            # - vLLM: inplace_final_state (default True, set False to avoid ssm_state_indices requirement)
            # - FLA: output_final_state
            if USE_VLLM_GDN_OPS:
                output, last_recurrent_state = fused_recurrent_gated_delta_rule(
                    query,
                    key,
                    value,
                    g=g,
                    beta=beta_gate,
                    initial_state=recurrent_state,
                    inplace_final_state=False,
                    use_qk_l2norm_in_kernel=True,
                )
            else:
                output, last_recurrent_state = fused_recurrent_gated_delta_rule(
                    query,
                    key,
                    value,
                    g=g,
                    beta=beta_gate,
                    initial_state=recurrent_state,
                    output_final_state=past_key_values is not None,
                    use_qk_l2norm_in_kernel=True,
                )
            self._debug_state_stats("DECODE out_state", last_recurrent_state, seq_len)

        self._debug_tensor("output (after FLA)", output)

        # Update recurrent state in cache
        if past_key_values is not None:
            past_key_values.recurrent_states[self.layer_idx] = last_recurrent_state

        # Apply gated normalization
        z_shape_og = z.shape
        output = output.reshape(-1, output.shape[-1])
        z_flat = z.reshape(-1, z.shape[-1])
        self._debug_tensor("output (before norm)", output)
        self._debug_tensor("z_flat (for norm)", z_flat)
        # Debug last token before norm (reshaped has tokens * heads rows)
        batch_size, num_tokens = hidden_states.shape[:2]
        if self._debug_layer and num_tokens > 0:
            num_heads = self.value_heads
            last_token_start = (num_tokens - 1) * num_heads
            last_out = output[last_token_start:last_token_start+1, :8]
            last_z = z_flat[last_token_start:last_token_start+1, :8]
            print(f"[TF-GDN layer={self.layer_idx}] output before norm (last token, head 0): [{', '.join(f'{v:.6f}' for v in last_out.flatten().float().tolist())}]")
            print(f"[TF-GDN layer={self.layer_idx}] z before norm (last token, head 0): [{', '.join(f'{v:.6f}' for v in last_z.flatten().float().tolist())}]")
        self._debug_tensor("norm.weight", self.norm.weight)
        self._debug_print(f"norm.eps={self.norm.eps}, norm.activation={self.norm.activation}")
        output = self.norm(output, z_flat)
        self._debug_tensor("output (after norm)", output)
        # Debug last token after norm
        if self._debug_layer and num_tokens > 0:
            last_out_after = output[last_token_start:last_token_start+1, :8]
            print(f"[TF-GDN layer={self.layer_idx}] output after norm (last token, head 0): [{', '.join(f'{v:.6f}' for v in last_out_after.flatten().float().tolist())}]")
        output = output.reshape(z_shape_og)
        output = output.reshape(output.shape[0], output.shape[1], -1)

        # Output projection
        output = self.out_proj(output)
        self._debug_tensor("output (final)", output)
        # Show last token specifically
        if self._debug_layer and output.dim() == 3:
            last_token = output[0, -1, :8]
            vals = ", ".join(f"{v:.6f}" for v in last_token.float().tolist())
            print(f"[TF-GDN layer={self.layer_idx}] output (last token): last_token_first8=[{vals}]")
        # Debug output hidden states during decode
        # Get decode step from cache
        decode_step = past_key_values.get_seq_length() if past_key_values is not None else 0
        if self._debug_output and use_precomputed_states and output.dim() == 3:
            flat = output.flatten()
            first8 = ", ".join(f"{v:.6f}" for v in flat[:8].float().tolist())
            print(f"[TF-GDN L{self.layer_idx}] STEP={decode_step} OUTPUT hs: mean={output.float().mean().item():.6f}, std={output.float().std().item():.6f}, first8=[{first8}]")
        self._debug_print("===== FORWARD END =====")

        return (output,)

    @classmethod
    def setup(
        cls,
        mixer_config: dict,
        hidden_size: int,
        max_position_embeddings: int,
    ) -> nn.ModuleDict:
        """GatedDeltaNet has no setup resources - returns empty ModuleDict."""
        return nn.ModuleDict()

    def preprocess(
        self,
        hidden_states: torch.Tensor,
        resources: Optional[nn.ModuleDict],
        **kwargs: Unpack[BlockSequenceKwargs],
    ) -> PreprocessingOutput:
        """GatedDeltaNet has no preprocessing - returns empty dict."""
        return {}


class KimiDeltaAttention(nn.Module):
    """
    Kimi Delta Attention (KDA) implementation matching Fast-LLM's kda.py.

    Weight names match Fast-LLM:
    - q_proj, k_proj, v_proj, o_proj - main projections
    - f_a_proj, f_b_proj - gate kernel (low-rank)
    - g_a_proj, g_b_proj - output gate (low-rank)
    - beta_proj - beta gating
    - q_conv, k_conv, v_conv - CausalConv1d modules
    - A_log, dt_bias - learnable parameters
    - norm - gated RMS normalization

    Uses fla.ops.kda.chunk_kda and fused_recurrent_kda kernels.
    Uses CausalConv1d for convolutions (requires causal_conv1d CUDA kernels).
    """

    def __init__(
        self,
        d_model,
        config_dict: dict,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        super().__init__()

        if chunk_kda is None or fused_kda_gate is None:
            raise ImportError(
                "KimiDeltaAttention requires the `fla` package. " "Please install it with `pip install -U fla-core`."
            )

        self.layer_idx = layer_idx
        self.hidden_size = d_model
        self.mode = "chunk"

        # Config params - match Fast-LLM naming
        self.num_heads = config_dict.get("heads", 32)
        self.head_dim = config_dict.get("head_dim", 64)
        conv_config = config_dict.get("convolution_layer", {})
        self.conv_kernel_size = conv_config.get("kernel_size", 4)
        norm_config = config_dict.get("normalization", {})
        self.norm_eps = norm_config.get("epsilon", 1e-5)
        self.norm_activation = norm_config.get(
            "activation", "silu"
        )  # default to silu to be consistent with Fast-LLM's default. Note, Kimi uses sigmoid.

        # Derived dimensions
        self.projection_size = self.head_dim * self.num_heads

        # Projection layers - names match Fast-LLM exactly
        self.q_proj = nn.Linear(d_model, self.projection_size, bias=False, device=device, dtype=dtype)
        self.k_proj = nn.Linear(d_model, self.projection_size, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(d_model, self.projection_size, bias=False, device=device, dtype=dtype)

        # Convolutions - use CausalConv1d for proper left-only padding
        # Named to match Fast-LLM (q_conv, k_conv, v_conv)
        self.q_conv = CausalConv1d(
            in_channels=self.projection_size,
            out_channels=self.projection_size,
            kernel_size=self.conv_kernel_size,
            groups=self.projection_size,  # depthwise
            bias=False,
            activation="silu",
            device=device,
            dtype=dtype,
        )
        self.k_conv = CausalConv1d(
            in_channels=self.projection_size,
            out_channels=self.projection_size,
            kernel_size=self.conv_kernel_size,
            groups=self.projection_size,
            bias=False,
            activation="silu",
            device=device,
            dtype=dtype,
        )
        self.v_conv = CausalConv1d(
            in_channels=self.projection_size,
            out_channels=self.projection_size,
            kernel_size=self.conv_kernel_size,
            groups=self.projection_size,
            bias=False,
            activation="silu",
            device=device,
            dtype=dtype,
        )

        # Gate kernel projections (low-rank: hidden -> head_dim -> projection)
        self.f_a_proj = nn.Linear(d_model, self.head_dim, bias=False, device=device, dtype=dtype)
        self.f_b_proj = nn.Linear(self.head_dim, self.projection_size, bias=False, device=device, dtype=dtype)

        # Output gate projections (low-rank)
        self.g_a_proj = nn.Linear(d_model, self.head_dim, bias=False, device=device, dtype=dtype)
        self.g_b_proj = nn.Linear(self.head_dim, self.projection_size, bias=False, device=device, dtype=dtype)

        # Beta projection - named beta_proj to match Fast-LLM (not b_proj)
        self.beta_proj = nn.Linear(d_model, self.num_heads, bias=False, device=device, dtype=dtype)

        # Output projection
        self.o_proj = nn.Linear(self.projection_size, d_model, bias=False, device=device, dtype=dtype)

        # Learnable parameters - match Fast-LLM shapes
        # A_log: 1D shape (num_heads,) to match Fast-LLM
        self.A_log = nn.Parameter(
            torch.zeros(self.num_heads, device=device, dtype=torch.float32).uniform_(1, 16).log()
        )
        self.dt_bias = nn.Parameter(torch.ones(self.projection_size, device=device, dtype=torch.float32))

        # Normalization - use GatedRMSNormalization (same wrapper as GDN, with sigmoid activation)
        self.norm = GatedRMSNormalization(self.head_dim, eps=self.norm_eps, activation=self.norm_activation)

    def _apply_conv(self, x: torch.Tensor, conv: CausalConv1d, conv_state: torch.Tensor | None, use_cache: bool):
        """
        Apply causal convolution with cache support.

        Args:
            x: Input tensor [batch, seq, dim]
            conv: CausalConv1d module
            conv_state: Previous conv state [batch, dim, kernel_size-1] or None
            use_cache: Whether to output final state for caching

        Returns:
            (output, new_conv_state) tuple
        """
        seq_len = x.shape[1]
        x = x.transpose(1, 2)  # [batch, dim, seq]

        # Single token decode with existing cache
        if conv_state is not None and seq_len == 1:
            out = conv.update(x.squeeze(2), conv_state)
            return out.unsqueeze(1), conv_state  # [batch, 1, dim]

        # Prefill mode
        if use_cache:
            out, final_state = conv(x, conv_state=conv_state, return_final_state=True)
        else:
            out = conv(x, conv_state=conv_state)
            final_state = None

        return out.transpose(1, 2), final_state  # [batch, seq, dim]

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values=None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        batch_size, seq_len, _ = hidden_states.shape
        mode = "fused_recurrent" if (seq_len <= 64 and not self.training) else self.mode

        # Get cache states if available
        conv_state_q, conv_state_k, conv_state_v = None, None, None
        recurrent_state = None
        use_cache = past_key_values is not None

        if past_key_values is not None:
            conv_states = past_key_values.conv_states[self.layer_idx]
            if conv_states is not None:
                conv_state_q, conv_state_k, conv_state_v = conv_states
            recurrent_state = past_key_values.recurrent_states[self.layer_idx]

        # Project Q, K, V and apply convolutions
        q, conv_state_q = self._apply_conv(self.q_proj(hidden_states), self.q_conv, conv_state_q, use_cache)
        k, conv_state_k = self._apply_conv(self.k_proj(hidden_states), self.k_conv, conv_state_k, use_cache)
        v, conv_state_v = self._apply_conv(self.v_proj(hidden_states), self.v_conv, conv_state_v, use_cache)

        # Gate kernel computation (raw g, gate applied inside kernel for chunk mode)
        g = self.f_b_proj(self.f_a_proj(hidden_states))
        g = rearrange(g, "... (h d) -> ... h d", d=self.head_dim)

        # Beta gating
        beta = self.beta_proj(hidden_states).float().sigmoid()

        # Reshape Q, K, V to head format
        q, k = map(lambda x: rearrange(x, "... (h d) -> ... h d", d=self.head_dim), (q, k))
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_dim)

        # Run KDA kernel
        if mode == "chunk":
            # For chunk mode: gate computed inside kernel (matches FLA reference)
            o, recurrent_state = chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                initial_state=recurrent_state,
                output_final_state=past_key_values is not None,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
            )
        else:
            # For fused_recurrent mode: pre-compute gate (matches FLA reference)
            g = fused_kda_gate(g, self.A_log.float(), dt_bias=self.dt_bias)
            o, recurrent_state = fused_recurrent_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )

        # Update cache
        if past_key_values is not None:
            past_key_values.recurrent_states[self.layer_idx] = recurrent_state
            past_key_values.conv_states[self.layer_idx] = (conv_state_q, conv_state_k, conv_state_v)

        # Output gating and normalization
        g_out = self.g_b_proj(self.g_a_proj(hidden_states))
        g_out = rearrange(g_out, "... (h d) -> ... h d", d=self.head_dim)

        # Flatten for normalization, then reshape back
        o_shape = o.shape
        o = self.norm(o.reshape(-1, o.shape[-1]), g_out.reshape(-1, g_out.shape[-1]))
        o = o.reshape(o_shape)

        # Reshape and project output
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)

        return (o,)

    @classmethod
    def setup(
        cls,
        mixer_config: dict,
        hidden_size: int,
        max_position_embeddings: int,
    ) -> nn.ModuleDict:
        """KimiDeltaAttention has no setup resources - returns empty ModuleDict."""
        return nn.ModuleDict()

    def preprocess(
        self,
        hidden_states: torch.Tensor,
        resources: Optional[nn.ModuleDict],
        **kwargs: Unpack[BlockSequenceKwargs],
    ) -> PreprocessingOutput:
        """KimiDeltaAttention has no preprocessing - returns empty dict."""
        return {}


class Apriel2BlockSequence(nn.Module):
    """
    Block sequence abstraction - mirrors Fast-LLM's BlockSequence.
    Used by both text decoder and vision encoder.

    Architecture:
    - Pure container for blocks (handles fixed/pattern types)
    - Delegates resource setup to mixers via mixer.setup()
    - Owns mixer_resources (ModuleDict from setup, deduplicated by block_name)
    - Delegates preprocessing to mixers via mixer.preprocess()
    - Caches preprocessing per unique block type (efficient)
    - Completely agnostic to mixer types (attention, mamba, etc.)

    Setup + Preprocessing pattern:
    1. Call mixer.setup() for each unique block type → collect resources (rotary_emb, etc.)
    2. Call mixer.preprocess() for each unique block type → compute tensors
    3. Cache preprocessing results indexed by block_name
    4. Reuse cached preprocessing for blocks of same type
    5. Merge preprocessing outputs into block kwargs
    """

    def __init__(
        self,
        sequence_config: dict,
        hidden_size: int,
        max_position_embeddings: int,
        config: Apriel2TextConfig,
    ):
        super().__init__()
        self.sequence_config = sequence_config
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.config = config

        # Build blocks (handles fixed/pattern)
        # NOTE: _build_blocks() calls classmethod setup() to create mixer_resources BEFORE instances
        self.blocks = self._build_blocks()

        # Extract unique mixer instances (one per unique block_name) for preprocessing
        self.unique_mixers: dict[str, nn.Module] = {}
        for layer_idx, block in enumerate(self.blocks):
            block_name = self.get_block_name(layer_idx)
            if block_name not in self.unique_mixers:
                self.unique_mixers[block_name] = block.mixer

    def _build_blocks(self) -> nn.ModuleList:
        """
        Build blocks based on fixed/pattern type.

        Phase 1: Setup resources (called once per block type, before instances)
        Phase 2: Create block instances (resources already available)
        """
        seq_type = self.sequence_config.get("type", "fixed")
        num_blocks = self.sequence_config.get("num_blocks")

        # PHASE 1: Setup resources BEFORE creating instances
        # Initialize mixer_resources container
        self.mixer_resources = nn.ModuleDict()

        # Extract unique block types and call setup for each
        if seq_type == "fixed":
            # Fixed: single block type repeated
            block_config = self.sequence_config.get("block", {})
            mixer_config = block_config.get("mixer", {})
            mixer_type = mixer_config.get("type", "attention")

            # Call classmethod setup
            mixer_class = get_mixer_class(mixer_type)
            resources = mixer_class.setup(mixer_config, self.hidden_size, self.max_position_embeddings)
            if len(resources) > 0:
                self.mixer_resources["block"] = resources

        elif seq_type == "pattern":
            # Pattern: multiple block types in repeating pattern
            blocks_config = self.sequence_config.get("blocks", {})
            for block_name, block_config in blocks_config.items():
                mixer_config = block_config.get("mixer", {})
                mixer_type = mixer_config.get("type", "attention")

                # Call classmethod setup
                mixer_class = get_mixer_class(mixer_type)
                resources = mixer_class.setup(mixer_config, self.hidden_size, self.max_position_embeddings)
                if len(resources) > 0:
                    self.mixer_resources[block_name] = resources
        else:
            raise ValueError(f"Unknown sequence type: {seq_type}")

        # PHASE 2: Create block instances (resources already set up)
        # Extract rms_norm_eps from config head.normalization.epsilon
        rms_norm_eps = self.config.head["normalization"]["epsilon"]

        blocks = []
        for layer_idx in range(num_blocks):
            # Get block_config for this layer
            if seq_type == "fixed":
                block_config = self.sequence_config.get("block", {})
            elif seq_type == "pattern":
                pattern = self.sequence_config.get("pattern", [])
                blocks_config = self.sequence_config.get("blocks", {})
                block_name = pattern[layer_idx % len(pattern)]
                block_config = blocks_config[block_name]
            else:
                raise ValueError(f"Unknown sequence type: {seq_type}")

            # Create block with explicit parameters (no fake config creation!)
            blocks.append(
                Apriel2Block(
                    block_config=block_config,
                    hidden_size=self.hidden_size,
                    layer_idx=layer_idx,
                    rms_norm_eps=rms_norm_eps,
                    config=self.config,
                )
            )

        return nn.ModuleList(blocks)

    def get_block_name(self, layer_idx: int) -> str:
        """Get block name for a specific layer (shared logic)."""
        seq_type = self.sequence_config.get("type", "fixed")
        if seq_type == "fixed":
            return "block"
        elif seq_type == "pattern":
            pattern = self.sequence_config.get("pattern", [])
            return pattern[layer_idx % len(pattern)]
        else:
            raise ValueError(f"Unknown sequence type: {seq_type}")

    def preprocess(
        self,
        hidden_states: torch.Tensor,
        **kwargs: Unpack[BlockSequenceKwargs],
    ) -> dict[str, PreprocessingOutput]:
        """
        Compute preprocessing for all unique block types.
        Aggregates preprocessing from all unique mixers.

        Args:
            hidden_states: Current hidden states (for shape/device)
            **kwargs: Metadata (position_ids, attention_mask, cache_position, etc.)

        Returns:
            Preprocessing cache keyed by block_name
        """
        preprocessing_cache: dict[str, PreprocessingOutput] = {}
        for block_name, mixer in self.unique_mixers.items():
            # Get resources for this block type (from setup)
            # Note: nn.ModuleDict doesn't have .get(), so we check membership first
            resources = self.mixer_resources[block_name] if block_name in self.mixer_resources else None

            # Mixer computes preprocessing using resources (read-only)
            # Returns PreprocessingOutput (position_embeddings, attention_mask, etc.)
            preprocessing_cache[block_name] = mixer.preprocess(hidden_states, resources, **kwargs)

        return preprocessing_cache

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs: Unpack[BlockSequenceKwargs],
    ) -> tuple[torch.Tensor, Optional[tuple], Optional[tuple]]:
        """
        Forward pass through block sequence.

        Args:
            hidden_states: Input tensor (data)
            **kwargs: Metadata (attention_mask, position_ids, etc.)

        Returns:
            (hidden_states, all_hidden_states, all_attentions)
        """
        # Compute preprocessing ONCE per unique block type
        # Delegates to self.preprocess() which aggregates from all mixers
        preprocessing_cache = self.preprocess(hidden_states, **kwargs)

        # Initialize output collections
        all_hidden_states = () if kwargs.get("output_hidden_states") else None
        all_attentions = () if kwargs.get("output_attentions") else None

        # Iterate through blocks - REUSE cached preprocessing
        for layer_idx, block in enumerate(self.blocks):
            # Collect intermediate hidden state if requested
            if all_hidden_states is not None:
                all_hidden_states += (hidden_states,)

            # Get preprocessing for this block type (reused for blocks of same type)
            block_name = self.get_block_name(layer_idx)
            preprocessing_kwargs = preprocessing_cache[block_name]

            # Merge input kwargs with preprocessing outputs
            # Preprocessing can override (e.g., causal mask overrides attention_mask)
            block_kwargs = {**kwargs, **preprocessing_kwargs}

            # Pipe through: y = f(x, **kwargs)
            # Block extracts what it needs from kwargs
            layer_outputs = block(hidden_states, **block_kwargs)
            hidden_states = layer_outputs[0]

            # Collect attention if requested
            if all_attentions is not None:
                all_attentions += (layer_outputs[1] if len(layer_outputs) > 1 else None,)

        return hidden_states, all_hidden_states, all_attentions


class Apriel2Block(nn.Module):
    """
    Transformer block with mixer (attention/mamba/etc) and MLP.
    Used for both text decoder and vision encoder.
    """

    def __init__(
        self,
        block_config: dict,
        hidden_size: int,
        layer_idx: int,
        rms_norm_eps: float,
        config: Apriel2TextConfig,
    ):
        """
        Args:
            block_config: Dict with 'mixer', 'mlp', 'normalization' configs
            hidden_size: Model hidden size
            layer_idx: Layer index in the sequence
            rms_norm_eps: Epsilon for RMS normalization
            config: Model config (passed to mixers that need it)
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx

        # Create mixer based on type
        mixer_config = block_config.get("mixer", {"type": "attention"})
        self.mixer = create_mixer(mixer_config, hidden_size, layer_idx, config, allow_stochastic=True)

        # Create MLP
        mlp_config = block_config.get("mlp", {"type": "mlp"})
        self.mlp = self._create_mlp(mlp_config, hidden_size)

        # Create normalization layers
        norm_config = block_config.get("normalization", {"type": "rms_norm"})
        self.input_layernorm = self._create_norm(norm_config, hidden_size, rms_norm_eps)
        self.post_attention_layernorm = self._create_norm(norm_config, hidden_size, rms_norm_eps)

    def _create_mlp(self, mlp_config: dict, hidden_size: int):
        """Create MLP based on config.

        Supports per-layer bias configuration mirroring Fast-LLM:
        - add_linear_biases: default bias setting for all layers
        - layer_1.bias.enabled: override for up_proj/gate_proj
        - layer_2.bias.enabled: override for down_proj
        """
        mlp_type = mlp_config.get("type", "mlp")

        if mlp_type == "mlp":
            intermediate_size = mlp_config["intermediate_size"]
            activation = mlp_config.get("activation", "silu")
            gated = mlp_config.get("gated", False)

            # Per-layer bias configuration (mirrors Fast-LLM structure)
            default_bias = mlp_config.get("add_linear_biases", False)

            def get_layer_bias(layer_name: str) -> bool:
                layer_cfg = mlp_config.get(layer_name, {})
                bias_cfg = layer_cfg.get("bias", {})
                enabled = bias_cfg.get("enabled")
                return default_bias if enabled is None else enabled

            layer_1_bias = get_layer_bias("layer_1")
            layer_2_bias = get_layer_bias("layer_2")

            if gated:
                # MistralMLP uses gate_proj, up_proj, down_proj (all bias controlled together)
                # For now, we use the default bias setting for gated MLPs
                # TODO: Add per-layer bias support to gated MLP
                mlp_cfg = SimpleNamespace(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    hidden_act=activation,
                )
                return MistralMLP(mlp_cfg)
            else:
                return SimpleMLP(
                    hidden_size,
                    intermediate_size,
                    activation,
                    layer_1_bias=layer_1_bias,
                    layer_2_bias=layer_2_bias,
                )
        else:
            raise ValueError(f"Unknown MLP type: {mlp_type}")

    def _create_norm(self, norm_config: dict, hidden_size: int, rms_norm_eps: float):
        """Create normalization layer based on config."""
        norm_type = norm_config.get("type", "rms_norm")
        if norm_type == "rms_norm":
            return MistralRMSNorm(hidden_size, eps=rms_norm_eps)
        elif norm_type == "layer_norm":
            return nn.LayerNorm(hidden_size, eps=rms_norm_eps)
        else:
            raise ValueError(f"Unknown normalization type: {norm_type}")

    _debug_layer = False  # Set to True to debug layer outputs

    def _debug_tensor(self, name: str, t: torch.Tensor, show_last=False):
        if not self._debug_layer or t is None:
            return
        if show_last:
            # Show last token
            last = t[0, -1, :8]
            vals = ", ".join(f"{v:.6f}" for v in last.float().tolist())
            print(f"[TF Layer {self.layer_idx}] {name}: shape={t.shape}, last_token_first8=[{vals}]")
        else:
            flat = t.flatten()[:8]
            vals = ", ".join(f"{v:.6f}" for v in flat.float().tolist())
            print(f"[TF Layer {self.layer_idx}] {name}: shape={t.shape}, first8=[{vals}]")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Apriel2Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings=None,
        **kwargs,
    ) -> tuple:
        num_tokens = hidden_states.size(1)
        self._debug_layer = False  # Disabled for testing

        self._debug_tensor("input hidden_states", hidden_states)

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        self._debug_tensor("after input_layernorm", hidden_states)

        mixer_outputs = self.mixer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = mixer_outputs[0]
        self._debug_tensor("mixer output", hidden_states)

        hidden_states = residual + hidden_states
        self._debug_tensor("after residual add 1", hidden_states)

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        self._debug_tensor("after post_attention_layernorm", hidden_states)

        hidden_states = self.mlp(hidden_states)
        self._debug_tensor("after mlp", hidden_states)

        hidden_states = residual + hidden_states
        self._debug_tensor("after residual add 2 (final)", hidden_states)
        # Also show last token for final layer comparison
        self._debug_tensor("after residual add 2 (last token)", hidden_states, show_last=True)

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

    def __init__(self, mixer_config: dict, config: Apriel2TextConfig, layer_idx: int):
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
        past_key_values = kwargs.get("past_key_values")
        if past_key_values is not None and hasattr(past_key_values, "set_active_mixer"):
            past_key_values.set_active_mixer(self.layer_idx, mixer_name)

        mixer = self.mixers[mixer_name]
        mixer_position_embeddings = position_embeddings.get(mixer_name) if position_embeddings else None
        mixer_attention_mask = attention_mask.get(mixer_name) if isinstance(attention_mask, dict) else attention_mask
        return mixer(
            hidden_states, attention_mask=mixer_attention_mask, position_embeddings=mixer_position_embeddings, **kwargs
        )

    @classmethod
    def setup(
        cls,
        mixer_config: dict,
        hidden_size: int,
        max_position_embeddings: int,
    ) -> nn.ModuleDict:
        """
        Setup resources for stochastic mixer with nested mixers.
        Called before instance creation, recursively calls setup on nested mixer classes.

        Returns a ModuleDict where each key is a nested mixer name and value is its setup ModuleDict.
        """
        nested_resources = nn.ModuleDict()

        # Get nested mixers config
        mixers_config = mixer_config.get("mixers", {})

        for mixer_name, sub_mixer_config in mixers_config.items():
            # Get mixer class from type
            mixer_type = sub_mixer_config.get("type", "attention")
            mixer_class = get_mixer_class(mixer_type)

            # Call setup on nested mixer class
            mixer_resources = mixer_class.setup(sub_mixer_config, hidden_size, max_position_embeddings)
            if len(mixer_resources) > 0:
                nested_resources[mixer_name] = mixer_resources

        return nested_resources

    def preprocess(
        self,
        hidden_states: torch.Tensor,
        resources: Optional[nn.ModuleDict],
        **kwargs: Unpack[BlockSequenceKwargs],
    ) -> PreprocessingOutput:
        """
        Preprocess for stochastic mixer with nested mixers.

        Returns a PreprocessingOutput where position_embeddings and attention_mask
        are dicts mapping nested mixer names to their respective values.
        """
        nested_position_embeddings = {}
        nested_attention_masks = {}

        for mixer_name, nested_mixer in self.mixers.items():
            # Get resources for this nested mixer (if resources is a ModuleDict of ModuleDicts)
            # Note: nn.ModuleDict doesn't have .get(), so we check membership first
            nested_resources = resources[mixer_name] if resources is not None and mixer_name in resources else None

            # Get preprocessing for nested mixer
            nested_output = nested_mixer.preprocess(hidden_states, nested_resources, **kwargs)
            # Extract position_embeddings (may be None for some mixer types)
            if nested_output.get("position_embeddings") is not None:
                nested_position_embeddings[mixer_name] = nested_output["position_embeddings"]
            # Extract attention_mask (may be None for SDPA, or float for eager)
            # We include it even if None to override the original long int mask
            if "attention_mask" in nested_output:
                nested_attention_masks[mixer_name] = nested_output["attention_mask"]

        # Return PreprocessingOutput with nested position_embeddings and attention_mask dicts
        return PreprocessingOutput(
            position_embeddings=nested_position_embeddings if nested_position_embeddings else None,
            attention_mask=nested_attention_masks if nested_attention_masks else None,
        )


class Apriel2PreTrainedModel(PreTrainedModel):
    config_class = Apriel2TextConfig
    base_model_prefix = "model"
    _no_split_modules = ["Apriel2Block"]
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


class Apriel2TextModel(Apriel2PreTrainedModel):
    """Apriel2 text-only base model (without LM head)."""

    def __init__(self, config: Apriel2TextConfig):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        # Decoder block sequence (uses shared BlockSequence abstraction)
        # Causal behavior determined by mixer config (attention mixers have causal=True by default)
        self.decoder = Apriel2BlockSequence(
            sequence_config=config.decoder,
            hidden_size=config.hidden_size,
            max_position_embeddings=config.embeddings["max_position_embeddings"],
            config=config,
        )

        # Final norm (epsilon from head.normalization config)
        self.norm = MistralRMSNorm(config.hidden_size, eps=config.head["normalization"]["epsilon"])

        self.gradient_checkpointing = False
        self.post_init()

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

        # Forward through decoder block sequence (handles position embeddings, masks, and iteration)
        hidden_states, all_hidden_states, all_self_attns = self.decoder(
            inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            cache_position=cache_position,
            **flash_attn_kwargs,
        )

        # Apply final normalization
        # Debug final norm
        batch_size, seq_len = hidden_states.shape[:2]
        _debug_final = False  # seq_len <= 10
        if _debug_final:
            # Show LAST token (to match vLLM)
            last_token = hidden_states[0, -1, :8]
            vals = ", ".join(f"{v:.6f}" for v in last_token.float().tolist())
            print(f"[TF Final] hidden_states (before norm): shape={hidden_states.shape}, last_token_first8=[{vals}]")
            print(f"[TF Final] norm.weight: first8=[{', '.join(f'{v:.6f}' for v in self.norm.weight.flatten()[:8].float().tolist())}]")
            print(f"[TF Final] norm.variance_epsilon={self.norm.variance_epsilon}")

        hidden_states = self.norm(hidden_states)

        if _debug_final:
            last_token = hidden_states[0, -1, :8]
            vals = ", ".join(f"{v:.6f}" for v in last_token.float().tolist())
            print(f"[TF Final] hidden_states (after norm): shape={hidden_states.shape}, last_token_first8=[{vals}]")

        # Add final hidden state if requested
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_decoder_cache = past_key_values if use_cache else None

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
    """Apriel2 model with a language modeling head (text-only)."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Apriel2TextConfig):
        super().__init__(config)
        self.model = Apriel2TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        # post_init() calls init_weights() which calls tie_weights() if config.tie_word_embeddings
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

        # Debug LM head input
        batch_size, seq_len = hidden_states.shape[:2]
        _debug_lm_head = False  # seq_len <= 10
        if _debug_lm_head:
            # Show LAST token's first 8 features (to match vLLM which only passes last token)
            last_token_hs = hidden_states[0, -1, :8]
            vals = ", ".join(f"{v:.6f}" for v in last_token_hs.float().tolist())
            print(f"[TF LM Head] input hidden_states: shape={hidden_states.shape}, last_token_first8=[{vals}]")
            print(f"[TF LM Head] lm_head.weight: shape={self.lm_head.weight.shape}, first8=[{', '.join(f'{v:.6f}' for v in self.lm_head.weight.flatten()[:8].float().tolist())}]")

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        if _debug_lm_head:
            # Get last token logits
            last_logits = logits[0, -1]
            top_vals, top_idx = last_logits.topk(5)
            print(f"[TF LM Head] logits shape={logits.shape}")
            print(f"[TF LM Head] last token top-5 logits: {[(idx.item(), val.item()) for idx, val in zip(top_idx, top_vals)]}")

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


class Apriel2Embeddings(nn.Module):
    """Converts images to patch embeddings via 2D convolution."""

    def __init__(self, vision_hidden_size: int, embeddings_config: dict):
        super().__init__()

        # Extract parameters from config dict
        patch_height = embeddings_config.get("patch_height", 16)
        patch_width = embeddings_config.get("patch_width", 16)
        input_channels = embeddings_config.get("input_channels", 3)  # RGB

        # 2D convolution to create patch embeddings (internally named patch_embeddings to match Fast-LLM)
        self.patch_embeddings = nn.Conv2d(
            in_channels=input_channels,
            out_channels=vision_hidden_size,
            kernel_size=(patch_height, patch_width),
            stride=(patch_height, patch_width),
            bias=False,
        )

        # Normalization layer
        norm_config = embeddings_config.get("normalization", {"type": "layer_norm"})
        norm_type = norm_config.get("type", "layer_norm")
        norm_eps = norm_config.get("eps", 1e-5)

        if norm_type == "layer_norm":
            self.normalization = nn.LayerNorm(vision_hidden_size, eps=norm_eps)
        elif norm_type == "rms_norm":
            self.normalization = MistralRMSNorm(vision_hidden_size, eps=norm_eps)
        else:
            raise ValueError(f"Unknown normalization type: {norm_type}")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [batch, channels, height, width]
        Returns:
            patch_embeddings: [batch, num_patches, hidden_size]
        """
        # Apply convolution: [batch, channels, height, width] -> [batch, hidden, num_patches_h, num_patches_w]
        x = self.patch_embeddings(pixel_values)

        # Flatten spatial dimensions: [batch, hidden, num_patches_h, num_patches_w] -> [batch, hidden, num_patches]
        batch_size, hidden_size, h, w = x.shape
        x = x.view(batch_size, hidden_size, h * w)

        # Transpose to sequence format: [batch, hidden, num_patches] -> [batch, num_patches, hidden]
        # NOTE: .contiguous() is required to match Pixtral's numerical behavior.
        # Pixtral concatenates patches before normalization, which makes the tensor contiguous.
        # Without this, RMSNorm produces slightly different results (~4.7e-7) due to
        # floating-point computation order differences on non-contiguous tensors.
        x = x.transpose(1, 2).contiguous()

        # Apply normalization
        x = self.normalization(x)

        return x


def _generate_block_attention_mask(
    patch_counts: list[int],
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Generate block diagonal attention mask to isolate images.

    Like Pixtral's generate_block_attention_mask: each image can only attend
    to its own patches, preventing cross-image attention.

    Args:
        patch_counts: List of patch counts per image [n1, n2, ...]
        hidden_states: Hidden states tensor for dtype/device [1, total_patches, hidden]

    Returns:
        attention_mask: [1, 1, total_patches, total_patches] with 0 for allowed, -inf for blocked
    """
    dtype = hidden_states.dtype
    device = hidden_states.device
    seq_len = hidden_states.shape[1]
    d_min = torch.finfo(dtype).min

    # Start with all blocked
    mask = torch.full((seq_len, seq_len), fill_value=d_min, dtype=dtype, device=device)

    # Unblock each image's diagonal block
    block_end_idx = torch.tensor(patch_counts, device=device).cumsum(-1)
    block_start_idx = torch.cat([torch.tensor([0], device=device), block_end_idx[:-1]])

    for start, end in zip(block_start_idx, block_end_idx):
        mask[start:end, start:end] = 0

    return mask[None, None, :, :]


def _compute_2d_position_ids(
    patch_embeds_list: list[torch.Tensor],
    max_patches_per_side: int,
    patch_size: int,
) -> torch.Tensor:
    """Compute 2D position IDs for concatenated patches.

    Like Pixtral's position_ids_in_meshgrid: computes position_id = h * max_width + w
    for each patch, then concatenates across all images.

    Args:
        patch_embeds_list: List of patch embeddings [patches_i, hidden] per image
        max_patches_per_side: Maximum patches per side for position encoding
        patch_size: Size of each patch

    Returns:
        position_ids: [total_patches] tensor of position IDs
    """
    positions = []
    for patch_embed in patch_embeds_list:
        # Infer grid dimensions from number of patches
        # This assumes patches are flattened from a grid
        num_patches = patch_embed.shape[0]

        # For now, assume square grid or use the stored dimensions
        # We'll get actual h, w from the caller
        height = width = int(num_patches**0.5)
        if height * width != num_patches:
            # Non-square: will be handled by caller passing dimensions
            height = width = int(num_patches**0.5)

        mesh = torch.meshgrid(
            torch.arange(height, device=patch_embed.device),
            torch.arange(width, device=patch_embed.device),
            indexing="ij",
        )
        h_grid, w_grid = torch.stack(mesh, dim=-1).reshape(-1, 2).chunk(2, -1)
        ids = h_grid * max_patches_per_side + w_grid
        positions.append(ids[:, 0])

    return torch.cat(positions)


class Apriel2VisionEncoder(nn.Module):
    """Vision encoder with embeddings, transformer blocks, and adapter.

    Uses Pixtral-style processing: concatenates all image patches into one sequence.
    Computes position_ids for 2D rotary embeddings and sequence_lengths for image
    isolation - these are passed to encoder blocks. Mixer-specific handling (rotary
    cos/sin, cu_seqlens) is delegated to each mixer's preprocess() method.
    """

    def __init__(self, vision_encoder_config: dict, text_config: Apriel2Config):
        super().__init__()

        self.hidden_size = vision_encoder_config["hidden_size"]

        # Build embeddings layer
        embeddings_config = vision_encoder_config["embeddings"]
        self.embeddings = Apriel2Embeddings(self.hidden_size, embeddings_config)

        # Store patch size for computing patch grid dimensions
        self.patch_size = embeddings_config["patch_height"]

        # Get max_image_size for 2D position encoding (vision encoder owns this)
        # Priority: encoder-level config > rotary config in any attention block > default
        self.max_image_size = self._get_max_image_size(vision_encoder_config)
        self.max_patches_per_side = self.max_image_size // self.patch_size

        # Build vision transformer encoder using shared BlockSequence abstraction
        encoder_config = vision_encoder_config.get("encoder", {})

        # Get norm epsilon from text config's head.normalization.epsilon
        norm_epsilon = text_config.head["normalization"]["epsilon"]

        # Create a minimal config for vision blocks (hierarchical structure)
        vision_block_config = Apriel2TextConfig(
            hidden_size=self.hidden_size,
            embeddings={"max_position_embeddings": 1024},  # Large enough for typical vision use cases
            head={"normalization": {"type": "rms_norm", "epsilon": norm_epsilon}},
            _attn_implementation=getattr(text_config, "_attn_implementation", "eager"),
        )

        # Vision encoder block sequence - supports any mixer type
        self.encoder = Apriel2BlockSequence(
            sequence_config=encoder_config,
            hidden_size=self.hidden_size,
            max_position_embeddings=1024,
            config=vision_block_config,
        )

        # Build adapter/projector
        adapter_config = vision_encoder_config.get("adapter", {})
        self.adapter = self._build_adapter(adapter_config, text_config.hidden_size)

    def _build_adapter(self, adapter_config: dict, text_hidden_size: int) -> nn.Module:
        """Build adapter/projector from config dict."""
        adapter_type = adapter_config.get("type", "mlp")

        if adapter_type == "mlp":
            # 2-layer MLP projector (mirrors Fast-LLM's adapter)
            intermediate_size = adapter_config.get("intermediate_size", text_hidden_size)
            activation = adapter_config.get("activation", "gelu")

            return Apriel2MultiModalProjector(
                vision_hidden_size=self.hidden_size,
                text_hidden_size=text_hidden_size,
                intermediate_size=intermediate_size,
                activation=activation,
            )
        else:
            raise ValueError(f"Unknown adapter type: {adapter_type}")

    def _get_max_image_size(self, config: dict) -> int:
        """Extract max_image_size from config with fallback chain.

        This is a vision encoder concern - determines 2D position encoding grid size.

        Priority:
        1. Encoder-level config: config["max_image_size"]
        2. From any attention block's rotary config (for backward compatibility)
        3. Default: 4096 (supports up to ~292x292 patches with patch_size=14)
        """
        # Priority 1: encoder-level config
        if "max_image_size" in config:
            return config["max_image_size"]

        # Priority 2: search through blocks for rotary config
        encoder_config = config.get("encoder", {})
        for block_config in self._iter_block_configs(encoder_config):
            mixer_config = block_config.get("mixer", {})
            rotary_config = mixer_config.get("rotary", {})
            if "max_image_size" in rotary_config:
                return rotary_config["max_image_size"]

        # Default fallback
        return 4096

    def _iter_block_configs(self, encoder_config: dict):
        """Iterate over all block configs in encoder (handles fixed/pattern types)."""
        seq_type = encoder_config.get("type", "fixed")

        if seq_type == "fixed":
            block_config = encoder_config.get("block", {})
            if block_config:
                yield block_config
        elif seq_type == "pattern":
            blocks_config = encoder_config.get("blocks", {})
            for block_config in blocks_config.values():
                yield block_config

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Process images through vision encoder using Pixtral-style concatenation.

        All image patches are concatenated into ONE sequence. Vision encoder computes:
        - position_ids: 2D position encoding (row * max_patches_per_side + col)
        - sequence_lengths: patches per image (for image isolation)

        These are passed to encoder blocks. Mixer-specific handling (rotary cos/sin,
        cu_seqlens/masks) is delegated to each mixer's preprocess() method.

        Args:
            pixel_values: [batch, channels, height, width] - batch of images

        Returns:
            image_features: [batch, num_patches, text_hidden_size]
        """
        batch_size = pixel_values.shape[0]
        _, _, img_height, img_width = pixel_values.shape
        height_patches = img_height // self.patch_size
        width_patches = img_width // self.patch_size
        num_patches_per_image = height_patches * width_patches

        # Process each image through embeddings independently, then concatenate
        # This mirrors Pixtral's approach of processing conv independently
        patch_embeds_list = []
        for i in range(batch_size):
            # [1, channels, H, W] -> [1, num_patches, hidden]
            embed = self.embeddings(pixel_values[i : i + 1])
            # [num_patches, hidden]
            patch_embeds_list.append(embed.squeeze(0))

        # Concatenate all patches into one sequence: [1, total_patches, hidden]
        hidden_states = torch.cat(patch_embeds_list, dim=0).unsqueeze(0)

        # Compute position_ids for 2D rotary: position_id = row * max_patches_per_side + col
        # Vision encoder owns 2D position encoding - attention just uses position_ids
        positions = []
        for _ in range(batch_size):
            mesh = torch.meshgrid(
                torch.arange(height_patches, device=hidden_states.device),
                torch.arange(width_patches, device=hidden_states.device),
                indexing="ij",
            )
            h_grid, w_grid = torch.stack(mesh, dim=-1).reshape(-1, 2).chunk(2, -1)
            ids = h_grid * self.max_patches_per_side + w_grid
            positions.append(ids[:, 0])
        position_ids = torch.cat(positions).unsqueeze(0)  # [1, total_patches]

        # Sequence lengths: patches per image (for image isolation in attention)
        sequence_lengths = [num_patches_per_image] * batch_size

        # Forward through vision encoder block sequence
        hidden_states, _, _ = self.encoder(
            hidden_states,
            attention_mask=None,  # Attention computes masks from sequence_lengths if needed
            position_ids=position_ids,
            sequence_lengths=sequence_lengths,
            past_key_values=None,
            output_attentions=False,
            output_hidden_states=False,
            use_cache=False,
            cache_position=None,
        )

        # Adapter/projector: [1, total_patches, vision_hidden] -> [1, total_patches, text_hidden]
        image_features = self.adapter(hidden_states)

        # Reshape back to [batch, num_patches, text_hidden]
        image_features = image_features.squeeze(0).view(batch_size, num_patches_per_image, -1)

        return image_features


class SimpleMLP(nn.Module):
    """Non-gated MLP: up_proj -> activation -> down_proj.

    Supports per-layer bias configuration mirroring Fast-LLM:
    - layer_1_bias: bias for up_proj (layer_1 in Fast-LLM naming)
    - layer_2_bias: bias for down_proj (layer_2 in Fast-LLM naming)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "silu",
        layer_1_bias: bool = False,
        layer_2_bias: bool = False,
    ):
        super().__init__()
        from transformers.activations import ACT2FN

        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=layer_1_bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=layer_2_bias)
        self.act_fn = ACT2FN[activation]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.up_proj(x)))


class Apriel2MultiModalProjector(nn.Module):
    """Projects vision features to text embedding space (2-layer MLP)."""

    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        intermediate_size: Optional[int] = None,
        activation: str = "gelu",
    ):
        super().__init__()
        from transformers.activations import ACT2FN

        if intermediate_size is None:
            intermediate_size = text_hidden_size

        self.linear_1 = nn.Linear(vision_hidden_size, intermediate_size, bias=True)
        self.act = ACT2FN[activation]
        self.linear_2 = nn.Linear(intermediate_size, text_hidden_size, bias=True)

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class Apriel2Model(Apriel2TextModel):
    """
    Apriel2 multimodal base model (vision + text, without LM head).

    Inherits from Apriel2TextModel (which provides embed_tokens, decoder, norm)
    and adds vision_encoder. This mirrors Fast-LLM's VisionMultiModalModel(LanguageModel)
    inheritance pattern for trivial weight conversion.
    """

    config_class = Apriel2Config

    def __init__(self, config: Apriel2Config):
        super().__init__(config)

        # Add vision encoder (text components inherited from Apriel2TextModel)
        if config.vision_encoder is not None:
            self.vision_encoder = Apriel2VisionEncoder(config.vision_encoder, config)
        else:
            self.vision_encoder = None

        # Re-run post_init to handle any vision encoder initialization
        self.post_init()

    def get_image_features(self, pixel_values, image_sizes=None):
        """Extract and project image features.

        Args:
            pixel_values: [num_images, channels, height, width] - batch of images (possibly padded)
            image_sizes: Optional[num_images, 2] - actual (height, width) of each image for cropping

        Returns:
            image_features: [num_images, num_patches, hidden_size] or concatenated features
        """
        if self.vision_encoder is None:
            raise ValueError("Cannot extract image features: vision_encoder is None")

        if image_sizes is None:
            # No cropping needed - process as batch
            return self.vision_encoder(pixel_values)

        # Get patch size from embeddings layer to determine minimum valid image size
        patch_height = self.vision_encoder.embeddings.patch_embeddings.kernel_size[0]
        patch_width = self.vision_encoder.embeddings.patch_embeddings.kernel_size[1]

        # Process each image individually with its actual size
        all_features = []
        for i, (image, (height, width)) in enumerate(zip(pixel_values, image_sizes)):
            height, width = int(height), int(width)
            # Skip images that are too small to produce any patches
            if height < patch_height or width < patch_width:
                continue
            # Crop to actual image size
            cropped = image[:, :height, :width]
            # Process single image - add batch dim
            features = self.vision_encoder(cropped.unsqueeze(0))
            # Remove batch dim and add to list
            all_features.append(features.squeeze(0))

        if not all_features:
            # No valid images - return empty tensor
            return torch.zeros(0, 0, self.config.hidden_size, device=pixel_values.device)

        # Concatenate all features along patch dimension
        return torch.cat(all_features, dim=0).unsqueeze(0)  # [1, total_patches, hidden]

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Apriel2Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[tuple, BaseModelOutputWithPast]:
        # If pixel_values provided, we need to merge vision and text embeddings
        if pixel_values is not None and input_ids is not None:
            # Encode and project images (with optional cropping based on image_sizes)
            image_features = self.get_image_features(pixel_values, image_sizes)

            # Get text embeddings (use inherited embed_tokens)
            inputs_embeds = self.embed_tokens(input_ids)

            # Merge image features into text embeddings
            image_token_index = self.config.image_token_index

            # Create mask for image token positions: [batch, seq_len]
            special_image_mask = input_ids == image_token_index

            # Validate token count matches feature count
            num_image_tokens = special_image_mask.sum().item()
            num_image_features = image_features.shape[0] * image_features.shape[1]

            if num_image_tokens != num_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: "
                    f"got {num_image_tokens} image tokens but {num_image_features} image features "
                    f"(shape: {image_features.shape})"
                )

            # Expand mask to match embedding dimension: [batch, seq_len, hidden_size]
            special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds)

            # Flatten image features to match the number of True values in mask
            image_features = image_features.view(-1, image_features.shape[-1])

            # Use masked_scatter for efficient vectorized merge
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

            # Clear input_ids since we're using inputs_embeds
            input_ids = None

        # Forward through inherited text model components
        return super().forward(
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


class Apriel2ForConditionalGeneration(Apriel2PreTrainedModel, GenerationMixin):
    """
    Apriel2 multimodal model with language modeling head (vision + text).

    Inherits from Apriel2PreTrainedModel to get proper cache handling.
    Uses Apriel2Model (which inherits from Apriel2TextModel) for the base model.
    """

    config_class = Apriel2Config
    _tied_weights_keys = []  # No weight tying by default, but can be configured

    def __init__(self, config: Apriel2Config):
        super().__init__(config)
        self.model = Apriel2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Handle weight tying if configured
        if config.tie_word_embeddings:
            self._tied_weights_keys = ["lm_head.weight"]

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_image_features(self, pixel_values):
        """Extract and project image features."""
        return self.model.get_image_features(pixel_values)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
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
            pixel_values=pixel_values,
            image_sizes=image_sizes,
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

        hidden_states = outputs.last_hidden_state if return_dict else outputs[0]

        # Compute logits
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            if attention_mask is not None:
                # Use the input attention mask to shift the logits and labels
                # Crop attention mask in case it is longer (e.g., in PrefixTuning with peft)
                shift_attention_mask = attention_mask[:, -shift_logits.shape[1] :].to(logits.device)
                shift_logits = shift_logits[shift_attention_mask != 0].contiguous()
                shift_labels = shift_labels[shift_attention_mask.to(shift_labels.device) != 0].contiguous()
            else:
                shift_logits = shift_logits.contiguous()
                shift_labels = shift_labels.contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            flat_logits = shift_logits.view(-1, self.vocab_size)
            flat_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(flat_logits, flat_labels)

        if not return_dict:
            output = (logits,) + (outputs[1:] if return_dict else outputs[1:])
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values if return_dict else outputs[1],
            hidden_states=outputs.hidden_states if return_dict else None,
            attentions=outputs.attentions if return_dict else None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        pixel_values=None,
        attention_mask=None,
        use_cache=True,
        logits_to_keep=None,
        **kwargs,
    ):
        """Prepare inputs for generation, handling multimodal inputs correctly."""
        # Overwritten -- custom handling for pixel_values during cached generation
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        # If we're in cached decoding stage, pixel_values should be None because input ids do not contain
        # special image tokens anymore. Otherwise pixel_values should be passed to model.
        # NOTE: use_cache=False always needs pixel_values
        if cache_position is not None and cache_position[0] == 0:
            model_inputs["pixel_values"] = pixel_values

        return model_inputs
