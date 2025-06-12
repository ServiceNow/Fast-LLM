from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from einops import rearrange, repeat
from mamba_ssm.ops.triton.selective_state_update import selective_state_update
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from torch import nn
from transformers import GenerationMixin
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import LossKwargs, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from transformers.utils.generic import ModelOutput

from fast_llm.models.ssm.external.apriel_hybrid.configuration_ssm_hybrid_apriel import (
    ROPE_INIT_FUNCTIONS,
    AprielSSMHybridConfig,
)

logger = logging.get_logger(__name__)


class HybridMambaAttentionStaticCache(Cache):
    def __init__(self, config: AprielSSMHybridConfig, batch_size, max_length, dtype=torch.float16, device=None):
        super().__init__()  # config, batch_size, max_length, device, dtype)
        self.dtype = dtype
        self.hybrid_override_pattern = config.hybrid_block_layout
        self.has_previous_state = False  # only used by mamba
        intermediate_size = config.ssm_cfg["d_inner"]
        ssm_state_size = config.ssm_cfg["d_state"]
        conv_kernel_size = config.ssm_cfg["d_conv"]
        self.n_qk_heads = config.ssm_cfg["n_qk_heads"]
        assert intermediate_size % self.n_qk_heads == 0, "d_inner must be divisible by n_qk_heads"
        self.head_d = intermediate_size // self.n_qk_heads
        self.conv_states = []
        self.ssm_states = []
        self.transformer_layers = []
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []

        self.batch_size = batch_size
        self.head_dim = (
            config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
        )
        self.max_cache_len = config.max_position_embeddings if max_length is None else max_length

        self.num_key_value_heads = (
            config.num_attention_heads
            if getattr(config, "num_key_value_heads", None) is None
            else config.num_key_value_heads
        )
        cache_shape = (self.batch_size, self.num_key_value_heads, max_length, self.head_dim)

        for i in range(config.num_hidden_layers):
            if self.hybrid_override_pattern[i] == "m2d":
                # Mamba layer
                new_layer_conv_state = torch.zeros(
                    batch_size,
                    conv_kernel_size,
                    intermediate_size + 2 * self.n_qk_heads * ssm_state_size,
                    device=device,
                    dtype=dtype,
                ).transpose(1, 2)

                new_layer_ssm_state = torch.zeros(
                    batch_size, self.n_qk_heads, self.head_d, ssm_state_size, device=device, dtype=dtype
                )
                new_layer_key_cache = None  # torch.zeros((0,), dtype=dtype, device=device)
                new_layer_value_cache = None  # torch.zeros((0,), dtype=dtype, device=device)
            else:
                # Attention or MLP layer
                new_layer_conv_state = None  # torch.tensor((0,), dtype=dtype, device=device)
                new_layer_ssm_state = None  # torch.tensor((0,), dtype=dtype, device=device)
                new_layer_key_cache = torch.zeros(cache_shape, dtype=dtype, device=device)
                new_layer_value_cache = torch.zeros(cache_shape, dtype=dtype, device=device)
                self.transformer_layers.append(i)

            # if not is_torchdynamo_compiling():
            #     self.register_buffer(f"key_cache_{i}", torch.zeros(cache_shape, dtype=dtype, device=device))
            #     self.register_buffer(f"value_cache_{i}", torch.zeros(cache_shape, dtype=dtype, device=device))
            #     new_layer_key_cache = getattr(self, f"key_cache_{i}")
            #     new_layer_value_cache = getattr(self, f"value_cache_{i}")
            #     torch._dynamo.mark_static_address(new_layer_key_cache)
            #     torch._dynamo.mark_static_address(new_layer_value_cache)
            #     self.register_buffer(f"conv_states_{i}", new_layer_conv_state)
            #     self.register_buffer(f"ssm_states_{i}", new_layer_ssm_state)
            #     torch._dynamo.mark_static_address(new_layer_conv_state)
            #     torch._dynamo.mark_static_address(new_layer_ssm_state)
            #     new_layer_ssm_state = getattr(self, f"ssm_states_{i}")
            #     new_layer_conv_state = getattr(self, f"conv_states_{i}")

            self.key_cache.append(new_layer_key_cache)
            self.value_cache.append(new_layer_value_cache)
            self.conv_states.append(new_layer_conv_state)
            self.ssm_states.append(new_layer_ssm_state)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        It is VERY important to index using a tensor, otherwise you introduce a copy to the device.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The `StaticCache` needs the `cache_position` input
                to know how where to write in the cache.

        Return:
            A tuple containing the updated key and value states.
        """

        cache_position = cache_kwargs.get("cache_position")

        k_out = self.key_cache[layer_idx]
        v_out = self.value_cache[layer_idx]
        key_states = key_states.to(k_out.dtype)
        value_states = value_states.to(v_out.dtype)

        if cache_position is None:
            k_out.copy_(key_states)
            v_out.copy_(value_states)
        else:
            # Note: here we use `tensor.index_copy_(dim, index, tensor)` that is equivalent to
            # `tensor[:, :, index] = tensor`, but the first one is compile-friendly and it does explicitly an in-place
            # operation, that avoids copies and uses less memory.
            try:
                k_out.index_copy_(2, cache_position, key_states)
                v_out.index_copy_(2, cache_position, value_states)
            except NotImplementedError:
                # The operator 'aten::index_copy.out' is not currently implemented for the MPS device.
                k_out[:, :, cache_position] = key_states
                v_out[:, :, cache_position] = value_states

        return k_out, v_out

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))

            device = self.conv_states[layer_idx].device
            self.conv_states[layer_idx] = self.conv_states[layer_idx].index_select(0, beam_idx.to(device))
            device = self.ssm_states[layer_idx].device
            self.ssm_states[layer_idx] = self.ssm_states[layer_idx].index_select(0, beam_idx.to(device))

    def get_seq_length(self, layer_idx: Optional[int] = None) -> int:
        """Returns the sequence length of the cached states that were seen by the model."""
        # Occupied cache == any slot in the 3rd dim (sequence length) holds a non-zero value. To save on compute, let's
        # limit the check to the first batch member and head dimension.
        # TODO: deprecate this function in favor of `cache_position`
        if layer_idx is None:
            layer_idx = self.transformer_layers[0]
        return (self.key_cache[layer_idx][0, 0].any(dim=-1)).sum()

    def get_max_cache_shape(self) -> Optional[int]:
        return self.max_cache_len

    # Copied from modeling_mamba2.py
    def update_conv_state(
        self, layer_idx: int, new_conv_state: torch.Tensor, cache_init: bool = False
    ) -> torch.Tensor:
        if cache_init:
            self.conv_states[layer_idx] = new_conv_state.to(self.conv_states.device)
        else:
            self.conv_states[layer_idx] = self.conv_states[layer_idx].roll(shifts=-1, dims=-1)
            self.conv_states[layer_idx][:, :, -1] = new_conv_state[:, 0, :].to(self.conv_states.device)
        return self.conv_states[layer_idx]

    def update_ssm_state(self, layer_idx: int, new_ssm_state: torch.Tensor):
        self.ssm_states[layer_idx] = new_ssm_state.to(self.ssm_states.device)
        return self.ssm_states[layer_idx]

    def reset(self):
        self.conv_states.zero_()
        self.ssm_states.zero_()


# Copied from https://github.com/huggingface/transformers/blob/main/src/transformers/models/jamba/modeling_jamba.py
class HybridMambaAttentionDynamicCache(DynamicCache):
    """
    A dynamic cache that can handle both the attention cache (which has a seq_len dimension) and the mamba cache
    (which has a constant shape regardless of seq_len).
    This cache has two sets of lists of tensors: `key_cache` and `value_cache` for attention cache and `conv_states`
    and `ssm_states` for mamba cache. Each of these lists has `num_layers` tensors. The expected shape for each tensor
    For attention layers, `key_cache` and `value_cache` have a shape of `(batch_size, num_heads, seq_len, head_dim)`,
    while `conv_states` and `ssm_states` have a shape of `(batch_size, 0)` (empty tensors).
    For mamba layers, `key_cache` and `value_cache` have a shape of `(batch_size, 0)` (empty tensors),
    while `conv_states` represents the convolution state and has a shape of `(batch_size, d_inner, d_conv)`,
    and `ssm_states` represents the ssm state and has a shape of `(batch_size, d_inner, d_state)`.
    """

    def __init__(self, config: AprielSSMHybridConfig, batch_size, dtype=torch.float16, device=None):
        super().__init__()
        self.dtype = dtype
        self.hybrid_override_pattern = config.hybrid_block_layout
        self.has_previous_state = False  # only used by mamba
        intermediate_size = config.ssm_cfg["d_inner"]
        ssm_state_size = config.ssm_cfg["d_state"]
        conv_kernel_size = config.ssm_cfg["d_conv"]
        self.n_qk_heads = config.ssm_cfg["n_qk_heads"]
        assert intermediate_size % self.n_qk_heads == 0, "d_inner must be divisible by n_qk_heads"
        self.head_d = intermediate_size // self.n_qk_heads
        self.conv_states = []
        self.ssm_states = []
        self.transformer_layers = []
        for i in range(config.num_hidden_layers):
            if self.hybrid_override_pattern[i] == "m2d":
                # Mamba layer
                self.conv_states += [
                    torch.zeros(
                        batch_size,
                        conv_kernel_size,
                        intermediate_size + 2 * self.n_qk_heads * ssm_state_size,
                        device=device,
                        dtype=dtype,
                    ).transpose(1, 2)
                ]
                self.ssm_states += [
                    torch.zeros(batch_size, self.n_qk_heads, self.head_d, ssm_state_size, device=device, dtype=dtype)
                ]
            else:
                # Attention or MLP layer
                self.conv_states += [torch.tensor([[]] * batch_size, device=device)]
                self.ssm_states += [torch.tensor([[]] * batch_size, device=device)]
                self.transformer_layers.append(i)

        self.key_cache = [torch.tensor([[]] * batch_size, device=device) for _ in range(config.num_hidden_layers)]
        self.value_cache = [torch.tensor([[]] * batch_size, device=device) for _ in range(config.num_hidden_layers)]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Update the cache
        if self.key_cache[layer_idx].shape[-1] == 0:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))

            device = self.conv_states[layer_idx].device
            self.conv_states[layer_idx] = self.conv_states[layer_idx].index_select(0, beam_idx.to(device))
            device = self.ssm_states[layer_idx].device
            self.ssm_states[layer_idx] = self.ssm_states[layer_idx].index_select(0, beam_idx.to(device))

    def to_legacy_cache(self) -> tuple[tuple[torch.Tensor], tuple[torch.Tensor]]:
        raise NotImplementedError("HybridMambaAttentionDynamicCache does not have a legacy cache equivalent.")

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[tuple[tuple[torch.FloatTensor]]] = None) -> "DynamicCache":
        raise NotImplementedError("HybridMambaAttentionDynamicCache does not have a legacy cache equivalent.")

    # Copied from modeling_mamba2.py
    def update_conv_state(
        self, layer_idx: int, new_conv_state: torch.Tensor, cache_init: bool = False
    ) -> torch.Tensor:
        if cache_init:
            self.conv_states[layer_idx] = new_conv_state.to(self.conv_states.device)
        else:
            self.conv_states[layer_idx] = self.conv_states[layer_idx].roll(shifts=-1, dims=-1)
            self.conv_states[layer_idx][:, :, -1] = new_conv_state[:, 0, :].to(self.conv_states.device)
        return self.conv_states[layer_idx]

    def update_ssm_state(self, layer_idx: int, new_ssm_state: torch.Tensor):
        self.ssm_states[layer_idx] = new_ssm_state.to(self.ssm_states.device)
        return self.ssm_states[layer_idx]

    def reset(self):
        self.conv_states.zero_()
        self.ssm_states.zero_()


@dataclass
class AprielHybridCausalOutput(ModelOutput):
    """Custom output class for MambaLMHeadModel."""

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    all_hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    attention_weights: Optional[torch.FloatTensor] = None
    past_key_values: Optional[HybridMambaAttentionDynamicCache] = None


class AprielRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, device=None, dtype=None, **kwargs):
        """
        AprielRMSNorm is equivalent to T5LayerNorm
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, **factory_kwargs))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


ALL_LAYERNORM_LAYERS.append(AprielRMSNorm)


class AprielMLP(nn.Module):
    def __init__(self, config, device=None, dtype=None, **kwargs):
        super().__init__(**kwargs)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias, **factory_kwargs)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias, **factory_kwargs)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias, **factory_kwargs)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class AprielRotaryEmbedding(nn.Module):
    def __init__(self, config: AprielSSMHybridConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=seq_len)
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            # This .to() is needed if the model has been moved to a device after being initialized (because
            # the buffer is automatically moved, but not the original copy)
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float().to(x.device) @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.
    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


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


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class AprielAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: AprielSSMHybridConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


def segsum(x):
    """More stable segment sum calculation."""
    # [1, 2, 3]
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    # [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    # [[0, 0, 0], [2, 0, 0], [3, 3, 0]]
    x_segsum = torch.cumsum(x, dim=-2)
    # [[0, 0, 0], [2, 0, 0], [5, 3, 0]]
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def materialize_mixer(A_log, B, C, D):
    """
    Since the transfer matrix will be equated to the attention matrix,
    we need to support the form: torch.matmul(attn_weights, value_states).
    Thus, y = torch.matmul(T, X)
    Arguments:
        A_log: (batch, length, n_heads)
        B: (batch, length, n_heads, d_state)
        C: (batch, length, n_heads, d_state)
    Return:
        T: (batch, n_heads, length, length)
    """
    batch_size, length, n_heads, d_state = B.shape
    assert A_log.shape == (batch_size, length, n_heads)
    assert B.shape == C.shape == (batch_size, length, n_heads, d_state)

    # Compute:
    A_log = rearrange(-F.softplus(A_log), "b l h -> b h l")
    powers = torch.exp(segsum(A_log))
    T = torch.einsum("blhn,bshn,bhls->bhsl", C, B, powers)

    # Add D:
    if D is not None:
        T[:, :, torch.arange(length), torch.arange(length)] += D.view(1, n_heads, 1)

    T = rearrange(T, "b h z l -> b h l z")
    return T


# This is from LLmaba/Mohawk: https://github.com/cartesia-ai/edge/blob/main/cartesia-pytorch/cartesia_pytorch/Llamba/mixers/discrete_mamba2.py
class DiscreteMamba2(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=64,
        n_qk_heads=32,
        n_v_heads=32,
        d_conv=4,
        expand=1,
        activation="identity",
        bias=False,
        conv_bias=True,
        chunk_size=128,
        layer_idx=None,
        device=None,
        dtype=None,
        d_inner=None,
        **kwargs,  # Absorb kwarg for general module
    ):
        """
        See the class .kernel.SSKernel for the kernel constructor which accepts kernel_args.
        Relevant options that are worth considering and tuning include "mode" + "measure", "dt_min", "dt_max", "lr"

        Other options are all experimental and should not need to be configured
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = self.expand * self.d_model if d_inner is None else d_inner
        self.n_qk_heads = n_qk_heads
        self.n_v_heads = n_v_heads
        self.headdim = self.d_inner // self.n_v_heads
        assert self.n_v_heads == self.d_inner // self.headdim
        assert self.d_inner % self.headdim == 0
        assert self.n_v_heads % self.n_qk_heads == 0
        self.activation = activation
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx
        self.bias = bias
        self.kwargs = kwargs

        # Projections
        self.in_proj = nn.Linear(
            self.d_model,
            2 * self.d_inner + 2 * self.n_qk_heads * self.d_state + self.n_v_heads,
            bias=bias,
            **factory_kwargs,
        )
        self.z_bias = (
            nn.Parameter(torch.zeros(self.d_inner, device=device)) if not bias else 0
        )  # make sure z_bias always exists

        # Convolutional layer
        conv_dim = self.d_inner + 2 * self.n_qk_heads * self.d_state
        self.conv_bias = conv_bias
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        # Activation after conv
        if self.activation == "identity":
            self.act = nn.Identity()
        elif self.activation in ["silu", "swish"]:
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation {self.activation}")

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.n_v_heads, device=device))
        self.D._optim = {"weight_decay": 0.0}

        # out_proj
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    @property
    def d_output(self):
        return self.d_model

    @property
    def state_to_tensor(self):
        return self.layer.state_to_tensor

    def forward(
        self,
        u,
        return_mixer_matrix=False,
        past_key_value: Optional[HybridMambaAttentionDynamicCache] = None,
        inference_params=None,
        **kwargs,
    ):
        """
        u: (B, L, D)
        Returns: same shape as u
        """
        outputs = {}
        # assert state is None
        batch, seqlen, dim = u.shape

        ssm_state, conv_state = None, None
        if past_key_value is not None:
            ssm_state, conv_state = self._get_states_from_cache(past_key_value, batch)
            if inference_params is not None and inference_params.seqlen_offset > 0:
                # States are updated inplace
                # TODO: make sure inference_params with seqlen_offset are properly initialized
                u = u.squeeze(1) if len(u.shape) == 3 else u
                out, _, _ = self.step(u, ssm_state, conv_state)
                out = out.unsqueeze(1) if len(u.shape) == 2 else out
                return {"hidden_states": out}

        # Hacky way to initialize state during inference
        chunk_size = self.chunk_size if ssm_state is None else seqlen

        # Pad input to nearest multiple of chunklen
        padded_len = (1 + (seqlen - 1) // chunk_size) * chunk_size
        u = F.pad(u, (0, 0, 0, padded_len - seqlen))

        # Project input
        xBCzA_log = self.in_proj(u)
        xBC, z, A_log = torch.split(
            xBCzA_log,
            [
                self.d_inner + 2 * self.n_qk_heads * self.d_state,
                self.d_inner,
                self.n_v_heads,
            ],
            dim=-1,
        )

        if ssm_state is not None:
            # If we just take xBC[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
            # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
            xBC_t = rearrange(xBC[:, :seqlen, :], "b l d -> b d l")
            conv_state.copy_(F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0)))  # Update state (B D W)

        # Convolutional layer
        xBC = self.convolutional_forward(xBC, padded_len)

        x, B, C = torch.split(
            xBC,
            [
                self.d_inner,
                self.n_qk_heads * self.d_state,
                self.n_qk_heads * self.d_state,
            ],
            dim=-1,
        )

        x = rearrange(x, "b l (h n) -> b l h n", h=self.n_v_heads)
        B = rearrange(B, "b l (h n) -> b l h n", h=self.n_qk_heads)
        C = rearrange(C, "b l (h n) -> b l h n", h=self.n_qk_heads)

        # SSM forward
        result = mamba_chunk_scan_combined(
            x=x / F.softplus(A_log).to(x.dtype).unsqueeze(-1),
            dt=A_log,
            dt_softplus=True,
            A=-torch.ones(self.n_v_heads, device=A_log.device),
            B=B,
            C=C,
            chunk_size=chunk_size,
            # initial_states=(state["ssm"] if state is not None else None), # currently not supported by mamba_ssm.utils.generation
            return_final_states=(ssm_state is not None),
        )

        if ssm_state is not None:
            y, ssm_state_update = result
            ssm_state.copy_(ssm_state_update)
        else:
            y = result

        Du = torch.einsum("h,blhp->blhp", self.D, x)
        y = rearrange(y + Du, "b l h p -> b l (h p)")

        # Norm and gate
        out = self.out_proj(y * F.silu(z + self.z_bias))
        outputs["hidden_states"] = out[:, :seqlen, :]

        if return_mixer_matrix:
            outputs["transfer_matrix"] = materialize_mixer(A_log=A_log, B=B, C=C, D=self.D)[..., :seqlen, :seqlen]
        return outputs

    def step(self, u, ssm_state, conv_state, **kwargs):
        """
        u: (B D)
        state: dict of states
        Returns: same shape as u
        """

        # Project input
        xBCzA_log = self.in_proj(u)
        xBC, z, A_log = torch.split(
            xBCzA_log,
            [
                self.d_inner + 2 * self.n_qk_heads * self.d_state,
                self.d_inner,
                self.n_v_heads,
            ],
            dim=-1,
        )

        xBC, conv_state_new = self.convolutional_step(xBC, conv_state)
        conv_state.copy_(conv_state_new)  # update state in place

        x, B, C = torch.split(
            xBC,
            [
                self.d_inner,
                self.n_qk_heads * self.d_state,
                self.n_qk_heads * self.d_state,
            ],
            dim=-1,
        )

        x = rearrange(x, "b (h s) -> b h s", h=self.n_v_heads)
        B = rearrange(B, "b (h s) -> b h s", h=self.n_qk_heads)
        C = rearrange(C, "b (h s) -> b h s", h=self.n_qk_heads)

        ssm_state = ssm_state.to(x.dtype)
        zeros = torch.zeros((self.n_v_heads, self.headdim), device=A_log.device).to(dtype=x.dtype)
        ones = torch.ones((self.n_v_heads, self.headdim, self.d_state), device=A_log.device).to(dtype=x.dtype)
        y = selective_state_update(
            x=x / F.softplus(A_log).to(x.dtype).unsqueeze(-1),
            dt=repeat(A_log, "b h -> b h p", p=self.headdim),
            dt_softplus=True,
            A=-ones,
            B=B,
            C=C,
            state=ssm_state,  # will be updated in place
            dt_bias=zeros,
            D=zeros,
        )

        y = y + self.D[:, None] * x
        y = rearrange(y, "b h p -> b (h p)")

        # Norm and gate
        out = self.out_proj(y * F.silu(z + self.z_bias))

        return out, ssm_state, conv_state

    # def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
    #     device = self.in_proj.weight.device
    #     # conv_state:
    #     conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
    #     conv_state = torch.zeros(
    #         batch_size,
    #         self.d_conv,
    #         self.conv1d.weight.shape[0],
    #         device=device,
    #         dtype=conv_dtype,
    #     ).transpose(1, 2)
    #     # ssm_state:
    #     ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
    #     ssm_state = torch.zeros(
    #         batch_size,
    #         self.n_v_heads,
    #         self.headdim,
    #         self.d_state,
    #         device=device,
    #         dtype=ssm_dtype,
    #     )
    #     return {"conv": conv_state, "ssm": ssm_state}

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        """
        conv_state: (batch, d_conv, conv1d.weight.shape[0])
        ssm_state: (batch, n_qk_heads, headdim, d_state)
        """
        assert self.layer_idx is not None
        # Allocate memory if not exists
        # if self.layer_idx not in inference_params.ssm_states:
        #     inference_params.key_value_memory_dict[self.layer_idx] = self.allocate_inference_cache(
        #         batch_size, inference_params.max_seqlen, dtype=torch.float32
        #     )
        # Get states
        ssm_states = inference_params.ssm_states[self.layer_idx]
        conv_states = inference_params.conv_states[self.layer_idx]
        if initialize_states:
            ssm_states.zero_()
            conv_states.zero_()
        return ssm_states, conv_states

    def convolutional_forward(self, xBC, padded_len):
        if causal_conv1d_fn is None or self.activation not in [
            "silu",
            "swish",
            "identity",
        ]:
            xBC = self.act(self.conv1d(xBC.transpose(1, 2))[..., :padded_len].transpose(1, 2))
        else:
            xBC = causal_conv1d_fn(
                xBC.transpose(1, 2),
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                activation=None if self.activation == "identity" else self.activation,
            ).transpose(1, 2)
        return xBC

    def convolutional_step(self, xBC, conv_state):
        # Convolutional layer
        conv_state = conv_state.to(xBC.dtype)
        if causal_conv1d_update:
            xBC = causal_conv1d_update(
                xBC,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation if self.activation != "identity" else None,
            )
        else:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = xBC
            xBC = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv_bias:
                xBC = xBC + self.conv1d.bias
            xBC = self.act(xBC).to(xBC.dtype)  # Some activations change dtype

        return xBC, conv_state


class AprielDecoderLayer(nn.Module):
    def __init__(self, config: AprielSSMHybridConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = AprielAttention(config=config, layer_idx=layer_idx)

        self.mlp = AprielMLP(config)
        self.input_layernorm = AprielRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = AprielRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class AprielSSMDecoderLayer(nn.Module):
    def __init__(self, config: AprielSSMHybridConfig, layer_idx: int, device=None, dtype=None, **kwargs):
        super().__init__(**kwargs)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.hidden_size = config.hidden_size

        self.mixer = DiscreteMamba2(
            d_model=config.hidden_size,
            layer_idx=layer_idx,
            **config.ssm_cfg,
            **factory_kwargs,
        )

        self.mlp = AprielMLP(config, **factory_kwargs)
        self.input_layernorm = AprielRMSNorm(config.hidden_size, eps=config.rms_norm_eps, **factory_kwargs)
        self.post_attention_layernorm = AprielRMSNorm(config.hidden_size, eps=config.rms_norm_eps, **factory_kwargs)

    def forward(
        self, hidden_states: torch.Tensor, **kwargs
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:

        outputs = {}
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        mixer_outputs = self.mixer(
            hidden_states,
            **kwargs,
        )

        hidden_states = mixer_outputs["hidden_states"].to(residual.dtype) + residual

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # outputs["hidden_states"] = hidden_states
        outputs = (hidden_states,)

        return outputs

    # def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
    #     """Allocate inference cache for the model."""
    #     if getattr(self.mixer, "allocate_inference_cache", None) is None:
    #         return
    #     return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


APRIEL_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)
    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.
    Parameters:
        config ([`AprielSSMHybridConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Apriel Model outputting raw hidden-states without any specific head on top.",
    APRIEL_START_DOCSTRING,
)
class AprielSSMPreTrainedModel(PreTrainedModel):
    config_class = AprielSSMHybridConfig
    base_model_prefix = "model"
    _no_split_modules = ["AprielDecoderLayer", "AprielSSMDecoderLayer"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    # def allocate_inference_cache(self, *args, **kwargs):
    #     """Allocate inference cache for the model."""
    #     return getattr(self, self.base_model_prefix).allocate_inference_cache(*args, **kwargs)


APRIEL_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).
            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.
            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.
            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance, see our
            [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache);
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.
            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.
            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare Apriel Model outputting raw hidden-states without any specific head on top.",
    APRIEL_START_DOCSTRING,
)
class AprielSSMHybridModel(AprielSSMPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`AprielDecoderLayer`, `AprielSSMDecoderLayer`]
    Args:
        config: AprielSSMHybridConfig
    """

    def __init__(self, config: AprielSSMHybridConfig, device=None, dtype=None, **kwargs):
        super().__init__(config, device=device, dtype=dtype, **kwargs)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        factory_kwargs = {"device": device, "dtype": dtype}
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx, **factory_kwargs)
        blocks = []
        logger.info(f"Loading hyubrid model with the following layout: {config.hybrid_block_layout}")
        for layer_idx, type in enumerate(config.hybrid_block_layout):
            if type == "m2d":
                blocks.append(AprielSSMDecoderLayer(config, layer_idx, **factory_kwargs))
            elif type == "t":
                blocks.append(AprielDecoderLayer(config, layer_idx))
            else:
                raise ValueError(f"Invalid block type: {type}")
        self.layers = nn.ModuleList(blocks)
        self.norm = AprielRMSNorm(config.hidden_size, eps=config.rms_norm_eps, **factory_kwargs)
        self.gradient_checkpointing = False
        self.rotary_emb = AprielRotaryEmbedding(config=config)
        self.has_transformer_layers = any(type == "t" for type in config.hybrid_block_layout)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # def allocate_inference_cache(self, *args, **kwargs):
    #     """Allocate inference cache for the model."""
    #     cache = {}
    #     for i, layer in enumerate(self.layers):
    #         if isinstance(layer, AprielSSMDecoderLayer):
    #             cache[i] = layer.allocate_inference_cache(*args, **kwargs)
    #     return cache

    @add_start_docstrings_to_model_forward(APRIEL_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[HybridMambaAttentionDynamicCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inference_params=None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            # past_key_values = HybridMambaAttentionDynamicCache()
            logger.warning_once(
                "Hybrid Apriel requires an initialized `HybridMambaAttentionDynamicCache` to return a cache. None was "
                "provided, so no cache will be returned."
            )

        if cache_position is None and self.has_transformer_layers:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None and self.has_transformer_layers:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = (
            self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions)
            if self.has_transformer_layers
            else None
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids) if self.has_transformer_layers else None

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                inference_params=inference_params,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions and isinstance(decoder_layer, AprielDecoderLayer):
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache) or isinstance(
            past_key_values, HybridMambaAttentionStaticCache
        )

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.
        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to place the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


class AprielSSMHybridForCausalLM(AprielSSMPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.model = AprielSSMHybridModel(config)
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

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        output_router_logits=False,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        # Overwritten -- has a unique cache type, `HybridMambaAttentionDynamicCache`

        empty_past_kv = past_key_values is None or not isinstance(past_key_values, HybridMambaAttentionDynamicCache)

        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        # Exception 3: with synced GPUs cache_position may go out of bounds, but we only want dummy token in that case.
        #              (we can't check exception 3 while compiling)
        if not empty_past_kv:
            if inputs_embeds is not None or cache_position[-1] >= input_ids.shape[1]:  # Exception 1  # Exception 3
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]
        else:
            past_key_values = HybridMambaAttentionDynamicCache(
                self.config, input_ids.shape[0], self.dtype, device=self.device
            )

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if not empty_past_kv:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and empty_past_kv:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}  # `contiguous()` needed for compilation use cases

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "output_router_logits": output_router_logits,
                # "logits_to_keep": self.config.num_logits_to_keep,
                "cache_position": cache_position,
            }
        )
        return model_inputs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids=None,
        return_hidden_states=False,
        return_logits=True,
        num_last_tokens=0,
        past_key_values: Optional[HybridMambaAttentionDynamicCache] = None,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[tuple, CausalLMOutputWithPast]:

        # past_key_values is None if prepare_inputs_for_generation is not called, which is the case when we evaluate without calling generate (non-generation tasks)
        # Its generally ok if cache is nto instantiated in this case, since we do single pass per sample anyways, a warning will be triggered in the model
        outputs: BaseModelOutputWithPast = self.model(
            input_ids,
            return_hidden_states=return_hidden_states,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )

        if outputs["last_hidden_state"] is not None and return_logits:
            logits = self.lm_head(outputs["last_hidden_state"]).float()
            outputs["logits"] = logits if num_last_tokens == 0 else logits[:, -num_last_tokens:]
        else:
            outputs["logits"] = None

        return AprielHybridCausalOutput(
            loss=None,
            logits=outputs["logits"],
            all_hidden_states=outputs.hidden_states,
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
        )


__all__ = [
    "AprielSSMHybridForCausalLM",
    "AprielSSMHybridModel",
    "AprielSSMPreTrainedModel",
]
