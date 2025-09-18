import copy
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
from transformers import GenerationMixin
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer, MistralMLP, MistralModel, MistralRMSNorm
from transformers.processing_utils import Unpack
from transformers.utils import LossKwargs, logging
from transformers.utils.generic import ModelOutput

from fast_llm_external_models.apriel_hybrid_ssm.configuration_apriel_hybrid_ssm import AprielHybridSSMConfig

# from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_scan_fn as varlen_selective_scan_fn
# from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn as varlen_causal_conv1d_fn


logger = logging.get_logger(__name__)


is_fast_path_available = all((selective_state_update, causal_conv1d_fn, causal_conv1d_update))


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


class HybridMambaAttentionStaticCache(Cache):
    def __init__(self, config: AprielHybridSSMConfig, batch_size, max_length, dtype=torch.float16, device=None):
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
            if len(self.transformer_layers) > 0:
                layer_idx = self.transformer_layers[0]
            else:
                return 0
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

    def __init__(self, config: AprielHybridSSMConfig, batch_size, dtype=torch.float16, device=None):
        super().__init__()
        self.dtype = dtype
        self.hybrid_override_pattern = config.hybrid_block_layout
        self.has_previous_state = False  # only used by mamba
        intermediate_size = (
            config.ssm_cfg["d_inner"]
            if config.ssm_cfg["d_inner"] is not None
            else config.ssm_cfg["expand"] * config.hidden_size
        )
        ssm_state_size = config.ssm_cfg["d_state"]
        conv_kernel_size = config.ssm_cfg["d_conv"]
        self.n_qk_heads = config.ssm_cfg["n_qk_heads"]
        self.num_C_head = intermediate_size // ssm_state_size  # mamba2
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
            elif self.hybrid_override_pattern[i] == "m2":
                if "repeat_kv_before_conv" in config.ssm_cfg:
                    assert (
                        config.ssm_cfg["repeat_kv_before_conv"] == True
                    ), "Only support repeat_kv_before_conv=True for m2 for now"

                self.conv_states += [
                    torch.zeros(
                        batch_size,
                        intermediate_size,
                        conv_kernel_size,
                        device=device,
                        dtype=dtype,
                    )
                ]
                self.ssm_states += [
                    torch.zeros(
                        batch_size,
                        self.num_C_head,
                        intermediate_size // self.num_C_head,
                        ssm_state_size,
                        device=device,
                        dtype=dtype,
                    )
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

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # take any layer that contains cache and not empty tensor
        layer_idx = self.transformer_layers[0] if layer_idx not in self.transformer_layers else layer_idx
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def reset(self):
        self.conv_states.zero_()
        self.ssm_states.zero_()

    def to_legacy_cache(self) -> tuple[tuple[torch.Tensor], tuple[torch.Tensor]]:
        raise NotImplementedError("HybridMambaAttentionDynamicCache does not have a legacy cache equivalent.")

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[tuple[tuple[torch.FloatTensor]]] = None) -> "DynamicCache":
        raise NotImplementedError("HybridMambaAttentionDynamicCache does not have a legacy cache equivalent.")


@dataclass
class AprielHybridCausalOutput(ModelOutput):
    """Custom output class for MambaLMHeadModel."""

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    all_hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    attention_weights: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None


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


def apply_mask_to_padding_states(hidden_states, attention_mask):
    """
    Tunes out the hidden states for padding tokens, see https://github.com/state-spaces/mamba/issues/66
    """
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

    return hidden_states


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
        # In __init__, pre-allocate these tensors
        # self.zeros_buffer = torch.zeros((self.n_v_heads, self.headdim), device=device, dtype=dtype)
        # self.ones_buffer = torch.ones((self.n_v_heads, self.headdim, self.d_state), device=device, dtype=dtype)

    # @property
    # def d_output(self):
    #     return self.d_model

    # @property
    # def state_to_tensor(self):
    #     return self.layer.state_to_tensor

    def forward(
        self,
        u,
        past_key_value: Optional[HybridMambaAttentionDynamicCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_mixer_matrix=False,
        **kwargs,
    ):
        """
        u: (B, L, D)
        Returns: same shape as u
        For later refference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/bamba/modeling_bamba.py
        """
        assert is_fast_path_available and "cuda" in self.in_proj.weight.device.type, "Only support fast path on cuda"
        cache_position = kwargs.get("cache_position", None)
        batch, seqlen, dim = u.shape
        u = apply_mask_to_padding_states(u, attention_mask)
        ssm_state, conv_state = None, None
        use_precomputed_states = False
        #########################################################
        # Quick and dirty to work with CG
        if "inference_params" in kwargs:
            seqlen_offset = kwargs["inference_params"].seqlen_offset
            if seqlen_offset > 0:
                use_precomputed_states = True
        else:
            seqlen_offset = kwargs.get("seqlen_offset", cache_position[0]) if cache_position is not None else 0
            use_precomputed_states = (
                getattr(past_key_value, "has_previous_state", False)
                and seqlen == 1
                and past_key_value.conv_states[self.layer_idx].shape[0]
                == past_key_value.ssm_states[self.layer_idx].shape[0]
                == batch
                and cache_position is not None
                and seqlen_offset > 0
            )
        #########################################################
        ssm_state, conv_state = self._get_states_from_cache(past_key_value, batch)
        if use_precomputed_states:
            # ssm_state, conv_state = self._get_states_from_cache(past_key_value, batch)
            u = u.squeeze(1) if len(u.shape) == 3 else u
            out, _, _ = self.step(u, ssm_state, conv_state)
            out = out.unsqueeze(1) if len(u.shape) == 2 else out
            return {"hidden_states": out}
        else:
            outputs = {}
            # Hacky way to initialize state during inference
            chunk_size = self.chunk_size  # if ssm_state is None else seqlen

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

            if conv_state is not None:
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
        if conv_state_new is not None:
            raise NotImplementedError("Should not end up here snce only support fast path.")
            # conv_state.copy_(conv_state_new)  # update state in place, only for slow pass

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
        # does nto work with CG, probably becuase zeros and ones are on CPU
        # zeros = self.zeros_buffer.to(A_log.device).to(x.dtype)  # Just cast, don't allocate
        # ones = self.ones_buffer.to(A_log.device).to(x.dtype)
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
            return xBC, None
        else:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = xBC
            xBC = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv_bias:
                xBC = xBC + self.conv1d.bias
            xBC = self.act(xBC).to(xBC.dtype)  # Some activations change dtype

            return xBC, conv_state


class Mamba2(nn.Module):
    def __init__(
        self,
        d_model,
        d_inner,
        d_xb=None,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        repeat_kv_before_conv=True,
        conv_bias=True,
        bias=False,
        dt_proj_bias=True,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_xb = d_xb if d_xb is not None else d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_inner if d_inner is not None else int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
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
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[HybridMambaAttentionDynamicCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_mixer_matrix=False,
        **kwargs,
    ):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        cu_seqlens = None
        assert is_fast_path_available and "cuda" in self.in_proj.weight.device.type, "Only support fast path on cuda"
        cache_position = kwargs.get("cache_position", None)
        batch, seqlen, dim = hidden_states.shape

        ssm_state, conv_state = None, None
        use_precomputed_states = False

        #########################################################
        # Quick and dirty to work with CG
        if "inference_params" in kwargs:
            seqlen_offset = kwargs["inference_params"].seqlen_offset
            if seqlen_offset > 0:
                use_precomputed_states = True
        else:
            seqlen_offset = kwargs.get("seqlen_offset", cache_position[0]) if cache_position is not None else 0
            use_precomputed_states = (
                past_key_value is not None
                and past_key_value.has_previous_state
                and seqlen == 1
                and past_key_value.conv_states[self.layer_idx].shape[0]
                == past_key_value.ssm_states[self.layer_idx].shape[0]
                == batch
                and cache_position is not None
                and seqlen_offset > 0
            )
        #########################################################

        ssm_state, conv_state = self._get_states_from_cache(past_key_value, batch)
        if use_precomputed_states:
            # ssm_state, conv_state = self._get_states_from_cache(past_key_value, batch)
            out, _, _ = self.step(hidden_states, conv_state, ssm_state)
            return {"hidden_states": out}

        outputs = {}
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        zxbc = self.in_proj(hidden_states)
        z, x, B, C = torch.split(
            zxbc,
            [
                self.d_inner,
                self.d_xb,
                self.d_xb,
                self.d_inner,
            ],
            dim=-1,
        )

        x = rearrange(x, "b l d -> b d l")
        z = rearrange(z, "b l d -> b d l")

        B = rearrange(B, "b l (n_group dstate) -> b n_group l dstate", dstate=self.d_state)
        B = repeat_kv(B, self.repeat_group)  # B, n_group, L, H
        B = rearrange(B, "b n_group l dstate -> b n_group dstate l").contiguous()
        C = rearrange(C, "b l (n_group dstate) -> b n_group dstate l", dstate=self.d_state).contiguous()

        dt = self.dt_proj(self.dt_in_proj(hidden_states))  # B, L, d_inner
        dt = rearrange(dt, "b l d -> b d l")  # B, d_inner, L

        if self.repeat_kv_before_conv:
            x = rearrange(x, "b (n_group dstate) l -> b n_group l dstate", dstate=self.d_state)
            x = repeat_kv(x, self.repeat_group)
            x = rearrange(x, "b n_group l dstate -> b (n_group dstate) l")

        if cu_seqlens is not None:
            # variable length path
            x = varlen_causal_conv1d_fn(
                x.squeeze(0) if cu_seqlens is not None else x,  # Add batch dimension
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_state,
                query_start_loc=cu_seqlens,
            )
            x = x.unsqueeze(0) if cu_seqlens is not None else x
        else:
            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                # Update state (B D W)
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))
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

        if cu_seqlens is not None:
            # use variable length decoding
            y = varlen_selective_scan_fn(
                x.squeeze(0),
                ssm_state,
                dt.squeeze(0),
                A,
                B.squeeze(0),
                C.squeeze(0),
                self.D.float(),
                z=z.squeeze(0),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                query_start_loc=cu_seqlens,
            )
            y = y.unsqueeze(0)
        else:
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
                return_last_state=(ssm_state is not None),
            )

            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(rearrange(last_state, "b (h d) n -> b h d n", h=self.num_C_head))

        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)

        outputs["hidden_states"] = out[:, :seqlen, :]
        return outputs

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"

        hidden_states_input = hidden_states.squeeze(1)

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        zxbcdt = self.in_proj(hidden_states_input)
        z, x, B, C, dt = torch.split(zxbcdt, [self.d_inner, self.d_xb, self.d_xb, self.d_inner, self.dt_rank], dim=-1)

        B = rearrange(B, "b (n_group dstate) -> b n_group dstate", dstate=self.d_state)
        B = torch.repeat_interleave(B, dim=1, repeats=self.repeat_group)
        C = rearrange(C, "b (n_group dstate) -> b n_group dstate", dstate=self.d_state).contiguous()

        dt = self.dt_proj(dt)  # B, d_inner

        if self.repeat_kv_before_conv:
            x = rearrange(x, "b (n_group dstate) -> b n_group dstate", dstate=self.d_state)
            x = torch.repeat_interleave(x, dim=1, repeats=self.repeat_group)
            x = rearrange(x, "b n_group dstate -> b (n_group dstate)")

        # Conv step
        if causal_conv1d_update is None:
            # Update state (B D W)
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
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

        if not self.repeat_kv_before_conv:
            x = rearrange(x, "b (n_group dstate) -> b n_group dstate", dstate=self.d_state)
            x = torch.repeat_interleave(x, dim=1, repeats=self.repeat_group)
            x = rearrange(x, "b n_group dstate -> b (n_group dstate)")

        x = rearrange(x, "b (h d) -> b h d", h=self.num_C_head)
        dt = rearrange(dt, "b (h d) -> b h d", h=self.num_C_head)
        A = rearrange(A, "(h d) n -> h d n", h=self.num_C_head)
        D = rearrange(self.D, "(h d) -> h d", h=self.num_C_head)
        z = rearrange(z, "b (h d) -> b h d", h=self.num_C_head)
        dt_bias = rearrange(self.dt_proj.bias, "(h d) -> h d", h=self.num_C_head)

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


class AprielSSMDecoderLayer(nn.Module):
    _mixer_class = DiscreteMamba2

    def __init__(self, config: AprielHybridSSMConfig, layer_idx: int, device=None, dtype=None, **kwargs):
        super().__init__(**kwargs)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.hidden_size = config.hidden_size

        self.mixer = self._mixer_class(
            d_model=config.hidden_size,
            layer_idx=layer_idx,
            **config.ssm_cfg,
            **factory_kwargs,
        )

        self.mlp = MistralMLP(config)
        self.input_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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


class AprielSSMM2DecoderLayer(AprielSSMDecoderLayer):
    _mixer_class = Mamba2


class AprielHybridIdentity(nn.Module):
    def __init__(self, config: AprielHybridSSMConfig):
        super().__init__()
        self.config = config

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        return (hidden_states,)


class AprielHybridSSMModel(MistralModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`AprielDecoderLayer`, `AprielSSMDecoderLayer`]
    Args:
        config: AprielHybridSSMConfig
    """

    def __init__(self, config: AprielHybridSSMConfig, **kwargs):
        config_copy = copy.deepcopy(config)
        config_copy.num_hidden_layers = 0
        super().__init__(config_copy, **kwargs)
        self.config = config
        blocks = []
        logger.info(f"Loading hyubrid model with the following layout: {config.hybrid_block_layout}")
        for layer_idx, type in enumerate(config.hybrid_block_layout):
            if type == "m2d":
                blocks.append(AprielSSMDecoderLayer(config, layer_idx))
            elif type == "m2":
                blocks.append(AprielSSMM2DecoderLayer(config, layer_idx))
            elif type == "t":
                blocks.append(MistralDecoderLayer(config, layer_idx))
            elif type == "i":
                blocks.append(AprielHybridIdentity(config))
            else:
                raise ValueError(f"Invalid block type: {type}")
        self.layers = nn.ModuleList(blocks)

        # Initialize weights and apply final processing
        self.post_init()


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


class AprielHybridSSMPreTrainedModel(PreTrainedModel):
    config_class = AprielHybridSSMConfig
    base_model_prefix = "model"
    _no_split_modules = ["MistralDecoderLayer", "AprielSSMDecoderLayer", "AprielSSMM2DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
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
        elif isinstance(module, MistralRMSNorm):
            module.weight.data.fill_(1.0)


class AprielHybridSSMForCausalLM(AprielHybridSSMPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}

    def __init__(self, config: AprielHybridSSMConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = AprielHybridSSMModel(config)
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
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[tuple, CausalLMOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, MistralForCausalLM

        >>> model = MistralForCausalLM.from_pretrained("meta-mistral/Mistral-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-mistral/Mistral-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return AprielHybridCausalOutput(
            loss=loss,
            logits=logits,
            all_hidden_states=outputs.hidden_states,
            past_key_values=outputs.past_key_values,
        )


__all__ = [
    "AprielHybridSSMForCausalLM",
    "AprielHybridSSMModel",
    "AprielHybridSSMPreTrainedModel",
]
