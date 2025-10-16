import copy
import math
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
import torch.nn.functional as F
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from configuration_ssm_hybrid_apriel15b import AprielSSMHybridConfig
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from mamba_ssm.ops.triton.selective_state_update import selective_state_update
from torch import nn
from transformers import GenerationMixin
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer, MistralMLP, MistralModel, MistralRMSNorm
from transformers.processing_utils import Unpack
from transformers.utils import LossKwargs, can_return_tuple, logging
from transformers.utils.generic import ModelOutput

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
        is_empty_layer = (
            len(self.key_cache) == 0  # no cache in any layer
            or len(self.key_cache) <= layer_idx  # skipped `layer_idx` and hasn't run a layer with cache after it
            or not self.key_cache[layer_idx].numel()  # the layer has no cache
        )
        return self.key_cache[layer_idx].shape[-2] if not is_empty_layer else 0
        # return self.key_cache[layer_idx].shape[-2]

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
        mamba_mask: Optional[torch.Tensor] = None,
        return_mixer_matrix=False,
        **kwargs,
    ):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
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

        # Compute short convolution
        if conv_state is not None:
            # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
            # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
            # Update state (B D W)
            conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))
        if causal_conv1d_fn is None:
            x = self.act(self.conv1d(x)[..., :seqlen]).transpose(1, 2)
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

        zxbc = self.in_proj(hidden_states_input)
        z, x, B, C = torch.split(zxbc, [self.d_inner, self.d_xb, self.d_xb, self.d_inner], dim=-1)

        B = rearrange(B, "b (n_group dstate) -> b n_group dstate", dstate=self.d_state)
        B = torch.repeat_interleave(B, dim=1, repeats=self.repeat_group)
        C = rearrange(C, "b (n_group dstate) -> b n_group dstate", dstate=self.d_state).contiguous()

        dt = self.dt_proj(self.dt_in_proj(hidden_states_input))  # B, d_inner

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

        # Get states
        ssm_states = inference_params.ssm_states[self.layer_idx]
        conv_states = inference_params.conv_states[self.layer_idx]
        if initialize_states:
            ssm_states.zero_()
            conv_states.zero_()
        return ssm_states, conv_states


class AprielSSMM2DecoderLayer(nn.Module):
    _mixer_class = Mamba2

    def __init__(self, config: AprielSSMHybridConfig, layer_idx: int, device=None, dtype=None, **kwargs):
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

        outputs = (hidden_states,)

        return outputs


class AprielHybridIdentity(nn.Module):
    def __init__(self, config: AprielSSMHybridConfig):
        super().__init__()
        self.config = config

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        return (hidden_states,)


class AprielThinkerSSMHybridModel(MistralModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`AprielDecoderLayer`, `AprielSSMDecoderLayer`]
    Args:
        config: AprielSSMHybridConfig
    """

    def __init__(self, config: AprielSSMHybridConfig, **kwargs):
        config_copy = copy.deepcopy(config)
        config_copy.num_hidden_layers = 0
        super().__init__(config_copy, **kwargs)
        self.config = config
        blocks = []
        logger.info(f"Loading hyubrid model with the following layout: {config.hybrid_block_layout}")
        for layer_idx, type in enumerate(config.hybrid_block_layout):
            if type == "m2":
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

    @can_return_tuple
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
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if use_cache and past_key_values is None:
            # for the case where prepare_inputs_for_generation is not called to create the cache (as in fast-llm test)
            batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
            past_key_values = HybridMambaAttentionDynamicCache(self.config, batch_size, self.dtype, device=self.device)
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **flash_attn_kwargs,
        )
        past_key_values: HybridMambaAttentionDynamicCache = output.past_key_values
        if past_key_values and not past_key_values.has_previous_state:
            past_key_values.has_previous_state = True
        return output


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


class AprielThinkerSSMHybridPreTrainedModel(PreTrainedModel):
    config_class = AprielSSMHybridConfig
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


class AprielThinkerSSMHybridForCausalLM(AprielThinkerSSMHybridPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}

    def __init__(self, config: AprielSSMHybridConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = AprielThinkerSSMHybridModel(config)
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
            mamba_mask=attention_mask,  # non-expended mask
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
    "AprielThinkerSSMHybridForCausalLM",
    "AprielThinkerSSMHybridModel",
    "AprielThinkerSSMHybridPreTrainedModel",
]
