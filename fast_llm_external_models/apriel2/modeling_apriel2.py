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
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import eager_attention_forward
from transformers.models.mistral.modeling_mistral import MistralMLP, MistralRMSNorm, apply_rotary_pos_emb
from transformers.processing_utils import Unpack
from transformers.utils import logging
from transformers.utils.import_utils import (
    is_causal_conv1d_available,
    is_mamba_ssm_available,
    is_torch_flex_attn_available,
)

from fast_llm_external_models.apriel2.cache import Apriel2Cache
from fast_llm_external_models.apriel2.configuration_apriel2 import Apriel2Config, Apriel2TextConfig

# GDN implementation - matches Fast-LLM's gdn.py exactly
try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
except ImportError:
    chunk_gated_delta_rule = None

try:
    from fla.modules.fused_norm_gate import rms_norm_gated
except ImportError:
    rms_norm_gated = None


is_fast_path_available = is_mamba_ssm_available() and is_causal_conv1d_available()

if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask
else:
    BlockMask = torch.Tensor

logger = logging.get_logger(__name__)

if not is_fast_path_available:
    logger.warning(
        "Mamba fast path not available. Requires CUDA, mamba_ssm, and causal_conv1d packages. "
        "Falling back to PyTorch implementation (slower, CPU-compatible)."
    )


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


@torch.compile
def torch_causal_conv1d_fn(x, weight, bias=None, activation="silu"):
    assert activation == "silu", f"Only silu activation is supported, got {activation}"

    seqlen = x.shape[-1]
    kernel_size = weight.shape[-1]

    # Causal padding and depthwise conv
    x = F.pad(x, (kernel_size - 1, 0))
    x = F.conv1d(x, weight.unsqueeze(1), bias=bias, groups=x.shape[1])
    x = x[..., :seqlen]

    return F.silu(x)


@torch.compile
def torch_causal_conv1d_update(x, conv_state, weight, bias=None, activation="silu"):
    assert activation == "silu", f"Only silu activation is supported, got {activation}"

    dtype = x.dtype
    conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
    conv_state[:, :, -1] = x
    x = torch.sum(conv_state * weight.unsqueeze(0), dim=-1)
    if bias is not None:
        x = x + bias
    return F.silu(x).to(dtype=dtype)


def torch_selective_scan_fn(
    u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=True, return_last_state=False
):
    raise NotImplementedError("torch_selective_scan_fn not yet implemented. Install mamba_ssm for CUDA kernels.")


def torch_selective_state_update(state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=True):
    raise NotImplementedError("torch_selective_state_update not yet implemented. Install mamba_ssm for CUDA kernels.")


if is_fast_path_available:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
else:
    causal_conv1d_fn = torch_causal_conv1d_fn
    causal_conv1d_update = torch_causal_conv1d_update
    selective_scan_fn = torch_selective_scan_fn
    selective_state_update = torch_selective_state_update


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

        # Whether to add biases to linear projections
        add_bias = mixer_config.get("add_linear_biases", False)

        # Projections (Fast-LLM weight names: q_proj, k_proj, v_proj, o_proj)
        self.q_proj = nn.Linear(d_model, self.num_heads * self.head_dim, bias=add_bias)
        self.k_proj = nn.Linear(d_model, self.num_key_value_heads * self.head_dim, bias=add_bias)
        self.v_proj = nn.Linear(d_model, self.num_key_value_heads * self.head_dim, bias=add_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, d_model, bias=add_bias)

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
    elif mixer_type == "kimi_linear_attention":
        return KimiLinearAttention
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
        # mamba, gdn, kimi_linear_attention all have same signature
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

        self.activation = "silu"  # Hardcoded for Mamba

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
        # Check for CUDA when using fast path
        if is_fast_path_available and "cuda" not in self.in_proj.weight.device.type:
            raise RuntimeError(
                "Mamba with CUDA kernels requires CUDA device. Current device: " + str(self.in_proj.weight.device)
            )

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

        if conv_state is not None:
            conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))

        # Compute short convolution
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


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """L2 normalization matching Fast-LLM's implementation."""
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    """Pure PyTorch fallback for chunk_gated_delta_rule - matches Fast-LLM's gdn.py."""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = (
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    )

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = (
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    )
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    # for each chunk
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


class GatedRMSNormalization(nn.Module):
    """
    Gated RMS normalization layer matching Fast-LLM's implementation.
    Uses fla.modules.fused_norm_gate.rms_norm_gated when available.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, input_: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        # Use PyTorch fallback on CPU since fla requires CUDA
        if rms_norm_gated is not None and input_.device.type != "cpu":
            return self._forward_fla(input_, gate)
        else:
            return self._forward_local(input_, gate)

    def _forward_fla(self, input_: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return rms_norm_gated(
            input_,
            gate,
            self.weight,
            None,
            activation="silu",
            eps=self.eps,
            residual=None,
            prenorm=False,
            residual_in_fp32=False,
        )

    def _forward_local(self, input_: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Pure PyTorch fallback for gated RMS normalization."""
        input_dtype = input_.dtype
        hidden_states = input_.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        return hidden_states * F.silu(gate)


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
        self.convolution = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
            device=device,
            dtype=dtype,
        )

        # Learnable parameters - match Fast-LLM initialization
        self.dt_bias = nn.Parameter(torch.ones(self.value_heads, device=device, dtype=dtype))
        self.A_log = nn.Parameter(torch.zeros(self.value_heads, device=device, dtype=dtype).uniform_(0, 16).log())

        # Normalization layer - named 'norm' with 'weight' param to match Fast-LLM
        self.norm = GatedRMSNormalization(self.value_head_dim, eps=self.norm_eps)

        # Select kernel implementation - fla if available, else torch fallback
        self._chunk_gated_delta_rule = chunk_gated_delta_rule or torch_chunk_gated_delta_rule

        if chunk_gated_delta_rule is None:
            logger.warning(
                "GatedDeltaNet fast path not available. Install fla library for optimized kernels. "
                "Falling back to PyTorch implementation."
            )

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

        # Project to QKVZ and BA
        mixed_qkvz = self.in_proj_qkvz(hidden_states)
        mixed_ba = self.in_proj_ba(hidden_states)

        # Split into components using Fast-LLM's flat layout
        query, key, value, z, beta, alpha = self._fix_query_key_value_ordering(mixed_qkvz, mixed_ba)

        # Flatten QKV for convolution (no Z in conv)
        query_flat = query.reshape(batch_size, seq_len, -1)
        key_flat = key.reshape(batch_size, seq_len, -1)
        value_flat = value.reshape(batch_size, seq_len, -1)
        mixed_qkv = torch.cat([query_flat, key_flat, value_flat], dim=-1)
        mixed_qkv = mixed_qkv.transpose(1, 2)  # [batch, conv_dim, seq]

        # Apply causal convolution
        if use_precomputed_states:
            # Single token update - use cached conv state
            # torch_causal_conv1d_update expects [batch, conv_dim] not [batch, conv_dim, 1]
            mixed_qkv = torch_causal_conv1d_update(
                mixed_qkv.squeeze(2),  # [batch, conv_dim, 1] -> [batch, conv_dim]
                conv_state,
                self.convolution.weight.squeeze(1),
                None,  # bias
                "silu",
            ).unsqueeze(
                2
            )  # [batch, conv_dim] -> [batch, conv_dim, 1]
        else:
            # Prefill - store padded state for future decoding
            if past_key_values is not None:
                # Pad to kernel size and store for future decoding
                padded = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                past_key_values.conv_states[self.layer_idx] = padded[:, :, -self.conv_kernel_size :]
            # Apply convolution
            mixed_qkv = F.silu(self.convolution(mixed_qkv)[:, :, :seq_len])

        mixed_qkv = mixed_qkv.transpose(1, 2)  # [batch, seq, conv_dim]

        # Split back after convolution
        query_flat, key_flat, value_flat = torch.split(mixed_qkv, (self.key_dim, self.key_dim, self.value_dim), dim=-1)
        query = query_flat.reshape(batch_size, seq_len, self.key_heads, self.key_head_dim)
        key = key_flat.reshape(batch_size, seq_len, self.key_heads, self.key_head_dim)
        value = value_flat.reshape(batch_size, seq_len, self.value_heads, self.value_head_dim)

        # Compute gating - match Fast-LLM exactly
        beta_gate = beta.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(alpha.float() + self.dt_bias)

        # Expand K heads to V heads if grouped query attention
        if self.value_heads_per_key > 1:
            query = query.repeat_interleave(self.value_heads_per_key, dim=2)
            key = key.repeat_interleave(self.value_heads_per_key, dim=2)

        # Run gated delta rule
        # Use PyTorch fallback on CPU since fla requires CUDA
        chunk_fn = self._chunk_gated_delta_rule
        if query.device.type == "cpu" and chunk_gated_delta_rule is not None:
            chunk_fn = torch_chunk_gated_delta_rule

        if not use_precomputed_states:
            # Chunked mode for prefill
            output, last_recurrent_state = chunk_fn(
                query,
                key,
                value,
                g=g,
                beta=beta_gate,
                initial_state=None,
                output_final_state=past_key_values is not None,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            # Recurrent mode for single token decode
            output, last_recurrent_state = self._recurrent_gated_delta_rule(
                query, key, value, g, beta_gate, recurrent_state
            )

        # Update recurrent state in cache
        if past_key_values is not None:
            past_key_values.recurrent_states[self.layer_idx] = last_recurrent_state

        # Apply gated normalization
        z_shape_og = z.shape
        output = output.reshape(-1, output.shape[-1])
        z_flat = z.reshape(-1, z.shape[-1])
        output = self.norm(output, z_flat)
        output = output.reshape(z_shape_og)
        output = output.reshape(output.shape[0], output.shape[1], -1)

        # Output projection
        output = self.out_proj(output)

        return (output,)

    def _recurrent_gated_delta_rule(self, query, key, value, g, beta, state):
        """Single-step recurrent update for cached inference."""
        # L2 normalize query and key
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)

        # Reshape for computation: [batch, heads, 1, dim] -> [batch, heads, dim]
        query = query.squeeze(2)
        key = key.squeeze(2)
        value = value.squeeze(2)
        g = g.squeeze(1)
        beta = beta.squeeze(1)

        # Update state: S = exp(g) * S + beta * k^T @ v
        decay = g.exp().unsqueeze(-1).unsqueeze(-1)  # [batch, heads, 1, 1]
        k_outer_v = torch.einsum("bhk,bhv->bhkv", key * beta.unsqueeze(-1), value)
        state = decay * state + k_outer_v

        # Output: o = q @ S
        output = torch.einsum("bhk,bhkv->bhv", query, state)
        output = output.unsqueeze(2)  # [batch, heads, 1, v_dim]

        return output, state

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

    @classmethod
    def setup(
        cls,
        mixer_config: dict,
        hidden_size: int,
        max_position_embeddings: int,
    ) -> nn.ModuleDict:
        """KimiLinearAttention setup not implemented."""
        raise NotImplementedError("KimiLinearAttention not yet implemented in apriel2")

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        raise NotImplementedError("KimiLinearAttention not yet implemented in apriel2")

    def preprocess(
        self,
        hidden_states: torch.Tensor,
        resources: Optional[nn.ModuleDict],
        **kwargs: Unpack[BlockSequenceKwargs],
    ) -> PreprocessingOutput:
        """KimiLinearAttention preprocessing not implemented."""
        raise NotImplementedError("KimiLinearAttention not yet implemented in apriel2")


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
        """Create MLP based on config."""
        mlp_type = mlp_config.get("type", "mlp")

        if mlp_type == "mlp":
            intermediate_size = mlp_config["intermediate_size"]
            activation = mlp_config.get("activation", "silu")
            gated = mlp_config["gated"]
            bias = mlp_config.get("add_linear_biases", False)

            if gated:
                mlp_cfg = SimpleNamespace(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    hidden_act=activation,
                )
                return MistralMLP(mlp_cfg)
            else:
                return SimpleMLP(hidden_size, intermediate_size, activation, bias)
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
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

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
        hidden_states = self.norm(hidden_states)

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

    def __init__(self, config: Apriel2TextConfig):
        super().__init__(config)
        self.model = Apriel2TextModel(config)
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
    """Non-gated MLP: up_proj -> activation -> down_proj."""

    def __init__(self, hidden_size: int, intermediate_size: int, activation: str = "silu", bias: bool = False):
        super().__init__()
        from transformers.activations import ACT2FN

        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
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
