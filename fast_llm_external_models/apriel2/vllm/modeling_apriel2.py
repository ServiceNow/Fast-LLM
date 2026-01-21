# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Apriel2 model compatible with HuggingFace weights.

Apriel2 is a hybrid model that supports multiple mixer types (attention, mamba,
GatedDeltaNet, KDA) with flexible block patterns. This implementation is
optimized for vLLM inference.
"""

import logging
import math
from collections.abc import Iterable
from itertools import islice

import torch

logger = logging.getLogger(__name__)
import triton
from einops import rearrange
from torch import nn
from transformers import PretrainedConfig
from transformers.activations import ACT2FN

from vllm.v1.attention.backend import AttentionMetadata
from vllm.attention.layer import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import (
    CacheConfig,
    ModelConfig,
    SpeculativeConfig,
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed import (
    divide,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fla.ops import (
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
)
from vllm.model_executor.models.qwen3_next import (
    fused_gdn_gating as qwen3_fused_gdn_gating,
)
from vllm.model_executor.layers.fla.ops.kda import (
    FusedRMSNormGated,
    chunk_kda,
    fused_kda_gate,
    fused_recurrent_kda,
)

# Import to register kda_attention custom op
import vllm.model_executor.layers.kda  # noqa: F401
from vllm.model_executor.layers.layernorm import RMSNorm, RMSNormGated
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.selector import get_mamba_attn_backend
from vllm.model_executor.layers.mamba.mamba_mixer import MambaMixer
from vllm.model_executor.layers.mamba.mamba_mixer2 import mamba_v2_sharded_weight_loader
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from vllm.model_executor.models.interfaces import HasInnerState, SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
)
from vllm.logger import init_logger

apriel2_logger = init_logger(__name__)

# =============================================================================
# Debug Flags
# =============================================================================
# Top-level debug flags that control all debug output in the module.
# Set these to True to enable debugging for specific components.

DEBUG_GDN_LAYER = False      # Debug GDN layer forward pass (tensors, shapes)
DEBUG_GDN_STATE = False      # Debug GDN recurrent state during decode
DEBUG_GDN_OUTPUT = False     # Debug GDN output hidden states during decode
DEBUG_KDA_LAYER = False      # Debug KDA layer outputs
DEBUG_DECODER_LAYER = False  # Debug decoder layer outputs (residual, norm)
DEBUG_FINAL_NORM = False     # Debug final norm before LM head
DEBUG_LM_HEAD = False        # Debug LM head input/output


# =============================================================================
# KV Cache Spec Computation
# =============================================================================

from dataclasses import dataclass
from typing import Literal


# Valid mamba_type values for MambaSpec
MambaType = Literal["gdn_attention", "kda_attention", "mamba"]


def _get_dtype_size(dtype: torch.dtype) -> int:
    """Get size in bytes for a torch dtype."""
    if isinstance(dtype, str):
        # Handle string dtype names (e.g., "auto", "bfloat16")
        if dtype == "auto":
            dtype = torch.bfloat16  # Default to bfloat16
        else:
            dtype = getattr(torch, dtype, torch.bfloat16)
    return torch.tensor([], dtype=dtype).element_size()


@dataclass
class AttentionBlockParams:
    """Parameters for an attention block's KV cache."""
    num_kv_heads: int
    head_size: int
    window_size: int | None
    dtype: torch.dtype

    @property
    def page_size_per_token(self) -> int:
        """Bytes per token for K + V."""
        return 2 * self.num_kv_heads * self.head_size * _get_dtype_size(self.dtype)


@dataclass
class MambaBlockParams:
    """Parameters for a mamba-like block's state cache."""
    shapes: tuple
    dtypes: tuple
    mamba_type: MambaType

    @property
    def natural_page_size(self) -> int:
        """Natural page size based on state shapes."""
        return sum(
            _get_dtype_size(dtype) * math.prod(shape)
            for shape, dtype in zip(self.shapes, self.dtypes)
        )


BlockParams = AttentionBlockParams | MambaBlockParams


def _create_mixer_params(
    mixer_config: dict,
    mixer_type: str,
    vllm_config: VllmConfig,
) -> BlockParams | None:
    """Create BlockParams for a single mixer config.

    This is the single source of truth for converting a mixer config dict
    into typed BlockParams. Used by both top-level and stochastic mixer handling.

    Args:
        mixer_config: The mixer configuration dict.
        mixer_type: The mixer type string (attention, gdn, kda, mamba).
        vllm_config: The vLLM config for cache/parallel settings.

    Returns:
        BlockParams for this mixer, or None if the mixer type doesn't need cache.
    """
    cache_config = vllm_config.cache_config
    model_dtype = vllm_config.model_config.dtype
    tp_size = vllm_config.parallel_config.tensor_parallel_size

    if mixer_type == "attention" or mixer_type == "sliding_window":
        cache_dtype = cache_config.cache_dtype
        if cache_dtype is None or cache_dtype == "auto":
            kv_cache_dtype = model_dtype
        elif isinstance(cache_dtype, str):
            kv_cache_dtype = getattr(torch, cache_dtype, model_dtype)
        else:
            kv_cache_dtype = cache_dtype

        return AttentionBlockParams(
            num_kv_heads=mixer_config["head_groups"],
            head_size=mixer_config["head_size"],
            window_size=mixer_config.get("window_size"),
            dtype=kv_cache_dtype,
        )

    elif mixer_type == "gdn":
        shapes = MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_world_size=tp_size,
            num_k_heads=mixer_config["key_heads"],
            num_v_heads=mixer_config["value_heads"],
            head_k_dim=mixer_config["key_head_dim"],
            head_v_dim=mixer_config["value_head_dim"],
            conv_kernel_size=mixer_config["convolution_layer"]["kernel_size"],
            num_spec=0,
        )
        dtypes = MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            model_dtype,
            cache_config.mamba_cache_dtype,
        )
        return MambaBlockParams(
            shapes=shapes,
            dtypes=dtypes,
            mamba_type="gdn_attention",
        )

    elif mixer_type == "kda":
        shapes = MambaStateShapeCalculator.kda_state_shape(
            tp_world_size=tp_size,
            num_heads=mixer_config["heads"],
            head_dim=mixer_config["head_dim"],
            conv_kernel_size=mixer_config["convolution_layer"]["kernel_size"],
        )
        dtypes = MambaStateDtypeCalculator.kda_state_dtype(
            model_dtype,
            cache_config.mamba_cache_dtype,
        )
        return MambaBlockParams(
            shapes=shapes,
            dtypes=dtypes,
            mamba_type="kda_attention",
        )

    elif mixer_type == "mamba":
        d_state = mixer_config["state_size"]
        d_conv = mixer_config["d_conv"]
        d_inner = mixer_config.get("d_inner")
        if d_inner is None:
            raise ValueError("Mamba mixer must specify 'd_inner'")
        shapes = MambaStateShapeCalculator.mamba1_state_shape(
            tp_world_size=tp_size,
            intermediate_size=d_inner,
            state_size=d_state,
            conv_kernel=d_conv,
        )
        dtypes = MambaStateDtypeCalculator.mamba1_state_dtype(
            model_dtype,
            cache_config.mamba_cache_dtype,
            cache_config.mamba_ssm_cache_dtype,
        )
        return MambaBlockParams(
            shapes=shapes,
            dtypes=dtypes,
            mamba_type="mamba",
        )

    # Unknown mixer type - may not need cache
    return None


def get_block_params(
    blocks_config: dict[str, dict],
    vllm_config: VllmConfig,
) -> dict[str, BlockParams]:
    """Parse block configs and compute cache parameters ONCE.

    This is the single source of truth for shapes, dtypes, and page sizes.
    Downstream functions use these precomputed params.

    Args:
        blocks_config: Dict mapping block names to their configs.
        vllm_config: The vLLM config for cache/parallel settings.

    Returns:
        Dict mapping block names to their BlockParams.
    """
    params: dict[str, BlockParams] = {}

    for block_name, block_config in blocks_config.items():
        mixer_config = block_config.get("mixer", {})
        mixer_type = mixer_config.get("type", "attention")

        if mixer_type == "stochastic":
            # For stochastic mixers, compute params for ALL sub-mixers
            # This creates the "convex hull" of cache requirements so the unified
            # page size is large enough for any mixer type
            mixers = mixer_config.get("mixers", {})
            for sub_mixer_name, sub_mixer_config in mixers.items():
                sub_mixer_type = sub_mixer_config.get("type", "attention")
                sub_block_name = f"{block_name}.{sub_mixer_name}"
                sub_params = _create_mixer_params(sub_mixer_config, sub_mixer_type, vllm_config)
                if sub_params is not None:
                    params[sub_block_name] = sub_params
        else:
            # Regular (non-stochastic) mixer
            mixer_params = _create_mixer_params(mixer_config, mixer_type, vllm_config)
            if mixer_params is not None:
                params[block_name] = mixer_params
            else:
                raise ValueError(f"Block '{block_name}': unknown mixer type '{mixer_type}'")

    return params


def get_block_page_sizes(
    block_params: dict[str, BlockParams],
) -> tuple[int | None, dict[str, int]]:
    """Extract page sizes from precomputed block params.

    Args:
        block_params: Dict mapping block names to their BlockParams.

    Returns:
        Tuple of:
        - attn_page_per_token: Bytes per token for attention (None if no attention).
        - mamba_page_sizes: Dict mapping mamba block names to natural page sizes.
    """
    attn_page_per_token: int | None = None
    mamba_page_sizes: dict[str, int] = {}

    for block_name, params in block_params.items():
        if isinstance(params, AttentionBlockParams):
            # All attention blocks should have same head config
            attn_page_per_token = params.page_size_per_token
        elif isinstance(params, MambaBlockParams):
            mamba_page_sizes[block_name] = params.natural_page_size

    return attn_page_per_token, mamba_page_sizes


def unify_block_page_sizes(
    attn_page_per_token: int | None,
    mamba_page_sizes: dict[str, int],
    default_block_size: int = 16,
    alignment: int = 16,
) -> tuple[int, int]:
    """Compute unified (block_size, page_size) for all block types.

    The unified page_size must work for both attention (which scales with
    block_size) and mamba-like blocks (fixed state sizes). We achieve this by:
    1. Finding max mamba page size
    2. Computing block_size so attention page >= max mamba page
    3. Padding mamba pages to match attention page

    Args:
        attn_page_per_token: Bytes per token for attention (None if no attention).
        mamba_page_sizes: Dict of mamba-like block names to natural page sizes.
        default_block_size: Minimum block size for attention.
        alignment: Block size alignment (FlashAttention needs 16).

    Returns:
        Tuple of (block_size, unified_page_size).
    """
    # Pure attention model
    if not mamba_page_sizes:
        block_size = max(default_block_size, alignment)
        if attn_page_per_token is None:
            return block_size, 0
        return block_size, block_size * attn_page_per_token

    # Pure mamba model
    if attn_page_per_token is None:
        max_mamba_page = max(mamba_page_sizes.values())
        return default_block_size, max_mamba_page

    # Hybrid model: need to align attention and mamba page sizes
    max_mamba_page = max(mamba_page_sizes.values())

    # Compute minimum block_size so attention page >= max mamba page
    # attn_page = block_size * attn_page_per_token >= max_mamba_page
    min_block_size = -(-max_mamba_page // attn_page_per_token)  # ceiling division

    # Align to kernel requirements
    aligned_block_size = alignment * -(-min_block_size // alignment)

    # Use larger of default and computed
    block_size = max(default_block_size, aligned_block_size)

    # Unified page size (attention page, mamba will be padded to match)
    unified_page_size = block_size * attn_page_per_token

    apriel2_logger.info(
        "Page size unification: max_mamba=%d, attn_per_token=%d, "
        "block_size=%d, unified_page=%d",
        max_mamba_page, attn_page_per_token, block_size, unified_page_size
    )

    return block_size, unified_page_size


def get_blocks_config(decoder_config: dict) -> dict[str, dict]:
    """Extract the blocks config dict from a decoder config.

    Handles both 'fixed' (single block) and 'pattern' (multiple blocks) modes.

    Args:
        decoder_config: The decoder config dict from model config.

    Returns:
        Dict mapping block names to their configs.
    """
    seq_type = decoder_config.get("type", "fixed")

    if seq_type == "fixed":
        # Single block type - synthesize a name
        block_config = decoder_config.get("block", {})
        return {"block": block_config}
    elif seq_type == "pattern":
        return decoder_config.get("blocks", {})
    else:
        return {}


def get_unified_page_size_for_config(
    config: PretrainedConfig,
    vllm_config: VllmConfig,
) -> tuple[int, int]:
    """Compute unified (block_size, page_size) for the model config.

    This is used by layer-level get_kv_cache_spec() methods to ensure
    all layers return specs with matching page_size_bytes, even when
    vLLM iterates over layers individually (e.g., TransformersForCausalLM).

    Args:
        config: The HuggingFace model config.
        vllm_config: The vLLM config.

    Returns:
        Tuple of (block_size, unified_page_size).
    """
    decoder_config = getattr(config, "decoder", {}) or {}
    blocks_config = get_blocks_config(decoder_config)
    block_params = get_block_params(blocks_config, vllm_config)
    attn_page_per_token, mamba_page_sizes = get_block_page_sizes(block_params)
    return unify_block_page_sizes(attn_page_per_token, mamba_page_sizes)


class Apriel2Config(PretrainedConfig):
    """Configuration for Apriel2 models.

    This config supports both text-only and multimodal variants with
    flexible decoder block patterns (attention, mamba, GDN, KDA).
    """

    model_type = "apriel2"

    def __init__(
        self,
        vocab_size: int = 131072,
        hidden_size: int = 4096,
        intermediate_size: int = 14336,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 131072,
        rms_norm_eps: float = 1e-5,
        tie_word_embeddings: bool = True,
        rope_theta: float = 500000.0,
        rope_scaling: dict | None = None,
        # Apriel2 specific
        decoder: dict | None = None,
        embeddings: dict | None = None,
        head: dict | None = None,
        vision_encoder: dict | None = None,
        image_token_index: int | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        # Apriel2 specific configs
        self.decoder = decoder or {}
        self.embeddings = embeddings or {
            "max_position_embeddings": max_position_embeddings
        }
        self.head = head or {"normalization": {"epsilon": rms_norm_eps}}
        self.vision_encoder = vision_encoder
        self.image_token_index = image_token_index

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    @property
    def layers_block_type(self) -> list[str]:
        """Return block types for each layer (for hybrid model detection)."""
        decoder_config = self.decoder
        seq_type = decoder_config.get("type", "fixed")
        num_blocks = decoder_config.get("num_blocks", self.num_hidden_layers)

        if seq_type == "fixed":
            block_config = decoder_config.get("block", {})
            mixer_type = block_config.get("mixer", {}).get("type", "attention")
            return [mixer_type] * num_blocks
        elif seq_type == "pattern":
            pattern = decoder_config.get("pattern", ["attention"])
            blocks_config = decoder_config.get("blocks", {})
            result = []
            for i in range(num_blocks):
                block_name = pattern[i % len(pattern)]
                mixer_type = (
                    blocks_config.get(block_name, {})
                    .get("mixer", {})
                    .get("type", "attention")
                )
                result.append(mixer_type)
            return result
        return ["attention"] * num_blocks


class Apriel2MLP(nn.Module):
    """Apriel2 MLP with gated activation (SwiGLU style)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = set()
        for name, weight in weights:
            if name == "gate_proj.weight":
                self.gate_up_proj.weight_loader(self.gate_up_proj.weight, weight, 0)
                loaded.add("gate_up_proj.weight")
            elif name == "up_proj.weight":
                self.gate_up_proj.weight_loader(self.gate_up_proj.weight, weight, 1)
                loaded.add("gate_up_proj.weight")
            elif name == "down_proj.weight":
                self.down_proj.weight_loader(self.down_proj.weight, weight)
                loaded.add("down_proj.weight")
        return loaded


class Apriel2Attention(nn.Module):
    """Apriel2 attention layer with rotary embeddings and GQA support."""

    def __init__(
        self,
        config: Apriel2Config,
        mixer_config: dict,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        # Extract from mixer config (required)
        self.total_num_heads = mixer_config["heads"]
        self.total_num_kv_heads = mixer_config["head_groups"]
        self.head_dim = mixer_config["head_size"]

        tp_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        # Bias configuration - supports per-layer overrides
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

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=q_bias or k_bias or v_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=self.hidden_size,
            bias=o_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Rotary embeddings
        rotary_config = mixer_config.get("rotary", {})
        rope_theta = rotary_config["theta"]
        max_pos = config.embeddings["max_position_embeddings"]

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_pos,
            rope_parameters={"rope_theta": rope_theta},
        )

        # Sliding window support
        self.window_size = mixer_config.get("window_size", None)

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=self.window_size,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = set()
        for name, weight in weights:
            if name == "q_proj.weight":
                self.qkv_proj.weight_loader(self.qkv_proj.weight, weight, "q")
                loaded.add("qkv_proj.weight")
            elif name == "k_proj.weight":
                self.qkv_proj.weight_loader(self.qkv_proj.weight, weight, "k")
                loaded.add("qkv_proj.weight")
            elif name == "v_proj.weight":
                self.qkv_proj.weight_loader(self.qkv_proj.weight, weight, "v")
                loaded.add("qkv_proj.weight")
            elif name == "q_proj.bias":
                self.qkv_proj.weight_loader(self.qkv_proj.bias, weight, "q")
                loaded.add("qkv_proj.bias")
            elif name == "k_proj.bias":
                self.qkv_proj.weight_loader(self.qkv_proj.bias, weight, "k")
                loaded.add("qkv_proj.bias")
            elif name == "v_proj.bias":
                self.qkv_proj.weight_loader(self.qkv_proj.bias, weight, "v")
                loaded.add("qkv_proj.bias")
            elif name == "o_proj.weight":
                self.o_proj.weight_loader(self.o_proj.weight, weight)
                loaded.add("o_proj.weight")
            elif name == "o_proj.bias":
                self.o_proj.weight_loader(self.o_proj.bias, weight)
                loaded.add("o_proj.bias")
        return loaded

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        """Return cache spec for attention with unified page size for hybrid models."""
        config = vllm_config.model_config.hf_config
        block_size, _ = get_unified_page_size_for_config(config, vllm_config)

        # Get dtype from cache config
        cache_dtype = vllm_config.cache_config.cache_dtype
        if cache_dtype is None or cache_dtype == "auto":
            kv_cache_dtype = vllm_config.model_config.dtype
        elif isinstance(cache_dtype, str):
            kv_cache_dtype = getattr(torch, cache_dtype, vllm_config.model_config.dtype)
        else:
            kv_cache_dtype = cache_dtype

        if self.window_size is not None:
            return SlidingWindowSpec(
                block_size=block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_dim,
                dtype=kv_cache_dtype,
                sliding_window=self.window_size,
            )
        else:
            return FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_dim,
                dtype=kv_cache_dtype,
            )


class Apriel2MambaMixer(nn.Module):
    """Apriel2 Mamba mixer layer wrapping vLLM's MambaMixer."""

    def __init__(
        self,
        config: Apriel2Config,
        mixer_config: dict,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        # Extract mamba params from config - architecture values required
        d_state = mixer_config["state_size"]
        d_conv = mixer_config["d_conv"]
        expand = mixer_config.get("expand", None)
        d_inner = mixer_config.get("d_inner", None)
        if d_inner is None:
            if expand is None:
                raise ValueError("mixer_config must specify either 'd_inner' or 'expand'")
            d_inner = int(expand * config.hidden_size)
        dt_rank = mixer_config.get("dt_rank", "auto")
        if dt_rank == "auto":
            dt_rank = math.ceil(config.hidden_size / 16)

        conv_bias = mixer_config.get("conv_bias", True)
        bias = mixer_config.get("add_linear_biases", False)

        self.mamba = MambaMixer(
            hidden_size=config.hidden_size,
            ssm_state_size=d_state,
            conv_kernel_size=d_conv,
            intermediate_size=d_inner,
            time_step_rank=dt_rank,
            use_conv_bias=conv_bias,
            use_bias=bias,
            use_rms_norm=False,
            activation=mixer_config.get("activation", "silu"),
            model_config=model_config,
            cache_config=cache_config,
            prefix=prefix,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        self.mamba(hidden_states, output)


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """L2 normalization."""
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


# ============================================================================
# GDN custom op registration
# ============================================================================


def apriel2_gdn_attention_core(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self._forward_core(
        mixed_qkv=mixed_qkv,
        b=b,
        a=a,
        core_attn_out=core_attn_out,
    )


def apriel2_gdn_attention_core_fake(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="apriel2_gdn_attention_core",
    op_func=apriel2_gdn_attention_core,
    mutates_args=["core_attn_out"],
    fake_impl=apriel2_gdn_attention_core_fake,
)


@triton.jit
def fused_gdn_gating_kernel(
    A_log_ptr,
    a_ptr,
    b_ptr,
    dt_bias_ptr,
    g_ptr,
    beta_ptr,
    num_heads: tl.constexpr,
    total_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
):
    """Fused kernel for GDN gating computation."""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < total_elements

    # Load and convert to fp32 for math operations (exp/log require fp32/fp64)
    A_log = tl.load(A_log_ptr + offset % num_heads, mask=mask).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + offset % num_heads, mask=mask).to(tl.float32)
    a = tl.load(a_ptr + offset, mask=mask).to(tl.float32)
    b = tl.load(b_ptr + offset, mask=mask).to(tl.float32)

    # g = -exp(A_log) * softplus(a + dt_bias)
    # Use numerically stable softplus: for large x, softplus(x) ≈ x
    A = tl.exp(A_log)
    x = a + dt_bias
    softplus_val = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
    g = -A * softplus_val

    # beta = sigmoid(b)
    beta = tl.sigmoid(b)

    tl.store(g_ptr + offset, g, mask=mask)
    tl.store(beta_ptr + offset, beta, mask=mask)


def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GDN gating: g = -exp(A_log) * softplus(a + dt_bias), beta = sigmoid(b)"""
    batch_size = a.shape[0]
    num_heads = a.shape[-1]

    g = torch.empty_like(a, dtype=torch.float32)
    beta = torch.empty_like(b)

    total_elements = batch_size * num_heads
    BLOCK_SIZE = 256
    grid = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    fused_gdn_gating_kernel[grid](
        A_log,
        a.reshape(-1),
        b.reshape(-1),
        dt_bias,
        g.reshape(-1),
        beta.reshape(-1),
        num_heads,
        total_elements,
        BLOCK_SIZE,
        softplus_threshold,
    )

    g = g.unsqueeze(0)  # Add batch dim for chunk_gated_delta_rule
    beta = beta.unsqueeze(0)

    return g, beta


class Apriel2GatedDeltaNet(nn.Module, AttentionLayerBase):
    """Gated Delta Net mixer for Apriel2 using vLLM infrastructure.

    Inherits from AttentionLayerBase directly (not MambaBase) to avoid
    the global mamba_page_size_padded assumption that breaks heterogeneous
    block models like Apriel2.
    """

    # State cache set by vLLM's bind_kv_cache
    kv_cache: tuple[torch.Tensor, ...]

    @property
    def mamba_type(self) -> str:
        return "gdn_attention"

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        if self.model_config is None or self.cache_config is None:
            raise ValueError("model_config and cache_config must be set")
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype
        )

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            self.tp_size,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            self.conv_kernel_size,
            self.num_spec,
        )

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        """Return cache spec with unified page size for hybrid models.

        The unified page size ensures all layers (attention, mamba, GDN, KDA)
        have matching page_size_bytes, which is required by vLLM's KV cache
        management.
        """
        config = vllm_config.model_config.hf_config
        _, unified_page_size = get_unified_page_size_for_config(config, vllm_config)

        block_size = (
            vllm_config.cache_config.mamba_block_size
            or vllm_config.model_config.max_model_len
        )
        return MambaSpec(
            shapes=self.get_state_shape(),
            dtypes=self.get_state_dtype(),
            block_size=block_size,
            page_size_padded=unified_page_size,
            mamba_type=self.mamba_type,
            num_speculative_blocks=self.num_spec,
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        """Get the attention backend for GDN."""
        return get_mamba_attn_backend(self.mamba_type)

    def __init__(
        self,
        config: Apriel2Config,
        mixer_config: dict,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        speculative_config: SpeculativeConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.hidden_size = config.hidden_size

        # Config params - required architecture values
        self.num_v_heads = mixer_config["value_heads"]
        self.num_k_heads = mixer_config["key_heads"]
        self.head_k_dim = mixer_config["key_head_dim"]
        self.head_v_dim = mixer_config["value_head_dim"]
        conv_config = mixer_config["convolution_layer"]
        self.conv_kernel_size = conv_config["kernel_size"]
        # Internal defaults for implementation details
        self.layer_norm_epsilon = mixer_config.get("norm_eps", 1e-5)
        self.activation = conv_config.get("activation", "silu")
        self.act = ACT2FN[self.activation]

        self.layer_idx = layer_idx
        self.prefix = prefix
        self.model_config = model_config
        self.cache_config = cache_config
        self.quant_config = quant_config
        self.speculative_config = speculative_config
        self.num_spec = (
            self.speculative_config.num_speculative_tokens
            if self.speculative_config
            else 0
        )

        # Derived dimensions
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.value_heads_per_key = self.num_v_heads // self.num_k_heads

        # Convolution layer using vLLM's ColumnParallelLinear pattern
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_kernel_size,
            output_size=self.conv_dim,
            bias=False,
            prefix=f"{prefix}.conv1d",
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        # Input projections
        self.projection_size_qkvz = self.key_dim * 2 + self.value_dim * 2
        self.projection_size_ba = self.num_v_heads * 2

        self.in_proj_qkvz = ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=self.projection_size_qkvz,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj_qkvz",
        )
        self.in_proj_ba = ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=self.projection_size_ba,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj_ba",
        )

        # Set up weight loaders for conv1d
        query_key_settings = (self.key_dim, 0, False)
        value_settings = (self.value_dim, 0, False)

        delattr(self.conv1d.weight, "weight_loader")
        set_weight_attrs(
            self.conv1d.weight,
            {
                "weight_loader": mamba_v2_sharded_weight_loader(
                    [query_key_settings, query_key_settings, value_settings],
                    self.tp_size,
                    self.tp_rank,
                )
            },
        )

        # Time step and decay parameters
        self.dt_bias = nn.Parameter(
            torch.ones(self.num_v_heads // self.tp_size),
        )
        self.A_log = nn.Parameter(
            torch.empty(divide(self.num_v_heads, self.tp_size)),
        )

        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(0)})
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        # Output normalization and projection
        self.norm = RMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            group_size=None,
            norm_before_gate=True,
            device=current_platform.current_device(),
            dtype=config.torch_dtype if hasattr(config, 'torch_dtype') else None,
        )

        self.out_proj = RowParallelLinear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

        # Register with compilation context
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def fix_query_key_value_ordering(
        self,
        mixed_qkvz: torch.Tensor,
        mixed_ba: torch.Tensor,
    ):
        """Derives query, key, value, z, b, a tensors from projections.

        Uses flat layout matching Fast-LLM and transformers reference:
        - QKVZ: [Q_all | K_all | V_all | Z_all]
        - BA: [b_all | a_all]
        """
        num_tokens = mixed_qkvz.size(0)

        # Split QKVZ using flat layout (matching Fast-LLM/transformers reference)
        qkvz_sizes = (
            self.key_dim // self.tp_size,   # Q: key_heads * key_head_dim
            self.key_dim // self.tp_size,   # K: key_heads * key_head_dim
            self.value_dim // self.tp_size, # V: value_heads * value_head_dim
            self.value_dim // self.tp_size, # Z: value_heads * value_head_dim
        )
        query, key, value, z = torch.split(mixed_qkvz, qkvz_sizes, dim=-1)

        # Reshape to head format: [tokens, heads, head_dim]
        query = query.reshape(num_tokens, self.num_k_heads // self.tp_size, self.head_k_dim)
        key = key.reshape(num_tokens, self.num_k_heads // self.tp_size, self.head_k_dim)
        value = value.reshape(num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim)
        z = z.reshape(num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim)

        # Split BA using flat layout: [b_all | a_all]
        ba_sizes = (
            self.num_v_heads // self.tp_size,  # b (beta)
            self.num_v_heads // self.tp_size,  # a (alpha)
        )
        b, a = torch.split(mixed_ba, ba_sizes, dim=-1)

        return query, key, value, z, b, a

    def rearrange_mixed_qkv(self, mixed_qkv: torch.Tensor | None):
        if mixed_qkv is None:
            return None, None, None
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim // self.tp_size,
                self.key_dim // self.tp_size,
                self.value_dim // self.tp_size,
            ],
            dim=-1,
        )
        query, key = map(
            lambda x: rearrange(x, "l (h d) -> 1 l h d", d=self.head_k_dim),
            (query, key),
        )
        value = rearrange(value, "l (h d) -> 1 l h d", d=self.head_v_dim)
        return query.contiguous(), key.contiguous(), value.contiguous()

    def _debug_state_stats(self, name: str, state: torch.Tensor, seq_len: int):
        """Debug recurrent state with statistics."""
        if not DEBUG_GDN_STATE or state is None:
            return
        flat = state.flatten()
        first8 = ", ".join(f"{v:.6f}" for v in flat[:8].float().tolist())
        print(f"[vLLM-GDN {self.prefix}] {name} (seq_len={seq_len}): shape={state.shape}, "
              f"mean={state.float().mean().item():.6f}, std={state.float().std().item():.6f}, "
              f"min={state.float().min().item():.6f}, max={state.float().max().item():.6f}, "
              f"first8=[{first8}]")

    def _debug_tensor(self, name: str, t: torch.Tensor):
        if not DEBUG_GDN_LAYER:
            return
        if t is None:
            print(f"[GDN {self.prefix}] {name}: None")
            return
        flat = t.flatten()[:8]
        vals = ", ".join(f"{v:.6f}" for v in flat.float().tolist())
        print(f"[GDN {self.prefix}] {name}: shape={t.shape}, dtype={t.dtype}, "
              f"mean={t.float().mean().item():.6f}, std={t.float().std().item():.6f}, "
              f"first8=[{vals}]")

    def _debug_print(self, msg: str):
        if not DEBUG_GDN_LAYER:
            return
        print(f"[GDN {self.prefix}] {msg}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        """Forward pass with custom op for core attention."""
        num_tokens = hidden_states.size(0)

        # self._cached_hidden_states = hidden_states  # Cache for debug in _forward_core
        # self._debug_print(f"===== FORWARD START (num_tokens={num_tokens}) =====")
        # self._debug_tensor("hidden_states", hidden_states)

        # Part 1: Input Projection
        projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
        projected_states_ba, _ = self.in_proj_ba(hidden_states)
        # self._debug_tensor("projected_states_qkvz", projected_states_qkvz)
        # self._debug_tensor("projected_states_ba", projected_states_ba)

        query, key, value, z, b, a = self.fix_query_key_value_ordering(
            projected_states_qkvz, projected_states_ba
        )
        # self._debug_tensor("query (after fix_ordering)", query)
        # self._debug_tensor("key (after fix_ordering)", key)
        # self._debug_tensor("value (after fix_ordering)", value)
        # self._debug_tensor("z (after fix_ordering)", z)
        # self._debug_tensor("b (after fix_ordering)", b)
        # self._debug_tensor("a (after fix_ordering)", a)

        # Flatten heads: [tokens, heads, head_dim] -> [tokens, heads * head_dim]
        query = query.reshape(query.size(0), -1)
        key = key.reshape(key.size(0), -1)
        value = value.reshape(value.size(0), -1)
        mixed_qkv = torch.cat((query, key, value), dim=-1)
        # self._debug_tensor("mixed_qkv (flattened)", mixed_qkv)

        # Part 2: Core Attention (Custom Op)
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        torch.ops.vllm.apriel2_gdn_attention_core(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            self.prefix,
        )
        # self._debug_tensor("core_attn_out (after custom op)", core_attn_out)

        # Part 3: Output Projection
        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        # self._debug_tensor("core_attn_out (before norm)", core_attn_out)
        # self._debug_tensor("z (before norm)", z)
        # Debug last token before norm (reshaped has tokens * heads rows)
        if DEBUG_GDN_LAYER and num_tokens > 0:
            num_heads = self.num_v_heads // self.tp_size
            last_token_start = (num_tokens - 1) * num_heads
            last_attn = core_attn_out[last_token_start:last_token_start+1, :8]
            last_z = z[last_token_start:last_token_start+1, :8]
            print(f"[GDN {self.prefix}] core_attn_out before norm (last token, head 0): [{', '.join(f'{v:.6f}' for v in last_attn.flatten().float().tolist())}]")
            print(f"[GDN {self.prefix}] z before norm (last token, head 0): [{', '.join(f'{v:.6f}' for v in last_z.flatten().float().tolist())}]")
        # self._debug_tensor("norm.weight", self.norm.weight)
        # self._debug_print(f"norm.norm_before_gate={self.norm.norm_before_gate}, norm.eps={self.norm.eps}")
        core_attn_out = self.norm(core_attn_out, z)
        # self._debug_tensor("core_attn_out (after norm)", core_attn_out)
        # Debug last token after norm
        if DEBUG_GDN_LAYER and num_tokens > 0:
            last_attn_after = core_attn_out[last_token_start:last_token_start+1, :8]
            print(f"[GDN {self.prefix}] core_attn_out after norm (last token, head 0): [{', '.join(f'{v:.6f}' for v in last_attn_after.flatten().float().tolist())}]")
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        # self._debug_tensor("core_attn_out (before out_proj)", core_attn_out)
        output[:num_tokens], _ = self.out_proj(core_attn_out)
        # self._debug_tensor("output (final)", output[:num_tokens])
        # Show last token specifically
        if DEBUG_GDN_LAYER:
            last_token = output[num_tokens-1, :8]
            vals = ", ".join(f"{v:.6f}" for v in last_token.float().tolist())
            print(f"[GDN {self.prefix}] output (last token): last_token_first8=[{vals}]")
        # Debug output hidden states during decode (num_tokens == 1)
        if DEBUG_GDN_OUTPUT and num_tokens == 1:
            flat = output[:num_tokens].flatten()
            first8 = ", ".join(f"{v:.6f}" for v in flat[:8].float().tolist())
            print(f"[vLLM-GDN {self.prefix}] OUTPUT hs: shape={output[:num_tokens].shape}, mean={output[:num_tokens].float().mean().item():.6f}, std={output[:num_tokens].float().std().item():.6f}, first8=[{first8}]")
        # self._debug_print("===== FORWARD END =====")

    def _forward_core(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        """Core attention computation (called by custom op)."""
        # self._debug_print("===== _forward_core START =====")
        # self._debug_tensor("mixed_qkv (input to core)", mixed_qkv)
        # self._debug_tensor("b (input to core)", b)
        # self._debug_tensor("a (input to core)", a)

        forward_context = get_forward_context()
        attn_metadata: AttentionMetadata = forward_context.attn_metadata

        if attn_metadata is None:
            # self._debug_print("attn_metadata is None, returning early")
            return

        assert isinstance(attn_metadata, dict)
        attn_metadata = attn_metadata[self.prefix]
        assert isinstance(attn_metadata, GDNAttentionMetadata)

        has_initial_state = attn_metadata.has_initial_state
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor
        num_actual_tokens = attn_metadata.num_actual_tokens

        # self._debug_print(f"num_actual_tokens={num_actual_tokens}, num_prefills={attn_metadata.num_prefills}, num_decodes={attn_metadata.num_decodes}")
        # self._debug_print(f"has_initial_state={has_initial_state}")
        # self._debug_print(f"non_spec_query_start_loc={non_spec_query_start_loc}")

        self_kv_cache = self.kv_cache[forward_context.virtual_engine]
        conv_state = self_kv_cache[0].transpose(-1, -2)
        ssm_state = self_kv_cache[1]

        # self._debug_tensor("conv_state (from cache)", conv_state)
        # self._debug_tensor("ssm_state (from cache)", ssm_state)

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        # self._debug_tensor("mixed_qkv (truncated)", mixed_qkv)
        # self._debug_tensor("b (truncated)", b)
        # self._debug_tensor("a (truncated)", a)

        # Convolution
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        # self._debug_tensor("conv_weights", conv_weights)
        # self._debug_tensor("conv1d.bias", self.conv1d.bias)
        # self._debug_print(f"activation={self.activation}")

        if attn_metadata.num_prefills > 0:
            # self._debug_print("Using causal_conv1d_fn (prefill path)")
            mixed_qkv_T = mixed_qkv.transpose(0, 1)
            # self._debug_tensor("mixed_qkv_T (before conv)", mixed_qkv_T)
            mixed_qkv = causal_conv1d_fn(
                mixed_qkv_T,
                conv_weights,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                metadata=attn_metadata,
            ).transpose(0, 1)
        else:
            # self._debug_print("Using causal_conv1d_update (decode path)")
            mixed_qkv = causal_conv1d_update(
                mixed_qkv,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=non_spec_state_indices_tensor[:num_actual_tokens],
                validate_data=True,
            )

        # self._debug_tensor("mixed_qkv (after conv)", mixed_qkv)

        query, key, value = self.rearrange_mixed_qkv(mixed_qkv)
        # self._debug_tensor("query (after rearrange)", query)
        # self._debug_tensor("key (after rearrange)", key)
        # self._debug_tensor("value (after rearrange)", value)

        # Expand K heads to V heads for grouped query attention
        # (matches Fast-LLM and transformers reference implementations)
        # Always call repeat_interleave (no-op when value_heads_per_key == 1) to avoid
        # conditional branches that confuse torch.compile
        # self._debug_print(f"Expanding K heads to V heads (value_heads_per_key={self.value_heads_per_key})")
        query = query.repeat_interleave(self.value_heads_per_key, dim=2)
        key = key.repeat_interleave(self.value_heads_per_key, dim=2)
        # self._debug_tensor("query (after expand)", query)
        # self._debug_tensor("key (after expand)", key)

        # self._debug_tensor("A_log", self.A_log)
        # self._debug_tensor("dt_bias", self.dt_bias)
        g, beta = fused_gdn_gating(self.A_log, a, b, self.dt_bias)
        # self._debug_tensor("g (from gating)", g)
        # self._debug_tensor("beta (from gating)", beta)

        # Recurrent attention
        if attn_metadata.num_prefills > 0:
            # self._debug_print("Using chunk_gated_delta_rule (prefill)")
            initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()
            initial_state[~has_initial_state, ...] = 0
            # self._debug_tensor("initial_state", initial_state)
            # Debug PREFILL INPUTS before kernel call
            if DEBUG_GDN_STATE:
                print(f"[vLLM-GDN {self.prefix}] PREFILL INPUTS:")
                print(f"  hidden_states: shape={self._cached_hidden_states.shape}, first8={self._cached_hidden_states.flatten()[:8].tolist()}")
                print(f"  mixed_qkv (input): shape={mixed_qkv.shape}, first8={mixed_qkv.flatten()[:8].tolist()}")
                print(f"  q: shape={query.shape}, first8={query.flatten()[:8].tolist()}")
                print(f"  k: shape={key.shape}, first8={key.flatten()[:8].tolist()}")
                print(f"  v: shape={value.shape}, first8={value.flatten()[:8].tolist()}")
                print(f"  g: shape={g.shape}, first8={g.flatten()[:8].tolist()}")
                print(f"  beta: shape={beta.shape}, first8={beta.flatten()[:8].tolist()}")
                print(f"  initial_state: {initial_state}")
                print(f"  cu_seqlens: {non_spec_query_start_loc}")
            core_out, last_state = chunk_gated_delta_rule(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=non_spec_query_start_loc,
                head_first=False,
                use_qk_l2norm_in_kernel=True,
            )
            # self._debug_tensor("core_out (from chunk_gated_delta_rule)", core_out)
            # self._debug_tensor("last_state", last_state)
            # # Debug prefill state - get seq_len from query_start_loc
            # if non_spec_query_start_loc is not None and len(non_spec_query_start_loc) >= 2:
            #     prefill_seq_len = int(non_spec_query_start_loc[1] - non_spec_query_start_loc[0])
            # else:
            #     prefill_seq_len = num_actual_tokens
            # self._debug_state_stats("PREFILL out_state", last_state, prefill_seq_len)
            ssm_state[non_spec_state_indices_tensor] = last_state.to(ssm_state.dtype)
        else:
            # self._debug_print("Using fused_recurrent_gated_delta_rule (decode)")
            # # For decode, access the correct slot using state indices
            # if non_spec_state_indices_tensor is not None and len(non_spec_state_indices_tensor) > 0:
            #     slot_idx = int(non_spec_state_indices_tensor[0])
            #     actual_state = ssm_state[slot_idx:slot_idx+1]
            #     # self._debug_state_stats("DECODE in_state", actual_state, num_actual_tokens)
            # Debug decode inputs
            if DEBUG_GDN_STATE:
                print(f"[vLLM-GDN {self.prefix}] DECODE inputs: q={query.flatten()[:4].tolist()}, k={key.flatten()[:4].tolist()}, v={value.flatten()[:4].tolist()}, g={g.flatten()[:4].tolist()}, beta={beta.flatten()[:4].tolist()}")
            core_out, _ = fused_recurrent_gated_delta_rule(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                initial_state=ssm_state,
                inplace_final_state=True,
                cu_seqlens=non_spec_query_start_loc[:attn_metadata.num_decodes + 1],
                ssm_state_indices=non_spec_state_indices_tensor,
                use_qk_l2norm_in_kernel=True,
            )
            # self._debug_tensor("core_out (from fused_recurrent)", core_out)
            # if non_spec_state_indices_tensor is not None and len(non_spec_state_indices_tensor) > 0:
            #     actual_state = ssm_state[slot_idx:slot_idx+1]
            #     # self._debug_state_stats("DECODE out_state", actual_state, num_actual_tokens)

        core_attn_out[:num_actual_tokens] = core_out.squeeze(0)[:num_actual_tokens]
        # self._debug_tensor("core_attn_out (final output)", core_attn_out)
        # self._debug_print("===== _forward_core END =====")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Checkpoint uses "convolution", model uses "conv1d"
        loaded = set()
        for name, weight in weights:
            if name == "convolution.weight":
                self.conv1d.weight_loader(self.conv1d.weight, weight)
                loaded.add("conv1d.weight")
            elif name == "in_proj_qkvz.weight":
                self.in_proj_qkvz.weight_loader(self.in_proj_qkvz.weight, weight)
                loaded.add("in_proj_qkvz.weight")
            elif name == "in_proj_ba.weight":
                self.in_proj_ba.weight_loader(self.in_proj_ba.weight, weight)
                loaded.add("in_proj_ba.weight")
            elif name == "out_proj.weight":
                self.out_proj.weight_loader(self.out_proj.weight, weight)
                loaded.add("out_proj.weight")
            elif name == "norm.weight":
                self.norm.weight.data.copy_(weight)
                loaded.add("norm.weight")
            elif name == "A_log":
                self.A_log.data.copy_(weight)
                loaded.add("A_log")
            elif name == "dt_bias":
                self.dt_bias.data.copy_(weight)
                loaded.add("dt_bias")
        return loaded


class Apriel2KDAMixer(nn.Module, AttentionLayerBase):
    """Kimi Delta Attention mixer for Apriel2 using vLLM's KDA infrastructure.

    Inherits from AttentionLayerBase directly (not MambaBase) to avoid
    the global mamba_page_size_padded assumption that breaks heterogeneous
    block models like Apriel2.
    """

    # State cache set by vLLM's bind_kv_cache
    kv_cache: tuple[torch.Tensor, ...]

    @property
    def mamba_type(self) -> str:
        # Use "gdn_attention" to match vLLM's KDA backend registration
        return "gdn_attention"

    def get_state_dtype(
        self,
    ) -> tuple[torch.dtype, torch.dtype, torch.dtype, torch.dtype]:
        if self.model_config is None or self.cache_config is None:
            raise ValueError("model_config and cache_config must be set")
        return MambaStateDtypeCalculator.kda_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype
        )

    def get_state_shape(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.kda_state_shape(
            self.tp_size, self.num_heads, self.head_dim, conv_kernel_size=self.conv_size
        )

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        """Return cache spec with unified page size for hybrid models.

        The unified page size ensures all layers (attention, mamba, GDN, KDA)
        have matching page_size_bytes, which is required by vLLM's KV cache
        management.
        """
        config = vllm_config.model_config.hf_config
        _, unified_page_size = get_unified_page_size_for_config(config, vllm_config)

        block_size = (
            vllm_config.cache_config.mamba_block_size
            or vllm_config.model_config.max_model_len
        )
        return MambaSpec(
            shapes=self.get_state_shape(),
            dtypes=self.get_state_dtype(),
            block_size=block_size,
            page_size_padded=unified_page_size,
            mamba_type=self.mamba_type,
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        """Get the attention backend for KDA."""
        return get_mamba_attn_backend(self.mamba_type)

    def __init__(
        self,
        config: Apriel2Config,
        mixer_config: dict,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.hidden_size = config.hidden_size
        self.model_config = model_config
        self.cache_config = cache_config

        # Extract KDA config params - architecture values required
        self.num_heads = mixer_config["heads"]
        self.head_dim = mixer_config["head_dim"]
        conv_config = mixer_config["convolution_layer"]
        self.conv_size = conv_config["kernel_size"]
        # Internal defaults for implementation details
        norm_config = mixer_config.get("normalization", {})
        rms_norm_eps = norm_config.get("epsilon", 1e-6)
        norm_activation = norm_config.get("activation", "silu")

        self.layer_idx = layer_idx
        self.prefix = prefix

        assert self.num_heads % self.tp_size == 0
        self.local_num_heads = divide(self.num_heads, self.tp_size)
        projection_size = self.head_dim * self.num_heads

        # Use vLLM's parallel layers
        self.q_proj = ColumnParallelLinear(
            self.hidden_size,
            projection_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )
        self.k_proj = ColumnParallelLinear(
            self.hidden_size,
            projection_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.k_proj",
        )
        self.v_proj = ColumnParallelLinear(
            self.hidden_size,
            projection_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.v_proj",
        )

        self.f_a_proj = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.f_a_proj",
        )
        self.f_b_proj = ColumnParallelLinear(
            self.head_dim,
            projection_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.f_b_proj",
        )
        self.dt_bias = nn.Parameter(
            torch.empty(divide(projection_size, self.tp_size), dtype=torch.float32)
        )
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        self.b_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.b_proj",
        )

        # Convolutions as parallel linears
        self.q_conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.q_conv1d",
        )
        self.k_conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.k_conv1d",
        )
        self.v_conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.v_conv1d",
        )
        # Shape conv weights correctly
        self.q_conv1d.weight.data = self.q_conv1d.weight.data.unsqueeze(1)
        self.k_conv1d.weight.data = self.k_conv1d.weight.data.unsqueeze(1)
        self.v_conv1d.weight.data = self.v_conv1d.weight.data.unsqueeze(1)

        # Store A_log as 1D to match checkpoint format - fused_kda_gate accepts [H] or [1,1,H,1]
        self.A_log = nn.Parameter(
            torch.empty(self.local_num_heads, dtype=torch.float32)
        )
        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(0)})

        self.g_a_proj = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.g_a_proj",
        )
        self.g_b_proj = ColumnParallelLinear(
            self.head_dim,
            projection_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.g_b_proj",
        )
        self.o_norm = FusedRMSNormGated(
            self.head_dim, eps=rms_norm_eps, activation=norm_activation
        )
        self.o_proj = RowParallelLinear(
            projection_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Register with compilation context
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        num_tokens = hidden_states.size(0)
        q = self.q_proj(hidden_states)[0]
        k = self.k_proj(hidden_states)[0]
        v = self.v_proj(hidden_states)[0]

        beta = self.b_proj(hidden_states)[0].float().sigmoid()
        g1 = self.f_b_proj(self.f_a_proj(hidden_states)[0])[0]
        g1 = fused_kda_gate(g1, self.A_log, self.head_dim, g_bias=self.dt_bias)
        beta = beta.unsqueeze(0)
        g1 = g1.unsqueeze(0)

        g_proj_states = self.g_b_proj(self.g_a_proj(hidden_states)[0])[0]
        g2 = rearrange(g_proj_states, "... (h d) -> ... h d", d=self.head_dim)

        core_attn_out = torch.zeros(
            (1, num_tokens, self.local_num_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        torch.ops.vllm.kda_attention(
            q,
            k,
            v,
            g1,
            beta,
            core_attn_out,
            self.prefix,
        )
        core_attn_out = self.o_norm(core_attn_out, g2)
        core_attn_out = rearrange(core_attn_out, "1 n h d -> n (h d)")
        output[:] = self.o_proj(core_attn_out)[0]

    def _forward(
        self,
        q_proj_states: torch.Tensor,
        k_proj_states: torch.Tensor,
        v_proj_states: torch.Tensor,
        g1: torch.Tensor,
        beta: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata: AttentionMetadata = forward_context.attn_metadata

        if attn_metadata is None:
            return

        assert isinstance(attn_metadata, dict)
        attn_metadata = attn_metadata[self.prefix]
        assert isinstance(attn_metadata, GDNAttentionMetadata)
        has_initial_state = attn_metadata.has_initial_state
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor
        num_actual_tokens = attn_metadata.num_actual_tokens
        constant_caches = self.kv_cache[forward_context.virtual_engine]

        q_proj_states = q_proj_states[:num_actual_tokens]
        k_proj_states = k_proj_states[:num_actual_tokens]
        v_proj_states = v_proj_states[:num_actual_tokens]
        g1 = g1[:num_actual_tokens]
        beta = beta[:num_actual_tokens]

        (conv_state_q, conv_state_k, conv_state_v, recurrent_state) = constant_caches
        conv_state_q = conv_state_q.transpose(-1, -2)
        conv_state_k = conv_state_k.transpose(-1, -2)
        conv_state_v = conv_state_v.transpose(-1, -2)

        q_conv_weights = self.q_conv1d.weight.view(
            self.q_conv1d.weight.size(0), self.q_conv1d.weight.size(2)
        )
        k_conv_weights = self.k_conv1d.weight.view(
            self.k_conv1d.weight.size(0), self.k_conv1d.weight.size(2)
        )
        v_conv_weights = self.v_conv1d.weight.view(
            self.v_conv1d.weight.size(0), self.v_conv1d.weight.size(2)
        )

        if attn_metadata.num_prefills > 0:
            q_proj_states = q_proj_states.transpose(0, 1)
            k_proj_states = k_proj_states.transpose(0, 1)
            v_proj_states = v_proj_states.transpose(0, 1)
            q = causal_conv1d_fn(
                q_proj_states,
                q_conv_weights,
                self.q_conv1d.bias,
                activation="silu",
                conv_states=conv_state_q,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                metadata=attn_metadata,
            ).transpose(0, 1)
            k = causal_conv1d_fn(
                k_proj_states,
                k_conv_weights,
                self.k_conv1d.bias,
                activation="silu",
                conv_states=conv_state_k,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                metadata=attn_metadata,
            ).transpose(0, 1)
            v = causal_conv1d_fn(
                v_proj_states,
                v_conv_weights,
                self.v_conv1d.bias,
                activation="silu",
                conv_states=conv_state_v,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                metadata=attn_metadata,
            ).transpose(0, 1)
        else:
            decode_conv_indices = non_spec_state_indices_tensor[:num_actual_tokens]
            q = causal_conv1d_update(
                q_proj_states,
                conv_state_q,
                q_conv_weights,
                self.q_conv1d.bias,
                activation="silu",
                conv_state_indices=decode_conv_indices,
                validate_data=True,
            )
            k = causal_conv1d_update(
                k_proj_states,
                conv_state_k,
                k_conv_weights,
                self.k_conv1d.bias,
                activation="silu",
                conv_state_indices=decode_conv_indices,
                validate_data=True,
            )
            v = causal_conv1d_update(
                v_proj_states,
                conv_state_v,
                v_conv_weights,
                self.v_conv1d.bias,
                activation="silu",
                conv_state_indices=decode_conv_indices,
                validate_data=True,
            )

        q, k, v = map(
            lambda x: rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim), (q, k, v)
        )

        if attn_metadata.num_prefills > 0:
            zero_idx = non_spec_state_indices_tensor[~has_initial_state]
            recurrent_state[zero_idx] = 0
            initial_state = recurrent_state[non_spec_state_indices_tensor].contiguous()
            core_attn_out_non_spec, last_recurrent_state = chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g1,
                beta=beta,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=non_spec_query_start_loc,
            )
            recurrent_state[non_spec_state_indices_tensor] = last_recurrent_state
        else:
            core_attn_out_non_spec, _ = fused_recurrent_kda(
                q=q,
                k=k,
                v=v,
                g=g1,
                beta=beta,
                initial_state=recurrent_state,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=non_spec_query_start_loc[:attn_metadata.num_decodes + 1],
                ssm_state_indices=non_spec_state_indices_tensor,
            )
        core_attn_out[0, :num_actual_tokens] = core_attn_out_non_spec[0, :num_actual_tokens]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Checkpoint to model name translations:
        # beta_proj → b_proj, q_conv → q_conv1d, k_conv → k_conv1d, v_conv → v_conv1d, norm → o_norm
        loaded = set()
        for name, weight in weights:
            if name == "beta_proj.weight":
                self.b_proj.weight_loader(self.b_proj.weight, weight)
                loaded.add("b_proj.weight")
            elif name == "q_conv.weight":
                self.q_conv1d.weight_loader(self.q_conv1d.weight, weight)
                loaded.add("q_conv1d.weight")
            elif name == "k_conv.weight":
                self.k_conv1d.weight_loader(self.k_conv1d.weight, weight)
                loaded.add("k_conv1d.weight")
            elif name == "v_conv.weight":
                self.v_conv1d.weight_loader(self.v_conv1d.weight, weight)
                loaded.add("v_conv1d.weight")
            elif name == "norm.weight":
                self.o_norm.weight.data.copy_(weight)
                loaded.add("o_norm.weight")
            elif name == "q_proj.weight":
                self.q_proj.weight_loader(self.q_proj.weight, weight)
                loaded.add("q_proj.weight")
            elif name == "k_proj.weight":
                self.k_proj.weight_loader(self.k_proj.weight, weight)
                loaded.add("k_proj.weight")
            elif name == "v_proj.weight":
                self.v_proj.weight_loader(self.v_proj.weight, weight)
                loaded.add("v_proj.weight")
            elif name == "o_proj.weight":
                self.o_proj.weight_loader(self.o_proj.weight, weight)
                loaded.add("o_proj.weight")
            elif name == "f_a_proj.weight":
                self.f_a_proj.weight_loader(self.f_a_proj.weight, weight)
                loaded.add("f_a_proj.weight")
            elif name == "f_b_proj.weight":
                self.f_b_proj.weight_loader(self.f_b_proj.weight, weight)
                loaded.add("f_b_proj.weight")
            elif name == "g_a_proj.weight":
                self.g_a_proj.weight_loader(self.g_a_proj.weight, weight)
                loaded.add("g_a_proj.weight")
            elif name == "g_b_proj.weight":
                self.g_b_proj.weight_loader(self.g_b_proj.weight, weight)
                loaded.add("g_b_proj.weight")
            elif name == "A_log":
                self.A_log.data.copy_(weight)
                loaded.add("A_log")
            elif name == "dt_bias":
                self.dt_bias.data.copy_(weight)
                loaded.add("dt_bias")
        return loaded


class Apriel2AttentionDecoderLayer(nn.Module):
    """Attention-based decoder layer for Apriel2."""

    def __init__(
        self,
        config: Apriel2Config,
        layer_idx: int,
        block_config: dict,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        mixer_config = block_config.get("mixer", {})
        mlp_config = block_config.get("mlp", {})
        norm_config = block_config.get("normalization", {})

        self.mixer = Apriel2Attention(
            config=config,
            mixer_config=mixer_config,
            layer_idx=layer_idx,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.mixer",
        )

        intermediate_size = mlp_config["intermediate_size"]
        mlp_bias = mlp_config.get("add_linear_biases", False)
        hidden_act = mlp_config.get("activation", "silu")
        rms_norm_eps = norm_config["epsilon"]

        self.mlp = Apriel2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            quant_config=quant_config,
            bias=mlp_bias,
            prefix=f"{prefix}.mlp",
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.mixer(positions=positions, hidden_states=hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Apriel2MambaDecoderLayer(nn.Module):
    """Mamba-based decoder layer for Apriel2."""

    def __init__(
        self,
        config: Apriel2Config,
        layer_idx: int,
        block_config: dict,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        mixer_config = block_config.get("mixer", {})
        mlp_config = block_config.get("mlp", {})
        norm_config = block_config.get("normalization", {})

        self.mixer = Apriel2MambaMixer(
            config=config,
            mixer_config=mixer_config,
            layer_idx=layer_idx,
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.mixer",
        )

        intermediate_size = mlp_config["intermediate_size"]
        mlp_bias = mlp_config.get("add_linear_biases", False)
        hidden_act = mlp_config.get("activation", "silu")
        rms_norm_eps = norm_config["epsilon"]

        self.mlp = Apriel2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            quant_config=quant_config,
            bias=mlp_bias,
            prefix=f"{prefix}.mlp",
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        output = torch.empty_like(hidden_states)
        self.mixer(hidden_states, output)
        hidden_states, residual = self.post_attention_layernorm(output, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Apriel2GDNDecoderLayer(nn.Module):
    """GatedDeltaNet-based decoder layer for Apriel2."""

    def __init__(
        self,
        config: Apriel2Config,
        layer_idx: int,
        block_config: dict,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        speculative_config: SpeculativeConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        mixer_config = block_config.get("mixer", {})
        mlp_config = block_config.get("mlp", {})
        norm_config = block_config.get("normalization", {})

        self.mixer = Apriel2GatedDeltaNet(
            config=config,
            mixer_config=mixer_config,
            layer_idx=layer_idx,
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            speculative_config=speculative_config,
            prefix=f"{prefix}.mixer",
        )

        intermediate_size = mlp_config["intermediate_size"]
        mlp_bias = mlp_config.get("add_linear_biases", False)
        hidden_act = mlp_config.get("activation", "silu")
        rms_norm_eps = norm_config["epsilon"]

        self.mlp = Apriel2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            quant_config=quant_config,
            bias=mlp_bias,
            prefix=f"{prefix}.mlp",
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=rms_norm_eps
        )

    def _debug_tensor(self, name: str, t: torch.Tensor, show_last=False):
        if not DEBUG_DECODER_LAYER or t is None:
            return
        if show_last:
            # Show last token
            last = t[-1, :8] if t.dim() == 2 else t[0, -1, :8]
            vals = ", ".join(f"{v:.6f}" for v in last.float().tolist())
            print(f"[vLLM Layer] {name}: shape={t.shape}, last_token_first8=[{vals}]")
        else:
            flat = t.flatten()[:8]
            vals = ", ".join(f"{v:.6f}" for v in flat.float().tolist())
            print(f"[vLLM Layer] {name}: shape={t.shape}, first8=[{vals}]")

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # self._debug_tensor("input hidden_states", hidden_states)
        # self._debug_tensor("input residual", residual)

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # self._debug_tensor("after input_layernorm", hidden_states)
        # self._debug_tensor("residual after input_layernorm", residual)

        output = torch.empty_like(hidden_states)
        self.mixer(hidden_states, output)
        # self._debug_tensor("mixer output", output)

        hidden_states, residual = self.post_attention_layernorm(output, residual)
        # self._debug_tensor("after post_attention_layernorm", hidden_states)
        # self._debug_tensor("residual after post_attention_layernorm", residual)

        hidden_states = self.mlp(hidden_states)
        # self._debug_tensor("after mlp", hidden_states)
        # Also show last token for final layer comparison
        # self._debug_tensor("after mlp (last token)", hidden_states, show_last=True)
        # self._debug_tensor("residual (last token)", residual, show_last=True)

        return hidden_states, residual


class Apriel2KDADecoderLayer(nn.Module):
    """KDA-based decoder layer for Apriel2."""

    def __init__(
        self,
        config: Apriel2Config,
        layer_idx: int,
        block_config: dict,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        mixer_config = block_config.get("mixer", {})
        mlp_config = block_config.get("mlp", {})
        norm_config = block_config.get("normalization", {})

        self.mixer = Apriel2KDAMixer(
            config=config,
            mixer_config=mixer_config,
            layer_idx=layer_idx,
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.mixer",
        )

        intermediate_size = mlp_config["intermediate_size"]
        mlp_bias = mlp_config.get("add_linear_biases", False)
        hidden_act = mlp_config.get("activation", "silu")
        rms_norm_eps = norm_config["epsilon"]

        self.mlp = Apriel2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            quant_config=quant_config,
            bias=mlp_bias,
            prefix=f"{prefix}.mlp",
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        output = torch.empty_like(hidden_states)
        self.mixer(hidden_states, output)
        hidden_states, residual = self.post_attention_layernorm(output, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Apriel2StochasticMixer(nn.Module):
    """Stochastic mixer that contains multiple sub-mixers.

    At inference time, routes inputs to the active mixer (configurable).
    All sub-mixer weights are loaded and available for runtime switching.

    Each sub-mixer gets a unique virtual layer index for cache registration,
    similar to Falcon H1's approach. This allows each mixer type to have its
    own cache allocation without conflicts.
    """

    # Map mixer type to (mixer_class, needs_model_config, needs_speculative_config)
    MIXER_REGISTRY: dict[str, tuple[type, bool, bool]] = {
        "attention": (Apriel2Attention, False, False),
        "mamba": (Apriel2MambaMixer, True, False),
        "gdn": (Apriel2GatedDeltaNet, True, True),
        "kda": (Apriel2KDAMixer, True, False),
    }

    def __init__(
        self,
        config: Apriel2Config,
        mixer_config: dict,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        speculative_config: SpeculativeConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        # Get sub-mixer configs
        mixers_config = mixer_config.get("mixers", {})
        self.active_mixer_name = mixer_config.get("main_mixer_name", list(mixers_config.keys())[0])

        # Get total number of layers for computing virtual layer indices
        decoder_config = getattr(config, "decoder", {}) or {}
        num_layers = decoder_config["num_blocks"]

        # Parse the prefix to extract base path (e.g., "model.layers" from "model.layers.0.mixer")
        # prefix format: "model.layers.{layer_idx}.mixer"
        prefix_parts = prefix.rsplit(".", 2)  # ["model.layers", "0", "mixer"]
        if len(prefix_parts) >= 3:
            layers_base = prefix_parts[0]  # "model.layers"
        else:
            layers_base = "model.layers"

        # Create all sub-mixers with unique virtual layer indices
        # Each sub-mixer gets a unique offset based on its position (not type)
        # to avoid collisions when multiple mixers have the same type
        self.mixers = nn.ModuleDict()
        for mixer_index, (name, sub_mixer_config) in enumerate(mixers_config.items()):
            sub_mixer_type = sub_mixer_config.get("type", "attention")

            if sub_mixer_type not in self.MIXER_REGISTRY:
                raise ValueError(f"Unknown sub-mixer type '{sub_mixer_type}' in stochastic mixer")

            mixer_class, needs_model_config, needs_spec_config = self.MIXER_REGISTRY[sub_mixer_type]

            # Compute virtual layer index using mixer's position index (Falcon H1 style)
            # Each sub-mixer gets its own "virtual layer" range: layer_idx + (index+1) * num_layers
            # This ensures unique indices even when multiple mixers have the same type
            virtual_layer_idx = layer_idx + (mixer_index + 1) * num_layers

            # Build prefix with virtual layer index for cache registration
            # This only affects static_forward_context registration, not weight loading
            virtual_prefix = f"{layers_base}.{virtual_layer_idx}.stochastic_{name}"

            # Build kwargs based on what each mixer type needs
            kwargs = {
                "config": config,
                "mixer_config": sub_mixer_config,
                "layer_idx": layer_idx,  # Keep real layer_idx for any internal use
                "cache_config": cache_config,
                "quant_config": quant_config,
                "prefix": virtual_prefix,
            }
            if needs_model_config:
                kwargs["model_config"] = model_config
            if needs_spec_config:
                kwargs["speculative_config"] = speculative_config

            self.mixers[name] = mixer_class(**kwargs)
            logger.debug(
                f"Created sub-mixer '{name}' (type={sub_mixer_type}) at virtual layer {virtual_layer_idx} "
                f"(real layer {layer_idx}, prefix={virtual_prefix})"
            )

        self._mixer_names = list(self.mixers.keys())
        logger.info(
            f"Initialized Apriel2StochasticMixer at layer {layer_idx} with {len(self.mixers)} mixers: "
            f"{', '.join(self._mixer_names)} (active={self.active_mixer_name})"
        )

    def set_active_mixer(self, name: str) -> None:
        """Set the active mixer by name."""
        if name not in self.mixers:
            raise ValueError(f"Unknown mixer '{name}'. Available: {self._mixer_names}")
        self.active_mixer_name = name

    def get_active_mixer(self) -> str:
        """Get the name of the currently active mixer."""
        return self.active_mixer_name

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Forward through the active mixer."""
        mixer = self.mixers[self.active_mixer_name]
        return mixer(positions=positions, hidden_states=hidden_states, **kwargs)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights for all sub-mixers."""
        loaded = set()
        # Group weights by sub-mixer name
        weights_by_mixer: dict[str, list[tuple[str, torch.Tensor]]] = {name: [] for name in self.mixers}

        for name, weight in weights:
            # Weight names are like "mixers.attention.q_proj.weight"
            if name.startswith("mixers."):
                parts = name.split(".", 2)  # ["mixers", "attention", "q_proj.weight"]
                if len(parts) >= 3:
                    mixer_name = parts[1]
                    param_name = parts[2]
                    if mixer_name in weights_by_mixer:
                        weights_by_mixer[mixer_name].append((param_name, weight))

        # Load weights for each sub-mixer
        for mixer_name, mixer_weights in weights_by_mixer.items():
            if mixer_weights:
                sub_loaded = self.mixers[mixer_name].load_weights(mixer_weights)
                # Prefix the loaded names with the mixer path
                loaded.update(f"mixers.{mixer_name}.{n}" for n in sub_loaded)

        return loaded

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        """Return cache spec for the active mixer.

        Delegates to the active sub-mixer's get_kv_cache_spec method.
        """
        active_mixer = self.mixers[self.active_mixer_name]
        return active_mixer.get_kv_cache_spec(vllm_config)


class Apriel2StochasticDecoderLayer(nn.Module):
    """Stochastic decoder layer that can switch between multiple mixer types."""

    def __init__(
        self,
        config: Apriel2Config,
        layer_idx: int,
        block_config: dict,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        speculative_config: SpeculativeConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        mixer_config = block_config.get("mixer", {})
        mlp_config = block_config.get("mlp", {})
        norm_config = block_config.get("normalization", {})

        self.mixer = Apriel2StochasticMixer(
            config=config,
            mixer_config=mixer_config,
            layer_idx=layer_idx,
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            speculative_config=speculative_config,
            prefix=f"{prefix}.mixer",
        )

        intermediate_size = mlp_config["intermediate_size"]
        mlp_bias = mlp_config.get("add_linear_biases", False)
        hidden_act = mlp_config.get("activation", "silu")
        rms_norm_eps = norm_config["epsilon"]

        self.mlp = Apriel2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            quant_config=quant_config,
            bias=mlp_bias,
            prefix=f"{prefix}.mlp",
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=rms_norm_eps
        )

    def set_active_mixer(self, name: str) -> None:
        """Set the active mixer for this layer."""
        self.mixer.set_active_mixer(name)

    def get_active_mixer(self) -> str:
        """Get the name of the currently active mixer."""
        return self.mixer.get_active_mixer()

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.mixer(positions=positions, hidden_states=hidden_states, **kwargs)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


ALL_DECODER_LAYER_TYPES = {
    "attention": Apriel2AttentionDecoderLayer,
    "mamba": Apriel2MambaDecoderLayer,
    "gdn": Apriel2GDNDecoderLayer,
    "kda": Apriel2KDADecoderLayer,
    "stochastic": Apriel2StochasticDecoderLayer,
}


def get_block_config_for_layer(
    config: Apriel2Config, layer_idx: int
) -> tuple[str, dict]:
    """Get mixer type and block config for a specific layer."""
    decoder_config = config.decoder
    seq_type = decoder_config.get("type", "fixed")

    if seq_type == "fixed":
        block_config = decoder_config.get("block", {})
        mixer_type = block_config.get("mixer", {}).get("type", "attention")
        return mixer_type, block_config
    elif seq_type == "pattern":
        pattern = decoder_config.get("pattern", ["attention"])
        blocks_config = decoder_config.get("blocks", {})
        block_name = pattern[layer_idx % len(pattern)]
        block_config = blocks_config.get(block_name, {})
        mixer_type = block_config.get("mixer", {}).get("type", "attention")
        return mixer_type, block_config
    else:
        return "attention", {}


def apriel2_model_invariants(
    input_ids, positions, intermediate_tensors=None, inputs_embeds=None
):
    """Shape invariants for Apriel2 model compilation.

    These are translated to runtime assertions for unbacked dynamic shapes
    and are compiled away for backed shapes.
    """
    if input_ids is not None:
        torch._check(positions.size()[0] == input_ids.size()[0])


@support_torch_compile(shape_invariants=apriel2_model_invariants)
class Apriel2Model(nn.Module):
    """Apriel2 base model (decoder stack)."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
            )
        else:
            self.embed_tokens = None

        def get_layer(*, prefix: str):
            layer_idx = int(prefix.rsplit(".", 1)[1])
            mixer_type, block_config = get_block_config_for_layer(config, layer_idx)
            layer_class = ALL_DECODER_LAYER_TYPES.get(mixer_type)

            if layer_class is None:
                raise ValueError(f"Unknown mixer type: {mixer_type}")

            if mixer_type == "attention":
                return layer_class(
                    config=config,
                    layer_idx=layer_idx,
                    block_config=block_config,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=prefix,
                )
            elif mixer_type == "mamba":
                return layer_class(
                    config=config,
                    layer_idx=layer_idx,
                    block_config=block_config,
                    model_config=model_config,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=prefix,
                )
            elif mixer_type == "gdn":
                return layer_class(
                    config=config,
                    layer_idx=layer_idx,
                    block_config=block_config,
                    model_config=model_config,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    speculative_config=vllm_config.speculative_config,
                    prefix=prefix,
                )
            elif mixer_type == "stochastic":
                return layer_class(
                    config=config,
                    layer_idx=layer_idx,
                    block_config=block_config,
                    model_config=model_config,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    speculative_config=vllm_config.speculative_config,
                    prefix=prefix,
                )
            else:  # kda
                return layer_class(
                    config=config,
                    layer_idx=layer_idx,
                    block_config=block_config,
                    model_config=model_config,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=prefix,
                )

        num_layers = config.decoder["num_blocks"]
        self.start_layer, self.end_layer, self.layers = make_layers(
            num_layers,
            get_layer,
            prefix=f"{prefix}.layers" if prefix else "layers",
        )

        if get_pp_group().is_last_rank:
            head_norm_eps = config.head["normalization"]["epsilon"]
            self.norm = RMSNorm(config.hidden_size, eps=head_norm_eps)
        else:
            self.norm = None

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(
                hidden_states=hidden_states,
                residual=residual,
                positions=positions,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        # Debug final norm
        if DEBUG_FINAL_NORM:
            # Show LAST token (to match TF)
            last_hs = hidden_states[-1, :8]
            last_res = residual[-1, :8] if residual is not None else None
            hs_vals = ", ".join(f"{v:.6f}" for v in last_hs.float().tolist())
            res_vals = ", ".join(f"{v:.6f}" for v in last_res.float().tolist()) if last_res is not None else "None"
            print(f"[vLLM Final] hidden_states (before norm): shape={hidden_states.shape}, last_token_first8=[{hs_vals}]")
            print(f"[vLLM Final] residual (before norm): shape={residual.shape if residual is not None else None}, last_token_first8=[{res_vals}]")
            print(f"[vLLM Final] norm.weight: first8=[{', '.join(f'{v:.6f}' for v in self.norm.weight.flatten()[:8].float().tolist())}]")
            print(f"[vLLM Final] norm.variance_epsilon={self.norm.variance_epsilon}")

        hidden_states, _ = self.norm(hidden_states, residual)

        if DEBUG_FINAL_NORM:
            last_out = hidden_states[-1, :8]
            out_vals = ", ".join(f"{v:.6f}" for v in last_out.float().tolist())
            print(f"[vLLM Final] hidden_states (after norm): shape={hidden_states.shape}, last_token_first8=[{out_vals}]")

        return hidden_states


class Apriel2ForCausalLM(nn.Module, HasInnerState, SupportsPP):
    """Apriel2 model for causal language modeling.

    Supports hybrid architectures with attention, mamba, GDN, and KDA mixers.
    """

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            "model.decoder.blocks.": "model.layers.",
        },
    )

    # For hybrid models
    has_inner_state = True
    # Don't use is_hybrid=True - it triggers HybridAttentionMambaModelConfig
    # which assumes all mamba-like layers have the same shape.
    # Apriel2 has heterogeneous blocks, each with its own get_kv_cache_spec().
    is_hybrid = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.config = config
        self.vllm_config = vllm_config

        self.model = Apriel2Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
            self.logits_processor = LogitsProcessor(config.vocab_size)
        else:
            self.lm_head = None

        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        # Debug LM head input
        if DEBUG_LM_HEAD:
            flat = hidden_states.flatten()[:8]
            vals = ", ".join(f"{v:.6f}" for v in flat.float().tolist())
            print(f"[vLLM LM Head] input hidden_states: shape={hidden_states.shape}, first8=[{vals}]")
            if self.lm_head is not None:
                lm_weight = self.lm_head.weight
                print(f"[vLLM LM Head] lm_head.weight: shape={lm_weight.shape}, first8=[{', '.join(f'{v:.6f}' for v in lm_weight.flatten()[:8].float().tolist())}]")

        logits = self.logits_processor(self.lm_head, hidden_states)

        if DEBUG_LM_HEAD and logits is not None:
            # Get last token logits
            last_logits = logits[-1] if logits.dim() == 2 else logits[0, -1]
            top_vals, top_idx = last_logits.topk(5)
            print(f"[vLLM LM Head] logits shape={logits.shape}")
            print(f"[vLLM LM Head] last token top-5 logits: {[(idx.item(), val.item()) for idx, val in zip(top_idx, top_vals)]}")

        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
