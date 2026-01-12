# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Apriel2 model compatible with HuggingFace weights.

Apriel2 is a hybrid model that supports multiple mixer types (attention, mamba,
GatedDeltaNet, KDA) with flexible block patterns. This implementation is
optimized for vLLM inference.
"""

import math
from collections.abc import Iterable
from itertools import islice

import torch
from einops import rearrange
from torch import nn
from transformers import PretrainedConfig
from transformers.activations import ACT2FN

from vllm.attention.backends.abstract import AttentionMetadata
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
from vllm.model_executor.layers.fla.ops.kda import (
    FusedRMSNormGated,
    chunk_kda,
    fused_kda_gate,
    fused_recurrent_kda,
)
from vllm.model_executor.layers.layernorm import RMSNorm, RMSNormGated
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.abstract import MambaBase
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
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid, SupportsPP
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

        # Extract from mixer config or use defaults from main config
        self.total_num_heads = mixer_config.get("heads", config.num_attention_heads)
        self.total_num_kv_heads = mixer_config.get(
            "head_groups", config.num_key_value_heads
        )
        self.head_dim = mixer_config.get("head_size", config.head_dim)

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
        rope_theta = rotary_config.get("theta", config.rope_theta)
        max_pos = config.embeddings.get(
            "max_position_embeddings", config.max_position_embeddings
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_pos,
            base=rope_theta,
            rope_scaling=config.rope_scaling,
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

        # Extract mamba params from config
        d_state = mixer_config.get("state_size", 16)
        d_conv = mixer_config.get("d_conv", 4)
        expand = mixer_config.get("expand", 2)
        d_inner = mixer_config.get("d_inner", int(expand * config.hidden_size))
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
            activation=config.hidden_act,
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
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for GDN gating computation."""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < num_heads

    A_log = tl.load(A_log_ptr + offset % num_heads, mask=mask)
    dt_bias = tl.load(dt_bias_ptr + offset % num_heads, mask=mask)
    a = tl.load(a_ptr + offset, mask=mask).to(tl.float32)
    b = tl.load(b_ptr + offset, mask=mask).to(tl.float32)

    # g = -exp(A_log) * softplus(a + dt_bias)
    A = tl.exp(A_log)
    softplus_val = tl.log(1.0 + tl.exp(a + dt_bias))
    g = -A * softplus_val

    # beta = sigmoid(b)
    beta = 1.0 / (1.0 + tl.exp(-b))

    tl.store(g_ptr + offset, g, mask=mask)
    tl.store(beta_ptr + offset, beta, mask=mask)


def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GDN gating: g = -exp(A_log) * softplus(a + dt_bias), beta = sigmoid(b)"""
    batch_size = a.shape[0]
    num_heads = a.shape[-1]

    g = torch.empty_like(a)
    beta = torch.empty_like(b)

    # Use triton kernel for efficiency
    total_elements = batch_size * num_heads
    BLOCK_SIZE = 256
    grid = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    fused_gdn_gating_kernel[grid](
        A_log,
        a.view(-1),
        b.view(-1),
        dt_bias,
        g.view(-1),
        beta.view(-1),
        num_heads,
        BLOCK_SIZE,
    )

    g = g.unsqueeze(0)  # Add batch dim for chunk_gated_delta_rule
    beta = beta.unsqueeze(0)

    return g, beta


class Apriel2GatedDeltaNet(nn.Module, MambaBase):
    """Gated Delta Net mixer for Apriel2 using vLLM infrastructure.

    Follows the same pattern as Qwen3NextGatedDeltaNet.
    """

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

        # Config params - support Fast-LLM naming
        self.num_v_heads = mixer_config.get("value_heads", 32)
        self.num_k_heads = mixer_config.get("key_heads", 8)
        self.head_k_dim = mixer_config.get("key_head_dim", 64)
        self.head_v_dim = mixer_config.get("value_head_dim", 64)
        conv_config = mixer_config.get("convolution_layer", {})
        self.conv_kernel_size = conv_config.get("kernel_size", 4)
        self.layer_norm_epsilon = mixer_config.get("norm_eps", config.rms_norm_eps)
        self.activation = conv_config.get("activation", config.hidden_act)
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
        """Derives query, key, value, z, b, a tensors from projections."""
        new_tensor_shape_qkvz = mixed_qkvz.size()[:-1] + (
            self.num_k_heads // self.tp_size,
            (
                self.head_k_dim
                + self.head_k_dim
                + (self.head_v_dim + self.head_v_dim)
                * self.num_v_heads
                // self.num_k_heads
            ),
        )
        new_tensor_shape_ba = mixed_qkvz.size()[:-1] + (
            self.num_k_heads // self.tp_size,
            2 * self.num_v_heads // self.num_k_heads,
        )

        mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)

        split_arg_list_qkvz = [
            self.head_k_dim,
            self.head_k_dim,
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
        ]
        split_arg_list_ba = [
            self.num_v_heads // self.num_k_heads,
            self.num_v_heads // self.num_k_heads,
        ]

        (query, key, value, z) = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=2)
        (b, a) = torch.split(mixed_ba, split_arg_list_ba, dim=2)

        value = value.reshape(value.size(0), -1, self.head_v_dim)
        z = z.reshape(z.size(0), -1, self.head_v_dim)
        b = b.reshape(b.size(0), self.num_v_heads // self.tp_size)
        a = a.reshape(a.size(0), self.num_v_heads // self.tp_size)

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

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        """Forward pass with custom op for core attention."""
        num_tokens = hidden_states.size(0)

        # Part 1: Input Projection
        projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
        projected_states_ba, _ = self.in_proj_ba(hidden_states)
        query, key, value, z, b, a = self.fix_query_key_value_ordering(
            projected_states_qkvz, projected_states_ba
        )
        query, key, value = map(
            lambda x: rearrange(x, "l p d -> l (p d)"), (query, key, value)
        )
        mixed_qkv = torch.cat((query, key, value), dim=-1)

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

        # Part 3: Output Projection
        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        output[:num_tokens], _ = self.out_proj(core_attn_out)

    def _forward_core(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        """Core attention computation (called by custom op)."""
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

        self_kv_cache = self.kv_cache[forward_context.virtual_engine]
        conv_state = self_kv_cache[0].transpose(-1, -2)
        ssm_state = self_kv_cache[1]

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        # Convolution
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        if attn_metadata.num_prefills > 0:
            mixed_qkv_T = mixed_qkv.transpose(0, 1)
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
            mixed_qkv = causal_conv1d_update(
                mixed_qkv,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=non_spec_state_indices_tensor[:num_actual_tokens],
                validate_data=True,
            )

        query, key, value = self.rearrange_mixed_qkv(mixed_qkv)

        g, beta = fused_gdn_gating(self.A_log, a, b, self.dt_bias)

        # Recurrent attention
        if attn_metadata.num_prefills > 0:
            initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()
            initial_state[~has_initial_state, ...] = 0
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
            ssm_state[non_spec_state_indices_tensor] = last_state.to(ssm_state.dtype)
        else:
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

        core_attn_out[:num_actual_tokens] = core_out.squeeze(0)[:num_actual_tokens]


class Apriel2KDAMixer(nn.Module, MambaBase):
    """Kimi Delta Attention mixer for Apriel2 using vLLM's KDA infrastructure.

    This implements the KDA (Kimi Delta Attention) mixer following the same
    patterns as vLLM's KimiDeltaAttention and uses the fla ops for kernels.
    """

    @property
    def mamba_type(self) -> str:
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

        # Extract KDA config params
        self.num_heads = mixer_config.get("heads", 32)
        self.head_dim = mixer_config.get("head_dim", 64)
        conv_config = mixer_config.get("convolution_layer", {})
        self.conv_size = conv_config.get("kernel_size", 4)
        norm_config = mixer_config.get("normalization", {})
        rms_norm_eps = norm_config.get("epsilon", config.rms_norm_eps)

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

        self.A_log = nn.Parameter(
            torch.empty(1, 1, self.local_num_heads, 1, dtype=torch.float32)
        )
        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(2)})

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
            self.head_dim, eps=rms_norm_eps, activation="sigmoid"
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
        positions: torch.Tensor,
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

        self.self_attn = Apriel2Attention(
            config=config,
            mixer_config=mixer_config,
            layer_idx=layer_idx,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

        intermediate_size = mlp_config.get("intermediate_size", config.intermediate_size)
        mlp_bias = mlp_config.get("add_linear_biases", False)

        self.mlp = Apriel2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=mlp_bias,
            prefix=f"{prefix}.mlp",
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
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

        self.mixer = Apriel2MambaMixer(
            config=config,
            mixer_config=mixer_config,
            layer_idx=layer_idx,
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.mixer",
        )

        intermediate_size = mlp_config.get("intermediate_size", config.intermediate_size)
        mlp_bias = mlp_config.get("add_linear_biases", False)

        self.mlp = Apriel2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=mlp_bias,
            prefix=f"{prefix}.mlp",
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
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

        intermediate_size = mlp_config.get("intermediate_size", config.intermediate_size)
        mlp_bias = mlp_config.get("add_linear_biases", False)

        self.mlp = Apriel2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=mlp_bias,
            prefix=f"{prefix}.mlp",
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
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

        self.mixer = Apriel2KDAMixer(
            config=config,
            mixer_config=mixer_config,
            layer_idx=layer_idx,
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.mixer",
        )

        intermediate_size = mlp_config.get("intermediate_size", config.intermediate_size)
        mlp_bias = mlp_config.get("add_linear_biases", False)

        self.mlp = Apriel2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=mlp_bias,
            prefix=f"{prefix}.mlp",
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
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


ALL_DECODER_LAYER_TYPES = {
    "attention": Apriel2AttentionDecoderLayer,
    "mamba": Apriel2MambaDecoderLayer,
    "gdn": Apriel2GDNDecoderLayer,
    "kda": Apriel2KDADecoderLayer,
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


@support_torch_compile
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

        def get_layer(layer_prefix: str):
            layer_idx = int(layer_prefix.rsplit(".", 1)[1])
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
                    prefix=layer_prefix,
                )
            elif mixer_type == "mamba":
                return layer_class(
                    config=config,
                    layer_idx=layer_idx,
                    block_config=block_config,
                    model_config=model_config,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=layer_prefix,
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
                    prefix=layer_prefix,
                )
            else:  # kda
                return layer_class(
                    config=config,
                    layer_idx=layer_idx,
                    block_config=block_config,
                    model_config=model_config,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=layer_prefix,
                )

        num_layers = config.decoder.get("num_blocks", config.num_hidden_layers)
        self.start_layer, self.end_layer, self.layers = make_layers(
            num_layers,
            get_layer,
            prefix=f"{prefix}.layers" if prefix else "layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
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
            # Attention layers need positions for rotary embeddings
            if isinstance(layer, Apriel2AttentionDecoderLayer):
                hidden_states, residual = layer(
                    positions=positions,
                    hidden_states=hidden_states,
                    residual=residual,
                )
            else:
                hidden_states, residual = layer(
                    hidden_states=hidden_states,
                    residual=residual,
                )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue

            # Handle A_log -> A conversion for mamba
            if "A_log" in name:
                name = name.replace("A_log", "A")

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

            loaded_params.add(name)

        return loaded_params


class Apriel2ForCausalLM(nn.Module, HasInnerState, SupportsPP, IsHybrid):
    """Apriel2 model for causal language modeling.

    Supports hybrid architectures with attention, mamba, GDN, and KDA mixers.
    """

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            ".self_attn.": ".",
            ".A_log": ".A",
            "model.decoder.blocks.": "model.layers.",
        },
    )

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    # For hybrid models
    has_inner_state = True
    is_hybrid = True

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
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.mamba1_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        config = vllm_config.model_config.hf_config
        parallel_config = vllm_config.parallel_config

        # Get mamba config from decoder
        decoder_config = getattr(config, "decoder", {}) or {}
        mamba_config = {}

        # Find first mamba block config
        seq_type = decoder_config.get("type", "fixed")
        if seq_type == "fixed":
            block_config = decoder_config.get("block", {})
            if block_config.get("mixer", {}).get("type") == "mamba":
                mamba_config = block_config.get("mixer", {})
        elif seq_type == "pattern":
            blocks_config = decoder_config.get("blocks", {})
            for block_config in blocks_config.values():
                if block_config.get("mixer", {}).get("type") == "mamba":
                    mamba_config = block_config.get("mixer", {})
                    break

        d_state = mamba_config.get("state_size", 16)
        d_conv = mamba_config.get("d_conv", 4)
        expand = mamba_config.get("expand", 2)
        d_inner = mamba_config.get("d_inner", int(expand * config.hidden_size))

        return MambaStateShapeCalculator.mamba1_state_shape(
            tp_world_size=parallel_config.tensor_parallel_size,
            intermediate_size=d_inner,
            state_size=d_state,
            conv_kernel=d_conv,
        )

    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
        return self.mamba_cache.copy_inputs_before_cuda_graphs(input_buffers, **kwargs)

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        return self.mamba_cache.get_seqlen_agnostic_capture_inputs(batch_size)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
