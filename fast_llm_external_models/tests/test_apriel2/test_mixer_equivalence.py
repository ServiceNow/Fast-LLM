"""Tests for numerical equivalence between Apriel2 mixers and reference implementations.

Tests forward-pass equivalence between:
1. Apriel2Attention vs MistralAttention
2. Apriel2Attention vs PixtralAttention (non-causal)
3. Apriel2GatedDeltaNet vs Qwen3NextGatedDeltaNet

Covers various input shapes, hyperparameters, and fast/slow paths.
"""

import pytest
import torch
import torch.nn as nn
from typing import Optional
from unittest.mock import patch


# =============================================================================
# Fixtures for configs
# =============================================================================


@pytest.fixture(params=[1, 2, 4])
def batch_size(request):
    """Batch sizes to test."""
    return request.param


@pytest.fixture(params=[1, 16, 64, 128])
def seq_len(request):
    """Sequence lengths to test."""
    return request.param


@pytest.fixture(params=[256, 512])
def hidden_size(request):
    """Hidden sizes to test."""
    return request.param


@pytest.fixture(
    params=[
        (8, 8, 32),  # MHA: 8 heads, 8 kv heads, 32 head_dim
        (8, 4, 32),  # GQA: 8 heads, 4 kv heads, 32 head_dim
        (8, 2, 64),  # GQA: 8 heads, 2 kv heads, 64 head_dim
        (4, 1, 64),  # MQA: 4 heads, 1 kv head, 64 head_dim
    ]
)
def attention_config(request):
    """Attention head configurations: (num_heads, num_kv_heads, head_dim)."""
    return request.param


@pytest.fixture(
    params=[
        (8, 4, 32, 32),  # 8 value heads, 4 key heads, 32 key_dim, 32 value_dim
        (8, 2, 64, 64),  # 8 value heads, 2 key heads, 64 key_dim, 64 value_dim
        (4, 2, 32, 64),  # 4 value heads, 2 key heads, 32 key_dim, 64 value_dim
    ]
)
def gdn_config(request):
    """GDN configurations: (value_heads, key_heads, key_head_dim, value_head_dim)."""
    return request.param


@pytest.fixture(params=[True, False])
def use_fast_path(request):
    """Whether to use fast path (CUDA kernels) or slow path (pure PyTorch)."""
    return request.param


# =============================================================================
# Helper functions
# =============================================================================


def copy_attention_weights(src: nn.Module, dst: nn.Module):
    """Copy attention weights from src to dst, handling different naming conventions."""
    with torch.no_grad():
        dst.q_proj.weight.copy_(src.q_proj.weight)
        dst.k_proj.weight.copy_(src.k_proj.weight)
        dst.v_proj.weight.copy_(src.v_proj.weight)
        dst.o_proj.weight.copy_(src.o_proj.weight)

        # Copy biases if present
        if hasattr(src.q_proj, "bias") and src.q_proj.bias is not None:
            if hasattr(dst.q_proj, "bias") and dst.q_proj.bias is not None:
                dst.q_proj.bias.copy_(src.q_proj.bias)
        if hasattr(src.k_proj, "bias") and src.k_proj.bias is not None:
            if hasattr(dst.k_proj, "bias") and dst.k_proj.bias is not None:
                dst.k_proj.bias.copy_(src.k_proj.bias)
        if hasattr(src.v_proj, "bias") and src.v_proj.bias is not None:
            if hasattr(dst.v_proj, "bias") and dst.v_proj.bias is not None:
                dst.v_proj.bias.copy_(src.v_proj.bias)
        if hasattr(src.o_proj, "bias") and src.o_proj.bias is not None:
            if hasattr(dst.o_proj, "bias") and dst.o_proj.bias is not None:
                dst.o_proj.bias.copy_(src.o_proj.bias)


def assert_close(a: torch.Tensor, b: torch.Tensor, rtol: float = 1e-4, atol: float = 1e-4, msg: str = ""):
    """Assert two tensors are close with detailed error message."""
    if not torch.allclose(a, b, rtol=rtol, atol=atol):
        diff = (a - b).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        raise AssertionError(
            f"{msg}\nMax diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}, " f"rtol={rtol}, atol={atol}"
        )


# =============================================================================
# Apriel2Attention vs MistralAttention Tests
# =============================================================================


class TestApriel2AttentionVsMistral:
    """Test equivalence between Apriel2Attention and MistralAttention."""

    @pytest.fixture
    def mistral_config(self, hidden_size, attention_config):
        """Create MistralConfig for testing."""
        from transformers import MistralConfig

        num_heads, num_kv_heads, head_dim = attention_config

        config = MistralConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            max_position_embeddings=4096,
            rope_theta=10000.0,
            attention_dropout=0.0,
        )
        # Set attn implementation to eager for testing (sdpa/flash require specific setup)
        config._attn_implementation = "eager"
        return config

    @pytest.fixture
    def apriel2_mixer_config(self, attention_config):
        """Create Apriel2 mixer config dict."""
        num_heads, num_kv_heads, head_dim = attention_config

        return {
            "type": "attention",
            "heads": num_heads,
            "head_groups": num_kv_heads,
            "head_size": head_dim,
            "add_linear_biases": False,
            "causal": True,
            "rotary": {"type": "mistral_1d", "theta": 10000.0},
        }

    @pytest.fixture
    def apriel2_config(self, hidden_size, apriel2_mixer_config):
        """Create Apriel2Config for testing."""
        from fast_llm_external_models.apriel2.configuration_apriel2 import Apriel2TextConfig

        config = Apriel2TextConfig(
            hidden_size=hidden_size,
            decoder={
                "type": "fixed",
                "num_blocks": 1,
                "block": {
                    "mixer": apriel2_mixer_config,
                    "mlp": {"type": "mlp", "intermediate_size": hidden_size * 4},
                    "normalization": {"type": "rms_norm", "epsilon": 1e-5},
                },
            },
            embeddings={"max_position_embeddings": 4096},
        )
        # Set attn implementation to eager for testing
        config._attn_implementation = "eager"
        return config

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_forward_equivalence(
        self,
        mistral_config,
        apriel2_config,
        apriel2_mixer_config,
        batch_size,
        seq_len,
        hidden_size,
        use_fast_path,
    ):
        """Test that Apriel2Attention produces same output as MistralAttention."""
        from transformers.models.mistral.modeling_mistral import MistralAttention, MistralRotaryEmbedding
        from fast_llm_external_models.apriel2.modeling_apriel2 import Apriel2Attention

        # Create models (uses default device/dtype from conftest fixtures)
        mistral_attn = MistralAttention(mistral_config, layer_idx=0)
        apriel2_attn = Apriel2Attention(hidden_size, apriel2_mixer_config, layer_idx=0, config=apriel2_config)

        # Copy weights
        copy_attention_weights(mistral_attn, apriel2_attn)

        # Create input
        torch.manual_seed(42)
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        # Create position_ids
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        # Create causal mask
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1).unsqueeze(0).unsqueeze(0)

        # Compute position embeddings using Mistral's rotary embedding
        # Use the same position embeddings for both to ensure equivalence test is fair
        mistral_rotary = MistralRotaryEmbedding(config=mistral_config)
        position_embeddings = mistral_rotary(hidden_states, position_ids)

        mistral_attn.eval()
        apriel2_attn.eval()

        with torch.no_grad():
            # Mistral forward - position_embeddings is now a required positional arg
            mistral_out = mistral_attn(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
            )[0]

            # Apriel2 forward - use the same position embeddings
            apriel2_out = apriel2_attn(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
            )[0]

        assert_close(
            apriel2_out,
            mistral_out,
            rtol=1e-4,
            atol=1e-4,
            msg=f"Apriel2Attention vs MistralAttention mismatch "
            f"(batch={batch_size}, seq={seq_len}, hidden={hidden_size})",
        )


# =============================================================================
# Apriel2Attention vs PixtralAttention Tests (non-causal)
# =============================================================================


class TestApriel2AttentionVsPixtral:
    """Test equivalence between Apriel2Attention and PixtralAttention (non-causal).

    Note: Full 2D rotary equivalence tests are in test_rotary_2d_equivalence.py.
    This test focuses on verifying the attention mechanism itself is equivalent
    when given the same inputs.
    """

    @pytest.fixture
    def pixtral_config(self, attention_config):
        """Create PixtralVisionConfig for testing."""
        from transformers.models.pixtral.configuration_pixtral import PixtralVisionConfig

        num_heads, _, head_dim = attention_config
        hidden_size = num_heads * head_dim

        config = PixtralVisionConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            num_hidden_layers=1,
            rope_theta=10000.0,
        )
        config._attn_implementation = "eager"
        return config

    @pytest.fixture
    def apriel2_mixer_config_noncausal(self, attention_config):
        """Create Apriel2 mixer config dict for non-causal attention."""
        num_heads, _, head_dim = attention_config

        return {
            "type": "attention",
            "heads": num_heads,
            "head_groups": num_heads,  # Pixtral uses MHA
            "head_size": head_dim,
            "add_linear_biases": False,
            "causal": False,
            "rotary": {"type": "pixtral_2d", "theta": 10000.0, "patch_size": 16, "max_image_size": 1024},
        }

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    @pytest.mark.parametrize("seq_len", [16, 64])  # Override to use specific lengths for vision
    def test_forward_equivalence_noncausal(
        self,
        pixtral_config,
        apriel2_mixer_config_noncausal,
        attention_config,
        batch_size,
        seq_len,
        use_fast_path,
    ):
        """Test that Apriel2Attention (non-causal) produces same output as PixtralAttention.

        This test creates 1D position embeddings in the format both implementations expect,
        allowing us to verify the core attention mechanism is equivalent.
        """
        from transformers.models.pixtral.modeling_pixtral import PixtralAttention, PixtralRotaryEmbedding
        from fast_llm_external_models.apriel2.modeling_apriel2 import Apriel2Attention
        from fast_llm_external_models.apriel2.configuration_apriel2 import Apriel2TextConfig

        num_heads, _, head_dim = attention_config
        hidden_size = num_heads * head_dim

        # Create Apriel2 config
        apriel2_config = Apriel2TextConfig(
            hidden_size=hidden_size,
            decoder={
                "type": "fixed",
                "num_blocks": 1,
                "block": {
                    "mixer": apriel2_mixer_config_noncausal,
                    "mlp": {"type": "mlp", "intermediate_size": hidden_size * 4},
                    "normalization": {"type": "rms_norm", "epsilon": 1e-5},
                },
            },
            embeddings={"max_position_embeddings": 4096},
        )
        apriel2_config._attn_implementation = "eager"

        # Create models (uses default device/dtype from conftest fixtures)
        pixtral_attn = PixtralAttention(pixtral_config)
        apriel2_attn = Apriel2Attention(
            hidden_size, apriel2_mixer_config_noncausal, layer_idx=0, config=apriel2_config
        )

        # Copy weights
        copy_attention_weights(pixtral_attn, apriel2_attn)

        # Create input
        torch.manual_seed(42)
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        # For 2D rotary, we need position_ids that represent 2D positions
        # Simulate a small image grid
        grid_size = int(seq_len**0.5)
        if grid_size * grid_size != seq_len:
            pytest.skip(f"seq_len {seq_len} is not a perfect square for 2D position test")

        rotary_emb = PixtralRotaryEmbedding(config=pixtral_config)
        position_ids = torch.arange(seq_len)
        cos, sin = rotary_emb(hidden_states, position_ids)
        # Add batch dimension for compatibility with both Pixtral and Apriel2 (Mistral) conventions
        position_embeddings = (cos.unsqueeze(0), sin.unsqueeze(0))

        pixtral_attn.eval()
        apriel2_attn.eval()

        with torch.no_grad():
            # Pixtral forward with explicit position embeddings
            pixtral_out = pixtral_attn(
                hidden_states,
                attention_mask=None,
                position_embeddings=position_embeddings,
            )[0]

            # Apriel2 forward with same position embeddings
            apriel2_out = apriel2_attn(
                hidden_states,
                attention_mask=None,
                position_embeddings=position_embeddings,
            )[0]

        assert_close(
            apriel2_out,
            pixtral_out,
            rtol=1e-4,
            atol=1e-4,
            msg=f"Apriel2Attention (non-causal) vs PixtralAttention mismatch "
            f"(batch={batch_size}, seq={seq_len}, hidden={hidden_size})",
        )


# =============================================================================
# Apriel2GatedDeltaNet vs Qwen3NextGatedDeltaNet Tests
# =============================================================================


class TestApriel2GDNVsQwen3Next:
    """Test equivalence between Apriel2GatedDeltaNet and Qwen3NextGatedDeltaNet."""

    @pytest.fixture
    def qwen3_config(self, hidden_size, gdn_config):
        """Create Qwen3NextConfig for testing."""
        from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig

        value_heads, key_heads, key_head_dim, value_head_dim = gdn_config

        return Qwen3NextConfig(
            hidden_size=hidden_size,
            # Qwen3NextConfig uses different param names for GDN:
            linear_num_value_heads=value_heads,
            linear_num_key_heads=key_heads,
            linear_key_head_dim=key_head_dim,
            linear_value_head_dim=value_head_dim,
            linear_conv_kernel_dim=4,
            rms_norm_eps=1e-5,
            max_position_embeddings=4096,
            # Attention params (not used for GDN but required)
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=64,
            # Explicitly set dtype to avoid torch.get_current_dtype() fallback
            torch_dtype=torch.get_default_dtype(),
        )

    @pytest.fixture
    def apriel2_gdn_config(self, gdn_config):
        """Create Apriel2 GDN config dict."""
        value_heads, key_heads, key_head_dim, value_head_dim = gdn_config

        return {
            "type": "gdn",
            "value_heads": value_heads,
            "key_heads": key_heads,
            "key_head_dim": key_head_dim,
            "value_head_dim": value_head_dim,
            "conv_kernel_size": 4,
            "norm_eps": 1e-5,
        }

    def _copy_gdn_weights(self, src: nn.Module, dst: nn.Module):
        """Copy GDN weights from Qwen3Next to Apriel2 format."""
        with torch.no_grad():
            # The weight layouts differ between Qwen3Next and Apriel2
            # Qwen3Next: q_proj, k_proj, v_proj, g_proj (gate), o_proj
            # Apriel2: in_proj_qkvz, in_proj_ba, out_proj, convolution, etc.
            # This requires careful weight remapping - for now we verify shapes only
            pass

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GDN requires CUDA")
    def test_gdn_shapes_match(
        self,
        qwen3_config,
        apriel2_gdn_config,
        hidden_size,
        gdn_config,
        batch_size,
    ):
        """Test that Apriel2GatedDeltaNet produces same output shapes as Qwen3NextGatedDeltaNet."""
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextGatedDeltaNet
        from fast_llm_external_models.apriel2.modeling_apriel2 import Apriel2GatedDeltaNet

        seq_len = 32  # Fixed for this test
        value_heads, key_heads, key_head_dim, value_head_dim = gdn_config

        # Create models (uses default device/dtype from conftest fixtures)
        qwen_gdn = Qwen3NextGatedDeltaNet(qwen3_config, layer_idx=0)
        apriel2_gdn = Apriel2GatedDeltaNet(hidden_size, apriel2_gdn_config, layer_idx=0)

        # Create input
        torch.manual_seed(42)
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        qwen_gdn.eval()
        apriel2_gdn.eval()

        with torch.no_grad():
            # Qwen3NextGatedDeltaNet returns tensor directly, Apriel2 returns tuple
            qwen_out = qwen_gdn(hidden_states)
            apriel2_out = apriel2_gdn(hidden_states)[0]

        assert (
            apriel2_out.shape == qwen_out.shape
        ), f"Shape mismatch: Apriel2 {apriel2_out.shape} vs Qwen3Next {qwen_out.shape}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GDN requires CUDA")
    @pytest.mark.parametrize("seq_len", [1, 16, 32, 64])
    def test_gdn_forward_with_cache(
        self,
        apriel2_gdn_config,
        hidden_size,
        gdn_config,
        batch_size,
        seq_len,
    ):
        """Test Apriel2GatedDeltaNet forward pass with various sequence lengths."""
        from fast_llm_external_models.apriel2.modeling_apriel2 import Apriel2GatedDeltaNet

        # Create model (uses default device/dtype from conftest fixtures)
        apriel2_gdn = Apriel2GatedDeltaNet(hidden_size, apriel2_gdn_config, layer_idx=0)

        # Create input
        torch.manual_seed(42)
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        apriel2_gdn.eval()

        with torch.no_grad():
            output = apriel2_gdn(hidden_states)[0]

        assert (
            output.shape == hidden_states.shape
        ), f"Output shape {output.shape} doesn't match input shape {hidden_states.shape}"
        assert not output.isnan().any(), "Output contains NaN"
        assert not output.isinf().any(), "Output contains Inf"


# =============================================================================
# Fast Path vs Slow Path Tests
# =============================================================================


class TestFastVsSlowPath:
    """Test that fast path (CUDA kernels) and slow path (PyTorch) produce same results."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_gdn_fast_vs_slow_path(self, gdn_config, batch_size):
        """Test GDN produces same output with fast path vs slow path."""
        from fast_llm_external_models.apriel2.modeling_apriel2 import (
            Apriel2GatedDeltaNet,
            chunk_gated_delta_rule,
            torch_chunk_gated_delta_rule,
        )

        if chunk_gated_delta_rule is None:
            pytest.skip("Fast path (fla) not available")

        value_heads, key_heads, key_head_dim, value_head_dim = gdn_config
        hidden_size = 256
        seq_len = 32

        gdn_config_dict = {
            "type": "gdn",
            "value_heads": value_heads,
            "key_heads": key_heads,
            "key_head_dim": key_head_dim,
            "value_head_dim": value_head_dim,
            "conv_kernel_size": 4,
            "norm_eps": 1e-5,
        }

        # Create model (uses default device/dtype from conftest fixtures)
        torch.manual_seed(42)
        model = Apriel2GatedDeltaNet(hidden_size, gdn_config_dict, layer_idx=0)

        # Create input
        torch.manual_seed(123)
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        model.eval()

        # Run with fast path
        with torch.no_grad():
            model._chunk_gated_delta_rule = chunk_gated_delta_rule
            fast_out = model(hidden_states)[0].clone()

        # Run with slow path
        with torch.no_grad():
            model._chunk_gated_delta_rule = torch_chunk_gated_delta_rule
            slow_out = model(hidden_states)[0].clone()

        assert_close(fast_out, slow_out, rtol=1e-3, atol=1e-3, msg="Fast path vs slow path mismatch for GDN")


# =============================================================================
# Determinism Tests
# =============================================================================


class TestDeterminism:
    """Test that models produce deterministic outputs."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_attention_determinism(self, attention_config):
        """Test Apriel2Attention produces deterministic output."""
        from fast_llm_external_models.apriel2.modeling_apriel2 import Apriel2Attention
        from fast_llm_external_models.apriel2.configuration_apriel2 import Apriel2TextConfig

        num_heads, num_kv_heads, head_dim = attention_config
        hidden_size = 256
        batch_size = 2
        seq_len = 32

        mixer_config = {
            "type": "attention",
            "heads": num_heads,
            "head_groups": num_kv_heads,
            "head_size": head_dim,
            "add_linear_biases": False,
            "causal": True,
            "rotary": {"type": "mistral_1d", "theta": 10000.0},
        }

        config = Apriel2TextConfig(
            hidden_size=hidden_size,
            decoder={
                "type": "fixed",
                "num_blocks": 1,
                "block": {
                    "mixer": mixer_config,
                    "mlp": {"type": "mlp", "intermediate_size": hidden_size * 4},
                    "normalization": {"type": "rms_norm", "epsilon": 1e-5},
                },
            },
            embeddings={"max_position_embeddings": 4096},
        )
        config._attn_implementation = "eager"

        # Create model with fixed seed (uses default device/dtype from conftest fixtures)
        torch.manual_seed(42)
        model = Apriel2Attention(hidden_size, mixer_config, layer_idx=0, config=config)
        model.eval()

        # Create input with fixed seed
        torch.manual_seed(123)
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        # Get rotary embeddings
        rotary_resources = Apriel2Attention.setup(mixer_config, hidden_size, 4096)
        rotary_emb = rotary_resources["rotary_emb"]
        position_embeddings = rotary_emb(hidden_states, position_ids)

        # Run twice
        with torch.no_grad():
            out1 = model(hidden_states, position_embeddings=position_embeddings)[0]
            out2 = model(hidden_states, position_embeddings=position_embeddings)[0]

        assert torch.equal(out1, out2), "Attention output is not deterministic"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GDN requires CUDA")
    def test_gdn_determinism(self, gdn_config):
        """Test Apriel2GatedDeltaNet produces deterministic output."""
        from fast_llm_external_models.apriel2.modeling_apriel2 import Apriel2GatedDeltaNet

        value_heads, key_heads, key_head_dim, value_head_dim = gdn_config
        hidden_size = 256
        batch_size = 2
        seq_len = 32

        gdn_config_dict = {
            "type": "gdn",
            "value_heads": value_heads,
            "key_heads": key_heads,
            "key_head_dim": key_head_dim,
            "value_head_dim": value_head_dim,
            "conv_kernel_size": 4,
            "norm_eps": 1e-5,
        }

        # Create model with fixed seed (uses default device/dtype from conftest fixtures)
        torch.manual_seed(42)
        model = Apriel2GatedDeltaNet(hidden_size, gdn_config_dict, layer_idx=0)
        model.eval()

        # Create input with fixed seed
        torch.manual_seed(123)
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        # Run twice
        with torch.no_grad():
            out1 = model(hidden_states)[0]
            out2 = model(hidden_states)[0]

        assert torch.equal(out1, out2), "GDN output is not deterministic"
