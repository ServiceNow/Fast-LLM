"""Comprehensive tests for Apriel2Cache.

Architecture Overview
=====================
Apriel2Cache manages state for autoregressive generation across different mixer types:

1. **Attention Cache** (_AttentionCache): Stores key/value states
   - Supports sliding window (window_size) for SWA
   - Efficient roll optimization for single-token decode

2. **SSM Cache** (_SSMCache): Stores conv and recurrent states
   - Used by Mamba, GDN, KDA
   - KDA uses tuple conv states (q, k, v), others use single tensor

3. **Stochastic Mixer Routing**: For layers with multiple mixer options
   - Each mixer has independent cache (no sharing)
   - active_mixer pointer routes operations to correct sub-cache
   - Switching mixers preserves each mixer's independent state

Cache Invalidation Semantics
============================
When switching between mixers in a stochastic layer:
- Each mixer maintains its OWN independent history
- Switching does NOT invalidate the previous mixer's cache
- Switching does NOT copy state between mixers
- To invalidate: call reset() explicitly

This is intentional for training with stochastic sampling where each mixer
should learn from its own history. For inference, main_mixer_name is fixed.

Test Organization
=================
1. CREATION & PROPERTIES - Cache initialization, config parsing
2. ATTENTION CACHE - Updates, sliding window, concatenation
3. SSM CACHE - Conv states, recurrent states, KDA tuples
4. STOCHASTIC ROUTING - Active mixer, isolation, switching
5. CACHE INVALIDATION - Reset, per-mixer reset, coherence
6. BEAM SEARCH - batch_repeat, reorder, select
7. HF INTEGRATION - get_mask_sizes, indexing, properties
8. GENERATION PATTERNS - Prefill→decode, crop→continue
9. ERROR HANDLING - Guards, bounds, invalid operations
"""

import pytest
import torch

from fast_llm_external_models.apriel2.cache import (
    Apriel2Cache,
    _AttentionCache,
    _SSMCache,
)


# =============================================================================
# FIXTURES - Configs and Sample Data
# =============================================================================


@pytest.fixture
def tiny_attention_config():
    """Minimal config with pure attention layers."""
    from fast_llm_external_models.apriel2.configuration_apriel2 import Apriel2Config

    return Apriel2Config(
        vocab_size=100,
        hidden_size=64,
        decoder={
            "type": "fixed",
            "num_blocks": 2,
            "block": {
                "mixer": {"type": "attention", "heads": 4, "head_groups": 2, "head_size": 16},
                "mlp": {"type": "mlp", "intermediate_size": 256},
                "normalization": {"type": "rms_norm", "epsilon": 1e-5},
            },
        },
    )


@pytest.fixture
def swa_config():
    """Config with sliding window attention."""
    from fast_llm_external_models.apriel2.configuration_apriel2 import Apriel2Config

    return Apriel2Config(
        vocab_size=100,
        hidden_size=64,
        decoder={
            "type": "fixed",
            "num_blocks": 2,
            "block": {
                "mixer": {
                    "type": "attention",
                    "heads": 4,
                    "head_groups": 2,
                    "head_size": 16,
                    "window_size": 8,  # Small for testing
                },
                "mlp": {"type": "mlp", "intermediate_size": 256},
                "normalization": {"type": "rms_norm", "epsilon": 1e-5},
            },
        },
    )


@pytest.fixture
def ssm_config():
    """Config with pure SSM layers (mamba)."""
    from fast_llm_external_models.apriel2.configuration_apriel2 import Apriel2Config

    return Apriel2Config(
        vocab_size=100,
        hidden_size=64,
        decoder={
            "type": "fixed",
            "num_blocks": 2,
            "block": {
                "mixer": {
                    "type": "mamba",
                    "d_inner": 128,
                    "d_state": 16,
                    "dt_rank": 4,
                    "d_conv": 4,
                },
                "mlp": {"type": "mlp", "intermediate_size": 256},
                "normalization": {"type": "rms_norm", "epsilon": 1e-5},
            },
        },
    )


@pytest.fixture
def kda_config():
    """Config with pure KDA layers."""
    from fast_llm_external_models.apriel2.configuration_apriel2 import Apriel2Config

    return Apriel2Config(
        vocab_size=100,
        hidden_size=64,
        decoder={
            "type": "fixed",
            "num_blocks": 2,
            "block": {
                "mixer": {
                    "type": "kda",
                    "heads": 4,
                    "head_dim": 16,
                    "convolution_layer": {"kernel_size": 4},
                    "normalization": {"epsilon": 1e-5},
                },
                "mlp": {"type": "mlp", "intermediate_size": 256},
                "normalization": {"type": "rms_norm", "epsilon": 1e-5},
            },
        },
    )


@pytest.fixture
def stochastic_config():
    """Config with stochastic mixer (attention + mamba)."""
    from fast_llm_external_models.apriel2.configuration_apriel2 import Apriel2Config

    return Apriel2Config(
        vocab_size=100,
        hidden_size=64,
        decoder={
            "type": "pattern",
            "num_blocks": 2,
            "pattern": ["attn", "stochastic"],
            "blocks": {
                "attn": {
                    "mixer": {"type": "attention", "heads": 4, "head_groups": 2, "head_size": 16},
                    "mlp": {"type": "mlp", "intermediate_size": 256},
                    "normalization": {"type": "rms_norm", "epsilon": 1e-5},
                },
                "stochastic": {
                    "mixer": {
                        "type": "stochastic",
                        "main_mixer_name": "attention",
                        "mixers": {
                            "attention": {"type": "attention", "heads": 4, "head_groups": 2, "head_size": 16},
                            "mamba": {"type": "mamba", "d_inner": 128, "d_state": 16, "dt_rank": 4, "d_conv": 4},
                        },
                    },
                    "mlp": {"type": "mlp", "intermediate_size": 256},
                    "normalization": {"type": "rms_norm", "epsilon": 1e-5},
                },
            },
        },
    )


@pytest.fixture
def all_mixers_config():
    """Config with stochastic mixer containing all 5 mixer types."""
    from fast_llm_external_models.apriel2.configuration_apriel2 import Apriel2Config

    return Apriel2Config(
        vocab_size=100,
        hidden_size=64,
        decoder={
            "type": "pattern",
            "num_blocks": 2,
            "pattern": ["attn", "all_mixers"],
            "blocks": {
                "attn": {
                    "mixer": {"type": "attention", "heads": 4, "head_groups": 2, "head_size": 16},
                    "mlp": {"type": "mlp", "intermediate_size": 256},
                    "normalization": {"type": "rms_norm", "epsilon": 1e-5},
                },
                "all_mixers": {
                    "mixer": {
                        "type": "stochastic",
                        "main_mixer_name": "attention",
                        "mixers": {
                            "attention": {"type": "attention", "heads": 4, "head_groups": 2, "head_size": 16},
                            "swa": {
                                "type": "attention",
                                "heads": 4,
                                "head_groups": 2,
                                "head_size": 16,
                                "window_size": 1024,
                            },
                            "mamba": {"type": "mamba", "d_inner": 128, "d_state": 16, "dt_rank": 4, "d_conv": 4},
                            "gdn": {
                                "type": "gdn",
                                "value_heads": 4,
                                "key_heads": 2,
                                "key_head_dim": 16,
                                "value_head_dim": 16,
                                "convolution_layer": {"kernel_size": 4},
                            },
                            "kda": {
                                "type": "kda",
                                "heads": 4,
                                "head_dim": 16,
                                "convolution_layer": {"kernel_size": 4},
                                "normalization": {"epsilon": 1e-5},
                            },
                        },
                    },
                    "mlp": {"type": "mlp", "intermediate_size": 256},
                    "normalization": {"type": "rms_norm", "epsilon": 1e-5},
                },
            },
        },
    )


@pytest.fixture
def multi_window_config():
    """Config with multiple different window sizes."""
    from fast_llm_external_models.apriel2.configuration_apriel2 import Apriel2Config

    return Apriel2Config(
        vocab_size=100,
        hidden_size=64,
        decoder={
            "type": "pattern",
            "num_blocks": 3,
            "pattern": ["full", "small_window", "large_window"],
            "blocks": {
                "full": {
                    "mixer": {"type": "attention", "heads": 4, "head_groups": 2, "head_size": 16},
                    "mlp": {"type": "mlp", "intermediate_size": 256},
                    "normalization": {"type": "rms_norm", "epsilon": 1e-5},
                },
                "small_window": {
                    "mixer": {
                        "type": "attention",
                        "heads": 4,
                        "head_groups": 2,
                        "head_size": 16,
                        "window_size": 512,
                    },
                    "mlp": {"type": "mlp", "intermediate_size": 256},
                    "normalization": {"type": "rms_norm", "epsilon": 1e-5},
                },
                "large_window": {
                    "mixer": {
                        "type": "attention",
                        "heads": 4,
                        "head_groups": 2,
                        "head_size": 16,
                        "window_size": 2048,
                    },
                    "mlp": {"type": "mlp", "intermediate_size": 256},
                    "normalization": {"type": "rms_norm", "epsilon": 1e-5},
                },
            },
        },
    )


@pytest.fixture
def sample_kv():
    """Sample key/value tensors: [batch=2, heads=4, seq=10, head_dim=16]."""
    return torch.randn(2, 4, 10, 16), torch.randn(2, 4, 10, 16)


@pytest.fixture
def sample_conv_single():
    """Sample single-tensor conv state: [batch=2, d_inner=128, kernel=4]."""
    return torch.randn(2, 128, 4)


@pytest.fixture
def sample_conv_tuple():
    """Sample tuple conv state for KDA: (q, k, v) each [batch=2, d=64, kernel=3]."""
    return (torch.randn(2, 64, 3), torch.randn(2, 64, 3), torch.randn(2, 64, 3))


@pytest.fixture
def sample_recurrent():
    """Sample recurrent state: [batch=2, heads=4, head_dim=16, d_state=16]."""
    return torch.randn(2, 4, 16, 16)


# =============================================================================
# SECTION 1: CACHE CREATION & PROPERTIES
# =============================================================================


class TestCacheCreation:
    """Test cache initialization from config."""

    def test_attention_cache_creation(self, tiny_attention_config):
        """Create cache for pure attention config."""
        cache = Apriel2Cache(tiny_attention_config)

        assert len(cache) == 2
        assert cache.mixer_types == ["attention", "attention"]
        assert all(isinstance(l, _AttentionCache) for l in cache.layers)

    def test_ssm_cache_creation(self, ssm_config):
        """Create cache for pure SSM config."""
        cache = Apriel2Cache(ssm_config)

        assert len(cache) == 2
        assert cache.mixer_types == ["mamba", "mamba"]
        assert all(isinstance(l, _SSMCache) for l in cache.layers)

    def test_kda_cache_creation(self, kda_config):
        """Create cache for pure KDA config."""
        cache = Apriel2Cache(kda_config)

        assert len(cache) == 2
        assert cache.mixer_types == ["kda", "kda"]
        assert all(isinstance(l, _SSMCache) for l in cache.layers)

    def test_stochastic_cache_creation(self, stochastic_config):
        """Create cache for stochastic mixer config."""
        cache = Apriel2Cache(stochastic_config)

        assert len(cache) == 2
        # Layer 0: pure attention, Layer 1: stochastic (dict)
        assert isinstance(cache.layers[0], _AttentionCache)
        assert isinstance(cache.layers[1], dict)
        assert set(cache.layers[1].keys()) == {"attention", "mamba"}

    def test_swa_window_captured(self, swa_config):
        """Verify sliding window size is captured."""
        cache = Apriel2Cache(swa_config)

        assert cache.layers[0].window == 8
        assert cache.is_sliding == [True, True]

    def test_active_mixers_initialized_none(self, stochastic_config):
        """Verify active_mixers starts as None for all layers."""
        cache = Apriel2Cache(stochastic_config)

        assert cache.active_mixers == [None, None]


class TestCacheProperties:
    """Test cache property accessors."""

    def test_empty_cache_properties(self, tiny_attention_config):
        """Test properties of uninitialized cache."""
        cache = Apriel2Cache(tiny_attention_config)

        assert cache.is_initialized == False
        assert cache.has_previous_state == False
        assert cache.max_batch_size is None
        assert cache.max_cache_len is None
        assert cache.is_compileable == False

    def test_is_initialized_attention(self, tiny_attention_config, sample_kv):
        """is_initialized detects attention cache."""
        cache = Apriel2Cache(tiny_attention_config)
        cache.update(*sample_kv, layer_idx=0)

        assert cache.is_initialized == True

    def test_is_initialized_ssm(self, ssm_config, sample_conv_single):
        """is_initialized detects SSM cache."""
        cache = Apriel2Cache(ssm_config)
        cache.conv_states[0] = sample_conv_single

        assert cache.is_initialized == True

    def test_has_previous_state_ssm_only(self, ssm_config, sample_conv_single):
        """has_previous_state only looks at SSM conv states."""
        cache = Apriel2Cache(ssm_config)

        assert cache.has_previous_state == False
        cache.conv_states[0] = sample_conv_single
        assert cache.has_previous_state == True

    def test_has_previous_state_ignores_attention(self, tiny_attention_config, sample_kv):
        """has_previous_state ignores attention cache."""
        cache = Apriel2Cache(tiny_attention_config)
        cache.update(*sample_kv, layer_idx=0)

        # Attention cache is set, but has_previous_state only checks SSM
        assert cache.has_previous_state == False

    def test_max_batch_size_from_attention(self, tiny_attention_config, sample_kv):
        """max_batch_size from attention cache."""
        cache = Apriel2Cache(tiny_attention_config)
        cache.update(*sample_kv, layer_idx=0)

        assert cache.max_batch_size == 2

    def test_max_batch_size_from_ssm(self, ssm_config, sample_conv_single):
        """max_batch_size from SSM cache."""
        cache = Apriel2Cache(ssm_config)
        cache.conv_states[0] = sample_conv_single

        assert cache.max_batch_size == 2

    def test_max_batch_size_from_kda_tuple(self, kda_config, sample_conv_tuple):
        """max_batch_size from KDA tuple conv state."""
        cache = Apriel2Cache(kda_config)
        cache.conv_states[0] = sample_conv_tuple

        assert cache.max_batch_size == 2

    def test_max_cache_len_single_window(self, swa_config):
        """max_cache_len with single window size."""
        cache = Apriel2Cache(swa_config)
        assert cache.max_cache_len == 8

    def test_max_cache_len_multiple_windows(self, multi_window_config):
        """max_cache_len returns minimum window."""
        cache = Apriel2Cache(multi_window_config)
        assert cache.max_cache_len == 512  # min(512, 2048)

    def test_max_cache_len_no_windows(self, tiny_attention_config):
        """max_cache_len is None when no windows."""
        cache = Apriel2Cache(tiny_attention_config)
        assert cache.max_cache_len is None

    def test_is_sliding_mixed(self, multi_window_config):
        """is_sliding reflects per-layer window presence."""
        cache = Apriel2Cache(multi_window_config)
        assert cache.is_sliding == [False, True, True]


# =============================================================================
# SECTION 2: ATTENTION CACHE OPERATIONS
# =============================================================================


class TestAttentionCacheBasics:
    """Test basic attention cache operations."""

    def test_update_stores_kv(self, tiny_attention_config, sample_kv):
        """update() stores key/value states."""
        cache = Apriel2Cache(tiny_attention_config)
        key, value = sample_kv

        k_out, v_out = cache.update(key, value, layer_idx=0)

        torch.testing.assert_close(k_out, key)
        torch.testing.assert_close(v_out, value)
        assert cache.get_seq_length(0) == 10

    def test_update_concatenates(self, tiny_attention_config, sample_kv):
        """Subsequent updates concatenate."""
        cache = Apriel2Cache(tiny_attention_config)
        key, value = sample_kv

        cache.update(key, value, layer_idx=0)
        k_out, v_out = cache.update(key, value, layer_idx=0)

        assert k_out.shape[-2] == 20
        assert cache.get_seq_length(0) == 20

    def test_key_value_cache_accessors(self, tiny_attention_config, sample_kv):
        """Test key_cache and value_cache accessors."""
        cache = Apriel2Cache(tiny_attention_config)
        cache.update(*sample_kv, layer_idx=0)

        assert cache.key_cache[0] is not None
        assert cache.value_cache[0] is not None
        torch.testing.assert_close(cache.key_cache[0], sample_kv[0])


class TestSlidingWindowAttention:
    """Test sliding window attention behavior."""

    def test_initial_within_window(self, swa_config):
        """Initial sequence within window is kept."""
        cache = Apriel2Cache(swa_config)
        key = torch.randn(2, 4, 5, 16)  # seq=5 < window=8
        value = torch.randn(2, 4, 5, 16)

        cache.update(key, value, layer_idx=0)

        assert cache.get_seq_length(0) == 5

    def test_initial_exceeds_window(self, swa_config):
        """Initial sequence > window is truncated to last window tokens."""
        cache = Apriel2Cache(swa_config)
        key = torch.arange(12).float().view(1, 1, 12, 1).expand(2, 4, 12, 16)
        value = key.clone()

        k_out, v_out = cache.update(key, value, layer_idx=0)

        assert cache.get_seq_length(0) == 8
        # Should keep tokens 4-11 (last 8)
        assert k_out[0, 0, 0, 0].item() == 4.0

    def test_single_token_roll_path(self, swa_config):
        """Single token decode with full window uses efficient roll."""
        cache = Apriel2Cache(swa_config)

        # Fill window exactly
        key1 = torch.arange(8).float().view(1, 1, 8, 1).expand(2, 4, 8, 16)
        cache.update(key1, key1.clone(), layer_idx=0)

        # Decode single token
        key2 = torch.full((2, 4, 1, 16), 8.0)
        k_out, _ = cache.update(key2, key2.clone(), layer_idx=0)

        assert cache.get_seq_length(0) == 8
        assert k_out[0, 0, 0, 0].item() == 1.0  # Token 0 rolled out
        assert k_out[0, 0, 7, 0].item() == 8.0  # New token at end

    def test_multi_token_cat_slice_path(self, swa_config):
        """Multiple tokens use cat+slice path."""
        cache = Apriel2Cache(swa_config)

        # Fill window
        key1 = torch.randn(2, 4, 8, 16)
        cache.update(key1, key1.clone(), layer_idx=0)

        # Add 3 tokens
        key2 = torch.randn(2, 4, 3, 16)
        k_out, _ = cache.update(key2, key2.clone(), layer_idx=0)

        assert cache.get_seq_length(0) == 8
        torch.testing.assert_close(k_out[..., -3:, :], key2)

    def test_partial_then_fill_then_overflow(self, swa_config):
        """Progressive filling: partial → full → overflow."""
        cache = Apriel2Cache(swa_config)

        cache.update(torch.randn(2, 4, 5, 16), torch.randn(2, 4, 5, 16), layer_idx=0)
        assert cache.get_seq_length(0) == 5

        cache.update(torch.randn(2, 4, 3, 16), torch.randn(2, 4, 3, 16), layer_idx=0)
        assert cache.get_seq_length(0) == 8

        cache.update(torch.randn(2, 4, 2, 16), torch.randn(2, 4, 2, 16), layer_idx=0)
        assert cache.get_seq_length(0) == 8

    def test_contiguous_output(self, swa_config):
        """Outputs are contiguous after windowing."""
        cache = Apriel2Cache(swa_config)

        cache.update(torch.randn(2, 4, 10, 16), torch.randn(2, 4, 10, 16), layer_idx=0)
        cache.update(torch.randn(2, 4, 5, 16), torch.randn(2, 4, 5, 16), layer_idx=0)

        assert cache.layers[0].key.is_contiguous()
        assert cache.layers[0].value.is_contiguous()


# =============================================================================
# SECTION 3: SSM CACHE OPERATIONS
# =============================================================================


class TestSSMCacheBasics:
    """Test basic SSM cache operations."""

    def test_conv_states_accessor(self, ssm_config, sample_conv_single):
        """Test conv_states accessor."""
        cache = Apriel2Cache(ssm_config)

        cache.conv_states[0] = sample_conv_single
        torch.testing.assert_close(cache.conv_states[0], sample_conv_single)

    def test_recurrent_states_accessor(self, ssm_config, sample_recurrent):
        """Test recurrent_states accessor."""
        cache = Apriel2Cache(ssm_config)

        cache.recurrent_states[0] = sample_recurrent
        torch.testing.assert_close(cache.recurrent_states[0], sample_recurrent)

    def test_ssm_seq_length_always_zero(self, ssm_config, sample_conv_single):
        """get_seq_length returns 0 for SSM (no KV cache)."""
        cache = Apriel2Cache(ssm_config)
        cache.conv_states[0] = sample_conv_single

        assert cache.get_seq_length(0) == 0


class TestKDACache:
    """Test KDA-specific cache operations with tuple conv states."""

    def test_tuple_conv_storage(self, kda_config, sample_conv_tuple):
        """KDA stores tuple conv states."""
        cache = Apriel2Cache(kda_config)

        cache.conv_states[0] = sample_conv_tuple

        assert isinstance(cache.conv_states[0], tuple)
        assert len(cache.conv_states[0]) == 3
        for i in range(3):
            torch.testing.assert_close(cache.conv_states[0][i], sample_conv_tuple[i])

    def test_tuple_with_recurrent(self, kda_config, sample_conv_tuple, sample_recurrent):
        """KDA can have both tuple conv and recurrent states."""
        cache = Apriel2Cache(kda_config)

        cache.conv_states[0] = sample_conv_tuple
        cache.recurrent_states[0] = sample_recurrent

        assert isinstance(cache.conv_states[0], tuple)
        assert cache.recurrent_states[0] is not None

    def test_has_previous_state_detects_tuple(self, kda_config, sample_conv_tuple):
        """has_previous_state works with tuple conv states."""
        cache = Apriel2Cache(kda_config)

        assert cache.has_previous_state == False
        cache.conv_states[0] = sample_conv_tuple
        assert cache.has_previous_state == True


# =============================================================================
# SECTION 4: STOCHASTIC ROUTING
# =============================================================================


class TestStochasticRouting:
    """Test stochastic mixer cache routing."""

    def test_set_active_mixer(self, stochastic_config):
        """set_active_mixer sets the pointer."""
        cache = Apriel2Cache(stochastic_config)

        cache.set_active_mixer(1, "attention")
        assert cache.active_mixers[1] == "attention"

        cache.set_active_mixer(1, "mamba")
        assert cache.active_mixers[1] == "mamba"

    def test_operations_route_to_active(self, stochastic_config, sample_kv):
        """Operations route to currently active mixer."""
        cache = Apriel2Cache(stochastic_config)

        cache.set_active_mixer(1, "attention")
        cache.update(*sample_kv, layer_idx=1)
        attn_len = cache.get_seq_length(1)

        cache.set_active_mixer(1, "mamba")
        mamba_len = cache.get_seq_length(1)

        assert attn_len == 10
        assert mamba_len == 0  # Mamba cache is separate and empty

    def test_each_mixer_independent_cache(self, stochastic_config, sample_kv, sample_conv_single):
        """Each mixer maintains independent cache."""
        cache = Apriel2Cache(stochastic_config)

        # Fill attention cache
        cache.set_active_mixer(1, "attention")
        cache.update(*sample_kv, layer_idx=1)

        # Fill mamba cache
        cache.set_active_mixer(1, "mamba")
        cache.conv_states[1] = sample_conv_single

        # Both preserved
        cache.set_active_mixer(1, "attention")
        assert cache.get_seq_length(1) == 10

        cache.set_active_mixer(1, "mamba")
        torch.testing.assert_close(cache.conv_states[1], sample_conv_single)


class TestMixerSwitching:
    """Test behavior when switching between mixers mid-generation."""

    def test_switch_preserves_previous_state(self, stochastic_config, sample_kv):
        """Switching mixers preserves previous mixer's state."""
        cache = Apriel2Cache(stochastic_config)

        cache.set_active_mixer(1, "attention")
        cache.update(*sample_kv, layer_idx=1)
        original_key = cache.layers[1]["attention"].key.clone()

        # Switch to mamba, do something
        cache.set_active_mixer(1, "mamba")
        cache.conv_states[1] = torch.randn(2, 128, 4)

        # Switch back - attention unchanged
        cache.set_active_mixer(1, "attention")
        torch.testing.assert_close(cache.layers[1]["attention"].key, original_key)

    def test_switch_does_not_copy_state(self, stochastic_config, sample_kv):
        """Switching does NOT copy state between mixers."""
        cache = Apriel2Cache(stochastic_config)

        # Fill attention with 10 tokens
        cache.set_active_mixer(1, "attention")
        cache.update(*sample_kv, layer_idx=1)

        # Switch to mamba - it has NO history from attention
        cache.set_active_mixer(1, "mamba")
        assert cache.conv_states[1] is None
        assert cache.recurrent_states[1] is None

    def test_has_previous_state_checks_all_sub_caches(self, stochastic_config):
        """has_previous_state checks ALL sub-caches, not just active."""
        cache = Apriel2Cache(stochastic_config)

        cache.set_active_mixer(1, "mamba")
        cache.conv_states[1] = torch.randn(2, 128, 4)

        # Even if we switch away, has_previous_state still detects it
        cache.set_active_mixer(1, "attention")
        assert cache.has_previous_state == True


class TestAllMixerTypes:
    """Test cache isolation across all 5 mixer types."""

    def test_all_five_mixer_types_isolated(self, all_mixers_config):
        """All 5 mixer types maintain isolated caches."""
        cache = Apriel2Cache(all_mixers_config)
        layer_idx = 1  # Stochastic layer

        # Fill each mixer's cache
        cache.set_active_mixer(layer_idx, "attention")
        attn_kv = (torch.randn(2, 4, 10, 16), torch.randn(2, 4, 10, 16))
        cache.update(*attn_kv, layer_idx=layer_idx)

        cache.set_active_mixer(layer_idx, "swa")
        swa_kv = (torch.randn(2, 4, 5, 16), torch.randn(2, 4, 5, 16))
        cache.update(*swa_kv, layer_idx=layer_idx)

        cache.set_active_mixer(layer_idx, "mamba")
        mamba_conv = torch.randn(2, 128, 4)
        cache.conv_states[layer_idx] = mamba_conv

        cache.set_active_mixer(layer_idx, "gdn")
        gdn_conv = torch.randn(2, 64, 3)
        cache.conv_states[layer_idx] = gdn_conv

        cache.set_active_mixer(layer_idx, "kda")
        kda_conv = (torch.randn(2, 64, 3), torch.randn(2, 64, 3), torch.randn(2, 64, 3))
        cache.conv_states[layer_idx] = kda_conv

        # Verify all preserved
        cache.set_active_mixer(layer_idx, "attention")
        assert cache.get_seq_length(layer_idx) == 10

        cache.set_active_mixer(layer_idx, "swa")
        assert cache.get_seq_length(layer_idx) == 5

        cache.set_active_mixer(layer_idx, "mamba")
        torch.testing.assert_close(cache.conv_states[layer_idx], mamba_conv)

        cache.set_active_mixer(layer_idx, "gdn")
        torch.testing.assert_close(cache.conv_states[layer_idx], gdn_conv)

        cache.set_active_mixer(layer_idx, "kda")
        assert isinstance(cache.conv_states[layer_idx], tuple)


# =============================================================================
# SECTION 5: CACHE INVALIDATION
# =============================================================================


class TestCacheInvalidation:
    """Test cache invalidation and reset semantics.

    Key principle: Each mixer maintains independent state. To invalidate:
    - reset() clears ALL caches across ALL layers and mixers
    - There is no per-mixer reset (by design - each mixer is independent)
    """

    def test_reset_clears_attention(self, tiny_attention_config, sample_kv):
        """reset() clears attention cache."""
        cache = Apriel2Cache(tiny_attention_config)
        cache.update(*sample_kv, layer_idx=0)

        cache.reset()

        assert cache.is_initialized == False
        assert cache.get_seq_length(0) == 0

    def test_reset_clears_ssm(self, ssm_config, sample_conv_single, sample_recurrent):
        """reset() clears SSM cache."""
        cache = Apriel2Cache(ssm_config)
        cache.conv_states[0] = sample_conv_single
        cache.recurrent_states[0] = sample_recurrent

        cache.reset()

        assert cache.has_previous_state == False
        assert cache.conv_states[0] is None
        assert cache.recurrent_states[0] is None

    def test_reset_clears_kda_tuple(self, kda_config, sample_conv_tuple):
        """reset() clears KDA tuple conv states."""
        cache = Apriel2Cache(kda_config)
        cache.conv_states[0] = sample_conv_tuple

        cache.reset()

        assert cache.conv_states[0] is None

    def test_reset_clears_all_stochastic_mixers(self, all_mixers_config):
        """reset() clears ALL mixer caches in stochastic layer."""
        cache = Apriel2Cache(all_mixers_config)
        layer_idx = 1

        # Fill all mixers
        cache.set_active_mixer(layer_idx, "attention")
        cache.update(torch.randn(2, 4, 10, 16), torch.randn(2, 4, 10, 16), layer_idx=layer_idx)

        cache.set_active_mixer(layer_idx, "mamba")
        cache.conv_states[layer_idx] = torch.randn(2, 128, 4)

        cache.set_active_mixer(layer_idx, "kda")
        cache.conv_states[layer_idx] = (torch.randn(2, 64, 3),) * 3

        cache.reset()

        # All cleared
        assert cache.layers[layer_idx]["attention"].key is None
        assert cache.layers[layer_idx]["mamba"].conv is None
        assert cache.layers[layer_idx]["kda"].conv is None

    def test_crop_truncates_attention(self, tiny_attention_config, sample_kv):
        """crop() truncates attention cache to max_length."""
        cache = Apriel2Cache(tiny_attention_config)
        cache.update(*sample_kv, layer_idx=0)

        cache.crop(5)

        assert cache.get_seq_length(0) == 5

    def test_crop_affects_all_layers(self, tiny_attention_config, sample_kv):
        """crop() affects all layers."""
        cache = Apriel2Cache(tiny_attention_config)
        cache.update(*sample_kv, layer_idx=0)
        cache.update(*sample_kv, layer_idx=1)

        cache.crop(3)

        assert cache.get_seq_length(0) == 3
        assert cache.get_seq_length(1) == 3

    def test_crop_ignores_ssm(self, ssm_config, sample_conv_single):
        """crop() only affects attention, not SSM."""
        cache = Apriel2Cache(ssm_config)
        cache.conv_states[0] = sample_conv_single

        cache.crop(5)  # Should not crash

        # Conv state unchanged
        torch.testing.assert_close(cache.conv_states[0], sample_conv_single)


# =============================================================================
# SECTION 6: BEAM SEARCH OPERATIONS
# =============================================================================


class TestBatchRepeatInterleave:
    """Test batch_repeat_interleave for beam search expansion."""

    def test_repeat_attention(self, tiny_attention_config, sample_kv):
        """Repeat attention cache for beam search."""
        cache = Apriel2Cache(tiny_attention_config)
        cache.update(*sample_kv, layer_idx=0)

        cache.batch_repeat_interleave(3)

        assert cache.max_batch_size == 6  # 2 * 3

    def test_repeat_ssm(self, ssm_config, sample_conv_single, sample_recurrent):
        """Repeat SSM cache for beam search."""
        cache = Apriel2Cache(ssm_config)
        cache.conv_states[0] = sample_conv_single
        cache.recurrent_states[0] = sample_recurrent

        cache.batch_repeat_interleave(4)

        assert cache.conv_states[0].shape[0] == 8  # 2 * 4
        assert cache.recurrent_states[0].shape[0] == 8

    def test_repeat_kda_tuple(self, kda_config, sample_conv_tuple):
        """Repeat KDA tuple conv states."""
        cache = Apriel2Cache(kda_config)
        cache.conv_states[0] = sample_conv_tuple

        cache.batch_repeat_interleave(3)

        for c in cache.conv_states[0]:
            assert c.shape[0] == 6

    def test_repeat_stochastic_all_mixers(self, all_mixers_config):
        """Repeat all mixer caches in stochastic layer."""
        cache = Apriel2Cache(all_mixers_config)
        layer_idx = 1

        cache.set_active_mixer(layer_idx, "attention")
        cache.update(torch.randn(2, 4, 10, 16), torch.randn(2, 4, 10, 16), layer_idx=layer_idx)

        cache.set_active_mixer(layer_idx, "mamba")
        cache.conv_states[layer_idx] = torch.randn(2, 128, 4)

        cache.batch_repeat_interleave(2)

        cache.set_active_mixer(layer_idx, "attention")
        assert cache.layers[layer_idx]["attention"].key.shape[0] == 4

        cache.set_active_mixer(layer_idx, "mamba")
        assert cache.conv_states[layer_idx].shape[0] == 4

    def test_repeat_skips_none(self, tiny_attention_config):
        """Repeat gracefully skips None caches."""
        cache = Apriel2Cache(tiny_attention_config)
        # Don't fill anything

        cache.batch_repeat_interleave(3)  # Should not crash

        assert cache.max_batch_size is None


class TestReorderCache:
    """Test reorder_cache for beam search hypothesis selection."""

    def test_reorder_attention(self, tiny_attention_config, sample_kv):
        """Reorder attention cache."""
        cache = Apriel2Cache(tiny_attention_config)
        key, value = sample_kv
        # Make batches distinguishable
        key = torch.arange(2).float().view(2, 1, 1, 1).expand(2, 4, 10, 16)
        cache.update(key, key.clone(), layer_idx=0)

        beam_idx = torch.tensor([1, 0])
        cache.reorder_cache(beam_idx)

        assert cache.layers[0].key[0, 0, 0, 0].item() == 1.0
        assert cache.layers[0].key[1, 0, 0, 0].item() == 0.0

    def test_reorder_ssm(self, ssm_config):
        """Reorder SSM cache."""
        cache = Apriel2Cache(ssm_config)
        conv = torch.arange(2).float().view(2, 1, 1).expand(2, 128, 4)
        cache.conv_states[0] = conv.clone()

        beam_idx = torch.tensor([1, 0])
        cache.reorder_cache(beam_idx)

        assert cache.conv_states[0][0, 0, 0].item() == 1.0

    def test_reorder_kda_tuple(self, kda_config):
        """Reorder KDA tuple conv states."""
        cache = Apriel2Cache(kda_config)
        conv_q = torch.arange(2).float().view(2, 1, 1).expand(2, 64, 3)
        cache.conv_states[0] = (conv_q.clone(), conv_q.clone(), conv_q.clone())

        beam_idx = torch.tensor([1, 0])
        cache.reorder_cache(beam_idx)

        for c in cache.conv_states[0]:
            assert c[0, 0, 0].item() == 1.0


class TestBatchSelectIndices:
    """Test batch_select_indices for beam selection."""

    def test_select_attention(self, tiny_attention_config, sample_kv):
        """Select subset of attention cache."""
        cache = Apriel2Cache(tiny_attention_config)
        key = torch.arange(4).float().view(4, 1, 1, 1).expand(4, 4, 10, 16)
        cache.update(key, key.clone(), layer_idx=0)

        indices = torch.tensor([0, 3])
        cache.batch_select_indices(indices)

        assert cache.max_batch_size == 2
        assert cache.layers[0].key[0, 0, 0, 0].item() == 0.0
        assert cache.layers[0].key[1, 0, 0, 0].item() == 3.0

    def test_select_kda_tuple(self, kda_config):
        """Select subset of KDA tuple conv states."""
        cache = Apriel2Cache(kda_config)
        conv = tuple(torch.arange(4).float().view(4, 1, 1).expand(4, 64, 3).clone() for _ in range(3))
        cache.conv_states[0] = conv

        indices = torch.tensor([1, 2])
        cache.batch_select_indices(indices)

        for c in cache.conv_states[0]:
            assert c.shape[0] == 2
            assert c[0, 0, 0].item() == 1.0


# =============================================================================
# SECTION 7: HUGGINGFACE INTEGRATION
# =============================================================================


class TestGetMaskSizes:
    """Test get_mask_sizes() for attention mask computation."""

    def test_empty_cache(self, tiny_attention_config):
        """Mask sizes with empty cache."""
        cache = Apriel2Cache(tiny_attention_config)
        cache_position = torch.arange(10)

        kv_length, kv_offset = cache.get_mask_sizes(cache_position, layer_idx=0)

        assert kv_length == 10
        assert kv_offset == 0

    def test_with_cached_tokens(self, tiny_attention_config, sample_kv):
        """Mask sizes with cached tokens."""
        cache = Apriel2Cache(tiny_attention_config)
        cache.update(*sample_kv, layer_idx=0)  # 10 tokens

        cache_position = torch.arange(5)
        kv_length, kv_offset = cache.get_mask_sizes(cache_position, layer_idx=0)

        assert kv_length == 15  # 10 + 5
        assert kv_offset == 10

    def test_single_token_decode(self, tiny_attention_config, sample_kv):
        """Mask sizes for single token decode."""
        cache = Apriel2Cache(tiny_attention_config)
        cache.update(*sample_kv, layer_idx=0)

        cache_position = torch.arange(1)
        kv_length, kv_offset = cache.get_mask_sizes(cache_position, layer_idx=0)

        assert kv_length == 11
        assert kv_offset == 10

    def test_ssm_returns_query_only(self, ssm_config, sample_conv_single):
        """SSM layers return query_length (no KV cache)."""
        cache = Apriel2Cache(ssm_config)
        cache.conv_states[0] = sample_conv_single

        cache_position = torch.arange(5)
        kv_length, kv_offset = cache.get_mask_sizes(cache_position, layer_idx=0)

        assert kv_length == 5
        assert kv_offset == 0


class TestCacheIndexing:
    """Test cache[idx] indexing."""

    def test_attention_returns_kv(self, tiny_attention_config, sample_kv):
        """Indexing attention layer returns (key, value)."""
        cache = Apriel2Cache(tiny_attention_config)
        cache.update(*sample_kv, layer_idx=0)

        result = cache[0]

        assert isinstance(result, tuple)
        torch.testing.assert_close(result[0], sample_kv[0])

    def test_empty_returns_empty_tensors(self, tiny_attention_config):
        """Indexing empty layer returns empty tensors."""
        cache = Apriel2Cache(tiny_attention_config)

        result = cache[0]

        assert result[0].numel() == 0
        assert result[1].numel() == 0

    def test_ssm_returns_empty(self, ssm_config, sample_conv_single):
        """Indexing SSM layer returns empty (no KV)."""
        cache = Apriel2Cache(ssm_config)
        cache.conv_states[0] = sample_conv_single

        result = cache[0]

        assert result[0].numel() == 0

    def test_stochastic_attention_returns_kv(self, stochastic_config, sample_kv):
        """Indexing stochastic with attention active returns KV."""
        cache = Apriel2Cache(stochastic_config)
        cache.set_active_mixer(1, "attention")
        cache.update(*sample_kv, layer_idx=1)

        result = cache[1]

        torch.testing.assert_close(result[0], sample_kv[0])


# =============================================================================
# SECTION 8: GENERATION PATTERNS
# =============================================================================


class TestGenerationPatterns:
    """Test real-world generation patterns."""

    def test_prefill_then_decode(self, tiny_attention_config, sample_kv):
        """Prefill with long prompt, then decode token-by-token."""
        cache = Apriel2Cache(tiny_attention_config)
        cache.update(*sample_kv, layer_idx=0)  # Prefill 10 tokens

        for _ in range(5):
            new_kv = (torch.randn(2, 4, 1, 16), torch.randn(2, 4, 1, 16))
            cache.update(*new_kv, layer_idx=0)

        assert cache.get_seq_length(0) == 15

    def test_crop_then_continue(self, tiny_attention_config, sample_kv):
        """Crop old context, continue generation."""
        cache = Apriel2Cache(tiny_attention_config)
        cache.update(*sample_kv, layer_idx=0)
        cache.update(*sample_kv, layer_idx=0)  # 20 tokens

        cache.crop(5)  # Keep last 5
        cache.update(torch.randn(2, 4, 3, 16), torch.randn(2, 4, 3, 16), layer_idx=0)

        assert cache.get_seq_length(0) == 8

    def test_reset_between_generations(self, tiny_attention_config, sample_kv):
        """Reset between independent generations."""
        cache = Apriel2Cache(tiny_attention_config)

        # First generation
        cache.update(*sample_kv, layer_idx=0)
        assert cache.is_initialized == True

        # Reset
        cache.reset()
        assert cache.is_initialized == False

        # Second generation
        cache.update(*sample_kv, layer_idx=0)
        assert cache.get_seq_length(0) == 10

    def test_multi_layer_consistency(self, tiny_attention_config, sample_kv):
        """All layers updated consistently."""
        cache = Apriel2Cache(tiny_attention_config)

        for layer_idx in range(2):
            cache.update(*sample_kv, layer_idx=layer_idx)
            cache.update(torch.randn(2, 4, 1, 16), torch.randn(2, 4, 1, 16), layer_idx=layer_idx)

        for layer_idx in range(2):
            assert cache.get_seq_length(layer_idx) == 11


# =============================================================================
# SECTION 9: ERROR HANDLING
# =============================================================================


class TestErrorHandling:
    """Test error conditions and guards."""

    def test_stochastic_update_without_active_mixer(self, stochastic_config):
        """update() on stochastic without active_mixer raises."""
        cache = Apriel2Cache(stochastic_config)

        with pytest.raises(RuntimeError, match="needs active_mixer set"):
            cache.update(torch.randn(2, 4, 10, 16), torch.randn(2, 4, 10, 16), layer_idx=1)

    def test_stochastic_accessor_without_active_mixer(self, stochastic_config):
        """Accessing stochastic cache without active_mixer raises."""
        cache = Apriel2Cache(stochastic_config)

        with pytest.raises(RuntimeError, match="requires set_active_mixer"):
            _ = cache.conv_states[1]

    def test_accessor_error_lists_available_mixers(self, stochastic_config):
        """Error message lists available mixers."""
        cache = Apriel2Cache(stochastic_config)

        with pytest.raises(RuntimeError, match="Available mixers:"):
            _ = cache.key_cache[1]

    def test_invalid_mixer_name(self, stochastic_config):
        """Invalid mixer name raises KeyError on access."""
        cache = Apriel2Cache(stochastic_config)
        cache.set_active_mixer(1, "nonexistent")

        with pytest.raises(KeyError):
            cache.update(torch.randn(2, 4, 10, 16), torch.randn(2, 4, 10, 16), layer_idx=1)

    def test_layer_idx_out_of_bounds(self, tiny_attention_config):
        """Out-of-bounds layer_idx raises IndexError."""
        cache = Apriel2Cache(tiny_attention_config)

        with pytest.raises(IndexError):
            cache.update(torch.randn(2, 4, 10, 16), torch.randn(2, 4, 10, 16), layer_idx=999)


# =============================================================================
# SECTION 10: INTERNAL CLASSES
# =============================================================================


class TestAttentionCacheInternal:
    """Test internal _AttentionCache class directly."""

    def test_unbounded_growth(self):
        """No window allows unbounded growth."""
        cache = _AttentionCache(window=None)

        for _ in range(10):
            cache.update(torch.randn(2, 4, 100, 16), torch.randn(2, 4, 100, 16))

        assert cache.key.shape[-2] == 1000

    def test_window_enforced(self):
        """Window caps cache size."""
        cache = _AttentionCache(window=50)

        for _ in range(10):
            cache.update(torch.randn(2, 4, 100, 16), torch.randn(2, 4, 100, 16))

        assert cache.key.shape[-2] == 50


class TestSSMCacheInternal:
    """Test internal _SSMCache class directly."""

    def test_initial_none(self):
        """Initial states are None."""
        cache = _SSMCache()

        assert cache.conv is None
        assert cache.recurrent is None

    def test_stores_tuple(self):
        """Can store tuple (for KDA)."""
        cache = _SSMCache()
        cache.conv = (torch.randn(2, 64, 3),) * 3

        assert isinstance(cache.conv, tuple)
