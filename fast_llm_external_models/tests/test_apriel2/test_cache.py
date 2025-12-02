"""Unit tests for Apriel2Cache."""

import pytest
import torch
from fast_llm_external_models.apriel2.cache import Apriel2Cache


class TestCacheBasics:
    """Test basic cache creation and properties."""

    def test_cache_creation(self, apriel2_config_tiny):
        """Test cache creation from config."""
        cache = Apriel2Cache(apriel2_config_tiny)
        assert len(cache) == apriel2_config_tiny.num_hidden_layers
        assert cache.is_compileable == False
        assert cache.is_initialized == False
        assert isinstance(cache.is_sliding, list)
        assert len(cache.is_sliding) == apriel2_config_tiny.num_hidden_layers

    def test_cache_properties_empty(self, apriel2_cache):
        """Test cache properties when empty."""
        assert apriel2_cache.is_initialized == False
        assert apriel2_cache.has_previous_state == False
        assert apriel2_cache.max_batch_size is None
        assert apriel2_cache.max_cache_len is None


class TestAttentionCache:
    """Test attention cache operations."""

    def test_attention_update(self, apriel2_cache, sample_attention_states):
        """Test updating attention cache."""
        key, value = sample_attention_states
        k_out, v_out = apriel2_cache.update(key, value, layer_idx=0)

        assert k_out.shape == key.shape
        assert v_out.shape == value.shape
        assert apriel2_cache.is_initialized == True
        assert apriel2_cache.get_seq_length(0) == key.shape[2]

    def test_attention_concatenation(self, apriel2_cache, sample_attention_states):
        """Test that cache concatenates new states."""
        key1, value1 = sample_attention_states
        apriel2_cache.update(key1, value1, layer_idx=0)

        # Add more tokens
        key2 = torch.randn(2, 8, 5, 64)
        value2 = torch.randn(2, 8, 5, 64)
        k_out, v_out = apriel2_cache.update(key2, value2, layer_idx=0)

        assert k_out.shape[2] == 15  # 10 + 5
        assert apriel2_cache.get_seq_length(0) == 15


class TestSSMCache:
    """Test SSM cache operations."""

    def test_ssm_direct_access(self, apriel2_config_stochastic):
        """Test direct SSM state access."""
        cache = Apriel2Cache(apriel2_config_stochastic)

        # Set active mixer to mamba
        cache.set_active_mixer(1, "mamba")

        # Set conv states
        conv = torch.randn(2, 128, 4)
        cache.conv_states[1] = conv

        # Retrieve and verify
        retrieved = cache.conv_states[1]
        assert retrieved is not None
        assert torch.allclose(retrieved, conv)


class TestStochasticMixer:
    """Test stochastic mixer cache routing."""

    def test_set_active_mixer(self, apriel2_config_stochastic):
        """Test setting active mixer."""
        cache = Apriel2Cache(apriel2_config_stochastic)
        cache.set_active_mixer(1, "attention")
        assert cache.active_mixers[1] == "attention"

    def test_routing_to_different_mixers(self, apriel2_config_stochastic, sample_attention_states):
        """Test that different mixers use separate caches."""
        cache = Apriel2Cache(apriel2_config_stochastic)
        key, value = sample_attention_states

        # Use attention mixer
        cache.set_active_mixer(1, "attention")
        cache.update(key, value, layer_idx=1)
        attn_len = cache.get_seq_length(1)

        # Switch to mamba mixer - should have empty cache
        cache.set_active_mixer(1, "mamba")
        mamba_len = cache.get_seq_length(1)

        assert attn_len == 10
        assert mamba_len == 0  # Different cache


class TestBeamSearch:
    """Test beam search operations."""

    def test_batch_repeat_interleave(self, apriel2_cache, sample_attention_states):
        """Test repeating cache for beam search."""
        key, value = sample_attention_states
        apriel2_cache.update(key, value, layer_idx=0)

        apriel2_cache.batch_repeat_interleave(2)
        assert apriel2_cache.max_batch_size == 4  # 2 * 2

    def test_reorder_cache(self, apriel2_cache, sample_attention_states):
        """Test reordering cache for beam search."""
        key, value = sample_attention_states
        apriel2_cache.update(key, value, layer_idx=0)

        beam_idx = torch.tensor([1, 0])
        apriel2_cache.reorder_cache(beam_idx)

        # Cache should still be valid
        assert apriel2_cache.is_initialized == True


class TestCacheReset:
    """Test cache reset operations."""

    def test_reset(self, apriel2_cache, sample_attention_states):
        """Test resetting cache."""
        key, value = sample_attention_states
        apriel2_cache.update(key, value, layer_idx=0)

        assert apriel2_cache.is_initialized == True

        apriel2_cache.reset()

        assert apriel2_cache.is_initialized == False
        assert apriel2_cache.get_seq_length(0) == 0

    def test_crop(self, apriel2_cache, sample_attention_states):
        """Test cropping cache to max length."""
        key, value = sample_attention_states
        apriel2_cache.update(key, value, layer_idx=0)

        apriel2_cache.crop(5)
        assert apriel2_cache.get_seq_length(0) == 5
