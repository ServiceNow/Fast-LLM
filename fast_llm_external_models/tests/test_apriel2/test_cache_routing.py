"""Tests for stochastic mixer cache routing and bug fixes."""

import pytest
import torch
from fast_llm_external_models.apriel2.cache import Apriel2Cache


class TestHasPreviousState:
    """Test has_previous_state property with stochastic mixers."""

    def test_checks_all_sub_caches(self, apriel2_config_stochastic):
        """Test that has_previous_state checks ALL sub-caches, not just main mixer."""
        cache = Apriel2Cache(apriel2_config_stochastic)

        # Initially no SSM state
        assert cache.has_previous_state == False

        # Set active mixer to mamba (NOT the main mixer which is attention)
        cache.set_active_mixer(1, "mamba")
        cache.conv_states[1] = torch.randn(2, 128, 4)

        # Should detect SSM state even though main mixer is "attention"
        assert cache.has_previous_state == True

    def test_detects_any_ssm_cache(self, apriel2_config_multi_mixer):
        """Test that has_previous_state detects SSM state in any sub-cache."""
        cache = Apriel2Cache(apriel2_config_multi_mixer)

        # Fill mamba_v1
        cache.set_active_mixer(0, "mamba_v1")
        cache.conv_states[0] = torch.randn(2, 128, 4)

        # Fill mamba_v2
        cache.set_active_mixer(0, "mamba_v2")
        cache.conv_states[0] = torch.randn(2, 128, 4)

        # Should detect SSM state from either variant
        assert cache.has_previous_state == True


class TestPropertyAccessorGuards:
    """Test that property accessors guard against None active_mixer."""

    def test_get_raises_error_without_active_mixer(self, apriel2_config_stochastic):
        """Test that accessing cache without set_active_mixer raises clear error."""
        cache = Apriel2Cache(apriel2_config_stochastic)

        with pytest.raises(RuntimeError) as exc_info:
            _ = cache.conv_states[1]

        assert "requires set_active_mixer()" in str(exc_info.value)
        assert "Available mixers:" in str(exc_info.value)

    def test_set_raises_error_without_active_mixer(self, apriel2_config_stochastic):
        """Test that setting cache without set_active_mixer raises clear error."""
        cache = Apriel2Cache(apriel2_config_stochastic)

        with pytest.raises(RuntimeError) as exc_info:
            cache.conv_states[1] = torch.randn(2, 128, 4)

        assert "requires set_active_mixer()" in str(exc_info.value)

    def test_access_works_after_set_active_mixer(self, apriel2_config_stochastic):
        """Test that access works correctly after set_active_mixer."""
        cache = Apriel2Cache(apriel2_config_stochastic)

        # Set active mixer
        cache.set_active_mixer(1, "mamba")

        # Now access should work
        cache.conv_states[1] = torch.randn(2, 128, 4)
        retrieved = cache.conv_states[1]

        assert retrieved is not None


class TestMultipleMixersSameType:
    """Test multiple mixers of the same type with independent caches."""

    def test_attention_variants_independent(self, apriel2_config_multi_mixer):
        """Test that different attention mixers have independent caches."""
        cache = Apriel2Cache(apriel2_config_multi_mixer)

        # Fill attn_small cache
        cache.set_active_mixer(0, "attn_small")
        key_small = torch.randn(2, 8, 10, 64)
        value_small = torch.randn(2, 8, 10, 64)
        cache.update(key_small, value_small, 0)

        assert cache.get_seq_length(0) == 10

        # Switch to attn_large - should have empty cache
        cache.set_active_mixer(0, "attn_large")
        assert cache.get_seq_length(0) == 0

        # Fill attn_large
        key_large = torch.randn(2, 8, 5, 64)
        value_large = torch.randn(2, 8, 5, 64)
        cache.update(key_large, value_large, 0)

        assert cache.get_seq_length(0) == 5

        # Switch back to attn_small - should still have original data
        cache.set_active_mixer(0, "attn_small")
        assert cache.get_seq_length(0) == 10

    def test_ssm_variants_independent(self, apriel2_config_multi_mixer):
        """Test that different SSM mixers have independent caches."""
        cache = Apriel2Cache(apriel2_config_multi_mixer)

        # Fill mamba_v1
        cache.set_active_mixer(0, "mamba_v1")
        conv1 = torch.randn(2, 128, 4)
        cache.conv_states[0] = conv1

        # Fill mamba_v2
        cache.set_active_mixer(0, "mamba_v2")
        conv2 = torch.randn(2, 128, 4)
        cache.conv_states[0] = conv2

        # Verify they're different
        cache.set_active_mixer(0, "mamba_v1")
        retrieved1 = cache.conv_states[0]

        cache.set_active_mixer(0, "mamba_v2")
        retrieved2 = cache.conv_states[0]

        assert not torch.allclose(retrieved1, retrieved2)
        assert torch.allclose(retrieved1, conv1)
        assert torch.allclose(retrieved2, conv2)

    def test_different_window_sizes(self, apriel2_config_multi_mixer):
        """Test that attention mixers with different window sizes are independent."""
        cache = Apriel2Cache(apriel2_config_multi_mixer)

        # Check that attn_small and attn_large have different window sizes
        cache.set_active_mixer(0, "attn_small")
        window_small = cache.get_max_cache_shape(0)

        cache.set_active_mixer(0, "attn_large")
        window_large = cache.get_max_cache_shape(0)

        assert window_small == 2048
        assert window_large == 8192
