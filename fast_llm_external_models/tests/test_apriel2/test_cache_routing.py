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


class TestMixerSwitching:
    """Test cache behavior when switching between different mixers."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="SSM mixers require CUDA")
    def test_cache_preserves_state_across_mixer_switches(self, apriel2_config_all_mixers, device):
        """Verify cache maintains independent state for each mixer when switching.

        This is the critical test for stochastic mixers: when we switch which mixer
        is active, the cache must preserve previous mixer states while updating the
        current mixer's state.
        """
        if device.type != "cuda":
            pytest.skip("SSM mixers require CUDA device")

        from fast_llm_external_models.apriel2.modeling_apriel2 import Apriel2ForCausalLM

        model = Apriel2ForCausalLM(apriel2_config_all_mixers).to(device)
        model.eval()

        stochastic_layer_idx = 1  # Layer 1 is the stochastic layer
        stochastic_layer = model.model.decoder.blocks[stochastic_layer_idx]
        input_ids = torch.randint(0, apriel2_config_all_mixers.vocab_size, (2, 10), device=device)

        # Forward 1: Use attention (default main mixer)
        stochastic_layer.mixer.main_mixer_name = "attention"
        outputs1 = model(input_ids, use_cache=True)
        cache = outputs1.past_key_values

        # Verify: only attention has data
        layer_cache = cache.layers[stochastic_layer_idx]
        assert layer_cache['attention'].key is not None, "Attention cache should have KV states"
        assert layer_cache['swa'].key is None, "SWA cache should be empty"
        assert layer_cache['mamba'].conv is None, "Mamba cache should be empty"
        assert layer_cache['gdn'].conv is None, "GatedDeltaNet cache should be empty"
        attn_seq_len_1 = layer_cache['attention'].key.shape[-2]

        # Forward 2: Switch to mamba (new token)
        stochastic_layer.mixer.main_mixer_name = "mamba"
        new_token = torch.randint(0, apriel2_config_all_mixers.vocab_size, (2, 1), device=device)
        outputs2 = model(new_token, past_key_values=cache, use_cache=True)
        cache = outputs2.past_key_values

        # Verify: attention preserved, mamba added
        assert layer_cache['attention'].key is not None, "Attention cache should be preserved"
        assert layer_cache['attention'].key.shape[-2] == attn_seq_len_1, "Attention seq_len should not change"
        assert layer_cache['mamba'].conv is not None, "Mamba cache should now have SSM states"
        assert layer_cache['swa'].key is None, "SWA cache should still be empty"
        assert layer_cache['gdn'].conv is None, "GatedDeltaNet cache should still be empty"

        # Forward 3: Switch to swa
        stochastic_layer.mixer.main_mixer_name = "swa"
        outputs3 = model(new_token, past_key_values=cache, use_cache=True)
        cache = outputs3.past_key_values

        # Verify: attention + mamba preserved, swa added
        assert layer_cache['attention'].key is not None, "Attention cache should be preserved"
        assert layer_cache['mamba'].conv is not None, "Mamba cache should be preserved"
        assert layer_cache['swa'].key is not None, "SWA cache should now have KV states"
        assert layer_cache['gdn'].conv is None, "GatedDeltaNet cache should still be empty"

        # Forward 4: Switch to gated_delta_net
        stochastic_layer.mixer.main_mixer_name = "gdn"
        outputs4 = model(new_token, past_key_values=cache, use_cache=True)
        cache = outputs4.past_key_values

        # Verify: ALL mixers now have independent state
        assert layer_cache['attention'].key is not None, "Attention cache should be preserved"
        assert layer_cache['mamba'].conv is not None, "Mamba cache should be preserved"
        assert layer_cache['swa'].key is not None, "SWA cache should be preserved"
        assert layer_cache['gdn'].conv is not None, "GatedDeltaNet cache should now have SSM states"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="SSM mixers require CUDA")
    def test_cache_isolation_between_attention_and_ssm(self, apriel2_config_all_mixers, device):
        """Verify attention caches (KV) and SSM caches (conv/recurrent) don't interfere."""
        if device.type != "cuda":
            pytest.skip("SSM mixers require CUDA device")

        from fast_llm_external_models.apriel2.modeling_apriel2 import Apriel2ForCausalLM

        model = Apriel2ForCausalLM(apriel2_config_all_mixers).to(device)
        model.eval()

        stochastic_layer_idx = 1
        stochastic_layer = model.model.decoder.blocks[stochastic_layer_idx]
        input_ids = torch.randint(0, apriel2_config_all_mixers.vocab_size, (2, 10), device=device)

        # Forward with attention
        stochastic_layer.mixer.main_mixer_name = "attention"
        outputs1 = model(input_ids, use_cache=True)
        cache = outputs1.past_key_values

        # Get attention cache state
        attn_cache = cache.layers[stochastic_layer_idx]['attention']
        attn_key = attn_cache.key.clone()
        attn_value = attn_cache.value.clone()

        # Forward with mamba (using same cache)
        stochastic_layer.mixer.main_mixer_name = "mamba"
        new_token = torch.randint(0, apriel2_config_all_mixers.vocab_size, (2, 1), device=device)
        outputs2 = model(new_token, past_key_values=cache, use_cache=True)
        cache = outputs2.past_key_values

        # Verify attention cache unchanged
        assert torch.allclose(cache.layers[stochastic_layer_idx]['attention'].key, attn_key), \
            "Attention KV cache should not be modified when mamba is active"
        assert torch.allclose(cache.layers[stochastic_layer_idx]['attention'].value, attn_value), \
            "Attention KV cache should not be modified when mamba is active"

        # Verify mamba cache is populated
        assert cache.layers[stochastic_layer_idx]['mamba'].conv is not None, \
            "Mamba SSM cache should be populated"

    def test_seq_len_tracking_per_mixer(self, apriel2_config_all_mixers):
        """Verify seq_len is tracked independently for each mixer."""
        from fast_llm_external_models.apriel2.modeling_apriel2 import Apriel2ForCausalLM

        model = Apriel2ForCausalLM(apriel2_config_all_mixers)
        model.eval()

        stochastic_layer_idx = 1
        stochastic_layer = model.model.decoder.blocks[stochastic_layer_idx]

        # Forward with attention (10 tokens)
        input_ids1 = torch.randint(0, apriel2_config_all_mixers.vocab_size, (2, 10))
        stochastic_layer.mixer.main_mixer_name = "attention"
        outputs1 = model(input_ids1, use_cache=True)
        cache = outputs1.past_key_values

        cache.set_active_mixer(stochastic_layer_idx, "attention")
        assert cache.get_seq_length(stochastic_layer_idx) == 10

        # Forward with swa (5 tokens) - independent from attention
        input_ids2 = torch.randint(0, apriel2_config_all_mixers.vocab_size, (2, 5))
        stochastic_layer.mixer.main_mixer_name = "swa"
        outputs2 = model(input_ids2, use_cache=True)
        cache2 = Apriel2Cache(apriel2_config_all_mixers)  # Fresh cache for swa
        outputs2 = model(input_ids2, past_key_values=cache2, use_cache=True)
        cache2 = outputs2.past_key_values

        cache2.set_active_mixer(stochastic_layer_idx, "swa")
        assert cache2.get_seq_length(stochastic_layer_idx) == 5

        # Original cache should still have attention with seq_len=10
        cache.set_active_mixer(stochastic_layer_idx, "attention")
        assert cache.get_seq_length(stochastic_layer_idx) == 10


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
