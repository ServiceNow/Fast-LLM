"""Tests for Apriel2-specific cache behaviors with no HuggingFace equivalent.

This module tests features unique to Apriel2Cache that cannot be validated
against upstream HF implementations:

1. Stochastic mixer routing (switching between attention/SSM per layer)
2. Multi-mixer layer support
3. Error handling and guard rails
4. Beam search operations (batch_repeat, reorder, select)
5. Crop operation

Fixtures used from conftest.py:
    - stochastic_config: Stochastic mixer config with attention and mamba
    - attention_config: Pure attention config
    - ssm_config: Pure SSM config
"""

import pytest
import torch

from fast_llm_external_models.apriel2.cache import Apriel2Cache

# =============================================================================
# STOCHASTIC MIXER ROUTING
# =============================================================================


class TestStochasticMixerRouting:
    """Test routing operations to correct sub-cache in stochastic layers."""

    def test_set_active_mixer(self, stochastic_config):
        """set_active_mixer updates routing for layer."""
        cache = Apriel2Cache(stochastic_config)

        cache.set_active_mixer(0, "attention")
        assert cache.active_mixers[0] == "attention"

        cache.set_active_mixer(0, "mamba")
        assert cache.active_mixers[0] == "mamba"

    def test_update_routes_to_active_mixer(self, stochastic_config):
        """update() stores in correct sub-cache based on active_mixer."""
        cache = Apriel2Cache(stochastic_config)

        # Route to attention
        cache.set_active_mixer(0, "attention")
        cache.update(torch.randn(2, 4, 10, 16), torch.randn(2, 4, 10, 16), layer_idx=0)

        # Attention sub-cache should have data
        assert cache.layers[0]["attention"].key is not None
        # Mamba sub-cache should be empty
        assert cache.layers[0]["mamba"].conv is None

    def test_each_mixer_has_independent_cache(self, stochastic_config):
        """Each mixer in a stochastic layer has its own independent state."""
        cache = Apriel2Cache(stochastic_config)

        # Store in attention
        cache.set_active_mixer(0, "attention")
        cache.update(torch.randn(2, 4, 5, 16), torch.randn(2, 4, 5, 16), layer_idx=0)

        # Switch to mamba and store
        cache.set_active_mixer(0, "mamba")
        cache.layers[0]["mamba"].conv = torch.randn(2, 64, 4)

        # Attention data should be unchanged
        assert cache.layers[0]["attention"].cumulative_length == 5

    def test_switching_preserves_all_states(self, stochastic_config):
        """Switching active_mixer doesn't clear other mixer's state."""
        cache = Apriel2Cache(stochastic_config)

        # Build up attention state
        cache.set_active_mixer(0, "attention")
        cache.update(torch.randn(2, 4, 10, 16), torch.randn(2, 4, 10, 16), layer_idx=0)
        attn_key = cache.layers[0]["attention"].key.clone()

        # Switch to mamba
        cache.set_active_mixer(0, "mamba")

        # Attention state preserved
        torch.testing.assert_close(cache.layers[0]["attention"].key, attn_key)


# =============================================================================
# ERROR HANDLING
# =============================================================================


class TestErrorHandling:
    """Test guard rails and error messages."""

    def test_update_without_active_mixer_raises(self, stochastic_config):
        """update() on stochastic layer without active_mixer raises RuntimeError."""
        cache = Apriel2Cache(stochastic_config)

        with pytest.raises(RuntimeError, match="needs active_mixer set"):
            cache.update(torch.randn(2, 4, 5, 16), torch.randn(2, 4, 5, 16), layer_idx=0)

    def test_accessor_without_active_mixer_raises(self, stochastic_config):
        """Accessing key_cache/value_cache without active_mixer raises RuntimeError."""
        cache = Apriel2Cache(stochastic_config)

        with pytest.raises(RuntimeError, match="requires set_active_mixer"):
            _ = cache.key_cache[0]

    def test_error_message_lists_available_mixers(self, stochastic_config):
        """Error message includes list of available mixers."""
        cache = Apriel2Cache(stochastic_config)

        with pytest.raises(RuntimeError, match="attention.*mamba|mamba.*attention"):
            _ = cache.key_cache[0]


# =============================================================================
# BEAM SEARCH OPERATIONS
# =============================================================================


class TestBeamSearchOperations:
    """Test batch manipulation for beam search."""

    def test_batch_repeat_interleave_attention(self, attention_config):
        """batch_repeat_interleave expands batch dimension."""
        cache = Apriel2Cache(attention_config)
        cache.update(torch.randn(2, 4, 10, 16), torch.randn(2, 4, 10, 16), layer_idx=0)

        cache.batch_repeat_interleave(3)

        assert cache.layers[0].key.shape[0] == 6  # 2 * 3

    def test_batch_repeat_interleave_ssm(self, ssm_config):
        """batch_repeat_interleave works for SSM caches."""
        cache = Apriel2Cache(ssm_config)
        cache.layers[0].conv = torch.randn(2, 64, 4)

        cache.batch_repeat_interleave(3)

        assert cache.layers[0].conv.shape[0] == 6

    def test_batch_repeat_interleave_kda_tuple(self, ssm_config):
        """batch_repeat_interleave handles KDA tuple conv states."""
        cache = Apriel2Cache(ssm_config)
        cache.layers[0].conv = (torch.randn(2, 64, 4),) * 3

        cache.batch_repeat_interleave(3)

        assert cache.layers[0].conv[0].shape[0] == 6

    def test_reorder_cache_attention(self, attention_config):
        """reorder_cache reorders batch dimension."""
        cache = Apriel2Cache(attention_config)
        k = torch.arange(4).float().view(4, 1, 1, 1).expand(4, 4, 10, 16)
        cache.update(k, k.clone(), layer_idx=0)

        beam_idx = torch.tensor([3, 2, 1, 0])
        cache.reorder_cache(beam_idx)

        # Check reordering
        assert cache.layers[0].key[0, 0, 0, 0].item() == 3.0
        assert cache.layers[0].key[3, 0, 0, 0].item() == 0.0

    def test_batch_select_indices(self, attention_config):
        """batch_select_indices selects subset of batch."""
        cache = Apriel2Cache(attention_config)
        cache.update(torch.randn(4, 4, 10, 16), torch.randn(4, 4, 10, 16), layer_idx=0)

        indices = torch.tensor([0, 2])
        cache.batch_select_indices(indices)

        assert cache.layers[0].key.shape[0] == 2

    def test_reorder_cache_ssm_tuple(self, ssm_config):
        """reorder_cache handles KDA tuple conv states."""
        cache = Apriel2Cache(ssm_config)
        # Create distinguishable tensors for each batch position
        conv0 = torch.full((1, 64, 4), 0.0)
        conv1 = torch.full((1, 64, 4), 1.0)
        conv2 = torch.full((1, 64, 4), 2.0)
        cache.layers[0].conv = (
            torch.cat([conv0, conv1, conv2], dim=0),
            torch.cat([conv0, conv1, conv2], dim=0),
            torch.cat([conv0, conv1, conv2], dim=0),
        )

        beam_idx = torch.tensor([2, 1, 0])
        cache.reorder_cache(beam_idx)

        # Check reordering: batch[0] should now have value 2.0
        assert cache.layers[0].conv[0][0, 0, 0].item() == 2.0
        assert cache.layers[0].conv[0][2, 0, 0].item() == 0.0

    def test_batch_select_indices_ssm_tuple(self, ssm_config):
        """batch_select_indices handles KDA tuple conv states."""
        cache = Apriel2Cache(ssm_config)
        cache.layers[0].conv = (torch.randn(4, 64, 4),) * 3

        indices = torch.tensor([0, 2])
        cache.batch_select_indices(indices)

        assert cache.layers[0].conv[0].shape[0] == 2
        assert cache.layers[0].conv[1].shape[0] == 2
        assert cache.layers[0].conv[2].shape[0] == 2


# =============================================================================
# CROP OPERATION
# =============================================================================


class TestCropOperation:
    """Test cache truncation."""

    def test_crop_truncates_attention(self, attention_config):
        """crop() truncates attention cache."""
        cache = Apriel2Cache(attention_config)
        cache.update(torch.randn(2, 4, 10, 16), torch.randn(2, 4, 10, 16), layer_idx=0)

        cache.crop(5)

        assert cache.layers[0].key.shape[-2] == 5
        assert cache.get_seq_length(0) == 5

    def test_crop_affects_all_layers(self, attention_config):
        """crop() affects all layers."""
        cache = Apriel2Cache(attention_config)
        cache.update(torch.randn(2, 4, 10, 16), torch.randn(2, 4, 10, 16), layer_idx=0)
        cache.update(torch.randn(2, 4, 10, 16), torch.randn(2, 4, 10, 16), layer_idx=1)

        cache.crop(3)

        assert cache.layers[0].key.shape[-2] == 3
        assert cache.layers[1].key.shape[-2] == 3

    def test_crop_ignores_ssm(self, ssm_config):
        """crop() doesn't affect SSM caches (they don't have seq dimension)."""
        cache = Apriel2Cache(ssm_config)
        cache.layers[0].conv = torch.randn(2, 64, 4)

        # Should not raise
        cache.crop(5)

        # SSM state unchanged
        assert cache.layers[0].conv.shape == (2, 64, 4)


# =============================================================================
# CACHE PROPERTIES
# =============================================================================


class TestCacheProperties:
    """Test cache property methods."""

    def test_is_initialized_attention(self, attention_config):
        """is_initialized True after update."""
        cache = Apriel2Cache(attention_config)
        assert not cache.is_initialized

        cache.update(torch.randn(2, 4, 5, 16), torch.randn(2, 4, 5, 16), layer_idx=0)
        assert cache.is_initialized

    def test_is_initialized_ssm(self, ssm_config):
        """is_initialized True after setting conv state."""
        cache = Apriel2Cache(ssm_config)
        assert not cache.is_initialized

        cache.layers[0].conv = torch.randn(2, 64, 4)
        assert cache.is_initialized

    def test_has_previous_state_ssm_only(self, ssm_config):
        """has_previous_state checks SSM conv states."""
        cache = Apriel2Cache(ssm_config)
        assert not cache.has_previous_state

        cache.layers[0].conv = torch.randn(2, 64, 4)
        assert cache.has_previous_state

    def test_has_previous_state_ignores_attention(self, attention_config):
        """has_previous_state ignores attention caches."""
        cache = Apriel2Cache(attention_config)
        cache.update(torch.randn(2, 4, 10, 16), torch.randn(2, 4, 10, 16), layer_idx=0)

        # Attention-only cache returns False for has_previous_state
        assert not cache.has_previous_state

    def test_reset_clears_ssm_states(self, ssm_config):
        """reset() clears SSM conv and recurrent states."""
        cache = Apriel2Cache(ssm_config)
        cache.layers[0].conv = torch.randn(2, 64, 4)
        cache.layers[0].recurrent = torch.randn(2, 64, 16)

        cache.reset()

        assert cache.layers[0].conv is None
        assert cache.layers[0].recurrent is None

    def test_max_batch_size_from_ssm_tuple(self, ssm_config):
        """max_batch_size works with KDA tuple conv states."""
        cache = Apriel2Cache(ssm_config)
        cache.layers[0].conv = (torch.randn(3, 64, 4),) * 3

        assert cache.max_batch_size == 3

    def test_max_batch_size(self, attention_config):
        """max_batch_size returns batch dimension."""
        cache = Apriel2Cache(attention_config)
        cache.update(torch.randn(3, 4, 10, 16), torch.randn(3, 4, 10, 16), layer_idx=0)

        assert cache.max_batch_size == 3

    def test_len_returns_num_layers(self, attention_config):
        """__len__ returns number of layers."""
        cache = Apriel2Cache(attention_config)
        assert len(cache) == 2


# =============================================================================
# INDEXING
# =============================================================================


class TestCacheIndexing:
    """Test __getitem__ for HF compatibility."""

    def test_getitem_returns_kv_tuple(self, attention_config):
        """cache[idx] returns (key, value) tuple."""
        cache = Apriel2Cache(attention_config)
        cache.update(torch.randn(2, 4, 10, 16), torch.randn(2, 4, 10, 16), layer_idx=0)

        k, v = cache[0]
        assert k.shape == (2, 4, 10, 16)
        assert v.shape == (2, 4, 10, 16)

    def test_getitem_empty_returns_empty_tensors(self, attention_config):
        """cache[idx] on empty cache returns empty tensors."""
        cache = Apriel2Cache(attention_config)

        k, v = cache[0]
        assert k.numel() == 0
        assert v.numel() == 0
