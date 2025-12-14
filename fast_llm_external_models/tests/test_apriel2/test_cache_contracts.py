"""Contract tests for Apriel2Cache against HuggingFace cache implementations.

This module tests that Apriel2Cache components behave equivalently to their
HuggingFace counterparts. This ensures compatibility with HF's generation
infrastructure (mask creation, beam search, etc.).

Mapping:
    Apriel2 Component          HuggingFace Equivalent
    -----------------          ----------------------
    _AttentionCache (no window) -> DynamicLayer
    _AttentionCache (window)    -> DynamicSlidingWindowLayer
    _SSMCache                   -> MambaCache (different interface, same concept)

Apriel2-specific features (stochastic routing, multi-mixer layers) are tested
separately in test_cache_apriel2_specific.py since they have no HF equivalent.

Fixtures used from conftest.py:
    - batch_size, num_heads, head_dim: Tensor dimensions
    - hf_dynamic_layer: HuggingFace DynamicLayer
    - hf_sliding_layer: HuggingFace DynamicSlidingWindowLayer (parameterized by window_size)
    - apriel_attention_cache: Apriel2 _AttentionCache (no window)
    - apriel_sliding_cache: Apriel2 _AttentionCache (with window, parameterized)
    - window_size: Parameterized window sizes [4, 8, 16, 32]
    - attention_config, swa_config: Apriel2 configs
"""

import pytest
import torch

from fast_llm_external_models.apriel2.cache import _AttentionCache, _SSMCache, Apriel2Cache


# =============================================================================
# SECTION 1: FULL ATTENTION - _AttentionCache vs DynamicLayer
# =============================================================================


class TestFullAttentionContract:
    """Test _AttentionCache (no window) matches HuggingFace DynamicLayer.

    DynamicLayer is the standard cache for full causal attention.
    We test that our cache produces identical mask parameters.
    """

    # -------------------------------------------------------------------------
    # get_seq_length: Must match exactly for generation to work
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize("seq_len", [1, 5, 10, 50, 100])
    def test_get_seq_length_after_prefill(
        self, hf_dynamic_layer, apriel_attention_cache, batch_size, num_heads, head_dim, seq_len
    ):
        """After prefill, cumulative_length matches HF get_seq_length."""
        key = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim)

        hf_dynamic_layer.update(key.clone(), value.clone())
        apriel_attention_cache.update(key.clone(), value.clone())

        assert apriel_attention_cache.cumulative_length == hf_dynamic_layer.get_seq_length()

    @pytest.mark.parametrize("prefill_len", [1, 5, 10])
    @pytest.mark.parametrize("decode_steps", [1, 5, 10, 20])
    def test_get_seq_length_during_decode(
        self, hf_dynamic_layer, apriel_attention_cache, batch_size, num_heads, head_dim, prefill_len, decode_steps
    ):
        """During decode, cumulative_length tracks total tokens seen."""
        # Prefill
        key = torch.randn(batch_size, num_heads, prefill_len, head_dim)
        value = torch.randn(batch_size, num_heads, prefill_len, head_dim)
        hf_dynamic_layer.update(key.clone(), value.clone())
        apriel_attention_cache.update(key.clone(), value.clone())

        # Decode
        for step in range(decode_steps):
            key = torch.randn(batch_size, num_heads, 1, head_dim)
            value = torch.randn(batch_size, num_heads, 1, head_dim)
            hf_dynamic_layer.update(key.clone(), value.clone())
            apriel_attention_cache.update(key.clone(), value.clone())

            assert apriel_attention_cache.cumulative_length == hf_dynamic_layer.get_seq_length(), (
                f"Mismatch at decode step {step}"
            )

    # -------------------------------------------------------------------------
    # get_mask_sizes: Verify HF behavior for documentation
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize("prefill_len", [1, 5, 10])
    @pytest.mark.parametrize("decode_steps", [0, 1, 5, 10])
    def test_hf_mask_sizes_kv_length(
        self, hf_dynamic_layer, apriel_attention_cache, batch_size, num_heads, head_dim, prefill_len, decode_steps
    ):
        """Document HF's kv_length behavior and verify cumulative_length tracks correctly.

        For full attention, kv_length = cumulative_length + query_length.
        This test verifies our cache tracks tokens identically to HF.
        """
        # Prefill
        key = torch.randn(batch_size, num_heads, prefill_len, head_dim)
        value = torch.randn(batch_size, num_heads, prefill_len, head_dim)
        hf_dynamic_layer.update(key.clone(), value.clone())
        apriel_attention_cache.update(key.clone(), value.clone())

        # Decode
        for _ in range(decode_steps):
            key = torch.randn(batch_size, num_heads, 1, head_dim)
            value = torch.randn(batch_size, num_heads, 1, head_dim)
            hf_dynamic_layer.update(key.clone(), value.clone())
            apriel_attention_cache.update(key.clone(), value.clone())

        # Verify cumulative_length matches HF
        assert apriel_attention_cache.cumulative_length == hf_dynamic_layer.get_seq_length()

        # Verify HF's kv_length follows the expected formula
        cache_position = torch.arange(1)  # Single token decode
        hf_kv_len, hf_kv_offset = hf_dynamic_layer.get_mask_sizes(cache_position)
        expected_kv_len = hf_dynamic_layer.get_seq_length() + cache_position.shape[0]
        assert hf_kv_len == expected_kv_len

    def test_hf_kv_offset_always_zero(self, hf_dynamic_layer, batch_size, num_heads, head_dim):
        """Document that HF DynamicLayer always returns kv_offset=0.

        For full attention, all cached KV pairs map to absolute positions
        starting from 0, so kv_offset is always 0.
        """
        # Add many tokens
        for _ in range(20):
            key = torch.randn(batch_size, num_heads, 5, head_dim)
            value = torch.randn(batch_size, num_heads, 5, head_dim)
            hf_dynamic_layer.update(key.clone(), value.clone())

            cache_position = torch.arange(1)
            _, hf_kv_offset = hf_dynamic_layer.get_mask_sizes(cache_position)

            assert hf_kv_offset == 0, "DynamicLayer always returns kv_offset=0"

    # -------------------------------------------------------------------------
    # update: Output shape and values must match
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize("seq_len", [1, 5, 10])
    def test_update_returns_same_shape(
        self, hf_dynamic_layer, apriel_attention_cache, batch_size, num_heads, head_dim, seq_len
    ):
        """update() returns tensors with matching shapes."""
        key = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim)

        hf_k, hf_v = hf_dynamic_layer.update(key.clone(), value.clone())
        apr_k, apr_v = apriel_attention_cache.update(key.clone(), value.clone())

        assert hf_k.shape == apr_k.shape
        assert hf_v.shape == apr_v.shape

    def test_update_concatenates_identically(
        self, hf_dynamic_layer, apriel_attention_cache, batch_size, num_heads, head_dim
    ):
        """Multiple updates produce identical concatenated states."""
        # Use deterministic values for comparison
        k1 = torch.arange(10).float().view(1, 1, 10, 1).expand(batch_size, num_heads, 10, head_dim)
        v1 = k1.clone()

        hf_dynamic_layer.update(k1.clone(), v1.clone())
        apriel_attention_cache.update(k1.clone(), v1.clone())

        k2 = torch.arange(10, 15).float().view(1, 1, 5, 1).expand(batch_size, num_heads, 5, head_dim)
        v2 = k2.clone()

        hf_k, hf_v = hf_dynamic_layer.update(k2.clone(), v2.clone())
        apr_k, apr_v = apriel_attention_cache.update(k2.clone(), v2.clone())

        torch.testing.assert_close(hf_k, apr_k)
        torch.testing.assert_close(hf_v, apr_v)


# =============================================================================
# SECTION 2: SLIDING WINDOW - _AttentionCache vs DynamicSlidingWindowLayer
# =============================================================================


class TestSlidingWindowContract:
    """Test _AttentionCache (with window) matches HuggingFace DynamicSlidingWindowLayer.

    DynamicSlidingWindowLayer is used for sliding window attention (e.g., Mistral).
    Critical behaviors:
    - cumulative_length tracks ALL tokens seen (not just cached)
    - kv_offset increases once window is exceeded
    - kv_length is capped at window size

    Uses fixtures from conftest.py:
    - window_size: parameterized [4, 8, 16, 32]
    - hf_sliding_layer: DynamicSlidingWindowLayer
    - apriel_sliding_cache: _AttentionCache with window
    """

    # -------------------------------------------------------------------------
    # cumulative_length: Must track total tokens, not cached tokens
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize("prefill_len", [1, 3, 5, 10, 20])
    def test_cumulative_length_matches_after_prefill(
        self, hf_sliding_layer, apriel_sliding_cache, batch_size, num_heads, head_dim, prefill_len
    ):
        """cumulative_length matches HF get_seq_length after prefill."""
        key = torch.randn(batch_size, num_heads, prefill_len, head_dim)
        value = torch.randn(batch_size, num_heads, prefill_len, head_dim)

        hf_sliding_layer.update(key.clone(), value.clone())
        apriel_sliding_cache.update(key.clone(), value.clone())

        assert apriel_sliding_cache.cumulative_length == hf_sliding_layer.get_seq_length()

    def test_cumulative_length_continues_past_window(
        self, hf_sliding_layer, apriel_sliding_cache, window_size, batch_size, num_heads, head_dim
    ):
        """cumulative_length keeps growing even after window is full."""
        total_tokens = window_size * 3  # Way past window

        for i in range(total_tokens):
            key = torch.randn(batch_size, num_heads, 1, head_dim)
            value = torch.randn(batch_size, num_heads, 1, head_dim)

            hf_sliding_layer.update(key.clone(), value.clone())
            apriel_sliding_cache.update(key.clone(), value.clone())

            expected = i + 1
            assert apriel_sliding_cache.cumulative_length == expected
            assert hf_sliding_layer.get_seq_length() == expected

    # -------------------------------------------------------------------------
    # get_mask_sizes: kv_offset must increase once window is exceeded
    # -------------------------------------------------------------------------

    def test_kv_offset_zero_before_window_full(
        self, hf_sliding_layer, apriel_sliding_cache, window_size, batch_size, num_heads, head_dim
    ):
        """kv_offset is 0 while cumulative < window.

        Before the window is full, kv_offset should be 0 because all cached tokens
        correspond to absolute positions starting from 0.
        """
        # Add tokens up to window-1
        for i in range(window_size - 1):
            key = torch.randn(batch_size, num_heads, 1, head_dim)
            value = torch.randn(batch_size, num_heads, 1, head_dim)

            hf_sliding_layer.update(key.clone(), value.clone())
            apriel_sliding_cache.update(key.clone(), value.clone())

            cache_position = torch.arange(1)
            hf_kv_len, hf_kv_offset = hf_sliding_layer.get_mask_sizes(cache_position)

            # Verify HF returns 0 offset before window full
            assert hf_kv_offset == 0, f"HF offset should be 0 at step {i}"
            # Verify Apriel cache tracks cumulative correctly
            assert apriel_sliding_cache.cumulative_length == i + 1

    def test_kv_offset_increases_after_window_full(
        self, hf_sliding_layer, apriel_sliding_cache, window_size, batch_size, num_heads, head_dim
    ):
        """kv_offset increases once cumulative >= window.

        Once the window is full, the cache discards oldest tokens. kv_offset tracks
        which absolute position KV[0] corresponds to.
        """
        # Fill to exactly window
        for _ in range(window_size):
            key = torch.randn(batch_size, num_heads, 1, head_dim)
            value = torch.randn(batch_size, num_heads, 1, head_dim)
            hf_sliding_layer.update(key.clone(), value.clone())
            apriel_sliding_cache.update(key.clone(), value.clone())

        cache_position = torch.arange(1)
        hf_kv_len, hf_kv_offset = hf_sliding_layer.get_mask_sizes(cache_position)

        # At window boundary, offset should be 1
        assert hf_kv_offset == 1, "HF offset should be 1 at window boundary"
        assert apriel_sliding_cache.cumulative_length == window_size

        # Add more tokens and verify offset keeps increasing with HF
        for i in range(5):
            key = torch.randn(batch_size, num_heads, 1, head_dim)
            value = torch.randn(batch_size, num_heads, 1, head_dim)
            hf_sliding_layer.update(key.clone(), value.clone())
            apriel_sliding_cache.update(key.clone(), value.clone())

            hf_kv_len, hf_kv_offset = hf_sliding_layer.get_mask_sizes(cache_position)

            expected_offset = i + 2
            assert hf_kv_offset == expected_offset
            assert apriel_sliding_cache.cumulative_length == window_size + i + 1

    def test_kv_length_capped_at_window(
        self, hf_sliding_layer, apriel_sliding_cache, window_size, batch_size, num_heads, head_dim
    ):
        """kv_length is capped at window size once exceeded.

        For a query of length 1 after the window is full, kv_length = window
        (window-1 cached tokens + 1 query token).
        """
        # Way past window
        for _ in range(window_size * 2):
            key = torch.randn(batch_size, num_heads, 1, head_dim)
            value = torch.randn(batch_size, num_heads, 1, head_dim)
            hf_sliding_layer.update(key.clone(), value.clone())
            apriel_sliding_cache.update(key.clone(), value.clone())

        cache_position = torch.arange(1)
        hf_kv_len, _ = hf_sliding_layer.get_mask_sizes(cache_position)

        # HF returns window (window-1 cached + 1 query)
        assert hf_kv_len == window_size
        # Verify our cache tracked cumulative correctly
        assert apriel_sliding_cache.cumulative_length == window_size * 2

    # -------------------------------------------------------------------------
    # Full sequence length tracking through generation
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize("prefill_len", [1, 3, 5, 10, 20])
    def test_cumulative_length_tracks_all_tokens(
        self, hf_sliding_layer, apriel_sliding_cache, window_size, batch_size, num_heads, head_dim, prefill_len
    ):
        """cumulative_length tracks total tokens seen through prefill + decode.

        This is the foundation for correct mask size computation. We verify that
        our _AttentionCache tracks tokens identically to HuggingFace's DynamicSlidingWindowLayer.
        The actual get_mask_sizes computation is tested in TestApriel2CacheIntegration.
        """
        # Prefill
        key = torch.randn(batch_size, num_heads, prefill_len, head_dim)
        value = torch.randn(batch_size, num_heads, prefill_len, head_dim)
        hf_sliding_layer.update(key.clone(), value.clone())
        apriel_sliding_cache.update(key.clone(), value.clone())

        assert apriel_sliding_cache.cumulative_length == hf_sliding_layer.get_seq_length()

        # Decode past window
        for i in range(window_size + 10):
            key = torch.randn(batch_size, num_heads, 1, head_dim)
            value = torch.randn(batch_size, num_heads, 1, head_dim)
            hf_sliding_layer.update(key.clone(), value.clone())
            apriel_sliding_cache.update(key.clone(), value.clone())

            assert apriel_sliding_cache.cumulative_length == hf_sliding_layer.get_seq_length(), (
                f"cumulative_length mismatch at step {i}"
            )


# =============================================================================
# SECTION 3: SSM CACHE - _SSMCache vs MambaCache concept
# =============================================================================


class TestSSMCacheContract:
    """Document _SSMCache interface and verify basic contract.

    Unlike attention caches which have HF equivalents (DynamicLayer, DynamicSlidingWindowLayer),
    SSM caches have no direct HF counterpart with matching interface. HF's MambaCache uses
    different methods (update_conv_state, update_ssm_state), so we can't do direct comparison.

    These tests document the interface contract:
    1. `conv` and `recurrent` attributes for storing states
    2. Both support None (lazy initialization)
    3. `conv` can be tuple (for KDA which has separate q/k/v conv states)

    Higher-level operations (reorder, batch_repeat, reset) are tested in
    TestBeamSearchOperations in test_cache_apriel2_specific.py.
    """

    def test_conv_state_storage(self, ssm_cache):
        """conv attribute stores conv states (batch, intermediate, kernel_size)."""
        conv = torch.randn(2, 64, 4)
        ssm_cache.conv = conv
        torch.testing.assert_close(ssm_cache.conv, conv)

    def test_recurrent_state_storage(self, ssm_cache):
        """recurrent attribute stores SSM states (batch, intermediate, state_size)."""
        recurrent = torch.randn(2, 64, 16)
        ssm_cache.recurrent = recurrent
        torch.testing.assert_close(ssm_cache.recurrent, recurrent)

    def test_conv_state_tuple_for_kda(self, ssm_cache):
        """conv can be tuple for KDA's separate q/k/v convolutions."""
        conv_tuple = (torch.randn(2, 64, 4), torch.randn(2, 64, 4), torch.randn(2, 64, 4))
        ssm_cache.conv = conv_tuple
        assert isinstance(ssm_cache.conv, tuple)
        assert len(ssm_cache.conv) == 3

    def test_initial_states_none(self, ssm_cache):
        """States are None initially (lazy initialization pattern)."""
        assert ssm_cache.conv is None
        assert ssm_cache.recurrent is None

    def test_states_independent(self, ssm_cache):
        """conv and recurrent states are independent."""
        ssm_cache.conv = torch.randn(2, 64, 4)
        assert ssm_cache.recurrent is None  # recurrent unchanged

        ssm_cache.recurrent = torch.randn(2, 64, 16)
        assert ssm_cache.conv is not None  # conv unchanged


# =============================================================================
# SECTION 4: APRIEL2CACHE INTEGRATION
# =============================================================================


class TestApriel2CacheIntegration:
    """Test Apriel2Cache correctly delegates to underlying caches.

    Uses fixtures from conftest.py:
    - attention_config: Pure attention config
    - swa_config: Sliding window attention config (window=8)
    """

    def test_get_seq_length_matches_dynamic_layer(self, attention_config):
        """Apriel2Cache.get_seq_length matches DynamicLayer for full attention."""
        from transformers.cache_utils import DynamicLayer

        cache = Apriel2Cache(attention_config)
        hf_layer = DynamicLayer()

        key = torch.randn(2, 4, 10, 16)
        value = torch.randn(2, 4, 10, 16)

        cache.update(key.clone(), value.clone(), layer_idx=0)
        hf_layer.update(key.clone(), value.clone())

        assert cache.get_seq_length(0) == hf_layer.get_seq_length()

    def test_get_mask_sizes_matches_dynamic_layer(self, attention_config):
        """Apriel2Cache.get_mask_sizes matches DynamicLayer."""
        from transformers.cache_utils import DynamicLayer

        cache = Apriel2Cache(attention_config)
        hf_layer = DynamicLayer()

        key = torch.randn(2, 4, 10, 16)
        value = torch.randn(2, 4, 10, 16)

        cache.update(key.clone(), value.clone(), layer_idx=0)
        hf_layer.update(key.clone(), value.clone())

        cache_position = torch.arange(1)
        hf_kv_len, hf_kv_offset = hf_layer.get_mask_sizes(cache_position)
        apr_kv_len, apr_kv_offset = cache.get_mask_sizes(cache_position, layer_idx=0)

        assert apr_kv_len == hf_kv_len
        assert apr_kv_offset == hf_kv_offset

    def test_get_mask_sizes_matches_sliding_layer(self, swa_config):
        """Apriel2Cache.get_mask_sizes matches DynamicSlidingWindowLayer."""
        from transformers.cache_utils import DynamicSlidingWindowLayer

        cache = Apriel2Cache(swa_config)
        hf_layer = DynamicSlidingWindowLayer(sliding_window=8)

        # Fill past window
        for _ in range(15):
            key = torch.randn(2, 4, 1, 16)
            value = torch.randn(2, 4, 1, 16)
            cache.update(key.clone(), value.clone(), layer_idx=0)
            hf_layer.update(key.clone(), value.clone())

        cache_position = torch.arange(1)
        hf_kv_len, hf_kv_offset = hf_layer.get_mask_sizes(cache_position)
        apr_kv_len, apr_kv_offset = cache.get_mask_sizes(cache_position, layer_idx=0)

        assert apr_kv_len == hf_kv_len
        assert apr_kv_offset == hf_kv_offset

    def test_reset_clears_cumulative_length(self, attention_config):
        """reset() clears cumulative_length (matches DynamicLayer.reset)."""
        cache = Apriel2Cache(attention_config)

        cache.update(torch.randn(2, 4, 10, 16), torch.randn(2, 4, 10, 16), layer_idx=0)
        assert cache.get_seq_length(0) == 10

        cache.reset()
        assert cache.get_seq_length(0) == 0


# =============================================================================
# SECTION 5: MASK CORRECTNESS (SEMANTIC TESTS)
# =============================================================================


class TestMaskCorrectness:
    """Test that mask parameters produce semantically correct masks.

    These tests verify the END RESULT: masks created with our parameters
    allow the correct attention patterns.
    """

    def test_full_attention_decode_can_attend_to_all(self):
        """During decode, query can attend to all cached positions."""
        from transformers.masking_utils import sdpa_mask, causal_mask_function

        cache = _AttentionCache(window=None)

        # Prefill + decode
        for _ in range(10):
            cache.update(torch.randn(1, 1, 1, 16), torch.randn(1, 1, 1, 16))

        # Mask for decode step
        cache_position = torch.tensor([10])  # Position of new token
        kv_length = cache.cumulative_length + 1
        kv_offset = 0

        mask = sdpa_mask(
            batch_size=1,
            cache_position=cache_position,
            kv_length=kv_length,
            kv_offset=kv_offset,
            mask_function=causal_mask_function,
        )

        if mask is not None:
            # Query at position 10 should attend to positions 0-10
            query_mask = mask[0, 0, 0, :]
            for kv_idx in range(kv_length):
                assert query_mask[kv_idx].item() == True, f"Should attend to position {kv_idx}"

    @pytest.mark.parametrize("window_size", [4, 8, 16])
    def test_sliding_window_decode_respects_window(self, window_size):
        """During decode, query only attends within sliding window."""
        from transformers.masking_utils import sdpa_mask, sliding_window_causal_mask_function

        cache = _AttentionCache(window=window_size)

        # Fill way past window
        total_tokens = window_size * 2
        for _ in range(total_tokens):
            cache.update(torch.randn(1, 1, 1, 16), torch.randn(1, 1, 1, 16))

        # Mask for decode step
        cache_position = torch.tensor([total_tokens])
        cumulative = cache.cumulative_length
        kv_offset = max(cumulative - window_size + 1, 0)
        kv_length = window_size - 1 + 1  # cached + query

        mask = sdpa_mask(
            batch_size=1,
            cache_position=cache_position,
            kv_length=kv_length,
            kv_offset=kv_offset,
            mask_function=sliding_window_causal_mask_function(window_size),
        )

        if mask is not None:
            query_mask = mask[0, 0, 0, :]
            query_pos = cache_position[0].item()

            for kv_idx in range(kv_length):
                abs_pos = kv_offset + kv_idx
                in_window = abs_pos > query_pos - window_size
                causal = abs_pos <= query_pos
                expected = in_window and causal

                assert query_mask[kv_idx].item() == expected, (
                    f"Position {abs_pos}: expected {expected}, got {query_mask[kv_idx].item()}"
                )

    def test_prefill_has_causal_pattern(self):
        """During prefill, mask has proper causal (lower triangular) pattern."""
        from transformers.masking_utils import sdpa_mask, causal_mask_function

        cache = _AttentionCache(window=None)
        cache.update(torch.randn(1, 1, 5, 16), torch.randn(1, 1, 5, 16))

        cache_position = torch.arange(5)
        kv_length = cache.cumulative_length
        kv_offset = 0

        mask = sdpa_mask(
            batch_size=1,
            cache_position=cache_position,
            kv_length=kv_length,
            kv_offset=kv_offset,
            mask_function=causal_mask_function,
            allow_is_causal_skip=False,  # Force mask creation
        )

        if mask is not None:
            # Check causal pattern
            for q_idx in range(5):
                for kv_idx in range(5):
                    expected = kv_idx <= q_idx
                    actual = mask[0, 0, q_idx, kv_idx].item()
                    assert actual == expected, f"q={q_idx}, kv={kv_idx}: expected {expected}"
