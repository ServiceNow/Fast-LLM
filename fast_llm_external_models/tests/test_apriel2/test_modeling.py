"""Tests for Apriel2 model instantiation, forward pass, and generation."""

import pytest
import torch
from fast_llm_external_models.apriel2.modeling_apriel2 import Apriel2ForCausalLM


class TestApriel2Modeling:
    """End-to-end tests for Apriel2 model with different configurations."""

    @pytest.mark.parametrize("config_name", [
        "apriel2_config_tiny",
        "apriel2_config_stochastic",
        "apriel2_config_multi_mixer",
        "apriel2_config_all_mixers",  # Tests all 4 mixer types
        "apriel2_config_with_bias",  # Tests per-layer bias and non-gated MLP
    ])
    def test_model_end_to_end(self, config_name, request):
        """Test instantiation, forward pass, cache correctness, and generation.

        This comprehensive test validates:
        1. Model can be instantiated from config
        2. Forward pass produces correct output shapes
        3. Cache is actually being used (not dormant)
        4. Cache produces numerically identical results to non-cached computation
        5. Generation works end-to-end

        The cache correctness check is critical for stochastic mixer configs,
        as it validates that set_active_mixer() is called correctly and cache
        routing works in the actual model (not just in isolation).
        """
        config = request.getfixturevalue(config_name)

        # Use longer sequences for better cache validation
        seq_len = 50
        input_ids = torch.randint(0, config.vocab_size, (2, seq_len))

        # 1. Instantiation
        model = Apriel2ForCausalLM(config)
        model.eval()  # Disable dropout for deterministic results
        assert model is not None

        # 2. Forward pass - basic shape validation
        outputs = model(input_ids, use_cache=False)
        assert outputs.logits.shape == (2, seq_len, config.vocab_size)
        assert hasattr(outputs, 'logits')

        # 3. Verify cache is actually being used (not dormant)
        split_pos = 30

        # Forward with correct cache
        outputs_part1 = model(input_ids[:, :split_pos], use_cache=True)
        assert outputs_part1.past_key_values is not None

        outputs_correct_cache = model(
            input_ids[:, split_pos:split_pos+1],
            past_key_values=outputs_part1.past_key_values,
            use_cache=True
        )

        # Test 1: Empty cache should give different results than filled cache
        # This verifies cache is being used at all
        from fast_llm_external_models.apriel2.cache import Apriel2Cache, _AttentionCache
        empty_cache = Apriel2Cache(config)

        outputs_empty_cache = model(
            input_ids[:, split_pos:split_pos+1],
            past_key_values=empty_cache,
            use_cache=True
        )

        cache_affects_output = not torch.allclose(
            outputs_correct_cache.logits,
            outputs_empty_cache.logits,
            atol=1e-3
        )
        assert cache_affects_output, f"Cache appears dormant for {config_name} - empty cache gives same results as filled cache"

        # Test 2: Corrupted cache (zeros) should give different results than correct cache
        # This verifies the actual cache VALUES are being used
        corrupted_cache = Apriel2Cache(config)
        correct_cache = outputs_part1.past_key_values

        # Derive dimensions from actual cache (handles different attention implementations)
        for layer_idx in range(config.decoder["num_blocks"]):
            correct_layer = correct_cache.layers[layer_idx]
            corrupted_layer = corrupted_cache.layers[layer_idx]

            # Handle both direct attention cache and stochastic mixer dict
            if isinstance(correct_layer, _AttentionCache) and correct_layer.key is not None:
                # Use same shape as correct cache but fill with zeros
                corrupted_layer.key = torch.zeros_like(correct_layer.key)
                corrupted_layer.value = torch.zeros_like(correct_layer.value)
            elif isinstance(correct_layer, dict):
                # For stochastic mixers, corrupt attention sub-caches
                for name, correct_sub in correct_layer.items():
                    if isinstance(correct_sub, _AttentionCache) and correct_sub.key is not None:
                        corrupted_layer[name].key = torch.zeros_like(correct_sub.key)
                        corrupted_layer[name].value = torch.zeros_like(correct_sub.value)

        outputs_corrupted_cache = model(
            input_ids[:, split_pos:split_pos+1],
            past_key_values=corrupted_cache,
            use_cache=True
        )

        cache_values_matter = not torch.allclose(
            outputs_correct_cache.logits,
            outputs_corrupted_cache.logits,
            atol=1e-3
        )
        assert cache_values_matter, f"Cache values not used for {config_name} - zeroed cache gives same results as correct cache"

        # 4. Cache correctness - validate cache produces same results as no-cache
        # Compute full sequence without cache
        outputs_full = model(input_ids, use_cache=False)

        # Compute in two steps with cache
        outputs_part1 = model(input_ids[:, :split_pos], use_cache=True)
        outputs_part2 = model(
            input_ids[:, split_pos:split_pos+1],
            past_key_values=outputs_part1.past_key_values,
            use_cache=True
        )

        # Logits should match between cached and non-cached
        # Note: GPU execution with bfloat16/float16 has lower precision than CPU float32,
        # so we use a looser tolerance here.
        assert torch.allclose(
            outputs_full.logits[:, split_pos, :],
            outputs_part2.logits[:, 0, :],
            atol=1e-3
        ), f"Cache correctness failed for {config_name}: cached and non-cached logits differ"

        # 5. Generation - end-to-end validation
        prompt = input_ids[:, :10]
        generated = model.generate(prompt, max_new_tokens=10, use_cache=True)
        assert generated.shape == (2, 20)  # 10 prompt + 10 generated
        assert torch.all(generated[:, :10] == prompt)  # Prompt should be preserved
