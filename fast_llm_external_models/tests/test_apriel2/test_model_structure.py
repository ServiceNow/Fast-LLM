"""Tests for Apriel2 model structure and architecture validation."""

import torch

from fast_llm_external_models.apriel2.modeling_apriel2 import (
    Apriel2Cache,
    Apriel2ForCausalLM,
    _AttentionCache,
    _SSMCache,
)
from fast_llm_external_models.tests.test_apriel2.conftest import requires_cuda


@requires_cuda
class TestStochasticMixerStructure:
    """Validate stochastic mixer architecture matches configuration."""

    def test_all_submixers_present(self, apriel2_config_all_mixers):
        """Stochastic layer contains all 4 configured sub-mixers."""
        model = Apriel2ForCausalLM(apriel2_config_all_mixers)
        stochastic_layer = model.model.decoder.blocks[1]  # Layer 1 is the "all_mixers" layer

        assert hasattr(stochastic_layer.mixer, "mixers"), "Stochastic mixer should have 'mixers' attribute"
        assert set(stochastic_layer.mixer.mixers.keys()) == {
            "attention",
            "swa",
            "mamba",
            "gdn",
        }, "Stochastic mixer should contain all 4 configured mixer types"

        # Verify each mixer is the correct type
        from fast_llm_external_models.apriel2.modeling_apriel2 import (
            Apriel2Attention,
            Apriel2GatedDeltaNet,
            Apriel2Mamba,
        )

        assert isinstance(stochastic_layer.mixer.mixers["attention"], Apriel2Attention)
        assert isinstance(
            stochastic_layer.mixer.mixers["swa"], Apriel2Attention
        )  # SWA is Apriel2Attention with sliding_window
        assert isinstance(stochastic_layer.mixer.mixers["mamba"], Apriel2Mamba)
        assert isinstance(stochastic_layer.mixer.mixers["gdn"], Apriel2GatedDeltaNet)

    def test_main_mixer_is_configured(self, apriel2_config_all_mixers):
        """Verify main_mixer_name is set correctly."""
        model = Apriel2ForCausalLM(apriel2_config_all_mixers)
        stochastic_layer = model.model.decoder.blocks[1]

        assert stochastic_layer.mixer.main_mixer_name == "attention"
        assert stochastic_layer.mixer.main_mixer_name in stochastic_layer.mixer.mixers

    def test_cache_has_all_submixer_slots(self, apriel2_config_all_mixers):
        """Cache for stochastic layer has dict with all mixer slots."""
        cache = Apriel2Cache(apriel2_config_all_mixers)
        layer_cache = cache.layers[1]  # stochastic layer

        assert isinstance(layer_cache, dict), "Stochastic layer cache should be a dict"
        assert set(layer_cache.keys()) == {
            "attention",
            "swa",
            "mamba",
            "gdn",
        }, "Cache should have slots for all 4 mixers"

    def test_attention_mixers_use_attention_cache(self, apriel2_config_all_mixers):
        """Attention and SWA use _AttentionCache, SSMs use _SSMCache."""
        cache = Apriel2Cache(apriel2_config_all_mixers)
        layer_cache = cache.layers[1]

        # Attention-based mixers use AttentionCache
        assert isinstance(layer_cache["attention"], _AttentionCache)
        assert isinstance(layer_cache["swa"], _AttentionCache)

        # SSM-based mixers use SSMCache
        assert isinstance(layer_cache["mamba"], _SSMCache)
        assert isinstance(layer_cache["gdn"], _SSMCache)

    def test_parameter_counts_differ_by_config(self):
        """Different configs create models with different parameter counts."""
        from fast_llm_external_models.apriel2.configuration_apriel2 import Apriel2Config

        rotary_config = {"type": "mistral_1d", "theta": 10000.0}
        attn_config = {
            "type": "attention",
            "heads": 4,
            "head_groups": 2,
            "head_size": 16,
            "rotary": rotary_config,
        }

        config_tiny = Apriel2Config(
            vocab_size=100,
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            decoder={
                "type": "fixed",
                "num_blocks": 2,
                "block": {
                    "mixer": attn_config,
                    "mlp": {"type": "mlp", "intermediate_size": 256, "gated": True},
                    "normalization": {"type": "rms_norm"},
                },
            },
        )

        config_stochastic = Apriel2Config(
            vocab_size=100,
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            decoder={
                "type": "pattern",
                "num_blocks": 2,
                "pattern": ["attn", "stoch"],
                "blocks": {
                    "attn": {
                        "mixer": attn_config,
                        "mlp": {"type": "mlp", "intermediate_size": 256, "gated": True},
                        "normalization": {"type": "rms_norm"},
                    },
                    "stoch": {
                        "mixer": {
                            "type": "stochastic",
                            "main_mixer_name": "attention",
                            "mixers": {
                                "attention": attn_config,
                                "mamba": {"type": "mamba", "conv_bias": True, "dt_proj_bias": True},
                            },
                        },
                        "mlp": {"type": "mlp", "intermediate_size": 256, "gated": True},
                        "normalization": {"type": "rms_norm"},
                    },
                },
            },
        )

        model_tiny = Apriel2ForCausalLM(config_tiny)
        model_stochastic = Apriel2ForCausalLM(config_stochastic)

        params_tiny = sum(p.numel() for p in model_tiny.parameters())
        params_stochastic = sum(p.numel() for p in model_stochastic.parameters())

        assert (
            params_stochastic > params_tiny
        ), "Stochastic mixer should have more parameters (has both attention and mamba)"

    def test_weights_are_initialized(self, apriel2_config_all_mixers):
        """Verify model weights are initialized (not all zeros/constant)."""
        model = Apriel2ForCausalLM(apriel2_config_all_mixers)

        # Check that model has parameters
        stochastic_layer = model.model.decoder.blocks[1]
        total_params = sum(p.numel() for p in stochastic_layer.mixer.parameters())
        assert total_params > 0, "Stochastic mixer should have parameters"

        # Basic sanity: at least some parameters should be non-zero
        non_zero_params = sum(
            not torch.all(p == 0) for mixer in stochastic_layer.mixer.mixers.values() for p in mixer.parameters()
        )
        assert non_zero_params > 0, "At least some mixer parameters should be non-zero"

        # Note: We don't check detailed statistics because:
        # - SSMs use special initialization (dt_proj uses log-spaced values, high mean)
        # - Some parameters may be intentionally constant (e.g., bias terms)
