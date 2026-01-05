"""Tests for Llava to Apriel2 converter.

Tests cover:
- Config conversion (Llava -> Apriel2)
- Plan-based weight conversion
- Surgery operations (Apriel2 -> Apriel2)
- Weight loading verification
- Plan key matching

Note: Forward pass equivalence tests are in test_equivalence.py, which provides
comprehensive component-by-component and integration testing with strict tolerances.

Run with: pytest fast_llm_external_models/tests/test_apriel2/test_convert_from_llava.py
"""

import json

import torch
from safetensors import safe_open

from fast_llm_external_models.apriel2.configuration_apriel2 import Apriel2Config
from fast_llm_external_models.apriel2.conversion import convert_llava_config as convert_config
from fast_llm_external_models.apriel2.conversion import execute, plan_llava_to_apriel2, plan_surgery
from fast_llm_external_models.apriel2.modeling_apriel2 import Apriel2ForConditionalGeneration

# =============================================================================
# Config Conversion Tests
# =============================================================================


class TestConvertConfig:
    """Test pure config conversion (no surgery)."""

    def test_basic_conversion(self, llava_pixtral_config):
        """Test that Llava config converts to valid Apriel2 config."""
        llava_config = llava_pixtral_config
        result = convert_config(llava_config)

        # Check model metadata
        assert result["model_type"] == "apriel2"
        assert result["architectures"] == ["Apriel2ForConditionalGeneration"]

        # Check basic fields are transferred
        assert "hidden_size" in result
        assert "vocab_size" in result
        assert "bos_token_id" in result
        assert "eos_token_id" in result

        # Check decoder config
        assert result["decoder"]["type"] == "fixed"
        assert "num_blocks" in result["decoder"]
        assert result["decoder"]["block"]["mixer"]["type"] == "attention"
        assert result["decoder"]["block"]["mlp"]["type"] == "mlp"

        # Check vision encoder
        assert "vision_encoder" in result
        assert "embeddings" in result["vision_encoder"]
        assert "encoder" in result["vision_encoder"]
        assert "adapter" in result["vision_encoder"]

    def test_config_can_be_instantiated(self, llava_pixtral_config):
        """Test that converted config can create Apriel2Config object."""
        llava_config = llava_pixtral_config
        result = convert_config(llava_config)

        # Should be able to instantiate
        config = Apriel2Config(**result)
        assert config.hidden_size == result["hidden_size"]
        assert config.vocab_size == result["vocab_size"]

    def test_preserves_dimensions(self, llava_pixtral_config):
        """Test that dimensions are preserved correctly."""
        result = convert_config(llava_pixtral_config)
        text_config = llava_pixtral_config["text_config"]

        assert result["hidden_size"] == text_config["hidden_size"]
        assert result["vocab_size"] == text_config["vocab_size"]
        assert result["decoder"]["num_blocks"] == text_config["num_hidden_layers"]
        assert result["decoder"]["block"]["mlp"]["intermediate_size"] == text_config["intermediate_size"]


# =============================================================================
# Plan-Based Weight Conversion Tests
# =============================================================================


class TestPlanConversion:
    """Test plan-based weight conversion."""

    def test_plan_converts_all_weights(self, llava_pixtral_checkpoint):
        """Test that plan converts all weights."""
        with open(llava_pixtral_checkpoint / "config.json") as f:
            llava_config = json.load(f)
        with safe_open(llava_pixtral_checkpoint / "model.safetensors", framework="pt") as f:
            source_weights = {key: f.get_tensor(key) for key in f.keys()}

        plan = plan_llava_to_apriel2(llava_config)
        apriel2_weights = execute(plan, source_weights, seed=0)

        # Should have same number of weights (all mapped)
        assert len(apriel2_weights) == len(source_weights)

    def test_plan_weight_names_are_apriel2_format(self, llava_pixtral_checkpoint):
        """Test that plan produces Apriel2 format weight names."""
        with open(llava_pixtral_checkpoint / "config.json") as f:
            llava_config = json.load(f)
        with safe_open(llava_pixtral_checkpoint / "model.safetensors", framework="pt") as f:
            source_weights = {key: f.get_tensor(key) for key in f.keys()}

        plan = plan_llava_to_apriel2(llava_config)
        apriel2_weights = execute(plan, source_weights, seed=0)

        # Check decoder weights
        assert any("model.decoder.blocks.0.mixer" in k for k in apriel2_weights.keys())
        assert any("model.decoder.blocks.0.mlp" in k for k in apriel2_weights.keys())

        # Check vision weights
        assert any("model.vision_encoder.encoder.blocks" in k for k in apriel2_weights.keys())
        assert any("model.vision_encoder.adapter" in k for k in apriel2_weights.keys())

    def test_plan_weight_values_unchanged(self, llava_pixtral_checkpoint):
        """Test that weight values are not modified during conversion."""
        with open(llava_pixtral_checkpoint / "config.json") as f:
            llava_config = json.load(f)
        with safe_open(llava_pixtral_checkpoint / "model.safetensors", framework="pt") as f:
            source_weights = {key: f.get_tensor(key) for key in f.keys()}

        plan = plan_llava_to_apriel2(llava_config)
        apriel2_weights = execute(plan, source_weights, seed=0)

        # Check specific weights are identical
        source_embed = source_weights["language_model.model.embed_tokens.weight"]
        target_embed = apriel2_weights["model.embed_tokens.weight"]
        assert torch.equal(source_embed, target_embed)


# =============================================================================
# Surgery Tests (Plan-Based)
# =============================================================================


class TestSurgery:
    """Test surgery operations (Apriel2 -> Apriel2) via plans."""

    def test_identity_surgery(self, llava_pixtral_checkpoint):
        """Test surgery with same source and target config (identity)."""
        # Load and convert to Apriel2 base
        with open(llava_pixtral_checkpoint / "config.json") as f:
            llava_config = json.load(f)
        with safe_open(llava_pixtral_checkpoint / "model.safetensors", framework="pt") as f:
            source_weights = {key: f.get_tensor(key) for key in f.keys()}

        # Convert via plan
        conversion_plan = plan_llava_to_apriel2(llava_config)
        apriel2_config = convert_config(llava_config)
        apriel2_weights = execute(conversion_plan, source_weights, seed=0)

        # Surgery with same config = identity
        surgery_plan = plan_surgery(apriel2_config, apriel2_config)
        result_weights = execute(surgery_plan, apriel2_weights, seed=0)

        # Weights should be identical
        assert "model.embed_tokens.weight" in result_weights
        assert torch.allclose(
            result_weights["model.embed_tokens.weight"],
            apriel2_weights["model.embed_tokens.weight"],
        )

    def test_surgery_to_stochastic_mixer(self, llava_pixtral_checkpoint):
        """Test surgery that wraps attention with stochastic mixer."""
        # Load and convert to Apriel2 base
        with open(llava_pixtral_checkpoint / "config.json") as f:
            llava_config = json.load(f)
        with safe_open(llava_pixtral_checkpoint / "model.safetensors", framework="pt") as f:
            source_weights = {key: f.get_tensor(key) for key in f.keys()}

        conversion_plan = plan_llava_to_apriel2(llava_config)
        source_config = convert_config(llava_config)
        source_weights = execute(conversion_plan, source_weights, seed=0)

        # Target config with stochastic mixer
        target_config = json.loads(json.dumps(source_config))  # Deep copy
        target_config["decoder"]["block"]["mixer"] = {
            "type": "stochastic",
            "main_mixer_name": "attention",
            "mixers": {
                "attention": source_config["decoder"]["block"]["mixer"],
                "sliding_window": {
                    **source_config["decoder"]["block"]["mixer"],
                    "window_size": 512,
                },
            },
        }

        surgery_plan = plan_surgery(source_config, target_config)
        result_weights = execute(surgery_plan, source_weights, seed=0)

        # Should have weights for both sub-mixers
        attn_keys = [k for k in result_weights if ".mixers.attention." in k]
        sw_keys = [k for k in result_weights if ".mixers.sliding_window." in k]

        assert len(attn_keys) > 0, "No attention sub-mixer weights"
        assert len(sw_keys) > 0, "No sliding_window sub-mixer weights"
        assert len(attn_keys) == len(sw_keys), "Sub-mixer weight counts differ"

    def test_surgery_mamba_uses_mil(self, llava_pixtral_checkpoint):
        """Test surgery that adds mamba uses MIL initialization."""
        # Load and convert to Apriel2 base
        with open(llava_pixtral_checkpoint / "config.json") as f:
            llava_config = json.load(f)
        with safe_open(llava_pixtral_checkpoint / "model.safetensors", framework="pt") as f:
            source_weights = {key: f.get_tensor(key) for key in f.keys()}

        conversion_plan = plan_llava_to_apriel2(llava_config)
        source_config = convert_config(llava_config)
        source_weights_converted = execute(conversion_plan, source_weights, seed=0)
        hidden_size = source_config["hidden_size"]

        # Target config with mamba
        target_config = json.loads(json.dumps(source_config))  # Deep copy
        target_config["decoder"]["block"]["mixer"] = {
            "type": "stochastic",
            "main_mixer_name": "attention",
            "mixers": {
                "attention": source_config["decoder"]["block"]["mixer"],
                "mamba": {
                    "type": "mamba",
                    "d_state": 16,
                    "d_conv": 4,
                    "d_inner": 2 * hidden_size,
                    "d_xb": hidden_size // 4,
                    "dt_rank": hidden_size // 16,
                    "repeat_kv_before_conv": True,
                    "conv_bias": True,
                    "dt_proj_bias": True,
                    "dt_min": 0.001,
                    "dt_max": 0.1,
                    "dt_init_floor": 1e-4,
                },
            },
        }

        surgery_plan = plan_surgery(source_config, target_config)
        result_weights = execute(surgery_plan, source_weights_converted, seed=0)

        # Should have mamba weights
        mamba_keys = [k for k in result_weights if ".mixers.mamba." in k]
        assert len(mamba_keys) > 0, "No mamba weights created"

        # Check mamba weights exist and have correct structure
        for key in mamba_keys:
            assert result_weights[key] is not None
            assert result_weights[key].numel() > 0


# =============================================================================
# Weight Loading Tests
# =============================================================================


class TestWeightLoading:
    """Test weight loading after conversion."""

    def test_model_can_load_converted_weights(self, llava_pixtral_checkpoint, tmp_path):
        """Test that converted weights can be loaded into Apriel2 model."""
        with open(llava_pixtral_checkpoint / "config.json") as f:
            llava_config = json.load(f)
        with safe_open(llava_pixtral_checkpoint / "model.safetensors", framework="pt") as f:
            source_weights = {key: f.get_tensor(key) for key in f.keys()}

        apriel2_config_dict = convert_config(llava_config)
        plan = plan_llava_to_apriel2(llava_config)
        apriel2_weights = execute(plan, source_weights, seed=0)

        apriel2_config = Apriel2Config(**apriel2_config_dict)
        model = Apriel2ForConditionalGeneration(apriel2_config)

        missing, unexpected = model.load_state_dict(apriel2_weights, strict=False)

        assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"
        for key in missing:
            assert "cache" in key.lower() or "position" in key.lower() or "mask" in key.lower()


# =============================================================================
# Plan Integration Tests
# =============================================================================


class TestPlanIntegration:
    """Test plan-based conversion integration."""

    def test_plan_source_keys_match_llava_keys(self, llava_pixtral_checkpoint):
        """Plan source keys must exist in Llava checkpoint."""
        with open(llava_pixtral_checkpoint / "config.json") as f:
            llava_config = json.load(f)
        with safe_open(llava_pixtral_checkpoint / "model.safetensors", framework="pt") as f:
            llava_keys = set(f.keys())

        plan = plan_llava_to_apriel2(llava_config)
        plan_source_keys = plan.source_keys()

        missing = plan_source_keys - llava_keys
        assert not missing, f"Plan references non-existent source keys: {sorted(missing)[:10]}"

    def test_plan_keys_match_model_state_dict(self, llava_pixtral_checkpoint):
        """Plan target keys must match actual Apriel2 model state_dict keys."""
        with open(llava_pixtral_checkpoint / "config.json") as f:
            llava_config = json.load(f)

        # Get keys from plan
        plan = plan_llava_to_apriel2(llava_config)
        plan_keys = plan.target_keys()

        # Get keys from instantiated model
        apriel2_config_dict = convert_config(llava_config)
        config = Apriel2Config(**apriel2_config_dict)
        model = Apriel2ForConditionalGeneration(config)
        model_keys = set(model.state_dict().keys())

        missing_in_plan = model_keys - plan_keys
        extra_in_plan = plan_keys - model_keys

        # Filter out expected missing keys (caches, positions, etc.)
        missing_in_plan = {
            k for k in missing_in_plan if not any(skip in k.lower() for skip in ["cache", "position", "mask"])
        }

        assert not missing_in_plan, f"Model keys not in plan: {sorted(missing_in_plan)[:10]}"
        assert not extra_in_plan, f"Plan keys not in model: {sorted(extra_in_plan)[:10]}"
