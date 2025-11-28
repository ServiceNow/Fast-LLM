"""Tests for Llava to Apriel2 converter.

Tests cover:
- Config extraction and conversion
- Weight conversion with different target configs
- Stochastic mixer conversion
- Pattern-based heterogeneous conversion
- Forward pass equivalence between source and converted models
- Validation of incompatible parameter overrides

Run with: pytest fast_llm_external_models/tests/test_apriel2/test_convert_from_llava.py
Run slow tests: pytest -m slow ...
"""

import json
from pathlib import Path

import pytest
import torch
import yaml
from safetensors import safe_open

from fast_llm_external_models.apriel2.configuration_apriel2 import Apriel2Config
from fast_llm_external_models.apriel2.convert_from_llava import (
    build_component_config,
    build_decoder_config,
    convert_config,
    convert_weights,
    extract_source_mixer_config,
    extract_source_mlp_config,
    extract_source_norm_config,
    validate_transfer_overrides,
)
from fast_llm_external_models.apriel2.modeling_apriel2 import Apriel2ForConditionalGeneration


# =============================================================================
# Config Extraction Tests
# =============================================================================


class TestConfigExtraction:
    """Test source config extraction from Llava config."""

    @pytest.mark.parametrize(
        "config_fixture",
        [
            "llava_pixtral_config",
            pytest.param("apriel_1_5_config", marks=pytest.mark.slow),
        ],
    )
    def test_extract_source_mixer_config(self, config_fixture, request):
        llava_config = request.getfixturevalue(config_fixture)
        mixer = extract_source_mixer_config(llava_config)

        assert mixer["type"] == "attention"
        assert "heads" in mixer
        assert "head_groups" in mixer
        assert "head_size" in mixer
        assert mixer["rotary"]["theta"] > 0

    @pytest.mark.parametrize(
        "config_fixture",
        [
            "llava_pixtral_config",
            pytest.param("apriel_1_5_config", marks=pytest.mark.slow),
        ],
    )
    def test_extract_source_mlp_config(self, config_fixture, request):
        llava_config = request.getfixturevalue(config_fixture)
        mlp = extract_source_mlp_config(llava_config)

        assert mlp["type"] == "mlp"
        assert "intermediate_size" in mlp
        assert mlp["activation"] == "silu"
        assert mlp["gated"] is True

    @pytest.mark.parametrize(
        "config_fixture",
        [
            "llava_pixtral_config",
            pytest.param("apriel_1_5_config", marks=pytest.mark.slow),
        ],
    )
    def test_extract_source_norm_config(self, config_fixture, request):
        llava_config = request.getfixturevalue(config_fixture)
        norm = extract_source_norm_config(llava_config)

        assert norm["type"] == "rms_norm"
        assert norm["epsilon"] == 1e-5


# =============================================================================
# Validation Tests
# =============================================================================


class TestValidateTransferOverrides:
    """Test validation of overrides with init: transfer."""

    def test_shape_affecting_override_raises_error(self, llava_pixtral_config):
        """Shape-affecting overrides should raise ValueError."""
        source = extract_source_mixer_config(llava_pixtral_config)

        with pytest.raises(ValueError, match="Cannot override 'heads'"):
            validate_transfer_overrides({"heads": 16}, source, "test_mixer")

        with pytest.raises(ValueError, match="Cannot override 'head_groups'"):
            validate_transfer_overrides({"head_groups": 2}, source, "test_mixer")

        with pytest.raises(ValueError, match="Cannot override 'head_size'"):
            validate_transfer_overrides({"head_size": 64}, source, "test_mixer")

    def test_non_shape_affecting_override_ok(self, llava_pixtral_config):
        """Non-shape-affecting overrides should be allowed."""
        source = extract_source_mixer_config(llava_pixtral_config)

        # These should not raise
        validate_transfer_overrides({"window_size": 4096}, source, "test_mixer")
        validate_transfer_overrides({"causal": True}, source, "test_mixer")

    def test_behavior_affecting_override_warns(self, llava_pixtral_config, caplog):
        """Behavior-affecting overrides should log warning."""
        source = extract_source_mlp_config(llava_pixtral_config)

        import logging

        with caplog.at_level(logging.WARNING):
            validate_transfer_overrides({"activation": "gelu"}, source, "test_mlp")

        assert "Overriding 'activation'" in caplog.text

    def test_same_value_override_ok(self, llava_pixtral_config):
        """Overriding with same value should not raise."""
        source = extract_source_mixer_config(llava_pixtral_config)

        # Same value - no error
        validate_transfer_overrides({"heads": 8}, source, "test_mixer")


# =============================================================================
# Config Building Tests
# =============================================================================


class TestBuildComponentConfig:
    """Test component config building with init modes."""

    def test_transfer_inherits_source(self, llava_pixtral_config):
        source = extract_source_mixer_config(llava_pixtral_config)
        spec = {"init": "transfer"}

        result = build_component_config(spec, source, "test_mixer")

        assert result["type"] == "attention"
        assert result["heads"] == 8
        assert result["head_groups"] == 4

    def test_transfer_with_safe_override(self, llava_pixtral_config):
        source = extract_source_mixer_config(llava_pixtral_config)
        spec = {"init": "transfer", "window_size": 4096}

        result = build_component_config(spec, source, "test_mixer")

        assert result["type"] == "attention"
        assert result["heads"] == 8
        assert result["window_size"] == 4096

    def test_transfer_with_incompatible_override_raises(self, llava_pixtral_config):
        """Incompatible shape override with transfer should raise."""
        source = extract_source_mixer_config(llava_pixtral_config)
        spec = {"init": "transfer", "heads": 16}  # Different from source (8)

        with pytest.raises(ValueError, match="Cannot override 'heads'"):
            build_component_config(spec, source, "test_mixer")

    def test_random_requires_full_config(self, llava_pixtral_config):
        source = extract_source_mixer_config(llava_pixtral_config)
        spec = {"init": "random"}  # No type specified

        with pytest.raises(ValueError, match="must specify full config"):
            build_component_config(spec, source, "test_mixer")

    def test_random_with_full_config(self, llava_pixtral_config):
        source = extract_source_mixer_config(llava_pixtral_config)
        spec = {
            "init": "random",
            "type": "gdn",
            "heads": 16,
            "head_size": 32,
        }

        result = build_component_config(spec, source, "test_mixer")

        assert result["type"] == "gdn"
        assert result["heads"] == 16

    def test_random_allows_any_shape(self, llava_pixtral_config):
        """init: random should allow any shape params."""
        source = extract_source_mixer_config(llava_pixtral_config)
        spec = {
            "init": "random",
            "type": "attention",
            "heads": 16,  # Different from source
            "head_groups": 16,
            "head_size": 64,
        }

        # Should not raise - random init doesn't transfer weights
        result = build_component_config(spec, source, "test_mixer")
        assert result["heads"] == 16


class TestBuildDecoderConfig:
    """Test decoder config building."""

    def test_fixed_decoder_basic(self, llava_pixtral_config):
        target = {
            "type": "fixed",
            "block": {
                "mixer": {"init": "transfer"},
                "mlp": {"init": "transfer"},
                "normalization": {"init": "transfer"},
            },
        }

        result = build_decoder_config(target, llava_pixtral_config)

        assert result["type"] == "fixed"
        assert result["num_blocks"] == 5
        assert result["block"]["mixer"]["type"] == "attention"
        assert result["block"]["mlp"]["intermediate_size"] == 512

    def test_fixed_decoder_stochastic_mixer(self, llava_pixtral_config):
        target = {
            "type": "fixed",
            "block": {
                "mixer": {
                    "type": "stochastic",
                    "main_mixer_name": "attention",
                    "sampling_strategy": "uniform",
                    "mixers": {
                        "attention": {"init": "transfer"},
                        "sliding_window": {"init": "transfer", "window_size": 2048},
                    },
                },
                "mlp": {"init": "transfer"},
                "normalization": {"init": "transfer"},
            },
        }

        result = build_decoder_config(target, llava_pixtral_config)

        assert result["block"]["mixer"]["type"] == "stochastic"
        assert "attention" in result["block"]["mixer"]["mixers"]
        assert "sliding_window" in result["block"]["mixer"]["mixers"]
        assert result["block"]["mixer"]["mixers"]["sliding_window"]["window_size"] == 2048

    def test_pattern_decoder(self, llava_pixtral_config):
        target = {
            "type": "pattern",
            "pattern": ["full", "local"],
            "blocks": {
                "full": {
                    "mixer": {"init": "transfer"},
                    "mlp": {"init": "transfer"},
                    "normalization": {"init": "transfer"},
                },
                "local": {
                    "mixer": {"init": "transfer", "window_size": 1024},
                    "mlp": {"init": "transfer"},
                    "normalization": {"init": "transfer"},
                },
            },
        }

        result = build_decoder_config(target, llava_pixtral_config)

        assert result["type"] == "pattern"
        assert result["pattern"] == ["full", "local"]
        assert "full" in result["blocks"]
        assert "local" in result["blocks"]
        assert result["blocks"]["local"]["mixer"]["window_size"] == 1024


# =============================================================================
# Full Config Conversion Tests
# =============================================================================


class TestConvertConfig:
    """Test full config conversion."""

    @pytest.mark.parametrize(
        "config_fixture",
        [
            "llava_pixtral_config",
            pytest.param("apriel_1_5_config", marks=pytest.mark.slow),
        ],
    )
    def test_basic_conversion(self, config_fixture, request):
        llava_config = request.getfixturevalue(config_fixture)
        result = convert_config(llava_config)

        assert result["model_type"] == "apriel2"
        assert "hidden_size" in result
        assert "vocab_size" in result
        assert result["decoder"]["type"] == "fixed"
        assert "num_blocks" in result["decoder"]
        assert result["vision_encoder"] is not None

    @pytest.mark.parametrize(
        "config_fixture",
        [
            "llava_pixtral_config",
            pytest.param("apriel_1_5_config", marks=pytest.mark.slow),
        ],
    )
    def test_with_target_config(self, config_fixture, request):
        llava_config = request.getfixturevalue(config_fixture)
        target = {
            "decoder": {
                "type": "fixed",
                "block": {
                    "mixer": {"init": "transfer", "window_size": 512},
                    "mlp": {"init": "transfer"},
                    "normalization": {"init": "transfer"},
                },
            },
        }

        result = convert_config(llava_config, target)

        assert result["decoder"]["block"]["mixer"]["window_size"] == 512


# =============================================================================
# Weight Conversion Tests
# =============================================================================


class TestWeightConversion:
    """Test weight conversion."""

    def test_basic_conversion(self, llava_pixtral_checkpoint, tmp_path):
        """Test basic conversion without target config."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        llava_config = json.load(open(llava_pixtral_checkpoint / "config.json"))
        apriel2_config = convert_config(llava_config)

        convert_weights(llava_pixtral_checkpoint, output_dir, None, apriel2_config)

        # Check output exists
        assert (output_dir / "model.safetensors").exists()

        # Load and verify weights
        with safe_open(output_dir / "model.safetensors", framework="pt") as f:
            keys = list(f.keys())

        # Should have decoder layer weights
        assert any("model.decoder.blocks.0.mixer" in k for k in keys)
        assert any("model.decoder.blocks.0.mlp" in k for k in keys)

        # Should have vision encoder weights
        assert any("model.vision_encoder" in k for k in keys)

    def test_stochastic_mixer_conversion(self, llava_pixtral_checkpoint, tmp_path):
        """Test stochastic mixer conversion duplicates weights."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        llava_config = json.load(open(llava_pixtral_checkpoint / "config.json"))

        target_config = {
            "decoder": {
                "type": "fixed",
                "block": {
                    "mixer": {
                        "type": "stochastic",
                        "main_mixer_name": "attention",
                        "mixers": {
                            "attention": {"init": "transfer"},
                            "sliding_window": {"init": "transfer", "window_size": 512},
                        },
                    },
                    "mlp": {"init": "transfer"},
                    "normalization": {"init": "transfer"},
                },
            },
        }

        apriel2_config = convert_config(llava_config, target_config)
        convert_weights(llava_pixtral_checkpoint, output_dir, target_config, apriel2_config)

        with safe_open(output_dir / "model.safetensors", framework="pt") as f:
            keys = list(f.keys())

        # Should have weights for both mixers
        attn_keys = [k for k in keys if ".mixers.attention." in k]
        sw_keys = [k for k in keys if ".mixers.sliding_window." in k]

        assert len(attn_keys) > 0
        assert len(sw_keys) > 0
        assert len(attn_keys) == len(sw_keys)  # Same number of weights

    def test_random_init_skips_weights(self, llava_pixtral_checkpoint, tmp_path):
        """Test that init: random skips weight transfer."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        llava_config = json.load(open(llava_pixtral_checkpoint / "config.json"))

        target_config = {
            "decoder": {
                "type": "fixed",
                "block": {
                    "mixer": {
                        "type": "stochastic",
                        "main_mixer_name": "attention",
                        "mixers": {
                            "attention": {"init": "transfer"},
                            "new_mixer": {
                                "init": "random",
                                "type": "gdn",
                                "heads": 8,
                                "head_size": 32,
                            },
                        },
                    },
                    "mlp": {"init": "transfer"},
                    "normalization": {"init": "transfer"},
                },
            },
        }

        apriel2_config = convert_config(llava_config, target_config)
        convert_weights(llava_pixtral_checkpoint, output_dir, target_config, apriel2_config)

        with safe_open(output_dir / "model.safetensors", framework="pt") as f:
            keys = list(f.keys())

        # Should have attention weights
        assert any(".mixers.attention." in k for k in keys)

        # Should NOT have new_mixer weights (init: random)
        assert not any(".mixers.new_mixer." in k for k in keys)

    def test_pattern_conversion(self, llava_pixtral_checkpoint, tmp_path):
        """Test heterogeneous pattern conversion."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        llava_config = json.load(open(llava_pixtral_checkpoint / "config.json"))

        target_config = {
            "decoder": {
                "type": "pattern",
                "pattern": ["full", "local"],
                "blocks": {
                    "full": {
                        "mixer": {"init": "transfer"},
                        "mlp": {"init": "transfer"},
                        "normalization": {"init": "transfer"},
                    },
                    "local": {
                        "mixer": {"init": "transfer", "window_size": 256},
                        "mlp": {"init": "transfer"},
                        "normalization": {"init": "transfer"},
                    },
                },
            },
        }

        apriel2_config = convert_config(llava_config, target_config)
        convert_weights(llava_pixtral_checkpoint, output_dir, target_config, apriel2_config)

        # Verify output config
        assert apriel2_config["decoder"]["type"] == "pattern"
        assert apriel2_config["decoder"]["blocks"]["local"]["mixer"]["window_size"] == 256


# =============================================================================
# Weight Count Verification
# =============================================================================


class TestWeightCounts:
    """Verify correct number of weights are transferred."""

    def test_basic_weight_count(self, llava_pixtral_checkpoint, tmp_path):
        """Verify all weights are transferred in basic conversion."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Count source weights
        with safe_open(llava_pixtral_checkpoint / "model.safetensors", framework="pt") as f:
            source_count = len(list(f.keys()))

        llava_config = json.load(open(llava_pixtral_checkpoint / "config.json"))
        apriel2_config = convert_config(llava_config)
        convert_weights(llava_pixtral_checkpoint, output_dir, None, apriel2_config)

        # Count output weights
        with safe_open(output_dir / "model.safetensors", framework="pt") as f:
            output_count = len(list(f.keys()))

        # Should have same number of weights
        assert output_count == source_count

    def test_stochastic_weight_count(self, llava_pixtral_checkpoint, tmp_path):
        """Verify stochastic mixer has duplicated weights."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        llava_config = json.load(open(llava_pixtral_checkpoint / "config.json"))
        num_layers = llava_config["text_config"]["num_hidden_layers"]

        target_config = {
            "decoder": {
                "type": "fixed",
                "block": {
                    "mixer": {
                        "type": "stochastic",
                        "main_mixer_name": "attention",
                        "mixers": {
                            "attention": {"init": "transfer"},
                            "sliding_window": {"init": "transfer", "window_size": 512},
                        },
                    },
                    "mlp": {"init": "transfer"},
                    "normalization": {"init": "transfer"},
                },
            },
        }

        apriel2_config = convert_config(llava_config, target_config)
        convert_weights(llava_pixtral_checkpoint, output_dir, target_config, apriel2_config)

        with safe_open(output_dir / "model.safetensors", framework="pt") as f:
            keys = list(f.keys())

        # Each mixer should have 4 weights per layer (q, k, v, o projections)
        attn_weights = [k for k in keys if ".mixers.attention.self_attn" in k]
        sw_weights = [k for k in keys if ".mixers.sliding_window.self_attn" in k]

        assert len(attn_weights) == num_layers * 4
        assert len(sw_weights) == num_layers * 4


# =============================================================================
# YAML Config Tests
# =============================================================================


class TestYAMLConfigs:
    """Test loading and applying YAML configs."""

    def test_stochastic_supernet_yaml(self, llava_pixtral_checkpoint):
        """Test the stochastic_supernet.yaml example."""
        yaml_config = """
decoder:
  type: fixed
  block:
    mixer:
      type: stochastic
      main_mixer_name: attention
      sampling_strategy: uniform
      mixers:
        attention:
          init: transfer
        sliding_window:
          init: transfer
          window_size: 512
    mlp:
      init: transfer
    normalization:
      init: transfer
"""
        target_config = yaml.safe_load(yaml_config)

        llava_config = json.load(open(llava_pixtral_checkpoint / "config.json"))
        apriel2_config = convert_config(llava_config, target_config)

        assert apriel2_config["decoder"]["block"]["mixer"]["type"] == "stochastic"
        assert "attention" in apriel2_config["decoder"]["block"]["mixer"]["mixers"]
        assert "sliding_window" in apriel2_config["decoder"]["block"]["mixer"]["mixers"]

    def test_heterogeneous_pattern_yaml(self, llava_pixtral_checkpoint):
        """Test the heterogeneous_pattern.yaml example."""
        yaml_config = """
decoder:
  type: pattern
  pattern: [full_attention, sliding_window]
  blocks:
    full_attention:
      mixer:
        init: transfer
      mlp:
        init: transfer
      normalization:
        init: transfer
    sliding_window:
      mixer:
        init: transfer
        window_size: 256
      mlp:
        init: transfer
      normalization:
        init: transfer
"""
        target_config = yaml.safe_load(yaml_config)

        llava_config = json.load(open(llava_pixtral_checkpoint / "config.json"))
        apriel2_config = convert_config(llava_config, target_config)

        assert apriel2_config["decoder"]["type"] == "pattern"
        assert apriel2_config["decoder"]["pattern"] == ["full_attention", "sliding_window"]


# =============================================================================
# Forward Pass Equivalence Tests
# =============================================================================


def _load_models_for_comparison(llava_pixtral_checkpoint, tmp_path):
    """Helper to load source Llava and converted Apriel2 models."""
    from transformers import LlavaForConditionalGeneration

    output_dir = tmp_path / "output"
    output_dir.mkdir(exist_ok=True)

    # Load source model
    source_model = LlavaForConditionalGeneration.from_pretrained(llava_pixtral_checkpoint)
    source_model.eval()

    # Convert to Apriel2
    llava_config = json.load(open(llava_pixtral_checkpoint / "config.json"))
    apriel2_config_dict = convert_config(llava_config)
    convert_weights(llava_pixtral_checkpoint, output_dir, None, apriel2_config_dict)

    # Load Apriel2 model
    apriel2_config = Apriel2Config(**apriel2_config_dict)
    target_model = Apriel2ForConditionalGeneration(apriel2_config)

    with safe_open(output_dir / "model.safetensors", framework="pt") as f:
        target_weights = {key: f.get_tensor(key) for key in f.keys()}

    target_model.load_state_dict(target_weights, strict=False)
    target_model.eval()

    return source_model, target_model, llava_config


class TestComponentEquivalence:
    """Test individual components produce identical outputs.

    These tests isolate each component to help pinpoint where differences occur.
    """

    def test_text_embedding_equivalence(self, llava_pixtral_checkpoint, tmp_path):
        """Test text embedding layer produces identical outputs."""
        source_model, target_model, llava_config = _load_models_for_comparison(
            llava_pixtral_checkpoint, tmp_path
        )

        # Get embedding layers
        source_embed = source_model.model.language_model.embed_tokens
        target_embed = target_model.model.embed_tokens

        # Test input
        torch.manual_seed(42)
        input_ids = torch.randint(0, llava_config["text_config"]["vocab_size"], (2, 16))

        with torch.no_grad():
            source_out = source_embed(input_ids)
            target_out = target_embed(input_ids)

        assert torch.allclose(source_out, target_out, atol=1e-6, rtol=1e-5), (
            f"Embedding max diff: {(source_out - target_out).abs().max()}"
        )

    def test_lm_head_equivalence(self, llava_pixtral_checkpoint, tmp_path):
        """Test LM head produces identical outputs."""
        source_model, target_model, llava_config = _load_models_for_comparison(
            llava_pixtral_checkpoint, tmp_path
        )

        # Get LM heads
        source_head = source_model.lm_head
        target_head = target_model.lm_head

        # Test input (hidden states)
        torch.manual_seed(42)
        hidden_size = llava_config["text_config"]["hidden_size"]
        hidden_states = torch.randn(2, 16, hidden_size)

        with torch.no_grad():
            source_out = source_head(hidden_states)
            target_out = target_head(hidden_states)

        assert torch.allclose(source_out, target_out, atol=1e-6, rtol=1e-5), (
            f"LM head max diff: {(source_out - target_out).abs().max()}"
        )

    def test_vision_patch_embedding_equivalence(self, llava_pixtral_checkpoint, tmp_path):
        """Test vision patch embedding produces identical outputs."""
        source_model, target_model, llava_config = _load_models_for_comparison(
            llava_pixtral_checkpoint, tmp_path
        )

        # Get patch embedding layers
        source_conv = source_model.model.vision_tower.patch_conv
        source_norm = source_model.model.vision_tower.ln_pre
        target_patch = target_model.model.vision_encoder.patch_convolution

        # Test input (small image)
        torch.manual_seed(42)
        # 32x32 image (2x2 patches with patch_size=16)
        pixel_values = torch.randn(1, 3, 32, 32)

        with torch.no_grad():
            # Source: conv then norm
            source_out = source_conv(pixel_values)
            # Reshape from (B, C, H, W) to (B, H*W, C) for norm
            b, c, h, w = source_out.shape
            source_out = source_out.flatten(2).transpose(1, 2)  # (B, H*W, C)
            source_out = source_norm(source_out)

            # Target: patch_convolution handles both
            target_out = target_patch(pixel_values)

        assert torch.allclose(source_out, target_out, atol=1e-5, rtol=1e-5), (
            f"Patch embedding max diff: {(source_out - target_out).abs().max()}"
        )

    def test_multimodal_projector_equivalence(self, llava_pixtral_checkpoint, tmp_path):
        """Test multimodal projector produces identical outputs."""
        source_model, target_model, llava_config = _load_models_for_comparison(
            llava_pixtral_checkpoint, tmp_path
        )

        # Get projectors
        source_proj = source_model.model.multi_modal_projector
        target_proj = target_model.model.vision_encoder.adapter

        # Test input (vision hidden states)
        torch.manual_seed(42)
        vision_hidden_size = llava_config["vision_config"]["hidden_size"]
        vision_hidden = torch.randn(2, 16, vision_hidden_size)

        with torch.no_grad():
            source_out = source_proj(vision_hidden)
            target_out = target_proj(vision_hidden)

        assert torch.allclose(source_out, target_out, atol=1e-5, rtol=1e-5), (
            f"Projector max diff: {(source_out - target_out).abs().max()}"
        )


class TestFullModelEquivalence:
    """Test full model forward pass equivalence.

    These tests verify end-to-end equivalence for text-only and multimodal inputs.
    """

    def test_text_only_forward(self, llava_pixtral_checkpoint, tmp_path):
        """Test text-only forward pass produces identical outputs."""
        source_model, target_model, llava_config = _load_models_for_comparison(
            llava_pixtral_checkpoint, tmp_path
        )

        # Test input
        torch.manual_seed(42)
        vocab_size = llava_config["text_config"]["vocab_size"]
        input_ids = torch.randint(0, vocab_size, (2, 16))

        with torch.no_grad():
            source_out = source_model(input_ids)
            target_out = target_model(input_ids)

        source_logits = source_out.logits
        target_logits = target_out.logits

        assert torch.allclose(source_logits, target_logits, atol=1e-5, rtol=1e-5), (
            f"Text-only logits max diff: {(source_logits - target_logits).abs().max()}"
        )

    def test_multimodal_forward(self, llava_pixtral_checkpoint, tmp_path):
        """Test multimodal forward pass works on both models.

        Note: Full numerical equivalence is not tested because Pixtral and Apriel2
        vision encoders have different patch extraction (Pixtral produces (size/16)^2 - 1
        patches vs Apriel2's (size/16)^2 patches). This is an architectural difference,
        not a conversion issue. The component tests verify weight equivalence for
        patch_conv, layer_norm, and projector individually.

        This test verifies:
        1. Source Llava model can process multimodal input
        2. Target Apriel2 model can process multimodal input
        3. Both produce valid logits with expected shapes
        """
        source_model, target_model, llava_config = _load_models_for_comparison(
            llava_pixtral_checkpoint, tmp_path
        )

        # Get config parameters
        vision_config = llava_config["vision_config"]
        num_channels = vision_config.get("num_channels", 3)
        image_token_index = llava_config["image_token_index"]
        vocab_size = llava_config["text_config"]["vocab_size"]

        torch.manual_seed(42)
        batch_size = 1
        image_size = 64
        pixel_values = torch.randn(batch_size, num_channels, image_size, image_size)

        # Get patch counts for each model (they differ due to architecture)
        with torch.no_grad():
            source_features = source_model.get_image_features(pixel_values)
            target_features = target_model.get_image_features(pixel_values)

        source_patches = source_features[0].shape[0] if isinstance(source_features, list) else source_features.shape[1]
        target_patches = target_features.shape[1]

        # Test source model
        source_input_ids = self._create_multimodal_input_ids(
            vocab_size, image_token_index, source_patches, batch_size
        )
        with torch.no_grad():
            source_out = source_model(input_ids=source_input_ids, pixel_values=pixel_values)
        assert source_out.logits.shape == (batch_size, source_input_ids.shape[1], vocab_size)

        # Test target model
        target_input_ids = self._create_multimodal_input_ids(
            vocab_size, image_token_index, target_patches, batch_size
        )
        with torch.no_grad():
            target_out = target_model(input_ids=target_input_ids, pixel_values=pixel_values)
        assert target_out.logits.shape == (batch_size, target_input_ids.shape[1], vocab_size)

        # Both should produce finite logits
        assert torch.isfinite(source_out.logits).all(), "Source model produced non-finite logits"
        assert torch.isfinite(target_out.logits).all(), "Target model produced non-finite logits"

    def _create_multimodal_input_ids(self, vocab_size, image_token_index, num_patches, batch_size):
        """Helper to create input_ids with image token placeholders."""
        prefix_len = 5
        suffix_len = 5

        prefix = torch.randint(0, vocab_size, (batch_size, prefix_len))
        prefix = torch.where(prefix == image_token_index, torch.tensor(0), prefix)

        image_tokens = torch.full((batch_size, num_patches), image_token_index)

        suffix = torch.randint(0, vocab_size, (batch_size, suffix_len))
        suffix = torch.where(suffix == image_token_index, torch.tensor(0), suffix)

        return torch.cat([prefix, image_tokens, suffix], dim=1)

    def test_model_can_load_converted_weights(self, llava_pixtral_checkpoint, tmp_path):
        """Test that converted weights can be loaded into Apriel2 model."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        llava_config = json.load(open(llava_pixtral_checkpoint / "config.json"))
        apriel2_config_dict = convert_config(llava_config)
        convert_weights(llava_pixtral_checkpoint, output_dir, None, apriel2_config_dict)

        # Create Apriel2 model
        apriel2_config = Apriel2Config(**apriel2_config_dict)
        model = Apriel2ForConditionalGeneration(apriel2_config)

        # Load converted weights
        with safe_open(output_dir / "model.safetensors", framework="pt") as f:
            converted_weights = {key: f.get_tensor(key) for key in f.keys()}

        # Should load without errors
        missing, unexpected = model.load_state_dict(converted_weights, strict=False)

        # No unexpected keys
        assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"

        # Only missing keys should be from caches or buffers (non-weight parameters)
        for key in missing:
            assert "cache" in key.lower() or "position" in key.lower() or "mask" in key.lower(), (
                f"Unexpected missing key: {key}"
            )


# =============================================================================
# Apriel 1.5 Full Conversion Tests (slow - requires large download)
# =============================================================================


@pytest.mark.slow
class TestApriel15Conversion:
    """Test conversion with the real Apriel 1.5 checkpoint.

    These tests require downloading the Apriel 1.5 model (~30GB).
    Run with: pytest -m slow
    """

    def test_apriel_1_5_config_conversion(self, apriel_1_5_config, tmp_path):
        """Test config conversion produces valid Apriel2 config."""
        apriel2_config_dict = convert_config(apriel_1_5_config)

        # Verify expected values for Apriel 1.5
        assert apriel2_config_dict["hidden_size"] == 5120
        assert apriel2_config_dict["vocab_size"] == 131072
        assert apriel2_config_dict["decoder"]["num_blocks"] == 48

        # Verify config can be instantiated
        config = Apriel2Config(**apriel2_config_dict)
        assert config.hidden_size == 5120

    def test_apriel_1_5_stochastic_config(self, apriel_1_5_config):
        """Test stochastic mixer config with Apriel 1.5."""
        target_config = {
            "decoder": {
                "type": "fixed",
                "block": {
                    "mixer": {
                        "type": "stochastic",
                        "main_mixer_name": "attention",
                        "sampling_strategy": "uniform",
                        "mixers": {
                            "attention": {"init": "transfer"},
                            "sliding_window": {"init": "transfer", "window_size": 4096},
                        },
                    },
                    "mlp": {"init": "transfer"},
                    "normalization": {"init": "transfer"},
                },
            },
        }

        apriel2_config_dict = convert_config(apriel_1_5_config, target_config)

        # Verify stochastic config
        mixer = apriel2_config_dict["decoder"]["block"]["mixer"]
        assert mixer["type"] == "stochastic"
        assert mixer["mixers"]["attention"]["heads"] == 32
        assert mixer["mixers"]["sliding_window"]["window_size"] == 4096

    def test_apriel_1_5_weight_conversion(self, apriel_1_5_checkpoint, tmp_path):
        """Test full weight conversion of Apriel 1.5.

        Warning: This downloads ~30GB of weights!
        """
        from fast_llm_external_models.apriel2.convert_from_llava import (
            convert_config,
            convert_weights,
            resolve_input,
            copy_model_files,
        )

        output_dir = tmp_path / "apriel2_converted"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Resolve input (handles HF model ID)
        input_path = resolve_input(apriel_1_5_checkpoint)

        # Load source config
        with open(input_path / "config.json") as f:
            llava_config = json.load(f)

        # Convert config
        apriel2_config = convert_config(llava_config)

        # Save config
        with open(output_dir / "config.json", "w") as f:
            json.dump(apriel2_config, f, indent=2)

        # Convert weights
        convert_weights(input_path, output_dir, None, apriel2_config)

        # Copy model files (configuration_apriel2.py, modeling_apriel2.py)
        copy_model_files(output_dir)

        # Verify outputs exist
        assert (output_dir / "config.json").exists()
        assert (output_dir / "model.safetensors").exists()

        # Verify config
        with open(output_dir / "config.json") as f:
            config = json.load(f)

        assert config["model_type"] == "apriel2"
        assert config["hidden_size"] == 5120
