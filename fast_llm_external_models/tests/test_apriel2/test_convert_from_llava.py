"""Tests for Llava to Apriel2 converter and surgery.

Tests cover:
- Pure format conversion (Llava -> Apriel2)
- Surgery operations (Apriel2 -> Apriel2)
- Forward pass equivalence between source and converted models

Run with: pytest fast_llm_external_models/tests/test_apriel2/test_convert_from_llava.py
Run slow tests: pytest -m slow ...
"""

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from fast_llm_external_models.apriel2.configuration_apriel2 import Apriel2Config
from fast_llm_external_models.apriel2.convert_from_llava import (
    convert_config,
    convert_weights,
    map_weight_name,
)
from fast_llm_external_models.apriel2.modeling_apriel2 import Apriel2ForConditionalGeneration


# =============================================================================
# Config Conversion Tests
# =============================================================================


class TestConvertConfig:
    """Test pure config conversion (no surgery)."""

    @pytest.mark.parametrize(
        "config_fixture",
        [
            "llava_pixtral_config",
            pytest.param("apriel_1_5_config", marks=pytest.mark.slow),
        ],
    )
    def test_basic_conversion(self, config_fixture, request):
        """Test that Llava config converts to valid Apriel2 config."""
        llava_config = request.getfixturevalue(config_fixture)
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
        assert "patch_convolution" in result["vision_encoder"]
        assert "encoder" in result["vision_encoder"]
        assert "adapter" in result["vision_encoder"]

    @pytest.mark.parametrize(
        "config_fixture",
        [
            "llava_pixtral_config",
            pytest.param("apriel_1_5_config", marks=pytest.mark.slow),
        ],
    )
    def test_config_can_be_instantiated(self, config_fixture, request):
        """Test that converted config can create Apriel2Config object."""
        llava_config = request.getfixturevalue(config_fixture)
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
# Weight Name Mapping Tests
# =============================================================================


class TestMapWeightName:
    """Test weight name mapping."""

    def test_static_mappings(self):
        """Test static weight mappings."""
        assert map_weight_name("language_model.model.embed_tokens.weight") == "model.embed_tokens.weight"
        assert map_weight_name("language_model.model.norm.weight") == "model.norm.weight"
        assert map_weight_name("language_model.lm_head.weight") == "lm_head.weight"

    def test_decoder_layer_mappings(self):
        """Test decoder layer weight mappings."""
        assert map_weight_name(
            "language_model.model.layers.0.self_attn.q_proj.weight"
        ) == "model.decoder.blocks.0.mixer.self_attn.q_proj.weight"

        assert map_weight_name(
            "language_model.model.layers.5.mlp.gate_proj.weight"
        ) == "model.decoder.blocks.5.mlp.gate_proj.weight"

        assert map_weight_name(
            "language_model.model.layers.10.input_layernorm.weight"
        ) == "model.decoder.blocks.10.input_layernorm.weight"

    def test_vision_layer_mappings(self):
        """Test vision encoder layer mappings."""
        assert map_weight_name(
            "vision_tower.transformer.layers.0.attention.q_proj.weight"
        ) == "model.vision_encoder.encoder.blocks.0.mixer.self_attn.q_proj.weight"

        assert map_weight_name(
            "vision_tower.transformer.layers.2.feed_forward.gate_proj.weight"
        ) == "model.vision_encoder.encoder.blocks.2.mlp.gate_proj.weight"

    def test_vision_adapter_mappings(self):
        """Test vision adapter (projector) mappings."""
        assert map_weight_name(
            "multi_modal_projector.linear_1.weight"
        ) == "model.vision_encoder.adapter.linear_1.weight"

        assert map_weight_name(
            "multi_modal_projector.linear_2.bias"
        ) == "model.vision_encoder.adapter.linear_2.bias"

    def test_unknown_weight_returns_none(self):
        """Test that unknown weights return None."""
        assert map_weight_name("unknown.weight") is None
        assert map_weight_name("some.random.path") is None


# =============================================================================
# Weight Conversion Tests
# =============================================================================


class TestConvertWeights:
    """Test weight conversion."""

    def test_converts_all_weights(self, llava_pixtral_checkpoint):
        """Test that all weights are converted."""
        # Load source weights
        with safe_open(llava_pixtral_checkpoint / "model.safetensors", framework="pt") as f:
            source_weights = {key: f.get_tensor(key) for key in f.keys()}

        apriel2_weights = convert_weights(source_weights)

        # Should have same number of weights (all mapped)
        assert len(apriel2_weights) == len(source_weights)

    def test_weight_names_are_apriel2_format(self, llava_pixtral_checkpoint):
        """Test that converted weight names are in Apriel2 format."""
        with safe_open(llava_pixtral_checkpoint / "model.safetensors", framework="pt") as f:
            source_weights = {key: f.get_tensor(key) for key in f.keys()}

        apriel2_weights = convert_weights(source_weights)

        # Check decoder weights
        assert any("model.decoder.blocks.0.mixer" in k for k in apriel2_weights.keys())
        assert any("model.decoder.blocks.0.mlp" in k for k in apriel2_weights.keys())

        # Check vision weights
        assert any("model.vision_encoder.encoder.blocks" in k for k in apriel2_weights.keys())
        assert any("model.vision_encoder.adapter" in k for k in apriel2_weights.keys())

    def test_weight_values_unchanged(self, llava_pixtral_checkpoint):
        """Test that weight values are not modified during conversion."""
        with safe_open(llava_pixtral_checkpoint / "model.safetensors", framework="pt") as f:
            source_weights = {key: f.get_tensor(key) for key in f.keys()}

        apriel2_weights = convert_weights(source_weights)

        # Check a few specific weights are identical
        source_embed = source_weights["language_model.model.embed_tokens.weight"]
        target_embed = apriel2_weights["model.embed_tokens.weight"]
        assert torch.equal(source_embed, target_embed)


# =============================================================================
# Surgery Tests
# =============================================================================


class TestSurgery:
    """Test surgery operations (Apriel2 -> Apriel2)."""

    def test_identity_surgery(self, llava_pixtral_checkpoint, tmp_path):
        """Test surgery with same source and target config (identity)."""
        from fast_llm_external_models.apriel2.surgery import surgery

        # Load and convert to Apriel2 base
        with safe_open(llava_pixtral_checkpoint / "model.safetensors", framework="pt") as f:
            source_weights = {key: f.get_tensor(key) for key in f.keys()}

        llava_config = json.load(open(llava_pixtral_checkpoint / "config.json"))
        apriel2_config = convert_config(llava_config)
        apriel2_weights = convert_weights(source_weights)

        # Surgery with same config = identity
        result_weights = surgery(apriel2_config, apriel2_weights, apriel2_config)

        # Non-decoder weights should be identical
        assert "model.embed_tokens.weight" in result_weights
        assert torch.allclose(
            result_weights["model.embed_tokens.weight"],
            apriel2_weights["model.embed_tokens.weight"],
        )

    def test_surgery_to_stochastic_mixer(self, llava_pixtral_checkpoint, tmp_path):
        """Test surgery that wraps attention with stochastic mixer."""
        from fast_llm_external_models.apriel2.surgery import surgery

        # Load and convert to Apriel2 base
        with safe_open(llava_pixtral_checkpoint / "model.safetensors", framework="pt") as f:
            source_weights = {key: f.get_tensor(key) for key in f.keys()}

        llava_config = json.load(open(llava_pixtral_checkpoint / "config.json"))
        source_config = convert_config(llava_config)
        source_weights = convert_weights(source_weights)

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

        result_weights = surgery(source_config, source_weights, target_config)

        # Should have weights for both sub-mixers
        attn_keys = [k for k in result_weights if ".mixers.attention." in k]
        sw_keys = [k for k in result_weights if ".mixers.sliding_window." in k]

        assert len(attn_keys) > 0, "No attention sub-mixer weights"
        assert len(sw_keys) > 0, "No sliding_window sub-mixer weights"
        assert len(attn_keys) == len(sw_keys), "Sub-mixer weight counts differ"

    def test_surgery_mamba_random_init(self, llava_pixtral_checkpoint, tmp_path):
        """Test surgery that adds mamba (requires random init)."""
        from fast_llm_external_models.apriel2.surgery import surgery

        # Load and convert to Apriel2 base
        with safe_open(llava_pixtral_checkpoint / "model.safetensors", framework="pt") as f:
            source_weights = {key: f.get_tensor(key) for key in f.keys()}

        llava_config = json.load(open(llava_pixtral_checkpoint / "config.json"))
        source_config = convert_config(llava_config)
        source_weights = convert_weights(source_weights)
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
                    "expand": 2,
                },
            },
        }

        result_weights = surgery(source_config, source_weights, target_config)

        # Should have mamba weights (randomly initialized)
        mamba_keys = [k for k in result_weights if ".mixers.mamba." in k]
        assert len(mamba_keys) > 0, "No mamba weights created"

        # Mamba weights should exist and have correct shapes
        for key in mamba_keys:
            assert result_weights[key] is not None
            assert result_weights[key].numel() > 0


# =============================================================================
# Forward Pass Equivalence Tests
# =============================================================================


def _load_models_for_comparison(llava_pixtral_checkpoint, tmp_path):
    """Helper to load source Llava and converted Apriel2 models."""
    from transformers import LlavaForConditionalGeneration

    # Load source model
    source_model = LlavaForConditionalGeneration.from_pretrained(llava_pixtral_checkpoint)
    source_model.eval()

    # Load and convert weights
    with safe_open(llava_pixtral_checkpoint / "model.safetensors", framework="pt") as f:
        source_weights = {key: f.get_tensor(key) for key in f.keys()}

    llava_config = json.load(open(llava_pixtral_checkpoint / "config.json"))
    apriel2_config_dict = convert_config(llava_config)
    apriel2_weights = convert_weights(source_weights)

    # Load Apriel2 model
    apriel2_config = Apriel2Config(**apriel2_config_dict)
    target_model = Apriel2ForConditionalGeneration(apriel2_config)
    target_model.load_state_dict(apriel2_weights, strict=False)
    target_model.eval()

    return source_model, target_model, llava_config


class TestComponentEquivalence:
    """Test individual components produce identical outputs."""

    def test_text_embedding_equivalence(self, llava_pixtral_checkpoint, tmp_path):
        """Test text embedding layer produces identical outputs."""
        source_model, target_model, llava_config = _load_models_for_comparison(
            llava_pixtral_checkpoint, tmp_path
        )

        source_embed = source_model.model.language_model.embed_tokens
        target_embed = target_model.model.embed_tokens

        torch.manual_seed(42)
        input_ids = torch.randint(0, llava_config["text_config"]["vocab_size"], (2, 16))

        with torch.no_grad():
            source_out = source_embed(input_ids)
            target_out = target_embed(input_ids)

        assert torch.allclose(source_out, target_out, atol=1e-6, rtol=1e-5)

    def test_lm_head_equivalence(self, llava_pixtral_checkpoint, tmp_path):
        """Test LM head produces identical outputs."""
        source_model, target_model, llava_config = _load_models_for_comparison(
            llava_pixtral_checkpoint, tmp_path
        )

        source_head = source_model.lm_head
        target_head = target_model.lm_head

        torch.manual_seed(42)
        hidden_size = llava_config["text_config"]["hidden_size"]
        hidden_states = torch.randn(2, 16, hidden_size)

        with torch.no_grad():
            source_out = source_head(hidden_states)
            target_out = target_head(hidden_states)

        assert torch.allclose(source_out, target_out, atol=1e-6, rtol=1e-5)

    def test_vision_patch_embedding_equivalence(self, llava_pixtral_checkpoint, tmp_path):
        """Test vision patch embedding produces identical outputs."""
        source_model, target_model, llava_config = _load_models_for_comparison(
            llava_pixtral_checkpoint, tmp_path
        )

        source_conv = source_model.model.vision_tower.patch_conv
        source_norm = source_model.model.vision_tower.ln_pre
        target_patch = target_model.model.vision_encoder.patch_convolution

        torch.manual_seed(42)
        pixel_values = torch.randn(1, 3, 32, 32)

        with torch.no_grad():
            source_out = source_conv(pixel_values)
            b, c, h, w = source_out.shape
            source_out = source_out.flatten(2).transpose(1, 2)
            source_out = source_norm(source_out)

            target_out = target_patch(pixel_values)

        assert torch.allclose(source_out, target_out, atol=1e-5, rtol=1e-5)

    def test_multimodal_projector_equivalence(self, llava_pixtral_checkpoint, tmp_path):
        """Test multimodal projector produces identical outputs."""
        source_model, target_model, llava_config = _load_models_for_comparison(
            llava_pixtral_checkpoint, tmp_path
        )

        source_proj = source_model.model.multi_modal_projector
        target_proj = target_model.model.vision_encoder.adapter

        torch.manual_seed(42)
        vision_hidden_size = llava_config["vision_config"]["hidden_size"]
        vision_hidden = torch.randn(2, 16, vision_hidden_size)

        with torch.no_grad():
            source_out = source_proj(vision_hidden)
            target_out = target_proj(vision_hidden)

        assert torch.allclose(source_out, target_out, atol=1e-5, rtol=1e-5)


class TestFullModelEquivalence:
    """Test full model forward pass equivalence."""

    def test_text_only_forward(self, llava_pixtral_checkpoint, tmp_path):
        """Test text-only forward pass produces identical outputs."""
        source_model, target_model, llava_config = _load_models_for_comparison(
            llava_pixtral_checkpoint, tmp_path
        )

        torch.manual_seed(42)
        vocab_size = llava_config["text_config"]["vocab_size"]
        input_ids = torch.randint(0, vocab_size, (2, 16))

        with torch.no_grad():
            source_out = source_model(input_ids)
            target_out = target_model(input_ids)

        assert torch.allclose(source_out.logits, target_out.logits, atol=1e-5, rtol=1e-5)

    def test_multimodal_forward(self, llava_pixtral_checkpoint, tmp_path):
        """Test multimodal forward pass works on both models.

        Note: Full numerical equivalence is not tested due to architectural
        differences in patch extraction between Pixtral and Apriel2.
        """
        source_model, target_model, llava_config = _load_models_for_comparison(
            llava_pixtral_checkpoint, tmp_path
        )

        vision_config = llava_config["vision_config"]
        image_token_index = llava_config["image_token_index"]
        vocab_size = llava_config["text_config"]["vocab_size"]

        torch.manual_seed(42)
        batch_size = 1
        image_size = 64
        pixel_values = torch.randn(batch_size, 3, image_size, image_size)

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
        assert torch.isfinite(source_out.logits).all()

        # Test target model
        target_input_ids = self._create_multimodal_input_ids(
            vocab_size, image_token_index, target_patches, batch_size
        )
        with torch.no_grad():
            target_out = target_model(input_ids=target_input_ids, pixel_values=pixel_values)
        assert torch.isfinite(target_out.logits).all()

    def _create_multimodal_input_ids(self, vocab_size, image_token_index, num_patches, batch_size):
        """Helper to create input_ids with image token placeholders."""
        prefix = torch.randint(0, vocab_size, (batch_size, 5))
        prefix = torch.where(prefix == image_token_index, torch.tensor(0), prefix)
        image_tokens = torch.full((batch_size, num_patches), image_token_index)
        suffix = torch.randint(0, vocab_size, (batch_size, 5))
        suffix = torch.where(suffix == image_token_index, torch.tensor(0), suffix)
        return torch.cat([prefix, image_tokens, suffix], dim=1)

    def test_model_can_load_converted_weights(self, llava_pixtral_checkpoint, tmp_path):
        """Test that converted weights can be loaded into Apriel2 model."""
        with safe_open(llava_pixtral_checkpoint / "model.safetensors", framework="pt") as f:
            source_weights = {key: f.get_tensor(key) for key in f.keys()}

        llava_config = json.load(open(llava_pixtral_checkpoint / "config.json"))
        apriel2_config_dict = convert_config(llava_config)
        apriel2_weights = convert_weights(source_weights)

        apriel2_config = Apriel2Config(**apriel2_config_dict)
        model = Apriel2ForConditionalGeneration(apriel2_config)

        missing, unexpected = model.load_state_dict(apriel2_weights, strict=False)

        assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"
        for key in missing:
            assert "cache" in key.lower() or "position" in key.lower() or "mask" in key.lower()


# =============================================================================
# Apriel 1.5 Full Conversion Tests (slow)
# =============================================================================


@pytest.mark.slow
class TestApriel15Conversion:
    """Test conversion with the real Apriel 1.5 checkpoint."""

    def test_apriel_1_5_config_conversion(self, apriel_1_5_config):
        """Test config conversion produces valid Apriel2 config."""
        apriel2_config_dict = convert_config(apriel_1_5_config)

        assert apriel2_config_dict["hidden_size"] == 5120
        assert apriel2_config_dict["vocab_size"] == 131072
        assert apriel2_config_dict["decoder"]["num_blocks"] == 48

        config = Apriel2Config(**apriel2_config_dict)
        assert config.hidden_size == 5120

    def test_apriel_1_5_weight_conversion(self, apriel_1_5_checkpoint, tmp_path):
        """Test full weight conversion of Apriel 1.5."""
        from fast_llm_external_models.apriel2.convert_from_llava import (
            convert_config,
            convert_weights,
            resolve_input,
            copy_model_files,
        )
        from safetensors import safe_open

        output_dir = tmp_path / "apriel2_converted"
        output_dir.mkdir(parents=True, exist_ok=True)

        input_path = resolve_input(apriel_1_5_checkpoint)

        with open(input_path / "config.json") as f:
            llava_config = json.load(f)

        apriel2_config = convert_config(llava_config)

        with open(output_dir / "config.json", "w") as f:
            json.dump(apriel2_config, f, indent=2)

        # Load source weights
        safetensor_files = sorted(input_path.glob("*.safetensors"))
        all_weights = {}
        for model_file in safetensor_files:
            with safe_open(model_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    all_weights[key] = f.get_tensor(key)

        apriel2_weights = convert_weights(all_weights)
        save_file(apriel2_weights, output_dir / "model.safetensors")

        copy_model_files(output_dir)

        assert (output_dir / "config.json").exists()
        assert (output_dir / "model.safetensors").exists()

        with open(output_dir / "config.json") as f:
            config = json.load(f)

        assert config["model_type"] == "apriel2"
        assert config["hidden_size"] == 5120


# =============================================================================
# Converters Tests
# =============================================================================


class TestConverters:
    """Test converter registry and implementations."""

    def test_identity_converter(self):
        """Test identity conversion (same type)."""
        from fast_llm_external_models.apriel2.converters import get_converter

        converter = get_converter("attention", "attention")
        assert converter is not None

        weights = {"self_attn.q_proj.weight": torch.randn(256, 256)}
        result = converter(weights, {}, {}, 256)

        assert torch.allclose(weights["self_attn.q_proj.weight"], result["self_attn.q_proj.weight"])

    def test_attention_to_sliding_window(self):
        """Test attention to sliding window conversion."""
        from fast_llm_external_models.apriel2.converters import get_converter

        converter = get_converter("attention", "sliding_window")
        assert converter is not None

        weights = {"self_attn.q_proj.weight": torch.randn(256, 256)}
        result = converter(weights, {}, {"window_size": 512}, 256)

        # Should copy weights unchanged
        assert torch.allclose(weights["self_attn.q_proj.weight"], result["self_attn.q_proj.weight"])

    def test_no_converter_returns_none(self):
        """Test that missing converter returns None."""
        from fast_llm_external_models.apriel2.converters import get_converter

        # No converter for attention -> mamba
        converter = get_converter("attention", "mamba")
        assert converter is None

    def test_random_init_mamba(self):
        """Test random initialization for mamba."""
        from fast_llm_external_models.apriel2.converters import random_init_mixer

        config = {"type": "mamba", "d_state": 16, "d_conv": 4, "expand": 2}
        weights = random_init_mixer(config, 256)

        assert "in_proj.weight" in weights
        assert "conv1d.weight" in weights
        assert "out_proj.weight" in weights
        assert weights["in_proj.weight"].shape[0] == 2 * 2 * 256  # 2 * expand * hidden

    def test_random_init_attention(self):
        """Test random initialization for attention."""
        from fast_llm_external_models.apriel2.converters import random_init_mixer

        config = {"type": "attention", "heads": 8, "head_groups": 4, "head_size": 32}
        weights = random_init_mixer(config, 256)

        assert "self_attn.q_proj.weight" in weights
        assert "self_attn.k_proj.weight" in weights
        assert "self_attn.v_proj.weight" in weights
        assert "self_attn.o_proj.weight" in weights
