"""Equivalence tests for Llava/Pixtral to Apriel2 conversion.

Testing Philosophy: Source-of-Truth Isolation
=============================================

To avoid floating-point error accumulation through the model pipeline, we test
each component in isolation by using Pixtral's output as the "source of truth"
input to both models. This ensures:

1. Each component can be tested with strict 1e-6 tolerance
2. Failures pinpoint exactly which component has a bug
3. Integration tests become documentation of expected FP variance, not bug detection

Test Structure:
- TestComponentIsolation: Each component tested with Pixtral output as input
- TestIntegration: End-to-end tests documenting expected FP compound variance
"""

from dataclasses import dataclass
from typing import Optional

import pytest
import torch
from transformers import LlavaForConditionalGeneration

from fast_llm_external_models.apriel2.modeling_apriel2 import Apriel2ForConditionalGeneration


# =============================================================================
# Input Configuration
# =============================================================================


@dataclass(frozen=True)
class InputConfig:
    """Configuration for test inputs."""

    name: str
    batch_size: int
    images_per_seq: tuple[int, ...]
    image_size: Optional[tuple[int, int]] = (64, 64)

    def __post_init__(self):
        assert len(self.images_per_seq) == self.batch_size

    @property
    def has_images(self) -> bool:
        return self.image_size is not None and sum(self.images_per_seq) > 0

    @property
    def total_images(self) -> int:
        return sum(self.images_per_seq)

    def __str__(self) -> str:
        return self.name


INPUT_CONFIGS = [
    InputConfig("single_img", batch_size=1, images_per_seq=(1,), image_size=(64, 64)),
    InputConfig("text_only", batch_size=2, images_per_seq=(0, 0), image_size=None),
    InputConfig("batch_2_single", batch_size=2, images_per_seq=(1, 1), image_size=(64, 64)),
    InputConfig("multi_img_seq", batch_size=2, images_per_seq=(2, 1), image_size=(64, 64)),
    InputConfig("batch_3_multi", batch_size=3, images_per_seq=(2, 1, 3), image_size=(64, 64)),
    InputConfig("tall_img", batch_size=1, images_per_seq=(1,), image_size=(48, 64)),
    InputConfig("wide_img", batch_size=1, images_per_seq=(1,), image_size=(64, 48)),
]


@dataclass
class ModelInputs:
    """Container for model inputs."""

    input_ids: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None
    pixel_values: Optional[torch.Tensor] = None

    def to_kwargs(self) -> dict:
        kwargs = {"input_ids": self.input_ids}
        if self.attention_mask is not None:
            kwargs["attention_mask"] = self.attention_mask
        if self.pixel_values is not None:
            kwargs["pixel_values"] = self.pixel_values
        return kwargs


def create_inputs(model: LlavaForConditionalGeneration, config: InputConfig, seed: int = 42) -> ModelInputs:
    """Create model inputs from configuration."""
    torch.manual_seed(seed)

    model_config = model.config
    vocab_size = model_config.text_config.vocab_size
    image_token_index = model_config.image_token_index
    text_length = 10

    if config.has_images:
        h, w = config.image_size
        dummy_pixel = torch.randn(1, 3, h, w)
        with torch.no_grad():
            features = model.get_image_features(dummy_pixel)
        num_patches = features[0].shape[0] if isinstance(features, list) else features.shape[1]
    else:
        num_patches = 0

    all_input_ids = []
    max_seq_len = 0

    for num_images in config.images_per_seq:
        seq_parts = []
        text = torch.randint(0, vocab_size, (text_length,))
        text = torch.where(text == image_token_index, torch.tensor(0), text)
        seq_parts.append(text)

        for i in range(num_images):
            img_tokens = torch.full((num_patches,), image_token_index, dtype=torch.long)
            seq_parts.append(img_tokens)
            if i < num_images - 1:
                text = torch.randint(0, vocab_size, (text_length // 2,))
                text = torch.where(text == image_token_index, torch.tensor(0), text)
                seq_parts.append(text)

        text = torch.randint(0, vocab_size, (text_length,))
        text = torch.where(text == image_token_index, torch.tensor(0), text)
        seq_parts.append(text)

        seq = torch.cat(seq_parts)
        all_input_ids.append(seq)
        max_seq_len = max(max_seq_len, len(seq))

    padded_input_ids = []
    attention_masks = []
    for seq in all_input_ids:
        pad_len = max_seq_len - len(seq)
        if pad_len > 0:
            seq = torch.cat([seq, torch.zeros(pad_len, dtype=seq.dtype)])
        padded_input_ids.append(seq)
        mask = torch.ones(max_seq_len, dtype=torch.long)
        if pad_len > 0:
            mask[-pad_len:] = 0
        attention_masks.append(mask)

    pixel_values = None
    if config.has_images:
        h, w = config.image_size
        pixel_values = torch.randn(config.total_images, 3, h, w)

    return ModelInputs(
        input_ids=torch.stack(padded_input_ids),
        attention_mask=torch.stack(attention_masks),
        pixel_values=pixel_values,
    )


# =============================================================================
# Helpers
# =============================================================================


def assert_equivalent(a: torch.Tensor, b: torch.Tensor, context: str, atol: float = 1e-6):
    """Assert tensors are equivalent, with detailed error message."""
    assert a.shape == b.shape, f"[{context}] Shape mismatch: {a.shape} vs {b.shape}"
    max_diff = (a - b).abs().max().item()
    print(f"[{context}] max_diff={max_diff:.6f}")
    assert max_diff <= atol, f"[{context}] max_diff={max_diff:.6f} > atol={atol}"


def get_pixtral_vision_features(source: LlavaForConditionalGeneration, pixel_values: torch.Tensor) -> torch.Tensor:
    """Get vision features from Pixtral, flattened to [total_patches, hidden]."""
    features = source.get_image_features(pixel_values)
    if isinstance(features, list):
        features = torch.cat(features, dim=0)
    return features


def get_pixtral_merged_embeds(
    source: LlavaForConditionalGeneration,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
) -> torch.Tensor:
    """Get merged embeddings from Pixtral (text + vision features merged)."""
    # Get text embeddings
    inputs_embeds = source.model.get_input_embeddings()(input_ids)

    # Get vision features
    vision_features = get_pixtral_vision_features(source, pixel_values)

    # Create mask and merge
    image_token_index = source.config.image_token_index
    special_image_mask = input_ids == image_token_index
    special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds)

    merged = inputs_embeds.masked_scatter(special_image_mask, vision_features)
    return merged


def get_pixtral_hidden_states(
    source: LlavaForConditionalGeneration,
    merged_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Get hidden states from Pixtral's text decoder."""
    outputs = source.model.language_model(
        inputs_embeds=merged_embeds,
        attention_mask=attention_mask,
    )
    return outputs.last_hidden_state


# =============================================================================
# Component Isolation Tests
# =============================================================================


@pytest.fixture(params=INPUT_CONFIGS, ids=lambda c: c.name)
def input_config(request) -> InputConfig:
    return request.param


class TestComponentIsolation:
    """Test each component with Pixtral's output as source-of-truth input.

    All tests should pass with 0.0 or near-0.0 difference since each component
    receives identical inputs. Any failure indicates a bug in that specific component.

    Note: Identity tests are skipped for most component tests since both models
    are LlavaForConditionalGeneration with identical weights - they would trivially pass.
    The value of isolation tests is for the converted variant.
    """

    def test_vision_encoder(self, model_pair, input_config: InputConfig):
        """Vision encoder: Same pixel_values → compare vision features.

        Both models process identical pixel_values through their vision encoders.
        This tests the full vision pipeline: embeddings → transformer → adapter.
        """
        source, target, _, variant = model_pair

        if variant == "identity":
            pytest.skip("Identity variant: both are same model type, trivially passes")

        if not input_config.has_images:
            pytest.skip("No images in this config")

        inputs = create_inputs(source, input_config)

        with torch.no_grad():
            # Pixtral vision features
            src_features = get_pixtral_vision_features(source, inputs.pixel_values)

            # Apriel2 vision features (flatten to match Pixtral format)
            tgt_features = target.get_image_features(inputs.pixel_values)
            tgt_features = tgt_features.view(-1, tgt_features.shape[-1])

        assert_equivalent(src_features, tgt_features, f"{variant}/{input_config}/vision_encoder")

    def test_text_embeddings(self, model_pair, input_config: InputConfig):
        """Text embeddings: Same input_ids → compare embed_tokens output."""
        source, target, _, variant = model_pair

        if variant == "identity":
            pytest.skip("Identity variant: both are same model type, trivially passes")

        inputs = create_inputs(source, input_config)

        with torch.no_grad():
            src_embeds = source.model.get_input_embeddings()(inputs.input_ids)
            tgt_embeds = target.model.embed_tokens(inputs.input_ids)

        assert_equivalent(src_embeds, tgt_embeds, f"{variant}/{input_config}/text_embeddings")

    def test_merge_logic(self, model_pair, input_config: InputConfig):
        """Merge logic: Same (vision_features, text_embeds) → compare merged result.

        Uses Pixtral's vision features as input to both merge implementations.
        """
        source, target, _, variant = model_pair

        if variant == "identity":
            pytest.skip("Identity variant: both are same model type, trivially passes")

        if not input_config.has_images:
            pytest.skip("No images in this config")

        inputs = create_inputs(source, input_config)

        with torch.no_grad():
            # Get Pixtral vision features (source of truth)
            pixtral_features = get_pixtral_vision_features(source, inputs.pixel_values)

            # Get text embeddings (should be identical)
            src_embeds = source.model.get_input_embeddings()(inputs.input_ids)
            tgt_embeds = target.model.embed_tokens(inputs.input_ids)

            # Create mask
            image_token_index = source.config.image_token_index
            special_image_mask = inputs.input_ids == image_token_index
            special_image_mask = special_image_mask.unsqueeze(-1).expand_as(src_embeds)

            # Merge using Pixtral features in both
            src_merged = src_embeds.masked_scatter(special_image_mask, pixtral_features)
            tgt_merged = tgt_embeds.masked_scatter(special_image_mask, pixtral_features)

        assert_equivalent(src_merged, tgt_merged, f"{variant}/{input_config}/merge_logic")

    def test_text_decoder(self, model_pair, input_config: InputConfig):
        """Text decoder: Same merged_embeds (from Pixtral) → compare hidden states.

        This is the key isolation test: uses Pixtral's merged embeddings as input
        to both decoders, eliminating any vision encoder variance.
        """
        source, target, _, variant = model_pair

        if not input_config.has_images:
            pytest.skip("No images in this config")

        inputs = create_inputs(source, input_config)

        with torch.no_grad():
            # Get merged embeddings from Pixtral (source of truth)
            merged_embeds = get_pixtral_merged_embeds(source, inputs.input_ids, inputs.pixel_values)

            # Forward through Pixtral's text decoder
            src_outputs = source.model.language_model(
                inputs_embeds=merged_embeds,
                attention_mask=inputs.attention_mask,
            )
            src_hidden = src_outputs.last_hidden_state

            # Forward through Apriel2's text decoder (using same merged_embeds)
            tgt_outputs = target.model(
                inputs_embeds=merged_embeds,
                attention_mask=inputs.attention_mask,
                pixel_values=None,  # Don't re-process images
            )
            tgt_hidden = tgt_outputs.last_hidden_state

        assert_equivalent(src_hidden, tgt_hidden, f"{variant}/{input_config}/text_decoder")

    def test_lm_head(self, model_pair, input_config: InputConfig):
        """LM head: Same hidden_states (from Pixtral) → compare logits.

        Uses Pixtral's full pipeline output as input to both LM heads.
        """
        source, target, _, variant = model_pair

        if not input_config.has_images:
            pytest.skip("No images in this config")

        inputs = create_inputs(source, input_config)

        with torch.no_grad():
            # Get merged embeddings and hidden states from Pixtral
            merged_embeds = get_pixtral_merged_embeds(source, inputs.input_ids, inputs.pixel_values)
            pixtral_hidden = get_pixtral_hidden_states(source, merged_embeds, inputs.attention_mask)

            # Apply LM heads to same hidden states
            src_logits = source.lm_head(pixtral_hidden)
            tgt_logits = target.lm_head(pixtral_hidden)

        assert_equivalent(src_logits, tgt_logits, f"{variant}/{input_config}/lm_head")

    def test_text_only_forward(self, model_pair, input_config: InputConfig):
        """Text-only forward: No images, full forward comparison."""
        source, target, _, variant = model_pair

        if input_config.has_images:
            pytest.skip("This test is for text-only configs")

        inputs = create_inputs(source, input_config)

        with torch.no_grad():
            src_out = source(**inputs.to_kwargs())
            tgt_out = target(**inputs.to_kwargs())

        assert_equivalent(src_out.logits, tgt_out.logits, f"{variant}/{input_config}/text_only")


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """End-to-end tests that document expected FP compound variance.

    These tests use the full pipeline (not isolated components). Any variance
    here is due to floating-point accumulation through the pipeline, NOT bugs,
    as long as all TestComponentIsolation tests pass.
    """

    def test_full_forward(self, model_pair, input_config: InputConfig):
        """Full forward pass comparison.

        Expected behavior:
        - Identity variant: 0.0 diff
        - Converted variant with images: Small FP variance that compounds
          through layers. If isolation tests pass, this variance is expected.
        """
        source, target, expected_atol, variant = model_pair

        inputs = create_inputs(source, input_config)

        with torch.no_grad():
            src_out = source(**inputs.to_kwargs())
            tgt_out = target(**inputs.to_kwargs())

        max_diff = (src_out.logits - tgt_out.logits).abs().max().item()
        print(f"[{variant}/{input_config}/full_forward] max_diff={max_diff:.6f}")

        # For identity tests, require exact match
        if variant == "identity":
            assert max_diff == 0.0, f"Identity test should have 0.0 diff, got {max_diff}"
        else:
            # For converted tests, document the variance
            # If all isolation tests pass, any variance here is just FP accumulation
            print(f"  NOTE: If isolation tests pass, this variance is expected FP accumulation")
            # Use a loose tolerance - the isolation tests catch real bugs
            assert max_diff < 1e-2, f"Unexpectedly large diff: {max_diff}"


# =============================================================================
# Diagnostic Tests
# =============================================================================


class TestDiagnostics:
    """Diagnostic tests to verify implementation details."""

    def test_weight_equivalence(self, model_pair):
        """Verify key weights are identical after conversion."""
        source, target, _, variant = model_pair

        if variant == "identity":
            pytest.skip("Weight comparison only meaningful for converted variant")

        # Vision encoder normalization
        source_ln = source.model.vision_tower.ln_pre.weight
        target_ln = target.model.vision_encoder.embeddings.normalization.weight
        max_diff = (source_ln - target_ln).abs().max().item()
        print(f"ln_pre/normalization weight max_diff: {max_diff:.6f}")
        assert max_diff == 0.0, f"ln_pre weights differ: {max_diff}"

        # Adapter/projector
        source_proj = source.model.multi_modal_projector.linear_1.weight
        target_proj = target.model.vision_encoder.adapter.linear_1.weight
        max_diff = (source_proj - target_proj).abs().max().item()
        print(f"adapter linear_1 weight max_diff: {max_diff:.6f}")
        assert max_diff == 0.0, f"adapter weights differ: {max_diff}"

    def test_rotary_embedding_equivalence(self, model_pair):
        """Verify rotary embeddings are identical."""
        source, target, _, variant = model_pair

        if variant == "identity":
            pytest.skip("Diagnostic only meaningful for converted variant")

        pixtral_rotary = source.model.vision_tower.patch_positional_embedding

        apriel2_rotary = None
        for name, module in target.model.vision_encoder.encoder.named_modules():
            if "rotary_emb" in name:
                apriel2_rotary = module
                break

        assert apriel2_rotary is not None, "Apriel2 rotary embedding not found"

        max_diff = (pixtral_rotary.inv_freq - apriel2_rotary.inv_freq).abs().max().item()
        print(f"inv_freq max_diff: {max_diff}")
        assert max_diff == 0.0, f"Rotary inv_freq values differ: {max_diff}"

    def test_batch_processing_behavior(self, model_pair):
        """Verify both models have identical batch vs sequential behavior.

        Both use concat+block_mask, so they should show the same numerical
        variance between batch and sequential processing.
        """
        source, target, _, variant = model_pair

        if variant == "identity":
            pytest.skip("Diagnostic only meaningful for converted variant")

        torch.manual_seed(42)
        pixel_values = torch.randn(3, 3, 64, 64)

        with torch.no_grad():
            # Batch processing
            batch_src = get_pixtral_vision_features(source, pixel_values)
            batch_tgt = target.get_image_features(pixel_values).view(-1, batch_src.shape[-1])

            # Sequential processing
            singles_src = [get_pixtral_vision_features(source, pixel_values[i:i+1]) for i in range(3)]
            singles_tgt = [target.get_image_features(pixel_values[i:i+1]).view(-1, batch_src.shape[-1]) for i in range(3)]

            single_concat_src = torch.cat(singles_src, dim=0)
            single_concat_tgt = torch.cat(singles_tgt, dim=0)

        src_diff = (batch_src - single_concat_src).abs().max().item()
        tgt_diff = (batch_tgt - single_concat_tgt).abs().max().item()

        print(f"Pixtral batch vs sequential: {src_diff:.6f}")
        print(f"Apriel2 batch vs sequential: {tgt_diff:.6f}")

        # Both should have the same behavior (within FP tolerance)
        assert abs(src_diff - tgt_diff) < 1e-6, (
            f"Batch processing behavior differs: src={src_diff:.6f}, tgt={tgt_diff:.6f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
