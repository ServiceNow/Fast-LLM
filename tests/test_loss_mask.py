"""
Integration test that loss_mask correctly combines all masking sources:
- Negative labels (padding and image placeholders)
- loss_masking_spans

Tests the actual preprocess_batch code path in fast_llm/models/gpt/model.py
"""

import torch

from fast_llm.config import NoAutoValidate
from fast_llm.data.sample.language_model import LanguageModelBatch
from fast_llm.data.sample.range import RangeBatch
from fast_llm.data.sample.token import TokenBatch
from fast_llm.engine.distributed.config import PhaseType
from fast_llm.layers.language_model.config import LanguageModelKwargs
from fast_llm.models.gpt.config import GPTBatchConfig, GPTModelConfig
from tests.utils.utils import get_base_model, requires_cuda


def create_test_batch(
    tokens: torch.Tensor,
    lengths: list[list[int]] | None = None,
    loss_masking_spans: list[list[tuple[int, int]]] | None = None,
) -> LanguageModelBatch:
    """Create a LanguageModelBatch for testing."""
    token_batch = TokenBatch(tokens, lengths)

    if loss_masking_spans is not None:
        range_batch = RangeBatch(loss_masking_spans, sample_size=tokens.shape[1])
    else:
        range_batch = None

    return LanguageModelBatch(
        tokens=token_batch,
        loss_masking_spans=range_batch,
    )


def get_minimal_model():
    """Create a minimal GPT model for testing."""
    config = GPTModelConfig.from_dict(
        {
            "base_model": {
                "decoder": {"num_blocks": 1},
                "embeddings": {"vocab_size": 1000},
                "hidden_size": 64,
            },
            "distributed": {},
        },
    )
    model, distributed = get_base_model(config)
    return model, distributed


def run_preprocess_batch(model, distributed_config, batch: LanguageModelBatch, phase: PhaseType = PhaseType.training):
    """
    Run preprocess_batch with proper GPTBatchConfig metadata.

    This avoids the code path that accesses prediction_heads directly.
    """
    micro_batch_size, sequence_length = batch.tokens.tokens.shape

    # Create GPTBatchConfig for metadata with proper setup
    with NoAutoValidate():
        batch_config = GPTBatchConfig(
            batch_size=micro_batch_size,
            sequence_length=sequence_length,
        )
    batch_config.setup(distributed_config)
    batch_config.validate()

    # Get preprocessed metadata using GPTBatchConfig
    preprocessed_meta = model.preprocess_meta(batch_config, phase)

    # Run preprocess_batch with the actual batch data
    return model.preprocess_batch(
        batch,
        preprocessed_meta=preprocessed_meta,
        phase=phase,
        iteration=0,
    )


@requires_cuda
class TestLossMaskIntegration:
    """
    Integration tests for loss_mask computation in preprocess_batch.

    These tests verify the masking behavior by checking labels, since:
    1. loss_mask = labels >= 0 (masks negative labels)
    2. loss_masking_spans positions are also masked
    3. labels are set to -100 at all masked positions

    So if labels are -100 at expected positions, the masking is working.
    """

    def test_negative_labels_preserved(self):
        """Test that negative input tokens result in negative labels (shifted by 1)."""
        model, distributed = get_minimal_model()

        # Sequence: [text, text, IMG(-100), IMG(-100), text, text, text, text]
        # Labels (shifted by 1): [text, IMG, IMG, text, text, text, text, ?]
        tokens = torch.tensor(
            [
                [100, 101, -100, -100, 104, 105, 106, 107],
            ],
            dtype=torch.int64,
        )

        batch = create_test_batch(tokens)
        preprocessed = run_preprocess_batch(model, distributed.config, batch)

        assert len(preprocessed) == 1
        _, kwargs = preprocessed[0]

        labels = kwargs[LanguageModelKwargs.labels]
        # Flatten for easier indexing (handles sequence_first)
        labels_flat = labels.flatten()

        # Labels at positions 1,2 should be -100 (the next token after positions 0,1 is -100)
        assert labels_flat[1].item() == -100, f"Label at position 1 should be -100, got {labels_flat[1].item()}"
        assert labels_flat[2].item() == -100, f"Label at position 2 should be -100, got {labels_flat[2].item()}"

        # Labels at other positions should be positive
        assert labels_flat[0].item() > 0, "Label at position 0 should be positive"
        assert labels_flat[3].item() > 0, "Label at position 3 should be positive"

    def test_loss_masking_spans_set_labels_to_negative(self):
        """Test that loss_masking_spans positions have labels set to -100."""
        model, distributed = get_minimal_model()

        # All positive tokens
        tokens = torch.tensor(
            [
                [100, 101, 102, 103, 104, 105, 106, 107],
            ],
            dtype=torch.int64,
        )

        # loss_masking_spans are in TOKEN space, but labels are shifted by 1
        # Span (3, 5) in token space -> after cropping with labels_begin=1 -> (2, 4) in label space
        # This will mask label positions 2 and 3
        loss_masking_spans = [[(3, 5)]]

        batch = create_test_batch(tokens, loss_masking_spans=loss_masking_spans)
        preprocessed = run_preprocess_batch(model, distributed.config, batch)

        assert len(preprocessed) == 1
        _, kwargs = preprocessed[0]

        labels = kwargs[LanguageModelKwargs.labels]
        labels_flat = labels.flatten()

        # After cropping, positions 2,3 in label space should be masked (set to -100)
        assert labels_flat[2].item() == -100, f"Label at position 2 should be -100, got {labels_flat[2].item()}"
        assert labels_flat[3].item() == -100, f"Label at position 3 should be -100, got {labels_flat[3].item()}"

        # Positions outside the span should be positive
        assert labels_flat[0].item() > 0, "Label at position 0 should be positive"
        assert labels_flat[1].item() > 0, "Label at position 1 should be positive"
        assert labels_flat[4].item() > 0, "Label at position 4 should be positive"

    def test_combined_masking_negative_labels_and_spans(self):
        """Test that both negative labels AND loss_masking_spans result in -100 labels."""
        model, distributed = get_minimal_model()

        # Tokens with -100 at positions 4,5 (will affect labels at 3,4)
        tokens = torch.tensor(
            [
                [100, 101, 102, 103, -100, -100, 106, 107],
            ],
            dtype=torch.int64,
        )

        # loss_masking_spans in token space: (2, 3) -> after cropping to label space: (1, 2)
        # This will mask label position 1
        loss_masking_spans = [[(2, 3)]]

        batch = create_test_batch(tokens, loss_masking_spans=loss_masking_spans)
        preprocessed = run_preprocess_batch(model, distributed.config, batch)

        assert len(preprocessed) == 1
        _, kwargs = preprocessed[0]

        labels = kwargs[LanguageModelKwargs.labels]
        labels_flat = labels.flatten()

        # Position 1 should be -100 (from loss_masking_spans after cropping)
        assert labels_flat[1].item() == -100, f"Position 1 should be -100 (from spans), got {labels_flat[1].item()}"

        # Positions 3,4 should be -100 (from negative input tokens at positions 4,5)
        assert labels_flat[3].item() == -100, f"Position 3 should be -100 (from IMG), got {labels_flat[3].item()}"
        assert labels_flat[4].item() == -100, f"Position 4 should be -100 (from IMG), got {labels_flat[4].item()}"

        # Position 0, 2, 5 should be positive (not masked)
        assert labels_flat[0].item() > 0, "Position 0 should be positive"
        assert labels_flat[2].item() > 0, "Position 2 should be positive"
        assert labels_flat[5].item() > 0, "Position 5 should be positive"

    def test_all_padding_sample(self):
        """Test that a sample with all -100 tokens (padding) results in all -100 labels."""
        model, distributed = get_minimal_model()

        # Sample 0: normal tokens
        # Sample 1: all padding (-100)
        tokens = torch.tensor(
            [
                [100, 101, 102, 103, 104, 105, 106, 107],
                [-100, -100, -100, -100, -100, -100, -100, -100],
            ],
            dtype=torch.int64,
        )

        batch = create_test_batch(tokens)
        preprocessed = run_preprocess_batch(model, distributed.config, batch)

        assert len(preprocessed) == 1
        _, kwargs = preprocessed[0]

        labels = kwargs[LanguageModelKwargs.labels]

        # Get labels for sample 1 (all should be -100)
        # Handle sequence_first dimension ordering
        if labels.shape[0] > labels.shape[1]:
            # sequence_first=True: shape is (seq, batch)
            sample1_labels = labels[:, 1]
        else:
            # sequence_first=False: shape is (batch, seq)
            sample1_labels = labels[1, :]

        assert torch.all(sample1_labels == -100), f"All labels in padding sample should be -100, got {sample1_labels}"

    def test_image_placeholders_interleaved(self):
        """Test realistic scenario: text, image placeholders, text interleaved."""
        model, distributed = get_minimal_model()

        # Realistic sequence: [BOS, text, IMG, IMG, IMG, text, text, EOS]
        # Labels should be: [text, IMG(-100), IMG(-100), IMG(-100), text, text, EOS, ?]
        tokens = torch.tensor(
            [
                [1, 100, -100, -100, -100, 200, 201, 2],
            ],
            dtype=torch.int64,
        )

        batch = create_test_batch(tokens)
        preprocessed = run_preprocess_batch(model, distributed.config, batch)

        assert len(preprocessed) == 1
        _, kwargs = preprocessed[0]

        labels = kwargs[LanguageModelKwargs.labels]
        labels_flat = labels.flatten()

        # Labels at positions 1,2,3 should be -100 (next tokens are IMG)
        assert labels_flat[1].item() == -100, f"Position 1 should be -100, got {labels_flat[1].item()}"
        assert labels_flat[2].item() == -100, f"Position 2 should be -100, got {labels_flat[2].item()}"
        assert labels_flat[3].item() == -100, f"Position 3 should be -100, got {labels_flat[3].item()}"

        # Labels at positions 0, 4, 5 should be positive
        assert labels_flat[0].item() > 0, f"Position 0 should be positive, got {labels_flat[0].item()}"
        assert labels_flat[4].item() > 0, f"Position 4 should be positive, got {labels_flat[4].item()}"
        assert labels_flat[5].item() > 0, f"Position 5 should be positive, got {labels_flat[5].item()}"
