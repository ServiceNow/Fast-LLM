import pytest
import torch

from fast_llm.data.document.config import LanguageModelBatchPreprocessingConfig
from fast_llm.data.document.language_model import LanguageModelBatch, LanguageModelDocument
from fast_llm.data.document.range import RangeDocument
from fast_llm.data.document.token_data import TokenDataDocument
from fast_llm.engine.distributed.config import PhaseType
from fast_llm.utils import Assert


@pytest.mark.parametrize(
    ("tokens", "loss_masking_spans"),
    (
        ([[100, 101, 102, 103, 104, 105, 106, 107]], [None]),  # Simple case
        ([[100, 101, -100, -100, 104, 105, 106, 107]], [None]),  # Negative tokens
        ([[100, 101, 102, 103, 104, 105, 106, 107]], [[(3, 5)]]),  # Loss masking span
        ([[100, 101, 102, 103, -100, -100, 106, 107]], [[(2, 3)]]),  # Both
        (
            [
                [100, 101, -100, 103, -100, -100, 106, 107],
                [100, 101, 102, 103, 104, 105, 106, 107],
            ],
            [[(2, 3)], None],
        ),  # Two samples
    ),
)
def test_preprocessing(tokens, loss_masking_spans):
    documents = [
        LanguageModelDocument(
            tokens=torch.tensor(tokens_, dtype=torch.int64),
            loss_masking_spans=None if loss_masking_spans_ is None else RangeDocument(ranges=loss_masking_spans_),
        )
        for tokens_, loss_masking_spans_ in zip(tokens, loss_masking_spans, strict=True)
    ]

    (model_input,) = LanguageModelBatch.from_documents(documents).get_model_inputs(
        LanguageModelBatchPreprocessingConfig()
    )

    Assert.all_equal(model_input.tokens, torch.cat([document.tokens for document in documents])[:-1])

    label_tokens = []
    for document in documents:
        label_tokens_ = document.tokens.clone()
        # Mask cross-document attention
        label_tokens_[0] = -100
        # Loss masking spans
        if document.loss_masking_spans is not None:
            for begin, end in document.loss_masking_spans.ranges:
                label_tokens_[begin:end] = -100
        label_tokens.append(label_tokens_)

    Assert.eq(len(model_input.targets), 1)
    Assert.all_equal(model_input.targets[0].tokens, torch.cat(label_tokens)[1:])


def test_preprocessing_padding():
    # 5 real tokens padded to 8: padding tokens (-100) should be masked out of labels.
    tokens = [100, 101, 102, 103, 104]
    document = LanguageModelDocument(tokens=torch.tensor(tokens, dtype=torch.int64))

    batch = LanguageModelBatch.from_documents([document], pad_to_size=8)
    (model_input,) = batch.get_model_inputs(LanguageModelBatchPreprocessingConfig())

    # total_input_length = 8 - 1 = 7; tokens[0:7] = [100, 101, 102, 103, 104, -100, -100]
    Assert.all_equal(
        model_input.tokens,
        torch.tensor([100, 101, 102, 103, 104, -100, -100], dtype=torch.int64),
    )

    # labels = [100, 101, 102, 103, 104, -100, -100, -100], then labels[0]=-100 (cross-doc)
    # target = labels[1:8] = [101, 102, 103, 104, -100, -100, -100]
    Assert.all_equal(
        model_input.targets[0].tokens,
        torch.tensor([101, 102, 103, 104, -100, -100, -100], dtype=torch.int64),
    )


@pytest.mark.parametrize("predicted_tokens", [1, 2, 3])
def test_preprocessing_multi_token_prediction(predicted_tokens):
    # With predicted_tokens=d, there are d target sets.
    # Target for distance d is tokens[d : d + total_input_length].
    # Cross-doc masking for distance d falls at index d-1, just outside each target window.
    tokens = list(range(100, 111))  # 11 tokens
    document = LanguageModelDocument(tokens=torch.tensor(tokens, dtype=torch.int64))

    config = LanguageModelBatchPreprocessingConfig(predicted_tokens=predicted_tokens)
    (model_input,) = LanguageModelBatch.from_documents([document]).get_model_inputs(config)

    total_input = len(tokens) - predicted_tokens
    Assert.all_equal(model_input.tokens, torch.tensor(tokens[:total_input], dtype=torch.int64))
    Assert.eq(len(model_input.targets), predicted_tokens)

    for i, target in enumerate(model_input.targets):
        d = i + 1
        # Cross-doc masking for all distances <=d falls at indices 0..d-1, outside window [d:d+total_input].
        Assert.all_equal(target.tokens, torch.tensor(tokens[d : d + total_input], dtype=torch.int64))


def test_preprocessing_micro_batch_splits():
    # micro_batch_splits=2 produces two model inputs each covering half the sequence.
    tokens = list(range(100, 113))  # 13 tokens → total_input_length=12, each split=6
    document = LanguageModelDocument(tokens=torch.tensor(tokens, dtype=torch.int64))

    config = LanguageModelBatchPreprocessingConfig(micro_batch_splits=2)
    model_inputs = LanguageModelBatch.from_documents([document]).get_model_inputs(config)

    Assert.eq(len(model_inputs), 2)
    Assert.all_equal(model_inputs[0].tokens, torch.tensor(tokens[:6], dtype=torch.int64))
    Assert.all_equal(model_inputs[1].tokens, torch.tensor(tokens[6:12], dtype=torch.int64))

    # labels[0]=-100 (cross-doc); targets are labels[1:7] and labels[7:13]
    Assert.all_equal(model_inputs[0].targets[0].tokens, torch.tensor(tokens[1:7], dtype=torch.int64))
    Assert.all_equal(model_inputs[1].targets[0].tokens, torch.tensor(tokens[7:13], dtype=torch.int64))


def test_preprocessing_prediction_mask():
    # return_prediction_mask exposes the boolean mask of non-masked label positions.
    tokens = [100, 101, 102, 103, 104, 105]
    document = LanguageModelDocument(
        tokens=torch.tensor(tokens, dtype=torch.int64),
        loss_masking_spans=RangeDocument(ranges=[(2, 4)]),  # mask positions 2 and 3
    )

    config = LanguageModelBatchPreprocessingConfig(return_prediction_mask=True)
    (model_input,) = LanguageModelBatch.from_documents([document]).get_model_inputs(config)

    # labels = [100, 101, 102, 103, 104, 105]
    # after span masking: labels[2:4] = -100 → [100, 101, -100, -100, 104, 105]
    # after cross-doc:    labels[0]   = -100 → [-100, 101, -100, -100, 104, 105]
    # target = labels[1:6] = [101, -100, -100, 104, 105]
    # mask[1:6]           = [True, False, False, True, True]
    assert model_input.targets[0].mask is not None
    Assert.all_equal(
        model_input.targets[0].mask,
        torch.tensor([True, False, False, True, True]),
    )


def test_preprocessing_label_counts():
    # return_label_counts gives each token the total count of valid labels in its document.
    # Two documents each of length 4; cross-doc masking removes the first token of each,
    # leaving 3 valid labels per document.
    docs = [
        LanguageModelDocument(tokens=torch.tensor([100, 101, 102, 103], dtype=torch.int64)),
        LanguageModelDocument(tokens=torch.tensor([200, 201, 202, 203], dtype=torch.int64)),
    ]

    config = LanguageModelBatchPreprocessingConfig(return_label_counts=True)
    (model_input,) = LanguageModelBatch.from_documents(docs).get_model_inputs(config)

    # labels after cross-doc masking: [-100, 101, 102, 103, -100, 201, 202, 203]
    # doc1: 3 valid labels (indices 1,2,3); doc2: 3 valid labels (indices 5,6,7)
    # target window: labels[1:8] → label_counts[1:8] = [3, 3, 3, 3, 3, 3, 3]
    assert model_input.targets[0].label_counts is not None
    Assert.all_equal(
        model_input.targets[0].label_counts,
        torch.full((7,), 3, dtype=model_input.targets[0].label_counts.dtype),
    )


def test_preprocessing_grpo_data():
    # use_grpo_data attaches per-token advantages and log-probabilities to the target,
    # cropped to the label window [label_begin:label_end].
    tokens = [100, 101, 102, 103, 104, 105]
    advantages_data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    log_probs_data = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6]

    document = LanguageModelDocument(
        tokens=torch.tensor(tokens, dtype=torch.int64),
        advantages=TokenDataDocument(data=torch.tensor(advantages_data)),
        old_log_probabilities=TokenDataDocument(data=torch.tensor(log_probs_data)),
    )

    config = LanguageModelBatchPreprocessingConfig(use_grpo_data=True)
    (model_input,) = LanguageModelBatch.from_documents([document]).get_model_inputs(config)

    # total_input_length=5; label_begin=1, label_end=6
    target = model_input.targets[0]
    assert target.advantages is not None
    assert target.old_log_probabilities is not None
    Assert.rms_close(target.advantages, torch.tensor(advantages_data[1:]), 1e-6)
    Assert.rms_close(target.old_log_probabilities, torch.tensor(log_probs_data[1:]), 1e-6)


def test_preprocessing_position_index():
    # return_position_index gives the within-document position of each input token,
    # resetting to 0 at every document boundary.
    docs = [
        LanguageModelDocument(tokens=torch.tensor([100, 101, 102, 103], dtype=torch.int64)),  # len=4
        LanguageModelDocument(tokens=torch.tensor([200, 201, 202, 203], dtype=torch.int64)),  # len=4
    ]

    config = LanguageModelBatchPreprocessingConfig(return_position_index=True)
    (model_input,) = LanguageModelBatch.from_documents(docs).get_model_inputs(config)

    # total_input_length=7; input tokens: [100,101,102,103,200,201,202]
    # positions: doc1 → [0,1,2,3], doc2 (first 3 tokens) → [0,1,2]
    assert model_input.position_index is not None
    Assert.all_equal(
        model_input.position_index,
        torch.tensor([0, 1, 2, 3, 0, 1, 2], dtype=torch.int32),
    )


def test_preprocessing_inference():
    # In inference phase num_labels=0, so the full token sequence is the input and there are no targets.
    tokens = [100, 101, 102, 103, 104]
    document = LanguageModelDocument(tokens=torch.tensor(tokens, dtype=torch.int64))

    config = LanguageModelBatchPreprocessingConfig(phase=PhaseType.inference)
    (model_input,) = LanguageModelBatch.from_documents([document]).get_model_inputs(config)

    Assert.all_equal(model_input.tokens, torch.tensor(tokens, dtype=torch.int64))
    Assert.eq(len(model_input.targets), 0)


def test_preprocessing_document_count():
    # return_document_count records how many documents are in the batch (first split only).
    docs = [
        LanguageModelDocument(tokens=torch.tensor([100, 101, 102], dtype=torch.int64)),
        LanguageModelDocument(tokens=torch.tensor([200, 201, 202], dtype=torch.int64)),
    ]

    config = LanguageModelBatchPreprocessingConfig(return_document_count=True)
    (model_input,) = LanguageModelBatch.from_documents(docs).get_model_inputs(config)

    Assert.eq(model_input.num_documents, 2)


def test_preprocessing_cumulative_sequence_lengths():
    # return_cumulative_sequence_lengths produces cu_seqlens tensors for flash-attention style kernels.
    docs = [
        LanguageModelDocument(tokens=torch.tensor([100, 101, 102, 103], dtype=torch.int64)),  # len=4
        LanguageModelDocument(tokens=torch.tensor([200, 201, 202, 203], dtype=torch.int64)),  # len=4
    ]

    config = LanguageModelBatchPreprocessingConfig(return_cumulative_sequence_lengths=True)
    (model_input,) = LanguageModelBatch.from_documents(docs).get_model_inputs(config)

    # total_input_length=7; lengths in this input: [4, 3] (doc2 is cut to 3 by the -1 label offset)
    # cumulative_lengths_q = padded_cumsum([4, 3]) = [0, 4, 7]
    # cumulative_lengths_k = [0, 4, 7] (sequence_k_past=0, first_document_begin=0)
    assert model_input.cumulative_lengths_q is not None
    assert model_input.cumulative_lengths_k is not None
    Assert.all_equal(model_input.cumulative_lengths_q, torch.tensor([0, 4, 7], dtype=torch.int32))
    Assert.all_equal(model_input.cumulative_lengths_k, torch.tensor([0, 4, 7], dtype=torch.int32))
