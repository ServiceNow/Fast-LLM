import pytest
import torch

from fast_llm.data.document.config import LanguageModelBatchPreprocessingConfig
from fast_llm.data.document.language_model import LanguageModelBatch, LanguageModelDocument
from fast_llm.data.document.range import RangeDocument
from fast_llm.utils import Assert


# TODO: Test padding, more scenarios
# TODO: Check rest of preprocessing output
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
