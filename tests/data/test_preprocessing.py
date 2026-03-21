import pytest
import torch

from fast_llm.data.document.config import LanguageModelBatchPreprocessingConfig
from fast_llm.data.document.language_model import LanguageModelBatch, LanguageModelDocument
from fast_llm.data.document.range import RangeDocument
from fast_llm.data.document.token_data import TokenDataDocument
from fast_llm.engine.distributed.config import DistributedConfig
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


def _make_grpo_document(tokens, loss_masking_spans=None):
    """Helper: create a LanguageModelDocument with GRPO fields (advantages, old_log_probabilities)."""
    t = torch.tensor(tokens, dtype=torch.int64)
    n = len(t)
    return LanguageModelDocument(
        tokens=t,
        loss_masking_spans=None if loss_masking_spans is None else RangeDocument(ranges=loss_masking_spans),
        advantages=TokenDataDocument(data=torch.zeros(n)),
        old_log_probabilities=TokenDataDocument(data=torch.zeros(n)),
    )


@pytest.mark.parametrize(
    ("token_lists", "loss_masking_spans_list", "expected_num_docs"),
    (
        # Single doc, no masking — all tokens are response tokens except first (cross-doc mask)
        ([[1, 2, 3, 4, 5]], [None], 1),
        # Single doc fully masked by loss_masking_spans — no response tokens, num_docs = 0
        ([[1, 2, 3, 4, 5]], [[(0, 5)]], 0),
        # Two docs, both with response tokens
        ([[1, 2, 3], [4, 5, 6]], [None, None], 2),
        # Two docs, one fully masked — only 1 contributes
        ([[1, 2, 3], [4, 5, 6]], [[(0, 3)], None], 1),
        # Two docs, both fully masked
        ([[1, 2, 3], [4, 5, 6]], [[(0, 3)], [(0, 3)]], 0),
        # Padding: a short doc packed into a larger micro_batch_size leaves a padding segment
        ([[1, 2, 3]], [None], 1),  # with pad_to_size below
    ),
)
def test_num_docs_computation(token_lists, loss_masking_spans_list, expected_num_docs):
    """num_docs counts only documents that have at least one non-masked response token."""
    documents = [_make_grpo_document(tokens, spans) for tokens, spans in zip(token_lists, loss_masking_spans_list)]
    config = LanguageModelBatchPreprocessingConfig(use_grpo_data=True, return_label_counts=True)
    (model_input,) = LanguageModelBatch.from_documents(documents).get_model_inputs(config)
    Assert.eq(model_input.num_docs, expected_num_docs)


def test_num_docs_excludes_padding():
    """Padding appended by pad_to_size is a 0-label segment and must not count toward num_docs."""
    documents = [_make_grpo_document([1, 2, 3, 4])]
    config = LanguageModelBatchPreprocessingConfig(use_grpo_data=True, return_label_counts=True)
    # pad_to_size > total tokens forces a padding segment to be added
    (model_input,) = LanguageModelBatch.from_documents(documents, pad_to_size=10).get_model_inputs(config)
    # Only the real document counts; the padding segment (all -100) does not
    Assert.eq(model_input.num_docs, 1)


def test_num_docs_none_without_label_counts():
    """num_docs is None when return_label_counts is False (GRPO preprocessing not requested)."""
    documents = [_make_grpo_document([1, 2, 3, 4])]
    config = LanguageModelBatchPreprocessingConfig()
    (model_input,) = LanguageModelBatch.from_documents(documents).get_model_inputs(config)
    assert model_input.num_docs is None


def _make_sdp_config(sdp_rank: int, sdp_size: int = 2) -> LanguageModelBatchPreprocessingConfig:
    """Config simulating a given sequence-data-parallel rank."""
    return LanguageModelBatchPreprocessingConfig(
        use_grpo_data=True,
        return_label_counts=True,
        distributed=DistributedConfig(world_size=sdp_size, rank=sdp_rank, sequence_data_parallel=sdp_size),
    )


def test_num_docs_sdp_only_counted_on_rank0():
    """With SDP=2, num_docs is counted only on sequence_data_rank=0.

    The runner all_reduces the denominator across the data group (which includes all SDP
    ranks).  If both SDP ranks reported num_docs=1 for the same document, the all_reduce
    SUM would produce denominator=2 and halve the metric.  Only rank 0 must contribute
    to avoid this double-counting.
    """
    # 9 tokens → total_input_length = 8 (divisible by SDP=2)
    documents = [_make_grpo_document([1, 2, 3, 4, 5, 6, 7, 8, 9])]
    batch = LanguageModelBatch.from_documents(documents)

    (model_input_rank0,) = batch.get_model_inputs(_make_sdp_config(sdp_rank=0))
    (model_input_rank1,) = batch.get_model_inputs(_make_sdp_config(sdp_rank=1))

    # Rank 0 counts the document; rank 1 must not.
    Assert.eq(model_input_rank0.num_docs, 1)
    Assert.eq(model_input_rank1.num_docs, 0)

    # After all_reduce SUM across SDP ranks the denominator equals the true doc count.
    Assert.eq(model_input_rank0.num_docs + model_input_rank1.num_docs, 1)


def test_num_docs_sdp_fully_masked_excluded_on_rank0():
    """A fully-masked document is excluded from num_docs even on SDP rank 0."""
    # Doc 0 fully masked; doc 1 has response tokens. 9 tokens total → 8 input tokens (div by 2).
    documents = [
        _make_grpo_document([1, 2, 3, 4], loss_masking_spans=[(0, 4)]),
        _make_grpo_document([5, 6, 7, 8, 9]),
    ]
    batch = LanguageModelBatch.from_documents(documents)

    (model_input_rank0,) = batch.get_model_inputs(_make_sdp_config(sdp_rank=0))
    (model_input_rank1,) = batch.get_model_inputs(_make_sdp_config(sdp_rank=1))

    # Only the unmasked document counts; rank 1 always contributes 0.
    Assert.eq(model_input_rank0.num_docs, 1)
    Assert.eq(model_input_rank1.num_docs, 0)
    Assert.eq(model_input_rank0.num_docs + model_input_rank1.num_docs, 1)


def test_num_docs_sdp_two_docs_counted_once():
    """Two documents on SDP=2 are counted once in total (not once per SDP rank)."""
    # 9 tokens total → 8 input tokens, divisible by SDP=2.
    documents = [_make_grpo_document([1, 2, 3, 4]), _make_grpo_document([5, 6, 7, 8, 9])]
    batch = LanguageModelBatch.from_documents(documents)

    (model_input_rank0,) = batch.get_model_inputs(_make_sdp_config(sdp_rank=0))
    (model_input_rank1,) = batch.get_model_inputs(_make_sdp_config(sdp_rank=1))

    Assert.eq(model_input_rank0.num_docs, 2)
    Assert.eq(model_input_rank1.num_docs, 0)
    Assert.eq(model_input_rank0.num_docs + model_input_rank1.num_docs, 2)
