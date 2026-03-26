import dataclasses
import functools

import pytest
import torch

from fast_llm.data.document.config import LanguageModelBatchPreprocessingConfig
from fast_llm.data.document.language_model import LanguageModelBatch, LanguageModelDocument
from fast_llm.data.document.range import RangeDocument
from fast_llm.data.document.token_data import TokenDataDocument
from fast_llm.engine.distributed.config import PhaseType
from fast_llm.utils import Assert, div


def _get_cropped_lengths(batch_lengths: list[int], begin: int, end: int) -> tuple[list[int], int]:
    """Return (cropped_lengths, first_document_begin) for the token window [begin, end)."""
    doc_begin = 0
    cropped = []
    first_doc_begin = 0
    for length in batch_lengths:
        doc_end = doc_begin + length
        crop = min(doc_end, end) - max(doc_begin, begin)
        if crop > 0:
            if not cropped:
                first_doc_begin = doc_begin
            cropped.append(crop)
        if doc_end > end:
            break
        doc_begin = doc_end
    return cropped, first_doc_begin


def _compute_label_counts(batch_lengths: list[int], labels: list[int]) -> torch.Tensor:
    """For each token, compute the count of valid (non-negative) labels in its document."""
    result = []
    offset = 0
    for length in batch_lengths:
        count = sum(1 for label in labels[offset : offset + length] if label >= 0)
        result.extend([count] * length)
        offset += length
    return torch.tensor(result, dtype=torch.int64)


def _assert_tensor_equal_or_none(actual: torch.Tensor | None, expected: torch.Tensor | None) -> None:
    if expected is None:
        assert actual is None
    else:
        Assert.all_equal(actual, expected)


@dataclasses.dataclass
class PreprocessingTestConfig:
    name: str
    tokens: list[list[int]]
    loss_masking_spans: list[list[tuple[int, int]] | None] | None = None
    padding: int | None = None
    advantages: list[list[float]] | None = None
    log_probabilities: list[list[float]] | None = None
    phase: PhaseType = PhaseType.training
    predicted_tokens: int = 1
    micro_batch_splits: int = 1
    use_grpo_data: bool = False
    return_prediction_mask: bool = False
    return_label_counts: bool = False
    return_position_index: bool = False
    return_document_count: bool = False
    return_cumulative_sequence_lengths: bool = False

    @functools.cached_property
    def config_kwargs(self) -> dict:
        return {
            "phase": self.phase,
            "predicted_tokens": self.predicted_tokens,
            "micro_batch_splits": self.micro_batch_splits,
            "use_grpo_data": self.use_grpo_data,
            "return_prediction_mask": self.return_prediction_mask,
            "return_label_counts": self.return_label_counts,
            "return_position_index": self.return_position_index,
            "return_document_count": self.return_document_count,
            "return_cumulative_sequence_lengths": self.return_cumulative_sequence_lengths,
        }

    @functools.cached_property
    def padding_size(self) -> int:
        return 0 if self.padding is None else self.padding

    @functools.cached_property
    def unpadded_size(self) -> int:
        return sum(self.unpadded_lengths)

    @functools.cached_property
    def padded_size(self) -> int:
        return self.unpadded_size + self.padding_size

    @functools.cached_property
    def unpadded_lengths(self) -> list[int]:
        return [len(tokens) for tokens in self.tokens]

    @functools.cached_property
    def padded_lengths(self) -> list[int]:
        return self.unpadded_lengths + ([self.padding_size] if self.padding_size > 0 else [])

    @functools.cached_property
    def num_labels(self) -> int:
        return self.padded_size - self.predicted_tokens

    @functools.cached_property
    def split_size(self) -> int:
        return div(self.num_labels, self.micro_batch_splits)

    @functools.cached_property
    def all_flat_tokens(self) -> list[int]:
        return sum(self.tokens, []) + [-100] * self.padding_size

    @functools.cached_property
    def base_labels(self) -> list[int]:
        """Tokens with loss masking spans applied, but no cross-document masking."""
        labels = list(self.all_flat_tokens)
        if self.loss_masking_spans is not None:
            offset = 0
            for doc_tokens, spans in zip(self.tokens, self.loss_masking_spans):
                if spans is not None:
                    for begin, end in spans:
                        labels[offset + begin : offset + end] = [-100] * (end - begin)
                offset += len(doc_tokens)
        return labels

    @functools.cached_property
    def labels_per_distance(self) -> list[list[int]]:
        """For each prediction distance d, labels with cumulative cross-document masking."""
        result = []
        labels = list(self.base_labels)
        for d in range(1, self.predicted_tokens + 1):
            offset = 0
            for doc_tokens in self.tokens:
                if d <= len(doc_tokens):
                    labels[offset + d - 1] = -100
                offset += len(doc_tokens)
            result.append(list(labels))
        return result

    @functools.cached_property
    def _split_ranges(self) -> list[tuple[int, int]]:
        return [(i * self.split_size, (i + 1) * self.split_size) for i in range(self.micro_batch_splits)]

    @functools.cached_property
    def _cropped_lengths_per_split(self) -> list[tuple[list[int], int]]:
        return [_get_cropped_lengths(self.padded_lengths, begin, end) for begin, end in self._split_ranges]

    @functools.cached_property
    def expected_input_tokens(self) -> list[torch.Tensor]:
        all_tokens = torch.tensor(self.all_flat_tokens, dtype=torch.int64)
        return [all_tokens[begin:end] for begin, end in self._split_ranges]

    @functools.cached_property
    def expected_target_tokens(self) -> list[list[torch.Tensor]]:
        labels_tensors = [torch.tensor(labels, dtype=torch.int64) for labels in self.labels_per_distance]
        return [
            [
                labels_tensors[target_index][begin + d : end + d]
                for target_index, d in enumerate(range(1, self.predicted_tokens + 1))
            ]
            for begin, end in self._split_ranges
        ]

    @functools.cached_property
    def expected_target_mask(self) -> list[list[torch.Tensor | None]]:
        if not self.return_prediction_mask:
            return [[None] * self.predicted_tokens for _ in range(self.micro_batch_splits)]
        return [[tokens >= 0 for tokens in split_targets] for split_targets in self.expected_target_tokens]

    @functools.cached_property
    def expected_target_label_counts(self) -> list[list[torch.Tensor | None]]:
        if not self.return_label_counts:
            return [[None] * self.predicted_tokens for _ in range(self.micro_batch_splits)]
        return [
            [
                _compute_label_counts(self.padded_lengths, self.labels_per_distance[target_index])[begin + d : end + d]
                for target_index, d in enumerate(range(1, self.predicted_tokens + 1))
            ]
            for begin, end in self._split_ranges
        ]

    @functools.cached_property
    def expected_advantages(self) -> list[list[torch.Tensor | None]]:
        if self.advantages is None:
            return [[None] * self.predicted_tokens for _ in range(self.micro_batch_splits)]
        flat = torch.tensor(sum(self.advantages, []) + [0.0] * self.padding_size, dtype=torch.float32)
        return [
            [flat[begin + d : end + d] for d in range(1, self.predicted_tokens + 1)]
            for begin, end in self._split_ranges
        ]

    @functools.cached_property
    def expected_log_probabilities(self) -> list[list[torch.Tensor | None]]:
        if self.log_probabilities is None:
            return [[None] * self.predicted_tokens for _ in range(self.micro_batch_splits)]
        flat = torch.tensor(sum(self.log_probabilities, []) + [0.0] * self.padding_size, dtype=torch.float32)
        return [
            [flat[begin + d : end + d] for d in range(1, self.predicted_tokens + 1)]
            for begin, end in self._split_ranges
        ]

    @functools.cached_property
    def expected_position_index(self) -> list[torch.Tensor | None]:
        if not self.return_position_index:
            return [None] * self.micro_batch_splits
        result = []
        for split_index, (begin, _end) in enumerate(self._split_ranges):
            cropped_lengths, first_doc_begin = self._cropped_lengths_per_split[split_index]
            pos_in_doc = begin - first_doc_begin
            positions = []
            remaining = cropped_lengths[0] if cropped_lengths else 0
            doc_index = 0
            for _ in range(self.split_size):
                positions.append(pos_in_doc)
                pos_in_doc += 1
                remaining -= 1
                if remaining == 0 and doc_index + 1 < len(cropped_lengths):
                    doc_index += 1
                    remaining = cropped_lengths[doc_index]
                    pos_in_doc = 0
            result.append(torch.tensor(positions, dtype=torch.int32))
        return result

    @functools.cached_property
    def expected_cumulative_lengths(self) -> list[tuple[torch.Tensor | None, torch.Tensor | None]]:
        if not self.return_cumulative_sequence_lengths:
            return [(None, None)] * self.micro_batch_splits
        result = []
        for split_index, (begin, _end) in enumerate(self._split_ranges):
            cropped_lengths, first_doc_begin = self._cropped_lengths_per_split[split_index]
            cu_q = torch.tensor([0] + cropped_lengths, dtype=torch.int32).cumsum(dim=0)
            cu_k = (cu_q + begin).clone()
            cu_k[0] = first_doc_begin
            result.append((cu_q, cu_k))
        return result

    @functools.cached_property
    def expected_num_documents(self) -> list[int | None]:
        if self.return_document_count:
            return [len(self.tokens) if split_index == 0 else 0 for split_index in range(self.micro_batch_splits)]
        else:
            return [None] * self.micro_batch_splits


_BASE_TEST_CASES = [
    PreprocessingTestConfig(
        name="simple",
        tokens=[[100, 101, 102, 103, 104, 105, 106, 107, 108]],
    ),
    PreprocessingTestConfig(
        name="negative_tokens",
        tokens=[[100, 101, -100, -100, 104, 105, 106, 107, 108]],
    ),
    PreprocessingTestConfig(
        name="loss_masking_span",
        tokens=[[100, 101, 102, 103, 104, 105, 106, 107, 108]],
        loss_masking_spans=[[(3, 5)]],
    ),
    PreprocessingTestConfig(
        name="negative_tokens_and_loss_masking",
        tokens=[[100, 101, 102, 103, -100, -100, 106, 107, 108]],
        loss_masking_spans=[[(2, 3)]],
    ),
    PreprocessingTestConfig(
        name="two_documents",
        tokens=[[100, 101, -100, 103, -100, -100, 106, 107], [100, 101, 102, 103, 104, 105, 106, 107, 108]],
        loss_masking_spans=[[(2, 3)], None],
    ),
    PreprocessingTestConfig(
        name="three_documents",
        tokens=[[100, 101, 102], [103, 104, 105], [106, 107, 108]],
        loss_masking_spans=[[(1, 2)], None, [(0, 2)]],
    ),
    PreprocessingTestConfig(
        # Document of length 1 is shorter than predicted_tokens=3; cross-document masking must not go out of bounds.
        name="short_document",
        tokens=[[100], [101, 102, 103, 104, 105, 106, 107, 108]],
    ),
    PreprocessingTestConfig(
        name="multiple_loss_masking_spans",
        tokens=[[100, 101, 102, 103, 104, 105, 106, 107, 108]],
        loss_masking_spans=[[(1, 3), (5, 7)]],
    ),
    PreprocessingTestConfig(
        # use_grpo_data attaches per-token advantages and log-probabilities to the target.
        name="grpo_data",
        tokens=[[100, 101, 102, 103, 104, 105, 106]],
        advantages=[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]],
        log_probabilities=[[-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7]],
        use_grpo_data=True,
    ),
    PreprocessingTestConfig(
        name="two_documents_grpo_data",
        tokens=[[100, 101, 102, 103], [104, 105, 106, 107, 108]],
        advantages=[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8, 0.9]],
        log_probabilities=[[-0.1, -0.2, -0.3, -0.4], [-0.5, -0.6, -0.7, -0.8, -0.9]],
        use_grpo_data=True,
    ),
    PreprocessingTestConfig(
        # In inference phase num_labels=0, so the full token sequence is the input and there are no targets.
        name="inference",
        tokens=[[100, 101, 102, 103, 104]],
        phase=PhaseType.inference,
    ),
]

# Each base case is run with each return configuration and both values of micro_batch_splits,
# except inference which has no labels to return or split.
_RETURN_CONFIG_VARIANTS: dict[str, dict] = {
    "": {},
    "prediction_mask": {"return_prediction_mask": True},
    "label_counts": {"return_label_counts": True},
    "position_index": {"return_position_index": True},
    "document_count": {"return_document_count": True},
    "cumulative_sequence_lengths": {"return_cumulative_sequence_lengths": True},
    "all": {
        "return_prediction_mask": True,
        "return_label_counts": True,
        "return_position_index": True,
        "return_document_count": True,
        "return_cumulative_sequence_lengths": True,
    },
}


def _make_name(
    base_name: str, return_name: str, predicted_tokens: int, micro_batch_splits: int, padding: int | None
) -> str:
    parts = [base_name]
    if return_name:
        parts.append(f"return_{return_name}")
    if predicted_tokens > 1:
        parts.append(f"predicted_tokens_{predicted_tokens}")
    if micro_batch_splits > 1:
        parts.append(f"splits_{micro_batch_splits}")
    if padding is not None:
        parts.append(f"padding_{padding}")
    return "_".join(parts)


_PREPROCESSING_TEST_CASES = [
    dataclasses.replace(
        base_case,
        name=_make_name(base_case.name, return_name, predicted_tokens, micro_batch_splits, padding),
        predicted_tokens=predicted_tokens,
        micro_batch_splits=micro_batch_splits,
        padding=padding,
        **return_config,
    )
    for base_case in _BASE_TEST_CASES
    for return_name, return_config in _RETURN_CONFIG_VARIANTS.items()
    for predicted_tokens in (1, 3)
    for micro_batch_splits in (1, 2)
    for padding in (None, 0, 2)
    if base_case.phase != PhaseType.inference
    or (not return_config and predicted_tokens == 1 and micro_batch_splits == 1)
]


@pytest.mark.parametrize(
    "test_config", [pytest.param(test_config, id=test_config.name) for test_config in _PREPROCESSING_TEST_CASES]
)
def test_preprocessing(test_config: PreprocessingTestConfig):
    config = LanguageModelBatchPreprocessingConfig(**test_config.config_kwargs)

    documents = [
        LanguageModelDocument(
            tokens=torch.tensor(tokens, dtype=torch.int64),
            loss_masking_spans=None if spans is None else RangeDocument(ranges=spans),
            advantages=None if doc_advantages is None else TokenDataDocument(data=torch.tensor(doc_advantages)),
            old_log_probabilities=(
                None if doc_log_probs is None else TokenDataDocument(data=torch.tensor(doc_log_probs))
            ),
        )
        for tokens, spans, doc_advantages, doc_log_probs in zip(
            test_config.tokens,
            test_config.loss_masking_spans or [None] * len(test_config.tokens),
            test_config.advantages or [None] * len(test_config.tokens),
            test_config.log_probabilities or [None] * len(test_config.tokens),
            strict=True,
        )
    ]

    batch = LanguageModelBatch.from_documents(
        documents, pad_to_size=test_config.padded_size if test_config.padding is not None else None
    )
    model_inputs = batch.get_model_inputs(config)

    # Inference: full token sequence as input, no targets.
    if config.phase == PhaseType.inference:
        Assert.eq(len(model_inputs), 1)
        Assert.all_equal(model_inputs[0].tokens, batch.tokens)
        Assert.eq(len(model_inputs[0].targets), 0)
        return

    Assert.eq(len(model_inputs), test_config.micro_batch_splits)
    for split_index, model_input in enumerate(model_inputs):
        Assert.all_equal(model_input.tokens, test_config.expected_input_tokens[split_index])
        Assert.eq(len(model_input.targets), test_config.predicted_tokens)

        for target_index, target in enumerate(model_input.targets):
            Assert.all_equal(target.tokens, test_config.expected_target_tokens[split_index][target_index])
            _assert_tensor_equal_or_none(target.mask, test_config.expected_target_mask[split_index][target_index])
            _assert_tensor_equal_or_none(
                target.label_counts, test_config.expected_target_label_counts[split_index][target_index]
            )
            _assert_tensor_equal_or_none(target.advantages, test_config.expected_advantages[split_index][target_index])
            _assert_tensor_equal_or_none(
                target.old_log_probabilities, test_config.expected_log_probabilities[split_index][target_index]
            )

        _assert_tensor_equal_or_none(model_input.position_index, test_config.expected_position_index[split_index])
        cu_q, cu_k = test_config.expected_cumulative_lengths[split_index]
        _assert_tensor_equal_or_none(model_input.cumulative_lengths_q, cu_q)
        _assert_tensor_equal_or_none(model_input.cumulative_lengths_k, cu_k)
        Assert.eq(model_input.num_documents, test_config.expected_num_documents[split_index])
