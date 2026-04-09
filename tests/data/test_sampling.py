import dataclasses
import functools
import pathlib

import numpy as np
import pytest
import torch

from fast_llm.data.dataset.config import ShufflingType
from fast_llm.data.dataset.gpt.config import GPTDatasetFromFileConfig, GPTSamplingConfig
from fast_llm.data.dataset.indexed import IndexedDataset
from fast_llm.data.document.language_model import LanguageModelBatch, LanguageModelDocument
from fast_llm.utils import Assert
from tests.data.common import (
    get_dataset_config,
    get_sampling_config,
    get_test_data_and_compare_samples,
    validate_indexed_dataset_sampling,
)
from tests.utils.dataset import get_common_test_dataset

try:
    from fast_llm.csrc.data import build_padded_token_cumsum  # noqa

    _extension_available = True
except ImportError:
    _extension_available = False


GPT_MEMMAP_SAMPLES = [
    [49152, 46, 10, 819, 19, 45],
    [45, 69, 17, 86, 38826, 15],
    [15, 25, 51, 31, 32348, 64],
    [64, 17, 93, 78, 40, 1793],
    [1793, 1, 1746, 38, 27, 58],
    [58, 22885, 93, 37, 92, 76],
    [76, 29, 19, 17365, 93, 46],
    [46, 83, 17211, 1, 785, 1023],
]


class SimpleGPTIndexedDataset[DocumentType: LanguageModelDocument](IndexedDataset[DocumentType]):
    # TODO: worth adding to the main codebase?
    def __init__(self, samples):
        self._samples = samples

    def get_document(self, index: int, begin: int = 0, end: int | None = None) -> DocumentType:
        if end is None:
            end = len(self._samples[index])
        return LanguageModelDocument(tokens=torch.tensor(self._samples[index][begin:end], dtype=torch.int64))

    def __len__(self) -> int:
        return len(self._samples)

    def get_document_sizes(self) -> torch.Tensor:
        return torch.tensor([self.get_document_size(index) for index in range(len(self))], dtype=torch.int64)

    def get_document_size(self, index: int) -> int:
        return len(self._samples[index])

    @property
    def name(self) -> str:
        return "dataset"


TEST_DATASET = SimpleGPTIndexedDataset(
    [
        [0, 1, 2, 3],
        [4],
        [5, 6, 7],
        [8, 9, 10, 11, 12],
        [13, 14, 15, 16, 17, 18],
        [19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
    ]
)

# Document sizes: 3, 5, 2, 4, 6.
# With maximum_document_length=4, truncate_documents=False: docs of size 5 and 6 are dropped.
# With maximum_document_length=4, truncate_documents=True: docs of size 5 and 6 are split into chunks of ≤4.
TRUNCATE_DATASET = SimpleGPTIndexedDataset(
    [
        [0, 1, 2],  # length 3 — fits
        [3, 4, 5, 6, 7],  # length 5 — exceeds maximum_document_length=4
        [8, 9],  # length 2 — fits
        [10, 11, 12, 13],  # length 4 — exactly at limit
        [14, 15, 16, 17, 18, 19],  # length 6 — exceeds
    ]
)


@dataclasses.dataclass
class SamplingTestConfig:
    name: str
    num_samples: int
    sequence_length: int = 5
    seed: int = 54983
    shuffle: ShufflingType = ShufflingType.epoch
    truncate_documents: bool = True
    maximum_document_length: int | None = None
    expected_samples: list[list[int]] | None = None
    # Tokens that must not appear in any sample (validated for drop/filter cases).
    # Defaults to empty — the check is always run but trivially passes.
    forbidden_tokens: frozenset[int] = frozenset()
    # Tokens that must collectively appear across all samples (validated for truncate cases).
    # Defaults to empty — the check is always run but trivially passes.
    required_tokens: frozenset[int] = frozenset()
    requires_extension: bool = False
    dataset: SimpleGPTIndexedDataset | None = dataclasses.field(default=None, compare=False, repr=False)

    @functools.cached_property
    def sampling_config_overrides(self) -> dict:
        if self.maximum_document_length is not None:
            return {"maximum_document_length": self.maximum_document_length}
        return {}


_SAMPLING_TEST_CASES = [
    SamplingTestConfig(
        name="simple",
        num_samples=20,
    ),
    SamplingTestConfig(
        # With truncate_documents=False, documents exceeding maximum_document_length are dropped entirely.
        # Only the 3 docs with length ≤ 4 contribute tokens: [0,1,2], [8,9], [10,11,12,13] = 9 tokens.
        name="maximum_document_length_drop",
        num_samples=2,
        sequence_length=4,
        shuffle=ShufflingType.disabled,
        truncate_documents=False,
        maximum_document_length=4,
        forbidden_tokens=frozenset(range(3, 8)) | frozenset(range(14, 20)),
        dataset=TRUNCATE_DATASET,
        requires_extension=True,
    ),
    SamplingTestConfig(
        # With truncate_documents=True, documents exceeding maximum_document_length are split into chunks.
        # All tokens should appear in the output; none should be dropped.
        name="maximum_document_length_truncate",
        num_samples=10,
        sequence_length=4,
        shuffle=ShufflingType.disabled,
        truncate_documents=True,
        maximum_document_length=4,
        required_tokens=frozenset(range(20)),
        dataset=TRUNCATE_DATASET,
    ),
]


@pytest.mark.parametrize("test_config", [pytest.param(c, id=c.name) for c in _SAMPLING_TEST_CASES])
def test_sampling(test_config: SamplingTestConfig):
    if test_config.requires_extension and not _extension_available:
        pytest.skip("CPP Extension not available")

    dataset = test_config.dataset if test_config.dataset is not None else TEST_DATASET
    base_config, num_samples, seed = get_sampling_config(
        test_config.num_samples,
        sequence_length=test_config.sequence_length,
        seed=test_config.seed,
        shuffle=test_config.shuffle,
        truncate_documents=test_config.truncate_documents,
    )
    sampling_config = GPTSamplingConfig.from_dict(base_config.to_dict(), test_config.sampling_config_overrides)
    sampled = dataset.sample(sampling_config, num_samples, seed)

    # validate_indexed_dataset_sampling's reference implementation concatenates tokens without padding,
    # so it only applies when truncate_documents=True (no padding between documents).
    if test_config.truncate_documents:
        tokens = validate_indexed_dataset_sampling(sampled, test_config.expected_samples)
    else:
        tokens = torch.stack(
            [
                LanguageModelBatch.from_documents(sampled[i], test_config.sequence_length + 1).tokens
                for i in range(len(sampled))
            ]
        )

    valid_tokens = set(tokens[tokens >= 0].tolist())
    assert test_config.forbidden_tokens.isdisjoint(valid_tokens)
    assert test_config.required_tokens.issubset(valid_tokens)


def test_gpt_sampled(data_result_path: pathlib.Path):
    # Make sure the memmap dataset works and check for unintended changes in behavior.
    _, config, _, preprocessing = get_common_test_dataset()
    sampled = get_dataset_config(
        dataset_config := config, GPTDatasetFromFileConfig[LanguageModelDocument]
    ).build_and_sample(*get_sampling_config(8, sequence_length=5, preprocessing=preprocessing))
    validate_indexed_dataset_sampling(sampled, GPT_MEMMAP_SAMPLES)

    # Test in data.
    get_test_data_and_compare_samples(
        {"datasets": {"training": dataset_config}},
        8,
        sequence_length=5,
        expected_samples=GPT_MEMMAP_SAMPLES,
        preprocessing=preprocessing,
        cache_directory=data_result_path / "sampling/gpt_sampled",
    )


@pytest.mark.parametrize("seed", (0, 32, 88))
@pytest.mark.parametrize(
    "shuffle", (ShufflingType.full, ShufflingType.epoch, ShufflingType.skip_first_epoch, ShufflingType.disabled)
)
def test_gpt_sample(seed, shuffle):
    previous_samples = None
    # Loop instead of parametrizing for the check below.
    for num_samples in (20, 10, 6, 5, 2, 1):
        sampled = TEST_DATASET.sample(
            *get_sampling_config(
                num_samples,
                sequence_length=5,
                seed=seed,
                shuffle=shuffle,
            )
        )
        samples = validate_indexed_dataset_sampling(sampled)
        if previous_samples is not None and shuffle != ShufflingType.full:
            # Check that the sequence is independent of `num_sample`.
            Assert.all_equal(samples, previous_samples[: len(samples)])
        previous_samples = samples


@pytest.mark.parametrize("token_cumsum_rate", (1, 3, 7, 20))
def test_token_cumsum_rate(token_cumsum_rate):
    # Different token_cumsum_rate values are a performance/memory tradeoff only —
    # sampling output must be identical regardless of the rate chosen.
    config, num_samples, seed = get_sampling_config(20, sequence_length=5)
    reference = validate_indexed_dataset_sampling(TEST_DATASET.sample(config, num_samples, seed))

    config_with_rate = GPTSamplingConfig.from_dict(config.to_dict(), {"token_cumsum_rate": token_cumsum_rate})
    result = validate_indexed_dataset_sampling(TEST_DATASET.sample(config_with_rate, num_samples, seed))
    Assert.all_equal(result, reference)


def test_cache_directory(data_result_path: pathlib.Path):
    # Verify that the cache is written on first run and reused on subsequent runs.
    cache_dir = data_result_path / "sampling/cache_directory"
    config, num_samples, seed = get_sampling_config(20, sequence_length=5, cache_directory=cache_dir)

    first = validate_indexed_dataset_sampling(TEST_DATASET.sample(config, num_samples, seed))
    assert cache_dir.exists() and any(cache_dir.iterdir())

    # Second run with the same config must produce identical output (reads from cache).
    second = validate_indexed_dataset_sampling(TEST_DATASET.sample(config, num_samples, seed))
    Assert.all_equal(first, second)


def test_cache_invalidated_on_config_change(data_result_path: pathlib.Path):
    # Changing a sampling parameter should raise rather than silently return stale data.
    cache_dir = data_result_path / "sampling/cache_invalidation"
    config, num_samples, seed = get_sampling_config(20, sequence_length=5, cache_directory=cache_dir)
    TEST_DATASET.sample(config, num_samples, seed)

    config_changed = GPTSamplingConfig.from_dict(config.to_dict(), {"token_cumsum_rate": 3})
    with pytest.raises(RuntimeError, match="Invalid dataset cache"):
        TEST_DATASET.sample(config_changed, num_samples, seed)


@pytest.mark.skipif(not _extension_available, reason="CPP Extension not available")
def test_build_padded_token_cumsum():
    sizes = np.array([100, 256, 580, 600, 550, 89, 339, 430, 400, 795, 680, 50], dtype=np.int32)
    sequence_length = 768
    token_cumsum_rate = 4
    offset = 0
    # sequences with padding:
    # [100, 256, 413 padded, 580, 189 padded, 600, 169 padded, 550, 89, 130 padded, 339, 430, 400, 369 padded, 680, 50, 39 padded]
    # cumsums:
    # [100, 356, 1349, 2307, 2857, 2946, 3415, 3845, 4245, 5294, 5344, 5383]
    expected_cumsums = [0, 2307, 3845, 5383]
    token_cumsum = build_padded_token_cumsum(sizes, sequence_length + 1, token_cumsum_rate, offset)
    Assert.all_equal(token_cumsum, expected_cumsums)


@pytest.mark.skipif(not _extension_available, reason="CPP Extension not available")
def test_gpt_sample_padding():
    for _ in range(10):
        vocab_size = 30
        num_sequences = np.random.randint(1, 20)
        sequence_length = np.random.randint(1, 20)
        doc_sizes = np.random.randint(1, 2 * sequence_length, num_sequences)
        samples = [np.random.randint(0, vocab_size, size) for size in doc_sizes]
        expected_samples = []
        seq_size = 0
        token_ids = []
        total_tokens = 0
        for idx, sample in enumerate(samples):
            doc_size = len(sample)
            if doc_size > sequence_length + 1:
                continue
            elif doc_size + seq_size > sequence_length + 1:
                padding_tokens = sequence_length + 1 - seq_size
                token_ids.append([-100] * padding_tokens)
                expected_samples.append(list(np.concatenate(token_ids)))
                token_ids = [sample]
                seq_size = doc_size
                total_tokens += doc_size
            else:
                token_ids.append(sample)
                seq_size += doc_size
                total_tokens += doc_size
        dataset = SimpleGPTIndexedDataset(samples)
        sampling = get_sampling_config(
            num_samples=len(expected_samples),
            sequence_length=sequence_length,
            seed=np.random.randint(100000),
            shuffle=ShufflingType.disabled,
            truncate_documents=False,
        )
        if total_tokens == 0:
            with pytest.raises(RuntimeError):
                dataset.sample(*sampling)
        else:
            sampled = dataset.sample(*sampling)
            for idx in range(len(expected_samples)):
                Assert.all_equal(
                    LanguageModelBatch.from_documents(sampled[idx], sequence_length + 1).tokens,
                    np.array(expected_samples[idx]),
                )
