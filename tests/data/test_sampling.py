import numpy as np
import pytest
import torch

from fast_llm.data.dataset.config import SamplingParameters, ShufflingType
from fast_llm.data.dataset.gpt.config import GPTDatasetFromFileConfig
from fast_llm.data.dataset.indexed import IndexedDataset
from fast_llm.data.preprocessing.language_model import LanguageModelPreprocessingConfig
from fast_llm.data.sample.language_model import LanguageModelSample
from fast_llm.data.sample.token import TokenSample
from fast_llm.utils import Assert
from tests.data.common import (
    get_dataset_config,
    get_sampling_data,
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


def test_gpt_sampled():
    # Make sure the memmap dataset works and check for unintended changes in behavior.
    _, config, _, preprocessing = get_common_test_dataset()
    sampled = get_dataset_config(
        dataset_config := config, GPTDatasetFromFileConfig[LanguageModelSample]
    ).build_and_sample(get_sampling_data(8, sequence_length=5, preprocessing=preprocessing))
    validate_indexed_dataset_sampling(sampled, GPT_MEMMAP_SAMPLES)

    # Test in data.
    get_test_data_and_compare_samples(
        {"datasets": {"training": dataset_config}},
        8,
        sequence_length=5,
        expected_samples=GPT_MEMMAP_SAMPLES,
        preprocessing=preprocessing,
    )


class SimpleGPTIndexedDataset[SampleType: LanguageModelSample](IndexedDataset[SampleType]):
    # TODO: worth adding to the main codebase?
    def __init__(self, samples):
        self._samples = samples

    def get_document(
        self, index: int, begin: int = 0, end: int | None = None, parameters: SamplingParameters | None = None
    ) -> SampleType:
        if end is None:
            end = len(self._samples[index])
        return LanguageModelSample(TokenSample(torch.tensor(self._samples[index][begin:end], dtype=torch.int64)))

    def __len__(self) -> int:
        return len(self._samples)

    def get_document_sizes(self) -> torch.Tensor:
        return torch.tensor([self.get_document_size(index) for index in range(len(self))], dtype=torch.int64)

    def get_document_size(self, index: int) -> int:
        return len(self._samples[index])

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


@pytest.mark.parametrize("seed", (0, 32, 88))
@pytest.mark.parametrize(
    "shuffle", (ShufflingType.full, ShufflingType.epoch, ShufflingType.skip_first_epoch, ShufflingType.disabled)
)
def test_gpt_sample(seed, shuffle):
    previous_samples = None
    # Loop instead of parametrizing for the check below.
    for num_samples in (20, 10, 6, 5, 2, 1):
        sampled = TEST_DATASET.sample(
            get_sampling_data(
                num_samples,
                sequence_length=5,
                seed=seed,
                shuffle=shuffle,
                preprocessing=LanguageModelPreprocessingConfig(vocab_size=0),
            )
        )
        samples = validate_indexed_dataset_sampling(sampled)
        if previous_samples is not None and shuffle != ShufflingType.full:
            # Check that the sequence is independent of `num_sample`.
            Assert.all_equal(samples, previous_samples[: len(samples)])
        previous_samples = samples


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


def get_test_seeds(num_seeds):
    np.random.seed(42)
    seeds = np.random.randint(0, num_seeds * 100, num_seeds)
    return seeds.tolist()


@pytest.mark.skipif(not _extension_available, reason="CPP Extension not available")
def test_gpt_sample_padding():
    for seed in get_test_seeds(100):
        vocab_size = 30
        np.random.seed(seed)
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
        sampling = get_sampling_data(
            num_samples=len(expected_samples),
            sequence_length=sequence_length,
            seed=seed,
            shuffle=ShufflingType.disabled,
            truncate_documents=False,
            preprocessing=LanguageModelPreprocessingConfig(vocab_size=vocab_size),
        )
        if total_tokens == 0:
            with pytest.raises(RuntimeError):
                dataset.sample(sampling)
        else:
            sampled = dataset.sample(sampling)
            for idx in range(len(expected_samples)):
                Assert.all_equal(sampled[idx].tokens.tokens, np.array(expected_samples[idx]))
