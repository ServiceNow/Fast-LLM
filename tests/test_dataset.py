import pathlib
import typing

import numpy
import numpy as np

from fast_llm.data.dataset.gpt.config import (
    GPTConcatenatedDatasetConfig,
    GPTDatasetSliceConfig,
    GPTDummyDatasetConfig,
    GPTMemmapDatasetConfig,
    GPTSampledDatasetConfig,
    GPTSamplingConfig,
)
from fast_llm.data.tokenizer import Tokenizer
from fast_llm.engine.distributed.config import DistributedConfig, PhaseType
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.utils import Assert
from tests.common import DATASET_PREFIX, TEST_RESULTS_PATH, TEST_VOCAB_SIZE, get_test_dataset

DATASET_CACHE = TEST_RESULTS_PATH / "dataset" / "cache"


def get_sampling_config(
    num_samples: int,
    *,
    seed: int = 54983,
    cache_directory: pathlib.Path | None = None,
    distributed: Distributed = Distributed(DistributedConfig(), use_cpu=True),
    phase: PhaseType = PhaseType.training,
    sequence_length: int = 512,
    vocab_size=TEST_VOCAB_SIZE,
    tokenizer: Tokenizer | None = None,
) -> GPTSamplingConfig:
    # Config with convenient defaults.
    return GPTSamplingConfig(
        num_samples=num_samples,
        seed=seed,
        cache_directory=cache_directory,
        distributed=distributed,
        phase=phase,
        sequence_length=sequence_length,
        vocab_size=vocab_size,
        tokenizer=tokenizer,
    )


def _get_dataset_config[T: GPTSampledDatasetConfig](config: dict[str, typing.Any], cls: type[T]) -> T:
    dataset_config = GPTSampledDatasetConfig.from_dict(config)
    Assert.custom(isinstance, dataset_config, cls)
    return typing.cast(cls, dataset_config)


def test_gpt_dummy_dataset():
    # Make sure the dummy dataset works and check for unintended changes in behavior.
    config = _get_dataset_config({"type": "dummy"}, GPTDummyDatasetConfig)
    sampled = config.build_and_sample(get_sampling_config(4, sequence_length=7))
    Assert.eq(len(sampled), 4)
    Assert.all_equal(
        numpy.stack([sampled[i] for i in range(4)]),
        np.array(
            [
                [3954, 4105, 6766, 859, 5494, 1675, 1303, 6913],
                [1654, 5701, 32, 1662, 7053, 3487, 1861, 1502],
                [5409, 6240, 5504, 7458, 7667, 3955, 3151, 3912],
                [5640, 6131, 7750, 2699, 1349, 2585, 7113, 6981],
            ]
        ),
    )


# Most documents are too long to write here, we test a few known short ones.
_MEMMAP_DATASET_EXPECTED_LENGTH = 6153
_MEMMAP_DATASET_EXPECTED_TOKENS = 508327
_MEMMAP_DATASET_EXPECTED_SAMPLES = {
    9: [],
    10: [80, 85, 4295, 4182, 489, 727, 84, 698, 1197, 583],
    13: [78, 727, 74, 317, 1358, 89],
    15: [78],
}


def test_gpt_memmap_dataset():
    # Make sure the memmap dataset works and check for unintended changes in behavior.
    get_test_dataset()
    dataset = _get_dataset_config({"type": "memmap", "path": DATASET_PREFIX}, GPTMemmapDatasetConfig).build()
    Assert.eq(len(dataset), _MEMMAP_DATASET_EXPECTED_LENGTH)
    sizes = dataset.get_document_sizes()
    Assert.eq(sizes.sum(), _MEMMAP_DATASET_EXPECTED_TOKENS)
    Assert.all_equal([len(dataset.get(i)) for i in range(100)], sizes[:100])
    for i, sample in _MEMMAP_DATASET_EXPECTED_SAMPLES.items():
        Assert.all_equal(dataset.get(i), np.array(sample, dtype=numpy.uint16))


def test_gpt_dataset_concatenate():
    # Make sure the dataset concatenation works and check for unintended changes in behavior.
    get_test_dataset()
    dataset = _get_dataset_config(
        {
            "type": "concatenated",
            "datasets": [{"type": "memmap", "path": DATASET_PREFIX} for _ in range(3)],
        },
        GPTConcatenatedDatasetConfig,
    ).build()
    Assert.eq(len(dataset), 3 * _MEMMAP_DATASET_EXPECTED_LENGTH)
    sizes = dataset.get_document_sizes()
    Assert.eq(sizes.sum(), 3 * _MEMMAP_DATASET_EXPECTED_TOKENS)
    for i in range(3):
        begin = i * _MEMMAP_DATASET_EXPECTED_LENGTH
        Assert.all_equal([len(dataset.get(begin + i)) for i in range(100)], sizes[begin : begin + 100])
        for i, sample in _MEMMAP_DATASET_EXPECTED_SAMPLES.items():
            Assert.all_equal(dataset.get(begin + i), np.array(sample, dtype=numpy.uint16))


def test_gpt_dataset_slice():
    # Make sure dataset splitting works and check for unintended changes in behavior.
    get_test_dataset()
    # samples[9:18]
    dataset = _get_dataset_config(
        {"type": "slice", "dataset": {"type": "memmap", "path": DATASET_PREFIX}, "begin": 0.0015, "end": 0.003},
        GPTDatasetSliceConfig,
    ).build()
    Assert.eq(len(dataset), 9)
    sizes = dataset.get_document_sizes()
    Assert.all_equal([len(dataset.get(i)) for i in range(9)], sizes[:9])
    for i, sample in _MEMMAP_DATASET_EXPECTED_SAMPLES.items():
        Assert.all_equal(dataset.get(i - 9), np.array(sample, dtype=numpy.uint16))


def test_gpt_indexed_dataset_sampling():
    # Make sure the memmap dataset works and check for unintended changes in behavior.
    get_test_dataset()
    sampled = _get_dataset_config({"type": "memmap", "path": DATASET_PREFIX}, GPTMemmapDatasetConfig).build_and_sample(
        get_sampling_config(8, sequence_length=5)
    )
    Assert.all_equal(
        np.stack([sampled[i] for i in range(5)]),
        np.array(
            [
                [1725, 74, 207, 1635, 4440, 2774],
                [359, 489, 4266, 2052, 5351, 80],
                [374, 7534, 87, 1073, 79, 480],
                [8008, 498, 71, 727, 80, 315],
                [2210, 8179, 73, 2582, 897, 1178],
            ]
        ),
    )
