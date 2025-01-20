import pathlib
import typing

import numpy
import numpy as np
import pytest

from fast_llm.config import NoAutoValidate
from fast_llm.data.config import TokenizerConfig
from fast_llm.data.data.gpt.config import GPTDataConfig
from fast_llm.data.data.gpt.data import GPTData
from fast_llm.data.dataset.gpt.config import (
    GPTBlendedDatasetConfig,
    GPTConcatenatedDatasetConfig,
    GPTDatasetSliceConfig,
    GPTFimSampledDatasetConfig,
    GPTMemmapDatasetConfig,
    GPTRandomDatasetConfig,
    GPTSampledDatasetConfig,
    GPTSamplingConfig,
)
from fast_llm.data.tokenizer import Tokenizer
from fast_llm.engine.distributed.config import DistributedConfig, PhaseType
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.schedule.config import BatchConfig
from fast_llm.utils import Assert
from tests.common import DATASET_PREFIX, TEST_RESULTS_PATH, TEST_VOCAB_SIZE, TOKENIZER_PATH, get_test_dataset

DATASET_CACHE = TEST_RESULTS_PATH / "dataset" / "cache"


def get_sampling_config(
    num_samples: int,
    *,
    seed: int = 54983,
    cache_directory: pathlib.Path | None = None,
    distributed: Distributed = Distributed(DistributedConfig(), use_cpu=True),
    phase=PhaseType.training,
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


def get_test_data_and_samples(
    config: dict,
    samples_per_phase: dict[PhaseType, int],
    cache_directory: pathlib.Path | None = None,
    sequence_length: int = 512,
    vocab_size=TEST_VOCAB_SIZE,
):
    distributed_config = DistributedConfig()
    distributed = Distributed(distributed_config, use_cpu=True)
    data = GPTData(GPTDataConfig.from_dict(config), distributed_config, vocab_size, sequence_length)
    data.setup(distributed, samples_per_phase, cache_directory)
    with NoAutoValidate():
        batch_config = BatchConfig(batch_size=1, sequence_length=sequence_length)
    batch_config.setup(distributed_config)
    batch_config.validate()
    samples = {
        phase: [sample for sample in data.get_iterator(batch_config, phase, consumed_samples=0, num_workers=0)]
        for phase, samples in samples_per_phase.items()
    }
    return data, samples


RANDOM_DATASET_EXPECTED_SAMPLES = [
    [3954, 4105, 6766, 859, 5494, 1675, 1303, 6913],
    [1654, 5701, 32, 1662, 7053, 3487, 1861, 1502],
    [5409, 6240, 5504, 7458, 7667, 3955, 3151, 3912],
    [5640, 6131, 7750, 2699, 1349, 2585, 7113, 6981],
]


def test_gpt_random_dataset():
    # Make sure the random dataset works and check for unintended changes in behavior.
    sampled = _get_dataset_config({"type": "random"}, GPTRandomDatasetConfig).build_and_sample(
        get_sampling_config(4, sequence_length=7)
    )
    Assert.eq(len(sampled), 4)
    Assert.all_equal(
        numpy.stack([sampled[i] for i in range(4)]),
        np.array(RANDOM_DATASET_EXPECTED_SAMPLES),
    )


def test_gpt_random_data():
    _, samples = get_test_data_and_samples(
        {
            "datasets": {
                "Training": {
                    "type": "random",
                }
            }
        },
        {PhaseType.training: 4},
        sequence_length=7,
    )
    Assert.all_equal(
        numpy.stack(samples[PhaseType.training]),
        np.array(RANDOM_DATASET_EXPECTED_SAMPLES),
    )


def test_gpt_random_data_legacy():
    _, samples = get_test_data_and_samples({"format": "random"}, {PhaseType.training: 4}, sequence_length=7)
    Assert.all_equal(
        numpy.stack(samples[PhaseType.training]),
        np.array(RANDOM_DATASET_EXPECTED_SAMPLES),
    )


# Most documents are too long to write here, we test a few known short ones.
MEMMAP_DATASET_EXPECTED_LENGTH = 6153
MEMMAP_DATASET_EXPECTED_TOKENS = 508327
MEMMAP_DATASET_EXPECTED_SAMPLES = {
    9: [],
    10: [80, 85, 4295, 4182, 489, 727, 84, 698, 1197, 583],
    13: [78, 727, 74, 317, 1358, 89],
    15: [78],
}


@pytest.mark.parametrize("cache_directory", (None, pathlib.Path(DATASET_CACHE) / "test_memmap"))
def test_gpt_memmap(cache_directory):
    # Make sure the memmap dataset works and check for unintended changes in behavior.
    get_test_dataset()
    dataset = _get_dataset_config({"type": "memmap", "path": DATASET_PREFIX}, GPTMemmapDatasetConfig).build()
    Assert.eq(len(dataset), MEMMAP_DATASET_EXPECTED_LENGTH)
    sizes = dataset.get_document_sizes()
    Assert.eq(sizes.sum(), MEMMAP_DATASET_EXPECTED_TOKENS)
    Assert.all_equal([len(dataset.get(i)) for i in range(100)], sizes[:100])
    for i, sample in MEMMAP_DATASET_EXPECTED_SAMPLES.items():
        Assert.all_equal(dataset.get(i), np.array(sample, dtype=numpy.uint16))


GPT_SAMPLED_EXPECTED_SAMPLES = [
    [1725, 74, 207, 1635, 4440, 2774],
    [359, 489, 4266, 2052, 5351, 80],
    [374, 7534, 87, 1073, 79, 480],
    [8008, 498, 71, 727, 80, 315],
    [2210, 8179, 73, 2582, 897, 1178],
    [409, 5091, 328, 1378, 5483, 88],
    [83, 4457, 3316, 333, 489, 317],
    [330, 155, 2449, 1136, 1106, 5370],
]


def test_gpt_sampled():
    # Make sure the memmap dataset works and check for unintended changes in behavior.
    get_test_dataset()
    sampled = _get_dataset_config({"type": "memmap", "path": DATASET_PREFIX}, GPTMemmapDatasetConfig).build_and_sample(
        get_sampling_config(8, sequence_length=5)
    )
    Assert.eq(len(sampled), 8)
    Assert.all_equal(
        np.stack([sampled[i] for i in range(8)]),
        np.array(GPT_SAMPLED_EXPECTED_SAMPLES),
    )


def test_gpt_sampled_data():
    _, samples = get_test_data_and_samples(
        {
            "datasets": {
                "Training": {
                    "type": "memmap",
                    "path": DATASET_PREFIX,
                }
            }
        },
        {PhaseType.training: 8},
        sequence_length=5,
    )
    Assert.all_equal(
        numpy.stack(samples[PhaseType.training]),
        np.array(RANDOM_DATASET_EXPECTED_SAMPLES),
    )


def test_gpt_sampled_data_legacy():
    _, samples = get_test_data_and_samples(
        {"format": "list", "path": [DATASET_PREFIX], "split": [1]}, {PhaseType.training: 8}, sequence_length=5
    )
    Assert.all_equal(
        numpy.stack(samples[PhaseType.training]),
        np.array(RANDOM_DATASET_EXPECTED_SAMPLES),
    )


GPT_CONCATENATED_EXPECTED_SAMPLES = [
    [1725, 74, 207, 1635, 4440, 2774],
    [359, 489, 4266, 2052, 5351, 80],
    [374, 7534, 87, 1073, 79, 480],
    [8008, 498, 71, 727, 80, 315],
    [2210, 8179, 73, 2582, 897, 1178],
    [409, 5091, 328, 1378, 5483, 88],
    [83, 4457, 3316, 333, 489, 317],
    [330, 155, 2449, 1136, 1106, 5370],
]


def test_gpt_concatenate():
    # Make sure the dataset concatenation works and check for unintended changes in behavior.
    get_test_dataset()
    dataset = _get_dataset_config(
        {"type": "concatenated", "datasets": [{"type": "memmap", "path": DATASET_PREFIX} for _ in range(3)]},
        GPTConcatenatedDatasetConfig,
    ).build()
    Assert.eq(len(dataset), 3 * MEMMAP_DATASET_EXPECTED_LENGTH)
    sizes = dataset.get_document_sizes()
    Assert.eq(sizes.sum(), 3 * MEMMAP_DATASET_EXPECTED_TOKENS)
    for i in range(3):
        begin = i * MEMMAP_DATASET_EXPECTED_LENGTH
        Assert.all_equal([len(dataset.get(begin + i)) for i in range(100)], sizes[begin : begin + 100])
        for i, sample in MEMMAP_DATASET_EXPECTED_SAMPLES.items():
            Assert.all_equal(dataset.get(begin + i), np.array(sample, dtype=numpy.uint16))
    sampled = dataset.sample(get_sampling_config(8, sequence_length=5))
    Assert.eq(len(sampled), 8)
    Assert.all_equal(
        np.stack([sampled[i] for i in range(8)]),
        np.array(GPT_CONCATENATED_EXPECTED_SAMPLES),
    )


def test_gpt_concatenate_data():
    _, samples = get_test_data_and_samples(
        {
            "datasets": {
                "Training": {
                    "type": "concatenated",
                    "datasets": [{"type": "memmap", "path": DATASET_PREFIX} for _ in range(3)],
                }
            }
        },
        {PhaseType.training: 8},
        sequence_length=5,
    )
    Assert.all_equal(
        numpy.stack(samples[PhaseType.training]),
        np.array(RANDOM_DATASET_EXPECTED_SAMPLES),
    )


GPT_SLICE_EXPECTED_TRAINING_SAMPLES = [
    [1725, 74, 207, 1635, 4440, 2774],
    [359, 489, 4266, 2052, 5351, 80],
    [374, 7534, 87, 1073, 79, 480],
    [8008, 498, 71, 727, 80, 315],
    [2210, 8179, 73, 2582, 897, 1178],
    [409, 5091, 328, 1378, 5483, 88],
    [83, 4457, 3316, 333, 489, 317],
    [330, 155, 2449, 1136, 1106, 5370],
]
GPT_SLICE_EXPECTED_VALIDATION_SAMPLES = [
    [1725, 74, 207, 1635, 4440, 2774],
    [359, 489, 4266, 2052, 5351, 80],
    [374, 7534, 87, 1073, 79, 480],
    [8008, 498, 71, 727, 80, 315],
    [2210, 8179, 73, 2582, 897, 1178],
    [409, 5091, 328, 1378, 5483, 88],
    [83, 4457, 3316, 333, 489, 317],
    [330, 155, 2449, 1136, 1106, 5370],
]


def test_gpt_slice():
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
    for i, sample in MEMMAP_DATASET_EXPECTED_SAMPLES.items():
        Assert.all_equal(dataset.get(i - 9), np.array(sample, dtype=numpy.uint16))
    sampled = dataset.sample(get_sampling_config(8, sequence_length=5))
    Assert.eq(len(sampled), 8)
    Assert.all_equal(
        np.stack([sampled[i] for i in range(8)]),
        np.array(GPT_SLICE_EXPECTED_VALIDATION_SAMPLES),
    )


def test_gpt_slice_data():
    _, samples = get_test_data_and_samples(
        {
            "datasets": {
                "Training": {"type": "split", "dataset": {"type": "memmap", "path": DATASET_PREFIX}},
                "begin": 0,
                "end": 0.0015,
                "Validation": {
                    "type": "split",
                    "dataset": {"type": "memmap", "path": DATASET_PREFIX},
                    "begin": 0.0015,
                    "end": 0.003,
                },
                "Test": {
                    "type": "split",
                    "dataset": {"type": "memmap", "path": DATASET_PREFIX},
                    "begin": 0.003,
                    "end": 1,
                },
            }
        },
        {PhaseType.training: 8},
        sequence_length=5,
    )
    Assert.all_equal(
        numpy.stack(samples[PhaseType.training]),
        np.array(GPT_SLICE_EXPECTED_TRAINING_SAMPLES),
    )
    Assert.all_equal(
        numpy.stack(samples[PhaseType.validation]),
        np.array(GPT_SLICE_EXPECTED_VALIDATION_SAMPLES),
    )


def test_gpt_slice_data_legacy():
    _, samples = get_test_data_and_samples(
        {"format": "list", "path": [DATASET_PREFIX], "split": [0.0015, 0.0015, 0.997]},
        {PhaseType.training: 4, PhaseType.validation: 8, PhaseType.test: 5},
        sequence_length=5,
    )
    Assert.all_equal(
        numpy.stack(samples[PhaseType.training]),
        np.array(GPT_SLICE_EXPECTED_TRAINING_SAMPLES),
    )
    Assert.all_equal(
        numpy.stack(samples[PhaseType.validation]),
        np.array(GPT_SLICE_EXPECTED_VALIDATION_SAMPLES),
    )


GPT_BLENDED_EXPECTED_SAMPLES = [
    [1725, 74, 207, 1635, 4440, 2774],
    [5291, 3692, 4158, 503, 2201, 2587],
    [359, 489, 4266, 2052, 5351, 80],
    [5558, 4833, 2889, 7476, 1588, 226],
    [374, 7534, 87, 1073, 79, 480],
    [8008, 498, 71, 727, 80, 315],
    [786, 3161, 8179, 2300, 6160, 2531],
    [2210, 8179, 73, 2582, 897, 1178],
]


def test_gpt_blended():
    # Make sure dataset blending works and check for unintended changes in behavior.
    get_test_dataset()
    sampled = _get_dataset_config(
        {
            "type": "blended",
            "datasets": [{"type": "memmap", "path": DATASET_PREFIX} for _ in range(2)],
            "weights": [0.75, 0.25],
        },
        GPTBlendedDatasetConfig,
    ).build_and_sample(get_sampling_config(8, sequence_length=5))
    Assert.eq(len(sampled), 8)
    Assert.all_equal(
        np.stack([sampled[i] for i in range(8)]),
        np.array(GPT_BLENDED_EXPECTED_SAMPLES),
    )


def test_gpt_blended_data():
    _, samples = get_test_data_and_samples(
        {
            "datasets": {
                "Training": {
                    "type": "blended",
                    "datasets": [{"type": "memmap", "path": DATASET_PREFIX} for _ in range(2)],
                    "weights": [0.75, 0.25],
                }
            }
        },
        {PhaseType.training: 8},
        sequence_length=5,
    )
    Assert.all_equal(
        numpy.stack(samples[PhaseType.validation]),
        np.array(GPT_BLENDED_EXPECTED_SAMPLES),
    )


def test_gpt_blended_legacy_data():
    _, samples = get_test_data_and_samples(
        {"format": "list", "path": [0.75, DATASET_PREFIX, 0.25, DATASET_PREFIX]},
        {PhaseType.training: 8},
        sequence_length=5,
    )
    Assert.all_equal(
        numpy.stack(samples[PhaseType.validation]),
        np.array(GPT_BLENDED_EXPECTED_SAMPLES),
    )


GPT_BLENDED_MIXED_EXPECTED_SAMPLES = [
    [1725, 74, 207, 1635, 4440, 2774],
    [5291, 3692, 4158, 503, 2201, 2587],
    [359, 489, 4266, 2052, 5351, 80],
    [5558, 4833, 2889, 7476, 1588, 226],
    [374, 7534, 87, 1073, 79, 480],
    [8008, 498, 71, 727, 80, 315],
    [786, 3161, 8179, 2300, 6160, 2531],
    [2210, 8179, 73, 2582, 897, 1178],
]


def test_gpt_blended_mixed():
    # Make sure dataset blending works and check for unintended changes in behavior.
    get_test_dataset()
    sampled = _get_dataset_config(
        {
            "type": "blended",
            "datasets": [
                {"type": "memmap", "path": DATASET_PREFIX},
                {"type": "random"},
            ],
            "weights": [0.6, 0.4],
        },
        GPTBlendedDatasetConfig,
    ).build_and_sample(get_sampling_config(8, sequence_length=5, seed=109766))
    Assert.eq(len(sampled), 8)
    Assert.all_equal(
        np.stack([sampled[i] for i in range(8)]),
        np.array(GPT_BLENDED_MIXED_EXPECTED_SAMPLES),
    )


def test_gpt_blended_mixed_data():
    _, samples = get_test_data_and_samples(
        {
            "datasets": {
                "Training": {
                    "type": "blended",
                    "datasets": [{"type": "memmap", "path": DATASET_PREFIX}, {"type": "random"}],
                    "weights": [0.6, 0.4],
                }
            }
        },
        {PhaseType.training: 8},
        sequence_length=5,
    )
    Assert.all_equal(
        numpy.stack(samples[PhaseType.validation]),
        np.array(GPT_BLENDED_MIXED_EXPECTED_SAMPLES),
    )


GPT_FIM_EXPECTED_SAMPLES = [
    [1725, 74, 207, 1635, 4440, 2774],
    [359, 489, 4266, 2052, 5351, 80],
    [86, 89, 22255, 1073, 79, 480],
    [8008, 498, 71, 727, 80, 315],
    [2210, 8179, 73, 2582, 897, 1178],
    [86, 89, 88, 87, 409, 70],
    [86, 83, 744, 89, 64, 333],
    [86, 89, 1461, 87, 330, 7876],
]


def test_gpt_fim():
    # Make sure the FIM wrapper works in a simple case and check for unintended changes in behavior.
    get_test_dataset()
    # The test tokenizer doesn't have fim tokens, so we work around it.
    sampling_config = get_sampling_config(
        8, sequence_length=5, tokenizer=Tokenizer(TokenizerConfig.from_dict({"path": TOKENIZER_PATH}))
    )
    sampled = _get_dataset_config(
        {
            "type": "fim",
            "dataset": {"type": "memmap", "path": DATASET_PREFIX},
            "rate": 0.5,
            "prefix_token": "w",
            "middle_token": "x",
            "pad_token": "y",
            "suffix_token": "z",
        },
        GPTFimSampledDatasetConfig,
    ).build_and_sample(sampling_config)
    Assert.eq(len(sampled), 8)
    # TODO: Does this output make sense?
    Assert.all_equal(
        np.stack([sampled[i] for i in range(8)]),
        np.array(GPT_FIM_EXPECTED_SAMPLES),
    )


def test_gpt_fim_data():
    _, samples = get_test_data_and_samples(
        {
            "datasets": {
                "Training": {
                    "type": "fim",
                    "dataset": {"type": "memmap", "path": DATASET_PREFIX},
                    "rate": 0.5,
                    "prefix_token": "w",
                    "middle_token": "x",
                    "pad_token": "y",
                    "suffix_token": "z",
                }
            }
        },
        {PhaseType.training: 8},
        sequence_length=5,
    )
    Assert.all_equal(
        numpy.stack(samples[PhaseType.validation]),
        np.array(GPT_FIM_EXPECTED_SAMPLES),
    )


def test_gpt_fim_data_legacy():
    _, samples = get_test_data_and_samples(
        {
            "format": "list",
            "path": [DATASET_PREFIX],
            "fim": {"rate": 0.5, "prefix_token": "w", "middle_token": "x", "pad_token": "y", "suffix_token": "z"},
            "tokenizer": {"path": TOKENIZER_PATH},
        },
        {PhaseType.training: 8},
        sequence_length=5,
    )
    Assert.all_equal(
        numpy.stack(samples[PhaseType.validation]),
        np.array(GPT_FIM_EXPECTED_SAMPLES),
    )
