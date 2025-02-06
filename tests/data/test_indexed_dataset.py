import numpy as np

from fast_llm.data.dataset.gpt.config import (
    GPTConcatenatedDatasetConfig,
    GPTDatasetSliceConfig,
    GPTMemmapDatasetConfig,
)
from fast_llm.engine.distributed.config import PhaseType
from fast_llm.utils import Assert
from tests.common import DATASET_PREFIX, get_test_dataset
from tests.data.common import _check_sampling, get_dataset_config, get_sampling_data, get_test_data_and_samples
from tests.data.test_dataset import (
    MEMMAP_DATASET_EXPECTED_LENGTH,
    MEMMAP_DATASET_EXPECTED_SAMPLES,
    MEMMAP_DATASET_EXPECTED_TOKENS,
)

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
    dataset = get_dataset_config({"type": "memmap", "path": DATASET_PREFIX}, GPTMemmapDatasetConfig).build()
    sampled = dataset.sample(get_sampling_data(8, sequence_length=5))
    _check_sampling(sampled, GPT_SAMPLED_EXPECTED_SAMPLES)


def test_gpt_sampled_data():
    get_test_dataset()
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
        expected_samples={PhaseType.training: GPT_SAMPLED_EXPECTED_SAMPLES},
    )


def test_gpt_sampled_data_legacy():
    _, samples = get_test_data_and_samples(
        {"format": "list", "path": [str(DATASET_PREFIX)], "split": [1, 0, 0]},
        {PhaseType.training: 8},
        sequence_length=5,
        expected_samples={PhaseType.training: GPT_SAMPLED_EXPECTED_SAMPLES},
    )


GPT_CONCATENATED_EXPECTED_SAMPLES = [
    [243, 498, 7172, 777, 306, 74],
    [821, 6042, 89, 977, 4797, 499],
    [387, 74, 330, 328, 1858, 484],
    [7722, 3069, 819, 4266, 304, 80],
    [80, 634, 4913, 373, 207, 1046],
    [72, 65, 5570, 73, 2210, 5514],
    [7983, 977, 4147, 4739, 890, 386],
    [5375, 275, 69, 771, 593, 8171],
]


def test_gpt_concatenate():
    # Make sure the dataset concatenation works and check for unintended changes in behavior.
    get_test_dataset()
    dataset = get_dataset_config(
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
            Assert.all_equal(dataset.get(begin + i), np.array(sample, dtype=np.uint16))
    sampled = dataset.sample(get_sampling_data(8, sequence_length=5))
    _check_sampling(sampled, GPT_CONCATENATED_EXPECTED_SAMPLES)


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
        np.stack(samples[PhaseType.training]),
        np.array(GPT_CONCATENATED_EXPECTED_SAMPLES),
    )


GPT_SLICE_EXPECTED_TRAINING_SAMPLES = [
    [2625, 76, 2625, 2639, 74, 243],
    [207, 481, 5546, 74, 414, 498],
    [74, 333, 1963, 310, 5337, 3628],
    [79, 2361, 80, 2012, 84, 480],
]
GPT_SLICE_EXPECTED_VALIDATION_SAMPLES = [
    [2352, 3687, 2311, 4900, 542, 3732],
    [2551, 5283, 900, 3140, 328, 68],
    [7979, 2283, 329, 727, 2740, 2818],
    [4117, 8056, 79, 1798, 243, 498],
    [243, 542, 387, 6476, 6686, 785],
    [95, 6641, 207, 279, 2304, 602],
    [89, 4446, 947, 293, 947, 1544],
    [243, 3712, 86, 476, 80, 2547],
]


def test_gpt_slice():
    # Make sure dataset splitting works and check for unintended changes in behavior.
    get_test_dataset()
    # samples[9:18]
    dataset = get_dataset_config(
        {"type": "slice", "dataset": {"type": "memmap", "path": DATASET_PREFIX}, "begin": 0.0015, "end": 0.003},
        GPTDatasetSliceConfig,
    ).build()
    Assert.eq(len(dataset), 9)
    sizes = dataset.get_document_sizes()
    Assert.all_equal([len(dataset.get(i)) for i in range(9)], sizes[:9])
    for i, sample in MEMMAP_DATASET_EXPECTED_SAMPLES.items():
        Assert.all_equal(dataset.get(i - 9), np.array(sample, dtype=np.uint16))
    sampled = dataset.sample(get_sampling_data(8, sequence_length=5))
    _check_sampling(sampled, GPT_SLICE_EXPECTED_VALIDATION_SAMPLES)


def test_gpt_slice_data():
    _, samples = get_test_data_and_samples(
        {
            "datasets": {
                "Training": {
                    "type": "slice",
                    "dataset": {"type": "memmap", "path": DATASET_PREFIX},
                    "begin": 0,
                    "end": 0.0015,
                },
                "Validation": {
                    "type": "slice",
                    "dataset": {"type": "memmap", "path": DATASET_PREFIX},
                    "begin": 0.0015,
                    "end": 0.003,
                },
                "Test": {
                    "type": "slice",
                    "dataset": {"type": "memmap", "path": DATASET_PREFIX},
                    "begin": 0.003,
                    "end": 1,
                },
            }
        },
        {PhaseType.training: 4, PhaseType.validation: 8, PhaseType.test: 5},
        sequence_length=5,
    )
    Assert.all_equal(
        np.stack(samples[PhaseType.validation]),
        np.array(GPT_SLICE_EXPECTED_VALIDATION_SAMPLES),
    )
    Assert.all_equal(
        np.stack(samples[PhaseType.training]),
        np.array(GPT_SLICE_EXPECTED_TRAINING_SAMPLES),
    )


def test_gpt_slice_data_legacy():
    get_test_dataset()
    _, samples = get_test_data_and_samples(
        {"format": "list", "path": [str(DATASET_PREFIX)], "split": [0.0015, 0.0015, 0.997]},
        {PhaseType.training: 4, PhaseType.validation: 8, PhaseType.test: 5},
        sequence_length=5,
    )
    Assert.all_equal(
        np.stack(samples[PhaseType.validation]),
        np.array(GPT_SLICE_EXPECTED_VALIDATION_SAMPLES),
    )
    Assert.all_equal(
        np.stack(samples[PhaseType.training]),
        np.array(GPT_SLICE_EXPECTED_TRAINING_SAMPLES),
    )
