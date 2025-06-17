from fast_llm.data.dataset.gpt.config import GPTDatasetSliceConfig
from tests.data.common import (
    compare_indexed_dataset,
    get_dataset_config,
    get_sampling_data,
    get_test_data_and_compare_samples,
    validate_indexed_dataset_sampling,
)
from tests.data.test_memmap import MEMMAP_DATASET_SAMPLES
from tests.utils.dataset import DATASET_PREFIX, get_test_dataset

GPT_SLICE_TRAINING_SAMPLES = [
    [80, 268, 79, 260, 207, 3086],
    [3086, 80, 413, 4872, 4602, 207],
    [207, 7208, 1489, 776, 3514, 269],
    [269, 73, 7367, 267, 477, 3126],
]
GPT_SLICE_VALIDATION_SAMPLES = [
    [1886, 317, 5621, 3173, 330, 284],
    [284, 2846, 706, 89, 80, 2047],
    [2047, 207, 2449, 1423, 65, 985],
    [985, 683, 4917, 87, 477, 481],
    [481, 695, 947, 5871, 2344, 87],
    [87, 489, 207, 489, 269, 356],
    [356, 727, 7800, 4078, 243, 3712],
    [3712, 86, 476, 80, 2547, 7390],
]

GPT_SLICE_TRAINING_SAMPLES_LEGACY = [
    [2625, 76, 2625, 2639, 74, 243],
    [207, 481, 5546, 74, 414, 498],
    [74, 333, 1963, 310, 5337, 3628],
    [79, 2361, 80, 2012, 84, 480],
]
GPT_SLICE_VALIDATION_SAMPLES_LEGACY = [
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
    compare_indexed_dataset(dataset, 9, 544, {i - 9: sample for i, sample in MEMMAP_DATASET_SAMPLES.items()})
    sampled = dataset.sample(get_sampling_data(8, sequence_length=5))
    validate_indexed_dataset_sampling(sampled, GPT_SLICE_VALIDATION_SAMPLES)


def test_gpt_slice_data():
    get_test_data_and_compare_samples(
        {
            "datasets": {
                "training": {
                    "type": "slice",
                    "dataset": {"type": "memmap", "path": DATASET_PREFIX},
                    "begin": 0,
                    "end": 0.0015,
                },
                "validation": {
                    "type": "slice",
                    "dataset": {"type": "memmap", "path": DATASET_PREFIX},
                    "begin": 0.0015,
                    "end": 0.003,
                },
                "test": {
                    "type": "slice",
                    "dataset": {"type": "memmap", "path": DATASET_PREFIX},
                    "begin": 0.003,
                    "end": 1,
                },
            }
        },
        {"training": 4, "validation": 8, "test": 5},
        sequence_length=5,
        expected_samples={
            "training": GPT_SLICE_TRAINING_SAMPLES,
            "validation": GPT_SLICE_VALIDATION_SAMPLES,
        },
    )


def test_gpt_slice_data_legacy():
    get_test_dataset()
    get_test_data_and_compare_samples(
        {"format": "list", "path": [str(DATASET_PREFIX)], "split": [0.0015, 0.0015, 0.997]},
        {"training": 4, "validation": 8, "test": 5},
        sequence_length=5,
        expected_samples={
            "training": GPT_SLICE_TRAINING_SAMPLES_LEGACY,
            "validation": GPT_SLICE_VALIDATION_SAMPLES_LEGACY,
        },
        legacy=True,
    )
