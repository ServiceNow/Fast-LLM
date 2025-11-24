from fast_llm.data.dataset.gpt.config import GPTDatasetSliceConfig
from tests.data.common import (
    compare_indexed_dataset,
    get_dataset_config,
    get_sampling_data,
    get_test_data_and_compare_samples,
    validate_indexed_dataset_sampling,
)
from tests.data.test_memmap import MEMMAP_DATASET_SAMPLES
from tests.utils.dataset import get_test_dataset
from tests.utils.global_variables import DATASET_PREFIX

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
