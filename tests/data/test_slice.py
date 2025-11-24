from fast_llm.data.dataset.config import DatasetSliceConfig
from fast_llm.data.dataset.gpt.config import GPTDatasetFromFileConfig
from fast_llm.data.sample.language_model import LanguageModelSample
from tests.data.common import (
    compare_indexed_dataset_tokens,
    get_dataset_config,
    get_sampling_data,
    get_test_data_and_compare_samples,
    validate_indexed_dataset_sampling,
)
from tests.data.test_preparator import COMMON_DATASET_SAMPLES
from tests.utils.dataset import get_common_test_dataset

GPT_SLICE_TRAINING_SAMPLES = [
    [49152, 20, 59, 81, 15, 54],
    [54, 76, 7909, 44, 41, 1],
    [1, 71, 28, 10, 42, 15963],
    [15963, 80, 59, 86, 4, 74],
]
GPT_SLICE_VALIDATION_SAMPLES = [
    [49152, 3, 5621, 27, 7859, 13009],
    [13009, 73, 32, 29, 32, 3],
    [3, 89, 15, 45, 25, 75],
    [75, 52, 13366, 88, 54, 19],
    [19, 2, 74, 23, 92, 24747],
    [24747, 42, 6, 477, 21, 47],
    [47, 92, 31, 30, 463, 64],
    [64, 23, 11, 56, 23555, 85],
]


def test_gpt_slice():
    # Make sure dataset splitting works and check for unintended changes in behavior.
    _, config, _ = get_common_test_dataset()
    memmap_config = GPTDatasetFromFileConfig.from_dict(config)._load_config()
    # samples[9:18]
    dataset = get_dataset_config(
        {"type": "slice", "dataset": memmap_config, "begin": 0.025, "end": 0.1},
        DatasetSliceConfig[LanguageModelSample],
    ).build()
    compare_indexed_dataset_tokens(dataset, 75, 3399, {i - 25: sample for i, sample in COMMON_DATASET_SAMPLES.items()})
    sampled = dataset.sample(get_sampling_data(8, sequence_length=5))
    validate_indexed_dataset_sampling(sampled, GPT_SLICE_VALIDATION_SAMPLES)

    # Test in data with multiple phases.
    get_test_data_and_compare_samples(
        {
            "datasets": {
                "training": {
                    "type": "slice",
                    "dataset": memmap_config,
                    "begin": 0,
                    "end": 0.025,
                },
                "validation": {
                    "type": "slice",
                    "dataset": memmap_config,
                    "begin": 0.025,
                    "end": 0.1,
                },
                "test": {
                    "type": "slice",
                    "dataset": memmap_config,
                    "begin": 0.1,
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
