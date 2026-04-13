from fast_llm.data.dataset.config import DatasetSliceConfig
from fast_llm.data.dataset.gpt.config import GPTDatasetFromFileConfig
from fast_llm.data.document.language_model import LanguageModelDocument
from tests.data.common import (
    compare_indexed_dataset_tokens,
    get_dataset_config,
    get_sampling_config,
    get_test_data_and_compare_samples,
    validate_indexed_dataset_sampling,
)
from tests.data.test_preparator import COMMON_DATASET_SAMPLES
from tests.utils.dataset import get_common_test_dataset

GPT_SLICE_TRAINING_SAMPLES = [
    [50256, 20, 59, 81, 15, 54],
    [54, 76, 1026, 43421, 1, 71],
    [71, 28, 10, 42, 21016, 80],
    [80, 59, 86, 4, 74, 45],
]
GPT_SLICE_VALIDATION_SAMPLES = [
    [50256, 3, 381, 27, 62, 8],
    [8, 10503, 73, 32, 29, 32],
    [32, 3, 89, 15, 45, 25],
    [25, 75, 7340, 40, 88, 54],
    [54, 19, 2, 74, 23, 92],
    [92, 65, 85, 42, 6, 304],
    [304, 21, 47, 92, 31, 30],
    [30, 8455, 23, 11, 56, 12805],
]


def test_gpt_slice(data_result_path):
    # Make sure dataset splitting works and check for unintended changes in behavior.
    _, config, _, preprocessing = get_common_test_dataset()
    memmap_config = GPTDatasetFromFileConfig.from_dict(config)._load_config()
    # samples[9:18]
    dataset = get_dataset_config(
        {"type": "slice", "dataset": memmap_config, "begin": 0.025, "end": 0.1},
        DatasetSliceConfig[LanguageModelDocument],
    ).build()
    compare_indexed_dataset_tokens(dataset, 75, 3575, {i - 25: sample for i, sample in COMMON_DATASET_SAMPLES.items()})
    sampled = dataset.sample(*get_sampling_config(8, sequence_length=5, preprocessing=preprocessing))
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
        preprocessing=preprocessing,
        cache_directory=data_result_path / "slice",
    )
