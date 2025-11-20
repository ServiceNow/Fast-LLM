from fast_llm.data.dataset.config import ConcatenatedDatasetConfig
from fast_llm.data.dataset.gpt.config import GPTDatasetFromFileConfig
from fast_llm.data.sample.language_model import LanguageModelSample
from tests.data.common import (
    compare_indexed_dataset_tokens,
    compare_sampled_dataset,
    get_dataset_config,
    get_sampling_data,
    get_test_data_and_compare_samples,
)
from tests.data.test_preparator import COMMON_DATASET_LENGTH, COMMON_DATASET_SAMPLES, COMMON_DATASET_TOKENS
from tests.utils.dataset import get_common_test_dataset

GPT_CONCATENATED_SAMPLES = [
    [49152, 46, 10, 819, 19, 45],
    [45, 69, 17, 86, 38826, 15],
    [15, 25, 51, 31, 32348, 64],
    [64, 17, 93, 78, 40, 1793],
    [1793, 1, 1746, 38, 27, 58],
    [58, 22885, 93, 37, 92, 76],
    [76, 29, 19, 17365, 93, 46],
    [46, 83, 17211, 1, 785, 1023],
]


def test_gpt_concatenate():
    # Make sure the dataset concatenation works and check for unintended changes in behavior.
    _, config, _ = get_common_test_dataset()
    memmap_config = GPTDatasetFromFileConfig.from_dict(config)._load_config()
    dataset = get_dataset_config(
        dataset_config := {"type": "concatenated", "datasets": [memmap_config.to_dict() for _ in range(3)]},
        ConcatenatedDatasetConfig[LanguageModelSample],
    ).build()
    compare_indexed_dataset_tokens(
        dataset,
        3 * COMMON_DATASET_LENGTH,
        3 * COMMON_DATASET_TOKENS,
        {j * COMMON_DATASET_LENGTH + i: sample for j in range(3) for i, sample in COMMON_DATASET_SAMPLES.items()},
    )
    sampled = dataset.sample(get_sampling_data(8, sequence_length=5))
    compare_sampled_dataset(sampled, GPT_CONCATENATED_SAMPLES)

    # Test in data.
    get_test_data_and_compare_samples(
        {"datasets": {"training": dataset_config}},
        8,
        sequence_length=5,
        expected_samples=GPT_CONCATENATED_SAMPLES,
    )
