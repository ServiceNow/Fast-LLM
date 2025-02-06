from fast_llm.data.dataset.gpt.config import GPTConcatenatedDatasetConfig
from fast_llm.engine.distributed.config import PhaseType
from tests.common import DATASET_PREFIX, get_test_dataset
from tests.data.common import (
    compare_indexed_dataset,
    compare_sampled_dataset,
    get_dataset_config,
    get_sampling_config,
    get_test_data_and_compare_samples,
)
from tests.data.test_memmap import MEMMAP_DATASET_LENGTH, MEMMAP_DATASET_SAMPLES, MEMMAP_DATASET_TOKENS

GPT_CONCATENATED_SAMPLES = [
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
    compare_indexed_dataset(
        dataset,
        3 * MEMMAP_DATASET_LENGTH,
        3 * MEMMAP_DATASET_TOKENS,
        {j * MEMMAP_DATASET_LENGTH + i: sample for j in range(3) for i, sample in MEMMAP_DATASET_SAMPLES.items()},
    )
    sampled = dataset.sample(get_sampling_config(8, sequence_length=5))
    compare_sampled_dataset(sampled, GPT_CONCATENATED_SAMPLES)


def test_gpt_concatenate_data():
    get_test_data_and_compare_samples(
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
        expected_samples={PhaseType.training: GPT_CONCATENATED_SAMPLES},
    )
