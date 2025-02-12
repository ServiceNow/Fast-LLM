from fast_llm.data.dataset.gpt.config import GPTConcatenatedDatasetConfig
from fast_llm.engine.distributed.config import PhaseType
from tests.common import DATASET_PREFIX, get_test_dataset
from tests.data.common import (
    compare_indexed_dataset,
    compare_sampled_dataset,
    get_dataset_config,
    get_sampling_data,
    get_test_data_and_compare_samples,
)
from tests.data.test_memmap import MEMMAP_DATASET_LENGTH, MEMMAP_DATASET_SAMPLES, MEMMAP_DATASET_TOKENS

GPT_CONCATENATED_SAMPLES = [
    [4709, 819, 79, 207, 277, 1790],
    [1790, 80, 6506, 1735, 542, 88],
    [88, 4302, 269, 2794, 119, 80],
    [80, 207, 567, 498, 89, 207],
    [207, 4700, 549, 79, 417, 3036],
    [3036, 253, 207, 2968, 4536, 1178],
    [1178, 3291, 317, 277, 2679, 89],
    [89, 542, 395, 583, 684, 554],
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
    sampled = dataset.sample(get_sampling_data(8, sequence_length=5))
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
