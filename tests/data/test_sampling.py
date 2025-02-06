from fast_llm.data.dataset.gpt.config import GPTMemmapDatasetConfig
from fast_llm.engine.distributed.config import PhaseType
from tests.common import DATASET_PREFIX, get_test_dataset
from tests.data.common import (
    compare_sampled_dataset,
    get_dataset_config,
    get_sampling_config,
    get_test_data_and_compare_samples,
)

GPT_MEMMAP_SAMPLES = [
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
    sampled = get_dataset_config({"type": "memmap", "path": DATASET_PREFIX}, GPTMemmapDatasetConfig).build_and_sample(
        get_sampling_config(8, sequence_length=5)
    )
    compare_sampled_dataset(sampled, GPT_MEMMAP_SAMPLES)


def test_gpt_sampled_data():
    get_test_dataset()
    get_test_data_and_compare_samples(
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
        expected_samples={PhaseType.training: GPT_MEMMAP_SAMPLES},
    )


def test_gpt_sampled_data_legacy():
    get_test_data_and_compare_samples(
        {"format": "list", "path": [str(DATASET_PREFIX)], "split": [1, 0, 0]},
        {PhaseType.training: 8},
        sequence_length=5,
        expected_samples={PhaseType.training: GPT_MEMMAP_SAMPLES},
    )
