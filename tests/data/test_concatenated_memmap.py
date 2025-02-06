from fast_llm.data.dataset.gpt.config import GPTConcatenatedMemmapConfig
from fast_llm.engine.distributed.config import PhaseType
from tests.common import DATASET_CACHE, get_test_concatenated_memmap_dataset
from tests.data.common import (
    compare_indexed_dataset,
    compare_sampled_dataset,
    get_dataset_config,
    get_sampling_config,
    get_test_data_and_compare_samples,
)
from tests.data.test_memmap import MEMMAP_DATASET_SAMPLES

_DATASET_PREFIX_MIX_CONCATENATED_MEMMAP = DATASET_CACHE / "concatenated_memmap"


def _get_test_dataset_concatenated_memmap():
    return get_test_concatenated_memmap_dataset(_DATASET_PREFIX_MIX_CONCATENATED_MEMMAP, 4)


CONCATENATED_MEMMAP_DATASET_LENGTH = 24806
CONCATENATED_MEMMAP_DATASET_TOKENS = 2033639
CONCATENATED_MEMMAP_DATASET_SAMPLES = {
    **MEMMAP_DATASET_SAMPLES,
    6930: [65, 2327],
    11962: [7078, 2713, 1431],
    15958: [207],
    19362: [69],
    24098: [555, 668, 70],
}
CONCATENATED_MEMMAP_SAMPLES = [
    [1411, 819, 6791, 7022, 285, 249],
    [329, 328, 512, 1985, 3069, 7838],
    [5158, 1023, 8171, 798, 1431, 313],
    [1073, 3917, 275, 480, 74, 1752],
    [207, 317, 269, 6662, 4357, 498],
    [74, 310, 277, 7091, 668, 367],
    [7828, 480, 89, 116, 4604, 69],
    [79, 6042, 577, 225, 207, 207],
]


def test_gpt_concatenated_memmap():
    # Make sure dataset splitting works and check for unintended changes in behavior.
    _get_test_dataset_concatenated_memmap()
    # samples[9:18]
    dataset = get_dataset_config(
        {"type": "concatenated_memmap", "path": _DATASET_PREFIX_MIX_CONCATENATED_MEMMAP},
        GPTConcatenatedMemmapConfig,
    ).build()
    compare_indexed_dataset(
        dataset,
        CONCATENATED_MEMMAP_DATASET_LENGTH,
        CONCATENATED_MEMMAP_DATASET_TOKENS,
        CONCATENATED_MEMMAP_DATASET_SAMPLES,
    )
    sampled = dataset.sample(get_sampling_config(8, sequence_length=5))
    compare_sampled_dataset(sampled, CONCATENATED_MEMMAP_SAMPLES)


def test_gpt_concatenated_memmap_data():
    _get_test_dataset_concatenated_memmap()
    get_test_data_and_compare_samples(
        {
            "datasets": {
                "Training": {
                    "type": "concatenated_memmap",
                    "path": _DATASET_PREFIX_MIX_CONCATENATED_MEMMAP,
                }
            }
        },
        {PhaseType.training: 8},
        sequence_length=5,
        expected_samples={PhaseType.training: CONCATENATED_MEMMAP_SAMPLES},
    )
