import pytest

from fast_llm.data.dataset.gpt.config import GPTConcatenatedMemmapConfig
from tests.data.common import (
    compare_indexed_dataset,
    get_dataset_config,
    get_sampling_data,
    get_test_data_and_compare_samples,
    validate_indexed_dataset_sampling,
)
from tests.data.test_memmap import MEMMAP_DATASET_SAMPLES
from tests.utils.dataset import get_test_concatenated_memmap_dataset
from tests.utils.global_variables import DATASET_CACHE

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
    [7554, 80, 5970, 87, 477, 4119],
    [4119, 6506, 74, 447, 87, 277],
    [277, 320, 2597, 4117, 301, 727],
    [727, 330, 3067, 2740, 81, 417],
    [417, 1486, 542, 248, 540, 1364],
    [1364, 7072, 2516, 2455, 79, 207],
    [207, 727, 2204, 2379, 540, 1322],
    [1322, 365, 2009, 72, 489, 1886],
]


def test_gpt_concatenated_memmap():
    # Make sure dataset splitting works and check for unintended changes in behavior.
    _get_test_dataset_concatenated_memmap()
    # samples[9:18]
    with pytest.warns(DeprecationWarning):
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
    sampled = dataset.sample(get_sampling_data(8, sequence_length=5))
    validate_indexed_dataset_sampling(sampled, CONCATENATED_MEMMAP_SAMPLES)


def test_gpt_concatenated_memmap_data():
    _get_test_dataset_concatenated_memmap()
    with pytest.warns(DeprecationWarning):
        get_test_data_and_compare_samples(
            {
                "datasets": {
                    "Training": {
                        "type": "concatenated_memmap",
                        "path": _DATASET_PREFIX_MIX_CONCATENATED_MEMMAP,
                    }
                }
            },
            8,
            sequence_length=5,
            expected_samples=CONCATENATED_MEMMAP_SAMPLES,
        )
