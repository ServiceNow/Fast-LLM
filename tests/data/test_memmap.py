import pathlib

import pytest

from fast_llm.data.dataset.gpt.config import GPTMemmapDatasetConfig
from tests.data.common import compare_indexed_dataset, get_dataset_config
from tests.utils.dataset import get_test_dataset
from tests.utils.global_variables import DATASET_CACHE, DATASET_PREFIX, DATASET_SAMPLING_CACHE

MEMMAP_DATASET_LENGTH = 6153
MEMMAP_DATASET_TOKENS = 508327
MEMMAP_DATASET_SAMPLES = {
    9: [],
    10: [80, 85, 4295, 4182, 489, 727, 84, 698, 1197, 583],
    13: [78, 727, 74, 317, 1358, 89],
    15: [78],
}


@pytest.mark.parametrize("cache_directory", (None, pathlib.Path(DATASET_SAMPLING_CACHE) / "test_memmap"))
def test_gpt_memmap(cache_directory):
    # Make sure the memmap dataset works and check for unintended changes in behavior.
    get_test_dataset()
    dataset = get_dataset_config({"type": "memmap", "path": DATASET_PREFIX}, GPTMemmapDatasetConfig).build()
    compare_indexed_dataset(dataset, MEMMAP_DATASET_LENGTH, MEMMAP_DATASET_TOKENS, MEMMAP_DATASET_SAMPLES)


MEMMAP_DATASET_SPANS = {
    9: [],
    10: [[0, 4], [6, 8]],
    13: [[1, 2]],
    15: [],
}

_DATASET_PREFIX_SPANS = DATASET_CACHE / "with_spans" / "dataset"


def test_gpt_data_with_spans():
    get_test_dataset(prefix=_DATASET_PREFIX_SPANS, max_spans=5)
    dataset = get_dataset_config(
        {
            "type": "memmap",
            "path": _DATASET_PREFIX_SPANS,
        },
        GPTMemmapDatasetConfig,
    ).build()
    compare_indexed_dataset(
        dataset, MEMMAP_DATASET_LENGTH, MEMMAP_DATASET_TOKENS, MEMMAP_DATASET_SAMPLES, MEMMAP_DATASET_SPANS
    )
