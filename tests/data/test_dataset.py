import pathlib

import numpy as np
import pytest

from fast_llm.data.dataset.gpt.config import GPTMemmapDatasetConfig
from fast_llm.utils import Assert
from tests.common import DATASET_PREFIX, DATASET_SAMPLING_CACHE, get_test_dataset
from tests.data.common import get_dataset_config

# Most documents are too long to write here, we test a few known short ones.
MEMMAP_DATASET_EXPECTED_LENGTH = 6153
MEMMAP_DATASET_EXPECTED_TOKENS = 508327
MEMMAP_DATASET_EXPECTED_SAMPLES = {
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
    Assert.eq(len(dataset), MEMMAP_DATASET_EXPECTED_LENGTH)
    sizes = dataset.get_document_sizes()
    Assert.eq(sizes.sum(), MEMMAP_DATASET_EXPECTED_TOKENS)
    Assert.all_equal([len(dataset.get(i)) for i in range(100)], sizes[:100])
    for i, sample in MEMMAP_DATASET_EXPECTED_SAMPLES.items():
        Assert.all_equal(dataset.get(i), np.array(sample, dtype=np.uint16))
