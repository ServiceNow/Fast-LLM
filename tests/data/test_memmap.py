import pathlib
import tempfile

import numpy as np
import pytest

from fast_llm.data.dataset.gpt.config import GPTMemmapDatasetConfig
from fast_llm.data.dataset.gpt.memmap import GPTMemmapDataset
from fast_llm.data.preparator.gpt_memmap.config import MEMMAP_DTYPES
from tests.common import DATASET_PREFIX, DATASET_SAMPLING_CACHE, get_test_dataset
from tests.data.common import compare_indexed_dataset, get_dataset_config


@pytest.mark.parametrize("dtype", MEMMAP_DTYPES.values())
def test_write_memmap_dataset(dtype):
    documents = [np.random.randint(1000, size=np.random.randint(1, 100)).astype(dtype) for _ in range(100)]
    with tempfile.TemporaryDirectory() as temp_dir:
        prefix = pathlib.Path(temp_dir)
        GPTMemmapDataset.write_dataset(prefix=prefix, documents=documents)
        dataset = GPTMemmapDataset(name="foo", prefix=prefix)
        for i, document in enumerate(documents):
            assert np.array_equal(
                dataset.get(i), document, equal_nan=True
            ), f"Mismatch for document {i}: {document} != {dataset.get(i)}."


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
