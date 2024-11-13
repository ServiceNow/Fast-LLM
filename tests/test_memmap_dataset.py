import pathlib
import tempfile

import numpy as np
import pytest

from fast_llm.data.gpt.memmap import GPTMemmapDataset
from fast_llm.data.preparator.gpt_memmap.config import MEMMAP_DTYPES


@pytest.mark.parametrize("dtype", MEMMAP_DTYPES.values())
def test_gpt_memmap_dataset(dtype):
    documents = [np.random.randint(1000, size=np.random.randint(1, 100)).astype(dtype) for _ in range(100)]
    with tempfile.TemporaryDirectory() as temp_dir:
        prefix = pathlib.Path(temp_dir)
        GPTMemmapDataset.write_dataset(prefix=prefix, documents=documents)
        dataset = GPTMemmapDataset(name="foo", prefix=prefix)
        for i, document in enumerate(documents):
            assert np.array_equal(
                dataset.get(i), document, equal_nan=True
            ), f"Mismatch for document {i}: {document} != {dataset.get(i)}."
