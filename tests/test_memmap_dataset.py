import pathlib
import tempfile

import numpy as np
import pytest

from fast_llm.data.dataset.gpt.memmap import GPTMemmapDataset
from fast_llm.data.preparator.gpt_memmap.config import MEMMAP_DTYPES


@pytest.mark.parametrize("dtype", MEMMAP_DTYPES.values())
def test_gpt_memmap_dataset(dtype):
    documents = list(
        zip(
            [np.random.randint(1000, size=np.random.randint(1, 100)).astype(dtype) for _ in range(100)],
            np.array([[]] * 100, dtype=np.int32),
        )
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        prefix = pathlib.Path(temp_dir)
        GPTMemmapDataset.write_dataset(prefix=prefix, documents=documents)
        dataset = GPTMemmapDataset(name="foo", prefix=prefix)
        for i, (document, spans) in enumerate(documents):
            memmap_document, memmap_spans = dataset.get(i)
            assert np.array_equal(
                memmap_document, document, equal_nan=True
            ), f"Mismatch for document {i}: {document} != {dataset.get(i)}."
            if len(spans) > 0:
                assert np.array_equal(
                    memmap_spans, spans, equal_nan=True
                ), f"Mismatch for non-empty spans {i}: {spans} != {dataset.get(i)}."
            else:
                assert np.array_equal(
                    memmap_spans.flatten(), spans.flatten(), equal_nan=True
                ), f"Mismatch for empty spans {i}: {spans} != {dataset.get(i)}."
