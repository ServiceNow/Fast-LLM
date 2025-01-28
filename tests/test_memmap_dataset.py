import pathlib
import tempfile

import numpy as np
import pytest

from fast_llm.data.dataset.gpt.memmap import GPTMemmapDataset, GPTMemmapDocument
from fast_llm.data.preparator.gpt_memmap.config import MEMMAP_DTYPES


@pytest.mark.parametrize("dtype", MEMMAP_DTYPES.values())
def test_gpt_memmap_dataset(dtype):
    documents = [
        GPTMemmapDocument(text, spans)
        for text, spans in zip(
            [np.random.randint(1000, size=np.random.randint(1, 100)).astype(dtype) for _ in range(100)],
            np.array([[]] * 100, dtype=np.int32),
        )
    ]
    with tempfile.TemporaryDirectory() as temp_dir:
        prefix = pathlib.Path(temp_dir)
        GPTMemmapDataset.write_dataset(prefix=prefix, documents=documents)
        dataset = GPTMemmapDataset(name="foo", prefix=prefix)
        for i, document in enumerate(documents):
            memmap_sample = dataset.get(i)
            assert np.array_equal(
                memmap_sample.ids, document.text, equal_nan=True
            ), f"Mismatch for document {i}: {document.text} != {dataset.get(i)}."
            if len(document.spans) > 0:
                assert np.array_equal(
                    memmap_sample.spans, document.spans, equal_nan=True
                ), f"Mismatch for non-empty spans {i}: {document.spans} != {dataset.get(i)}."
            else:
                assert np.array_equal(
                    memmap_sample.spans.flatten(), document.spans.flatten(), equal_nan=True
                ), f"Mismatch for empty spans {i}: {document.spans} != {dataset.get(i)}."
