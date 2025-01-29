import pathlib
import tempfile

import numpy as np
import pytest

from fast_llm.data.dataset.gpt.memmap import GPTMemmapDataset
from fast_llm.data.dataset.gpt.sampled import GPTSample
from fast_llm.data.preparator.gpt_memmap.config import MEMMAP_DTYPES


@pytest.mark.parametrize("dtype", MEMMAP_DTYPES.values())
def test_gpt_memmap_dataset(dtype):
    documents = [
        GPTSample(text, spans)
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
                memmap_sample.token_ids, document.token_ids, equal_nan=True
            ), f"Mismatch for document {i}: {document.token_ids} != {dataset.get(i)}."
            if len(document.ignore_loss_spans) > 0:
                assert np.array_equal(
                    memmap_sample.ignore_loss_spans, document.ignore_loss_spans, equal_nan=True
                ), f"Mismatch for non-empty spans {i}: {document.ignore_loss_spans} != {dataset.get(i)}."
            else:
                assert np.array_equal(
                    memmap_sample.ignore_loss_spans.flatten(), document.ignore_loss_spans.flatten(), equal_nan=True
                ), f"Mismatch for empty spans {i}: {document.ignore_loss_spans} != {dataset.get(i)}."
