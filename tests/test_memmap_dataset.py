from hypothesis import given, strategies as st
from hypothesis.extra import numpy as npst
import numpy as np
from tempfile import TemporaryDirectory
from pathlib import Path
from fast_llm.data.gpt.memmap import GPTMemmapDataset
import pytest

def dtype_arrays(dtype: np.dtype, min_size: int=1, max_size: int=100) -> st.SearchStrategy:
    return st.lists(
        npst.arrays(dtype=dtype, shape=st.integers(1, 1000)),
        min_size=min_size,
        max_size=max_size,
    )

@pytest.mark.parametrize("dtype", GPTMemmapDataset._DTYPES.values())
def test_gpt_memmap_dataset(dtype):
    @given(documents=dtype_arrays(dtype))
    def inner_test(documents):
        with TemporaryDirectory() as temp_dir:
            prefix = Path(temp_dir)
            GPTMemmapDataset.write_dataset(prefix=prefix, documents=documents)
            dataset = GPTMemmapDataset(name="foo", prefix=prefix)
            for i, document in enumerate(documents):
                assert np.array_equal(dataset.get(i), document, equal_nan=True), f"Mismatch at index {i}"
    
    inner_test()
