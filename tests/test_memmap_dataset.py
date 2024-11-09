from hypothesis import given, strategies as st
from hypothesis.extra import numpy as npst
import numpy as np
from tempfile import TemporaryDirectory
from pathlib import Path
from fast_llm.data.gpt.memmap import GPTMemmapDataset

def dtype_arrays(dtype: type[np.integer], min_size: int=1, max_size: int=100) -> st.SearchStrategy:
    return st.lists(
        npst.arrays(
            dtype=dtype,
            shape=st.integers(1, 1000),
            elements=st.integers(
                min_value=np.iinfo(dtype).min,
                max_value=np.iinfo(dtype).max,
            ),
        ),
        min_size=min_size,
        max_size=max_size,
    )

for dtype in [np.int8, np.uint16, np.int16, np.int32, np.int64]:
    @given(arrays=dtype_arrays(dtype))
    def test_gpt_memmap_dataset(arrays: list[np.ndarray]):
        run_gpt_memmap_dataset_test(documents=arrays)

def run_gpt_memmap_dataset_test(documents: list[np.ndarray]) -> None:
    with TemporaryDirectory() as temp_dir:
        prefix = Path(temp_dir)
        GPTMemmapDataset.write_dataset(prefix=prefix, documents=documents)
        dataset = GPTMemmapDataset(name="foo", prefix=prefix)
        for i, document in enumerate(documents):
            assert np.array_equal(dataset.get(i), document), f"Mismatch at index {i}"
