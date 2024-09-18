import pathlib
import struct

import numpy as np

from fast_llm.utils import Assert, div


class MMapIndexedDataset:
    """
    A memory map dataset, which handles lazy loading of a pre-processed dataset in the Megatron-LM format,
    i.e. a pair of numpy file containing
    1. A data file (`{prefix}.bin`) containing a flat buffer containing the concatenated, tokenized documents.
    2. An index file (`{prefix}.idx`) containing a list of document sizes and pointers (start index) in the data file.
    See https://github.com/NVIDIA/Megatron-LM?tab=readme-ov-file#data-preprocessing for more details.
    """

    _DTYPES = {
        1: np.uint8,
        2: np.int8,
        3: np.int16,
        4: np.int32,
        5: np.int64,
        6: np.float32,
        7: np.float64,
        8: np.uint16,
    }
    _HDR_MAGIC = b"MMIDIDX\x00\x00"

    def __init__(self, prefix: pathlib.Path | str):
        self._init(prefix)

    def _init(self, prefix: pathlib.Path | str):
        super().__init__()
        self._prefix = pathlib.Path(prefix)

        with self._prefix.with_suffix(".idx").open("rb") as stream:
            Assert.eq(stream.read(9), self._HDR_MAGIC)
            Assert.eq(struct.unpack("<Q", stream.read(8))[0], 1)

            self._dtype = self._DTYPES[struct.unpack("<B", stream.read(1))[0]]
            self._num_documents = struct.unpack("<Q", stream.read(8))[0]
            self._doc_count = struct.unpack("<Q", stream.read(8))[0]
            offset = stream.tell()

        self._index_bin_buffer_mmap = np.memmap(self._prefix.with_suffix(".idx"), mode="r", order="C")
        self._index_bin_buffer = memoryview(self._index_bin_buffer_mmap)
        self.sizes = np.frombuffer(self._index_bin_buffer, dtype=np.int32, count=self._num_documents, offset=offset)
        self._pointers = np.frombuffer(
            self._index_bin_buffer, dtype=np.int64, count=self._num_documents, offset=offset + self.sizes.nbytes
        )
        self._doc_idx = np.frombuffer(
            self._index_bin_buffer,
            dtype=np.int64,
            count=self._doc_count,
            offset=offset + self.sizes.nbytes + self._pointers.nbytes,
        )

        self._bin_buffer_mmap = np.memmap(self._prefix.with_suffix(".bin"), mode="r", order="C")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def __getstate__(self):
        return self.prefix

    def __setstate__(self, state):
        self._init(state)

    def __del__(self):
        self._bin_buffer_mmap._mmap.close()  # noqa
        del self._bin_buffer_mmap
        self._index_bin_buffer_mmap._mmap.close()  # noqa
        del self._index_bin_buffer_mmap

    def __len__(self):
        return self._num_documents

    def get(self, idx, offset=0, length=None):
        return np.frombuffer(
            self._bin_buffer,
            dtype=self._dtype,
            count=self.sizes[idx] - offset if length is None else length,
            offset=self._pointers[idx] + offset * np.dtype(self._dtype).itemsize,
        )

    @property
    def num_documents(self):
        return self._num_documents

    @property
    def num_tokens(self):
        return div(self._bin_buffer_mmap.size, np.dtype(self._dtype).itemsize)

    @property
    def prefix(self):
        return self._prefix
