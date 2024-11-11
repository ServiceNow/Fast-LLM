import pathlib
import struct

import numpy as np

from fast_llm.data.gpt.config import GPTMemmapDatasetConfig
from fast_llm.data.gpt.dataset import GPTIndexedDataset
from fast_llm.utils import Assert, div, padded_cumsum


class GPTMemmapDataset(GPTIndexedDataset):
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
    _INDEX_HEADER = b"MMIDIDX\x00\x00"

    def __init__(self, config: GPTMemmapDatasetConfig):
        self._init(config)

    def _init(self, config: GPTMemmapDatasetConfig):
        super().__init__()
        self._config = config

        with self._config.path.with_suffix(".idx").open("rb") as stream:
            Assert.eq(stream.read(9), self._INDEX_HEADER)
            Assert.eq(struct.unpack("<Q", stream.read(8))[0], 1)

            self._dtype = self._DTYPES[struct.unpack("<B", stream.read(1))[0]]
            self._num_documents = struct.unpack("<Q", stream.read(8))[0]
            _ = struct.unpack("<Q", stream.read(8))[0]
            offset = stream.tell()

        self._index_bin_buffer_mmap = np.memmap(self._config.path.with_suffix(".idx"), mode="r", order="C")
        self._index_bin_buffer = memoryview(self._index_bin_buffer_mmap)
        self._document_sizes = np.frombuffer(
            self._index_bin_buffer, dtype=np.int32, count=self._num_documents, offset=offset
        )
        self._pointers = np.frombuffer(
            self._index_bin_buffer,
            dtype=np.int64,
            count=self._num_documents,
            offset=offset + self._document_sizes.nbytes,
        )
        self._bin_buffer_mmap = np.memmap(self._config.path.with_suffix(".bin"), mode="r", order="C")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def __getstate__(self):
        return self._config.to_serialized()

    def __setstate__(self, state):
        self._init(GPTMemmapDatasetConfig.from_dict(state))

    def __del__(self):
        self._bin_buffer_mmap._mmap.close()  # noqa
        del self._bin_buffer_mmap
        self._index_bin_buffer_mmap._mmap.close()  # noqa
        del self._index_bin_buffer_mmap

    def __len__(self):
        return self._num_documents

    def get(self, document: int, offset: int = 0, length: int | None = None):
        return np.frombuffer(
            self._bin_buffer,
            dtype=self._dtype,
            count=self._document_sizes[document] - offset if length is None else length,
            offset=self._pointers[document] + offset * np.dtype(self._dtype).itemsize,
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def num_documents(self) -> int:
        return self._num_documents

    @property
    def num_tokens(self) -> int:
        return div(self._bin_buffer_mmap.size, np.dtype(self._dtype).itemsize)

    def get_document_sizes(self) -> np.ndarray:
        """
        The size of each document in the dataset.
        The resulting array could be very large, so this method should be called cautiously,
        and derived classes should try to avoid holding the whole array im memory.
        """
        return self._document_sizes

    @classmethod
    def write_dataset(cls, prefix: pathlib.Path | str, documents: list[np.ndarray]):
        # Write index and binary files.
        dtype = documents[0].dtype
        num_documents = len(documents)
        lengths = np.array([len(document) for document in documents], dtype=np.int32)
        pointers = padded_cumsum(lengths[:-1].astype(np.int64) * 2)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        with prefix.with_suffix(".idx").open("wb") as stream:
            stream.write(cls._INDEX_HEADER)
            stream.write(struct.pack("<Q", 1))
            # Data type
            stream.write(struct.pack("<B", {y: x for x, y in cls._DTYPES.items()}[dtype.type]))
            # "Number of sequences", same as documents in our case.
            stream.write(struct.pack("<Q", num_documents))
            # "Number of documents", needs a +1 for some reason.
            stream.write(struct.pack("<Q", num_documents + 1))
            # Sequence (document) lengths
            stream.write(lengths.tobytes(order="C"))
            # Sequence (document) begin offsets in the bin file
            stream.write(pointers.tobytes(order="C"))
            # Document indices, unused but needed for compatibility with Megatron-LM
            stream.write(np.arange(num_documents + 1, dtype=np.int64).tobytes(order="C"))

        with prefix.with_suffix(".bin").open("wb") as stream:
            for document in documents:
                assert document.dtype == dtype
                stream.write(document.tobytes(order="C"))
