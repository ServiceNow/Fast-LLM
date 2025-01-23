import pathlib
import struct
import typing

import numpy as np

from fast_llm.data.dataset.gpt.indexed import GPTIndexedDataset
from fast_llm.data.preparator.gpt_memmap.config import MEMMAP_DTYPES, MEMMAP_DTYPES_INV, MEMMAP_INDEX_HEADER
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.utils import Assert, div


class GPTMemmapDataset(GPTIndexedDataset):
    """
    A memory map dataset, which handles lazy loading of a pre-processed dataset in the Megatron-LM format,
    i.e. a pair of numpy file containing
    1. A data file (`{prefix}.bin`) containing a flat buffer containing the concatenated, tokenized documents.
    2. An index file (`{prefix}.idx`) containing a list of document sizes and pointers (start index) in the data file.
    See https://github.com/NVIDIA/Megatron-LM?tab=readme-ov-file#data-preprocessing for more details.
    """

    def __init__(self, name: str, prefix: pathlib.Path | str):
        self._init(name, prefix)

    def _init(self, name: str, prefix: pathlib.Path | str) -> None:
        super().__init__()
        self._name = name
        self._prefix = pathlib.Path(prefix)

        with self._prefix.with_suffix(".idx").open("rb") as stream:
            Assert.eq(stream.read(9), MEMMAP_INDEX_HEADER)
            Assert.eq(struct.unpack("<Q", stream.read(8))[0], 1)

            self._dtype = MEMMAP_DTYPES[struct.unpack("<B", stream.read(1))[0]].numpy
            self._num_documents = struct.unpack("<Q", stream.read(8))[0]
            _ = struct.unpack("<Q", stream.read(8))[0]
            offset = stream.tell()

        self._index_bin_buffer_mmap = np.memmap(self._prefix.with_suffix(".idx"), mode="r", order="C")
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

        self._bin_buffer_mmap = np.memmap(self._prefix.with_suffix(".bin"), mode="r", order="C")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def __getstate__(self) -> tuple[str, pathlib.Path]:
        return (self._name, self._prefix)

    def __setstate__(self, state: tuple[str, pathlib.Path]):
        self._init(*state)

    def __del__(self):
        if hasattr(self, "_bin_buffer_mmap"):
            self._bin_buffer_mmap._mmap.close()  # noqa
            del self._bin_buffer_mmap
        if hasattr(self, "_index_bin_buffer"):
            self._index_bin_buffer_mmap._mmap.close()  # noqa
            del self._index_bin_buffer_mmap

    def get(self, idx, offset=0, length=None) -> np.ndarray:
        return np.frombuffer(
            self._bin_buffer,
            dtype=self._dtype,
            count=self._document_sizes[idx] - offset if length is None else length,
            offset=self._pointers[idx] + offset * np.dtype(self._dtype).itemsize,
        )

    @property
    def name(self) -> str:
        return self._name

    def __len__(self) -> int:
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
    def write_dataset(cls, prefix: pathlib.Path | str, documents: typing.Iterable[np.ndarray]):
        # Initialize metadata
        dtype = None
        num_documents = 0
        lengths = []
        pointers = []
        offset = 0

        prefix = pathlib.Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)

        # Write the binary data file (.bin) lazily
        with prefix.with_suffix(".bin").open("wb") as bin_stream:
            for document in documents:
                # Infer dtype from the first document
                if dtype is None:
                    dtype = document.dtype
                    assert dtype is not None, "Document dtype could not be inferred from the data."

                # Ensure all documents have the same dtype
                assert document.dtype == dtype, f"Expected dtype {dtype}, got {document.dtype}."

                # Write document to binary file
                bin_stream.write(document.tobytes(order="C"))

                # Update metadata
                doc_length = len(document)
                lengths.append(doc_length)
                pointers.append(offset)
                offset += doc_length * np.dtype(dtype).itemsize
                num_documents += 1

        # Finalize metadata arrays
        lengths = np.array(lengths, dtype=np.int32)
        pointers = np.array(pointers, dtype=np.int64)

        # Write the index file (.idx)
        with prefix.with_suffix(".idx").open("wb") as idx_stream:
            idx_stream.write(MEMMAP_INDEX_HEADER)
            idx_stream.write(struct.pack("<Q", 1))  # Version
            # Data type
            idx_stream.write(struct.pack("<B", MEMMAP_DTYPES_INV[DataType.from_numpy(dtype.type)]))
            # "Number of sequences", same as documents in our case
            idx_stream.write(struct.pack("<Q", num_documents))
            # "Number of documents", needs a +1 for some reason
            idx_stream.write(struct.pack("<Q", num_documents + 1))
            # Sequence (document) lengths
            idx_stream.write(lengths.tobytes(order="C"))
            # Sequence (document) begin offsets in the bin file
            idx_stream.write(pointers.tobytes(order="C"))
            # Document indices, unused but needed for compatibility with Megatron-LM
            idx_stream.write(np.arange(num_documents + 1, dtype=np.int64).tobytes(order="C"))
