import dataclasses
import pathlib
import struct
import typing

import numpy as np

from fast_llm.data.dataset.gpt.indexed import GPTIndexedDataset
from fast_llm.data.preparator.gpt_memmap.config import MEMMAP_DTYPES, MEMMAP_DTYPES_INV, MEMMAP_INDEX_HEADER
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.utils import Assert, div


@dataclasses.dataclass
class GPTMemmapDocument:
    text: np.ndarray
    spans: np.ndarray


@dataclasses.dataclass
class GPTMemmapSample:
    ids: np.ndarray
    spans: np.ndarray


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
            self._version = struct.unpack("<Q", stream.read(8))[0]

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

        # Spans are introduced in version 2. Datasets tokenized with version 1 do not contain span information and
        # compute loss on all tokens by default
        if self._version == 1:
            self._num_spans = np.zeros(self._num_documents, dtype=np.int32)
            self._spans = [np.array([], dtype=np.int32).reshape(-1, 2)] * self._num_documents
        elif self._version == 2:
            self._num_spans = np.frombuffer(
                self._index_bin_buffer,
                dtype=np.int32,
                count=self._num_documents,
                offset=offset + self._document_sizes.nbytes + self._pointers.nbytes,
            )
            spans = []
            offset = offset + self._document_sizes.nbytes + self._pointers.nbytes + self._num_spans.nbytes
            for n_spans in self._num_spans:
                span = np.frombuffer(
                    self._index_bin_buffer,
                    dtype=np.int32,
                    count=n_spans * 2,
                    offset=offset,
                ).reshape(-1, 2)
                spans.append(span)
                offset += span.nbytes
            self._spans = spans
        else:
            raise ValueError(f"Unsupported version for gpt_memmap dataset: {self._version}.")

        self._bin_buffer_mmap = np.memmap(self._prefix.with_suffix(".bin"), mode="r", order="C")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def __getstate__(self) -> tuple[str, pathlib.Path]:
        return (self._name, self._prefix)

    def __setstate__(self, state: tuple[str, pathlib.Path]):
        self._init(*state)

    def __del__(self):
        self._bin_buffer_mmap._mmap.close()  # noqa
        del self._bin_buffer_mmap
        self._index_bin_buffer_mmap._mmap.close()  # noqa
        del self._index_bin_buffer_mmap

    def get(self, idx, offset=0, length=None) -> GPTMemmapSample:
        ids = np.frombuffer(
            self._bin_buffer,
            dtype=self._dtype,
            count=self._document_sizes[idx] - offset if length is None else length,
            offset=self._pointers[idx] + offset * np.dtype(self._dtype).itemsize,
        )
        spans = []
        for span in self._spans[idx]:
            if span[0] < offset + len(ids) and span[1] >= offset:
                spans.append([max(span[0], offset) - offset, min(span[1], offset + len(ids) - 1) - offset])
        # return (ids, np.array(spans, dtype=np.int32).reshape(-1, 2))
        return GPTMemmapSample(ids=ids, spans=np.array(spans, dtype=np.int32).reshape(-1, 2))

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
    def write_dataset(cls, prefix: pathlib.Path | str, documents: typing.Iterable[GPTMemmapDocument]):
        # Initialize metadata
        dtype = None
        num_documents = 0
        lengths = []
        pointers = []
        offset = 0
        # number of spans for each document
        num_spans = []
        spans = []

        prefix = pathlib.Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)

        # Write the binary data file (.bin) lazily
        with prefix.with_suffix(".bin").open("wb") as bin_stream:
            for document in documents:
                # Infer dtype from the first document
                if dtype is None:
                    dtype = document.text.dtype
                    assert dtype is not None, "Document dtype could not be inferred from the data."

                # Ensure all documents have the same dtype
                assert document.text.dtype == dtype, f"Expected dtype {dtype}, got {document.text.dtype}."

                # Write document to binary file
                bin_stream.write(document.text.tobytes(order="C"))

                # Update metadata
                doc_length = len(document.text)
                lengths.append(doc_length)
                pointers.append(offset)
                num_spans.append(len(document.spans))
                spans.append(document.spans)
                offset += doc_length * np.dtype(dtype).itemsize
                num_documents += 1

        # Finalize metadata arrays
        lengths = np.array(lengths, dtype=np.int32)
        pointers = np.array(pointers, dtype=np.int64)
        num_spans = np.array(num_spans, dtype=np.int32)
        spans = np.vstack(spans, dtype=np.int32)

        # Write the index file (.idx)
        with prefix.with_suffix(".idx").open("wb") as idx_stream:
            idx_stream.write(MEMMAP_INDEX_HEADER)
            # Indicates the version
            # Version 2 adds number of spans and spans to the index file.
            idx_stream.write(struct.pack("<Q", 2))
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
            # Number of spans per document
            idx_stream.write(num_spans.tobytes(order="C"))
            # Span indices for each document
            idx_stream.write(spans.tobytes(order="C"))
            # Document indices, unused but needed for compatibility with Megatron-LM
            idx_stream.write(np.arange(num_documents + 1, dtype=np.int64).tobytes(order="C"))
