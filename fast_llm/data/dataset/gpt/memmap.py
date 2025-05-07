import pathlib
import struct
import typing

import numpy as np

from fast_llm.data.dataset.gpt.indexed import GPTIndexedDataset
from fast_llm.data.dataset.gpt.sampled import GPTSample
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

    def __init__(
        self,
        name: str,
        prefix: pathlib.Path | str,
        num_documents: int | None = None,
        num_tokens: int | None = None,
    ):
        self._init(name, prefix, num_documents, num_tokens)

    def _init(self, name: str, prefix: pathlib.Path | str, num_documents: int | None, num_tokens: int | None) -> None:
        super().__init__()
        self._name = name
        self._prefix = pathlib.Path(prefix)
        self._has_spans = 0
        self._has_preference_spans = False

        with self._prefix.with_suffix(".idx").open("rb") as stream:
            Assert.eq(stream.read(9), MEMMAP_INDEX_HEADER, msg=f"File: {stream.name}")
            self._version = struct.unpack("<Q", stream.read(8))[0]
            assert self._version in [1, 2, 3], f"Unsupported version for gpt_memmap dataset: {self._version}."
            if self._version >= 2:
                self._has_spans = struct.unpack("<B", stream.read(1))[0]
            if self._version >= 3:
                self._has_preference_spans = struct.unpack("<B", stream.read(1))[0]

            self._dtype = MEMMAP_DTYPES[struct.unpack("<B", stream.read(1))[0]].numpy
            self._num_documents = struct.unpack("<Q", stream.read(8))[0]
            _ = struct.unpack("<Q", stream.read(8))[0]
            offset = stream.tell()

        if num_documents is not None:
            assert self._num_documents == num_documents

        self._index_bin_buffer_mmap = np.memmap(self._prefix.with_suffix(".idx"), mode="r", order="C")
        self._index_bin_buffer = memoryview(self._index_bin_buffer_mmap)

        # read document sizes
        self._document_sizes = np.frombuffer(
            self._index_bin_buffer, dtype=np.int32, count=self._num_documents, offset=offset
        )

        # read pointers
        self._pointers = np.frombuffer(
            self._index_bin_buffer,
            dtype=np.int64,
            count=self._num_documents,
            offset=offset + self._document_sizes.nbytes,
        )

        # read spans
        self._spans = None
        if self._has_spans and self._version >= 2:
            self._spans = []
            self._num_spans = np.frombuffer(
                self._index_bin_buffer,
                dtype=np.int32,
                count=self._num_documents,
                offset=offset + self._document_sizes.nbytes + self._pointers.nbytes,
            )
            span_offset = offset + self._document_sizes.nbytes + self._pointers.nbytes + self._num_spans.nbytes
            self._num_spans_cumsum = np.r_[0, np.cumsum(self._num_spans[:-1], dtype=np.int64)]
            for idx in range(self._num_documents):
                self._spans.append(
                    np.frombuffer(
                        self._index_bin_buffer,
                        dtype=np.int32,
                        count=self._num_spans[idx] * 2,
                        offset=span_offset + self._num_spans_cumsum[idx] * 2 * np.dtype(np.int32).itemsize,
                    ).reshape(-1, 2)
                )

        # read preference spans
        self._chosen_spans = None
        self._rejected_spans = None
        if self._has_preference_spans and self._version >= 3:
            self._chosen_spans = []
            self._rejected_spans = []
            chosen_span_offset = offset + self._document_sizes.nbytes + self._pointers.nbytes
            for idx in range(self._num_documents):
                self._chosen_spans.append(
                    np.frombuffer(
                        self._index_bin_buffer,
                        dtype=np.int32,
                        count=2,
                        offset=chosen_span_offset + idx * 2 * np.dtype(np.int32).itemsize,
                    )
                )

            rejected_span_offset = (
                offset + self._document_sizes.nbytes + self._pointers.nbytes + np.array(self._chosen_spans).nbytes
            )
            for idx in range(self._num_documents):
                self._rejected_spans.append(
                    np.frombuffer(
                        self._index_bin_buffer,
                        dtype=np.int32,
                        count=2,
                        offset=rejected_span_offset + idx * 2 * np.dtype(np.int32).itemsize,
                    )
                )

        self._bin_buffer_mmap = np.memmap(self._prefix.with_suffix(".bin"), mode="r", order="C")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

        self._num_tokens = div(self._bin_buffer_mmap.size, np.dtype(self._dtype).itemsize)
        if num_tokens is not None:
            assert self._num_tokens == num_tokens

    def __getstate__(self) -> tuple[str, pathlib.Path, int | None, int | None]:
        return (self._name, self._prefix, self._num_documents, self._num_tokens)

    def __setstate__(self, state: tuple[str, pathlib.Path, int | None, int | None]):
        self._init(*state)

    def __del__(self):
        if hasattr(self, "_bin_buffer_mmap"):
            self._bin_buffer_mmap._mmap.close()  # noqa
            del self._bin_buffer_mmap
        if hasattr(self, "_index_bin_buffer"):
            self._index_bin_buffer_mmap._mmap.close()  # noqa
            del self._index_bin_buffer_mmap

    def get(
        self,
        idx: int,
        offset: int = 0,
        length: int | None = None,
        use_loss_masking_spans: bool = False,
        use_preference_loss_spans: bool = False,
    ) -> GPTSample:
        token_ids = np.frombuffer(
            self._bin_buffer,
            dtype=self._dtype,
            count=self._document_sizes[idx] - offset if length is None else length,
            offset=self._pointers[idx] + offset * np.dtype(self._dtype).itemsize,
        )
        sample_spans = None
        if use_loss_masking_spans and self._spans is not None:
            sample_spans = self._spans[idx]

            # filter spans that are outside the range of the selected tokens in the document
            sample_spans = sample_spans[
                (sample_spans[:, 0] < offset + len(token_ids)) & (sample_spans[:, 1] >= offset)
            ]

            # subtract by offset to normalize span boundaries
            sample_spans[:, 0] = np.maximum(sample_spans[:, 0], offset) - offset  # offset
            sample_spans[:, 1] = np.minimum(sample_spans[:, 1], offset + len(token_ids) - 1) - offset

        chosen_span = None
        rejected_span = None

        if use_preference_loss_spans:
            if not self._has_preference_spans:
                raise ValueError("No preference spans found in memmap dataset.")
            elif self._has_preference_spans and self._chosen_spans is None:
                raise ValueError("Failed to read chosen spans from memmap dataset.")
            elif self._has_preference_spans and self._rejected_spans is None:
                raise ValueError("Failed to read rejected spans from memmap dataset.")
            else:
                chosen_span = self._chosen_spans[idx]

                # filter spans that are outside the range of the selected tokens in the document
                chosen_span = chosen_span[(chosen_span[0] < offset + len(token_ids)) & (chosen_span[1] >= offset)][0]

                # subtract by offset to normalize span boundaries
                chosen_span[0] = np.maximum(chosen_span[0], offset) - offset  # offset
                chosen_span[1] = np.minimum(chosen_span[1], offset + len(token_ids) - 1) - offset

                rejected_span = self._rejected_spans[idx]

                # filter spans that are outside the range of the selected tokens in the document
                rejected_span = rejected_span[
                    (rejected_span[0] < offset + len(token_ids)) & (rejected_span[1] >= offset)
                ][0]

                # subtract by offset to normalize span boundaries
                rejected_span[0] = np.maximum(rejected_span[0], offset) - offset  # offset
                rejected_span[1] = np.minimum(rejected_span[1], offset + len(token_ids) - 1) - offset

        return GPTSample(
            token_ids=token_ids,
            loss_masking_spans=sample_spans,
            chosen_span=chosen_span,
            rejected_span=rejected_span,
        )

    @property
    def name(self) -> str:
        return self._name

    def __len__(self) -> int:
        return self._num_documents

    @property
    def num_tokens(self) -> int:
        return self._num_tokens

    def get_document_sizes(self) -> np.ndarray:
        """
        The size of each document in the dataset.
        The resulting array could be very large, so this method should be called cautiously,
        and derived classes should try to avoid holding the whole array im memory.
        """
        return self._document_sizes

    def get_document_size(self, index: int) -> int:
        return self._document_sizes[index].item()

    @classmethod
    def write_dataset(cls, prefix: pathlib.Path | str, documents: typing.Iterable[GPTSample]):
        # Initialize metadata
        dtype = None
        num_documents = 0
        lengths = []
        pointers = []
        offset = 0
        # number of spans for each document
        num_spans = []
        spans = []
        chosen_spans = []
        rejected_spans = []

        prefix = pathlib.Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)

        # Write the binary data file (.bin) lazily
        with prefix.with_suffix(".bin").open("wb") as bin_stream:
            for document in documents:
                # Infer dtype from the first document
                if dtype is None:
                    dtype = document.token_ids.dtype
                    assert dtype is not None, "Document dtype could not be inferred from the data."

                # Ensure all documents have the same dtype
                assert document.token_ids.dtype == dtype, f"Expected dtype {dtype}, got {document.token_ids.dtype}."

                # Write document to binary file
                bin_stream.write(document.token_ids.tobytes(order="C"))

                # Update metadata
                doc_length = len(document.token_ids)
                lengths.append(doc_length)
                pointers.append(offset)
                if document.loss_masking_spans is not None:
                    num_spans.append(len(document.loss_masking_spans))
                    spans.append(document.loss_masking_spans)
                if document.chosen_span is not None:
                    chosen_spans.append(document.chosen_span)
                if document.rejected_span is not None:
                    rejected_spans.append(document.rejected_span)
                offset += doc_length * np.dtype(dtype).itemsize
                num_documents += 1

        # Finalize metadata arrays
        lengths = np.array(lengths, dtype=np.int32)
        pointers = np.array(pointers, dtype=np.int64)
        num_spans = np.array(num_spans, dtype=np.int32)
        if len(spans) > 0:
            spans = np.vstack(spans, dtype=np.int32)
        else:
            spans = np.array(spans, dtype=np.int32)
        chosen_spans = np.array(chosen_spans, dtype=np.int32).reshape(-1, 2)
        rejected_spans = np.array(rejected_spans, dtype=np.int32).reshape(-1, 2)

        # Write the index file (.idx)
        with prefix.with_suffix(".idx").open("wb") as idx_stream:
            idx_stream.write(MEMMAP_INDEX_HEADER)
            # Indicates the version
            # Version 2 optionally adds loss-masking spans
            # Version 3 optionally adds chosen/rejected spans
            idx_stream.write(struct.pack("<Q", 3))
            # Flag to indicate whether loss-masking spans are present
            idx_stream.write(struct.pack("<B", 1 if spans.size > 0 else 0))
            # Flag to indicate whether preference loss-masking spans are present
            idx_stream.write(struct.pack("<B", 1 if chosen_spans.size > 0 and rejected_spans.size > 0 else 0))
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
            # Chosen indices for each document
            idx_stream.write(chosen_spans.tobytes(order="C"))
            # Rejected indices for each document
            idx_stream.write(rejected_spans.tobytes(order="C"))
            # Document indices, unused but needed for compatibility with Megatron-LM
            idx_stream.write(np.arange(num_documents + 1, dtype=np.int64).tobytes(order="C"))
