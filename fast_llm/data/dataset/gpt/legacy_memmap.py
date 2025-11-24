import pathlib
import struct

import numpy as np
import torch

from fast_llm.data.dataset.gpt.config import GPTSamplingParameters
from fast_llm.data.dataset.indexed import IndexedDataset
from fast_llm.data.sample.language_model import LanguageModelSample
from fast_llm.data.sample.range import RangeSample
from fast_llm.data.sample.token import TokenSample
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.utils import Assert, div

MEMMAP_DTYPES = {
    1: DataType.uint8,
    2: DataType.int8,
    3: DataType.int16,
    4: DataType.int32,
    5: DataType.int64,
    6: DataType.float32,
    7: DataType.float64,
    8: DataType.uint16,
}
MEMMAP_INDEX_HEADER = b"MMIDIDX\x00\x00"


class LegacyMemmapDataset[SampleType: LanguageModelSample](IndexedDataset[SampleType]):
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
    ):
        self._init(name, prefix)

    def _init(self, name: str, prefix: pathlib.Path | str) -> None:
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

            self._dtype = MEMMAP_DTYPES[struct.unpack("<B", stream.read(1))[0]].torch
            self._num_documents = struct.unpack("<Q", stream.read(8))[0]
            _ = struct.unpack("<Q", stream.read(8))[0]
            offset = stream.tell()

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

        self._num_tokens = div(self._bin_buffer_mmap.size, self._dtype.itemsize)

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

    def get_document(
        self, index: int, begin: int = 0, end: int | None = None, parameters: GPTSamplingParameters | None = None
    ) -> SampleType:
        if end is None:
            end = self.get_document_size(index)
        sample_size = self._document_sizes[index].item()
        assert 0 <= begin <= end <= sample_size, (0, begin, end, sample_size)
        token_ids = (
            torch.frombuffer(
                self._bin_buffer,
                dtype=self._dtype,
                count=end - begin,
                offset=self._pointers[index].item() + begin * self._dtype.itemsize,
            )
            if end > begin
            else torch.empty(0, dtype=self._dtype)
        )
        if not self._dtype.is_signed:
            # Needed because torch doesn't yet support type promotion between signed and unsigned types. TODO: Remove when supported.
            token_ids = token_ids.to(torch.int64)
        if parameters is not None and parameters.use_loss_masking_spans:
            assert self._spans is not None
            # Convert to in range format (begin, end).
            sample_spans = RangeSample(
                [(begin_, last_ + 1) for begin_, last_ in self._spans[index].tolist()], sample_size
            ).crop(begin, end)
        else:
            sample_spans = None

        if parameters is not None and parameters.use_preference_loss_spans:
            if not self._has_preference_spans:
                raise ValueError("No preference spans found in memmap dataset.")
            elif self._has_preference_spans and self._chosen_spans is None:
                raise ValueError("Failed to read chosen spans from memmap dataset.")
            elif self._has_preference_spans and self._rejected_spans is None:
                raise ValueError("Failed to read rejected spans from memmap dataset.")
            # Convert to in range format (begin, end).
            chosen_spans = RangeSample(
                [(self._chosen_spans[index][0].item(), self._chosen_spans[index][1].item() + 1)],
                sample_size,
            ).crop(begin, end)
            rejected_spans = RangeSample(
                [(self._rejected_spans[index][0].item(), self._rejected_spans[index][1].item() + 1)],
                sample_size,
            ).crop(begin, end)
        else:
            chosen_spans = rejected_spans = None

        return LanguageModelSample(
            tokens=TokenSample(token_ids),
            loss_masking_spans=sample_spans,
            chosen_spans=chosen_spans,
            rejected_spans=rejected_spans,
        )

    @property
    def name(self) -> str:
        return self._name

    def __len__(self) -> int:
        return self._num_documents

    @property
    def num_tokens(self) -> int:
        return self._num_tokens

    def get_document_sizes(self) -> torch.Tensor:
        """
        The size of each document in the dataset.
        The resulting array could be very large, so this method should be called cautiously,
        and derived classes should try to avoid holding the whole array im memory.
        """
        return torch.from_numpy(self._document_sizes)

    def get_document_size(self, index: int) -> int:
        return self._document_sizes[index].item()
