import io
import typing

import numpy as np

from fast_llm.data.dataset.gpt.components.config import GPTMemmapDatasetHeader
from fast_llm.data.dataset.gpt.memmap import BufferOffset, ShiftMap
from fast_llm.data.dataset.gpt.sampled import GPTSample
from fast_llm.utils import Assert


class GPTTokensDatasetComponent:
    def __init__(
        self,
        header: GPTMemmapDatasetHeader,
        index_binary_buffer: memoryview,
        binary_buffer: memoryview,
        offset: BufferOffset,
    ):
        self._header = header
        self._index_binary_buffer = index_binary_buffer
        self._binary_buffer = binary_buffer
        self.sizes = np.frombuffer(
            self._index_binary_buffer, dtype=np.int32, count=self._header.num_documents, offset=offset.value
        )
        self._item_size = self._header.token_data_type.numpy.itemsize
        offset.value += self.sizes.nbytes

    def get(
        self, index: int, start_offset: int, end_offset: int, shift_map: ShiftMap, buffer_offset: BufferOffset
    ) -> np.ndarray:
        unshifted_start_offset = shift_map.unshift(start_offset)
        token_ids = np.frombuffer(
            self._binary_buffer,
            dtype=self._header.token_data_type,
            count=shift_map.unshift(end_offset) - unshifted_start_offset,
            offset=buffer_offset.value + unshifted_start_offset * self._item_size,
        )
        buffer_offset.value += self.sizes[index] * self._item_size
        return token_ids

    @classmethod
    def write_document_and_gather_index(
        cls, document: GPTSample, index_data: dict[str, typing.Any], binary_stream: io.BufferedWriter
    ):
        if "token_data_type" in index_data:
            Assert.eq(document.token_ids.dtype, index_data["token_data_type"])
        else:
            index_data["token_data_type"] = document.token_ids.dtype
        if "document_lengths" not in index_data:
            index_data["document_lengths"] = []
        index_data["document_lengths"].append(document_length := len(document.token_ids))
        if "num_tokens" not in index_data:
            index_data["num_tokens"] = 0
        index_data["num_tokens"] += document_length

        # Write document to binary file
        binary_stream.write(document.token_ids.tobytes(order="C"))

    @classmethod
    def write_index(self, index_data: dict[str, typing.Any], index_stream: io.BufferedWriter):
        # Document (tokens) lengths.
        index_stream.write(np.array(index_data["document_lengths"], dtype=np.int32).tobytes(order="C"))
