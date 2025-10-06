import io
import typing

import numpy as np

from fast_llm.data.dataset.gpt.components.config import GPTMemmapDatasetHeader
from fast_llm.data.dataset.gpt.memmap import BufferOffset, ShiftMap
from fast_llm.data.dataset.gpt.sampled import GPTSample
from fast_llm.utils import Assert


class GPTSpansDatasetComponent:
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

        self._count_cumsum = np.frombuffer(
            self._index_binary_buffer,
            dtype=np.int32,
            count=self._header.num_documents + 1,
            offset=offset.value,
        )
        offset.value += self._count_cumsum.nbytes
        self._spans = np.frombuffer(
            self._index_binary_buffer,
            dtype=np.int32,
            count=self._count_cumsum[-1] * 2,
            offset=offset.value,
        ).reshape(-1, 2)
        offset.value += self._spans.nbytes

    def get(self, index: int, start_offset: int, end_offset: int, shift_map: ShiftMap) -> list[tuple[int, int]]:
        loss_masking_spans = []
        for span_begin, span_end in self._spans[self._count_cumsum[index] : self._count_cumsum[index + 1]].tolist():
            span_begin = max(shift_map.shift(span_begin), start_offset) - start_offset
            span_end = min(shift_map.shift(span_end), end_offset - 1) - start_offset
            if span_end > span_begin:
                loss_masking_spans.append((span_begin, span_end))
        return loss_masking_spans

    @classmethod
    def write_document_and_gather_index(
        cls, document: GPTSample, index_data: dict[str, typing.Any], binary_stream: io.BufferedWriter
    ):
        has_spans = document.loss_masking_spans is not None
        if "has_span" in index_data:
            Assert.eq(index_data["has_span"], has_spans)
        else:
            index_data["has_span"] = has_spans
        if has_spans:
            if "spans" not in index_data:
                index_data["spans"] = []
            index_data["spans"].extend(document.loss_masking_spans)
            if "spans_cumsum" not in index_data:
                index_data["spans_cumsum"] = [0]
            index_data["spans_cumsum"].append(len(index_data["spans"]))

    @classmethod
    def write_index(self, index_data: dict[str, typing.Any], index_stream: io.BufferedWriter):
        if index_data["has_spans"]:
            # Should be ok, checking just in case.
            Assert.leq(index_data["spans_cumsum"][-1], np.iinfo(np.int32).max)
            Assert.eq(len(index_data["spans_cumsum"]), index_data["num_documents"] + 1)
            Assert.eq(len(index_data["spans"]), index_data["spans_cumsum"][-1])
            index_stream.write(np.array(index_data["spans_cumsum"], dtype=np.int32).tobytes(order="C"))
            index_stream.write(np.vstack(index_data["spans"], dtype=np.int32).tobytes(order="C"))
