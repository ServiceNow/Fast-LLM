import typing

import numpy as np
import torch

from fast_llm.data.dataset.memmap.abstract import MemmapReader, MemmapWriter
from fast_llm.data.dataset.memmap.config import RangeReaderConfig
from fast_llm.data.document.abstract import Document
from fast_llm.data.document.range import RangeDocument
from fast_llm.data.preprocessing.abstract import PreprocessingConfig
from fast_llm.utils import Assert


class RangeReader[ConfigType: RangeReaderConfig](MemmapReader[ConfigType]):
    def __init__(self, config: ConfigType, buffer: memoryview, model_preprocessing: PreprocessingConfig | None = None):
        super().__init__(config, buffer, model_preprocessing)
        self._ranges = torch.frombuffer(
            self._buffer,
            dtype=torch.int32,
            count=self._config.num_ranges * 2,
        ).view(-1, 2)
        self._count_cumsums = torch.frombuffer(
            self._buffer,
            dtype=torch.int32,
            count=self._config.num_documents + 1,
            offset=self._ranges.nbytes,
        )

    def get_document(self, index: int, begin: int, end: int) -> Document:
        sample_size = end - begin
        cropped_ranges = (
            (max(begin_ - begin, 0), min(end_ - begin, sample_size))
            for begin_, end_ in self._ranges[self._count_cumsums[index] : self._count_cumsums[index + 1]].tolist()
        )
        return RangeDocument(ranges=[(begin_, end_) for begin_, end_ in cropped_ranges if end_ > begin_])

    def get_split(self, begin_index: int, end_index: int) -> dict[str, typing.Any]:
        Assert.custom(lambda x: x == sorted(x), [0, begin_index, end_index, self._config.num_documents])
        return {
            "num_documents": end_index - begin_index,
            "num_ranges": self._count_cumsums[end_index].item() - self._count_cumsums[begin_index].item(),
        }


class RangeWriter(MemmapWriter):
    def __enter__(self):
        super().__enter__()
        self._count_cumsum = [0]
        return self

    def write(self, document: RangeDocument):
        super().write(document)
        self._stream.write(np.array(document.ranges, dtype=np.int32).tobytes(order="C"))
        self._count_cumsum.append(self._count_cumsum[-1] + len(document.ranges))

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            Assert.lt(self._count_cumsum[-1], np.iinfo(np.int32).max)
            self._stream.write(np.array(self._count_cumsum, dtype=np.int32).tobytes(order="C"))
        super().__exit__(exc_type, exc_val, exc_tb)

    @classmethod
    def _get_config_class(cls) -> type[RangeReaderConfig]:
        return RangeReaderConfig

    def _get_config(self, begin: int, end: int):
        return RangeReaderConfig(
            begin=begin,
            end=end,
            num_documents=len(self._count_cumsum) - 1,
            num_ranges=self._count_cumsum[-1],
            preprocessing=self._preprocessing_config,
        )
