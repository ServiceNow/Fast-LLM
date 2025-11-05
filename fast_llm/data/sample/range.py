import typing

import numpy as np
import torch

from fast_llm.config import Field, config_class
from fast_llm.data.sample.abstract import (
    Batch,
    MemmapReader,
    MemmapReaderBaseConfig,
    MemmapReaderConfig,
    MemmapWriter,
    Sample,
)
from fast_llm.utils import Assert, get_unique


class RangeSample(Sample):
    """
    A reusable component holding a set of ranges in a sample.
    """

    def __init__(self, ranges: list[tuple[int, int]], sample_size: int):
        self.ranges = ranges
        self.sample_size = sample_size

    @classmethod
    def from_documents(cls, documents: typing.Iterable[typing.Self]) -> typing.Self:
        document: RangeSample
        ranges = []
        sample_size = 0
        for document in documents:
            for begin, end in document.ranges:
                ranges.extend((begin + sample_size, end + sample_size))
            sample_size += document.sample_size
        return cls(ranges, sample_size)

    def crop(self, begin: int, end: int) -> typing.Self:
        sample_size = end - begin
        cropped_ranges = ((max(begin_ - begin, 0), min(end_ - begin, sample_size)) for begin_, end_ in self.ranges)
        return self.__class__([(begin_, end_) for begin_, end_ in cropped_ranges if end_ > begin_], sample_size)

    def __len__(self) -> int:
        return self.sample_size

    def get_padding(self, size: int) -> typing.Self:
        return RangeSample([], size)


class RangeBatch(Batch):
    def __init__(self, ranges: list[list[tuple[int, int]]], sample_size: int):
        self.sample_size = sample_size
        self.ranges = ranges

    @classmethod
    def from_samples(cls, samples: typing.Iterable[RangeSample]) -> typing.Self:
        return cls([sample.ranges for sample in samples], get_unique(sample.sample_size for sample in samples))

    def to_samples(self) -> list[RangeSample]:
        return [RangeSample(sample_ranges, self.sample_size) for sample_ranges in self.ranges]


@config_class(dynamic_type={MemmapReaderBaseConfig: "range"})
class RangeReaderConfig(MemmapReaderConfig):
    _abstract = False
    header: typing.ClassVar[bytes] = b"range begin"
    footer: typing.ClassVar[bytes] = b"range end"
    num_documents: int = Field()
    num_ranges: int = Field()

    @property
    def reader_class(self) -> "type[RangeReader]":
        return RangeReader

    @property
    def writer_class(self) -> "type[RangeWriter]":
        return RangeWriter

    @property
    def _expected_buffer_size(self) -> int:
        return self.num_ranges * torch.int32.itemsize * 2 + (self.num_documents + 1) * torch.int32.itemsize


class RangeReader[ConfigType: RangeReaderConfig](MemmapReader[ConfigType]):
    def __init__(self, config: ConfigType, buffer: memoryview):
        super().__init__(config, buffer)
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

    def get_document(self, index: int, begin: int, end: int) -> Sample:
        sample_size = end - begin
        cropped_ranges = (
            (max(begin_ - begin, 0), min(end_ - begin, sample_size))
            for begin_, end_ in self._ranges[self._count_cumsums[index] : self._count_cumsums[index + 1]].tolist()
        )
        return RangeSample([(begin_, end_) for begin_, end_ in cropped_ranges if end_ > begin_], sample_size)


class RangeWriter(MemmapWriter):
    def __enter__(self):
        super().__enter__()
        self._count_cumsum = [0]
        return self

    def write(self, document: RangeSample):
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
        )
