import io
import typing

import numpy as np

from fast_llm.data.sample.abstract import Batch, MemmapReader, Sample
from fast_llm.data.sample.config import RangeReaderConfig
from fast_llm.utils import get_unique


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


class RangeReader[ConfigType: RangeReaderConfig](MemmapReader[ConfigType]):
    # ====== TODO: Standardize to end = 1 past last index (spans use last index) ======
    def __init__(self, config: ConfigType, buffer: memoryview):
        super().__init__(config, buffer)
        self._ranges = np.frombuffer(
            self._buffer,
            dtype=np.int32,
            count=self._config.num_ranges,
        ).reshape(-1, 2)
        self._count_cumsums = np.frombuffer(
            self._buffer,
            dtype=np.int32,
            count=self._config.num_documents + 1,
            offset=self._ranges.nbytes,
        )

    def get(self, index: int, begin: int, end: int) -> RangeSample:
        sample_size = end - begin
        cropped_ranges = (
            (max(begin_ - begin, 0), min(end_ - begin, sample_size))
            for begin_, end_ in self._ranges[self._count_cumsums[index] : self._count_cumsums[index + 1]].tolist()
        )
        return RangeSample(sample_size, tuple((begin_, end_) for begin_, end_ in cropped_ranges if end_ > begin_))

    @classmethod
    def write(cls, documents: typing.Iterable[RangeSample], stream: io.BufferedWriter) -> RangeReaderConfig:
        begin = stream.tell()
        count_cumsum = [0]
        for document in documents:
            stream.write(np.array(document.ranges, dtype=np.uint32).tobytes(order="C"))
            count_cumsum.append(count_cumsum[-1] + len(document.ranges))
        stream.write(np.array(count_cumsum, dtype=np.uint32).tobytes(order="C"))
        end = stream.tell()

        return RangeReaderConfig(
            begin=begin,
            end=end,
            num_documents=len(count_cumsum) - 1,
            num_ranges=count_cumsum[-1],
        )
