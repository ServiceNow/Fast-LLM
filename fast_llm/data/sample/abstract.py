import abc
import io
import typing

from fast_llm.config import Configurable
from fast_llm.data.sample.config import MemmapIndexDatasetReaderConfig, MemmapReaderBaseConfig

if typing.TYPE_CHECKING:
    import torch


class Sample(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def from_documents(cls, documents: typing.Iterable[typing.Self]) -> typing.Self:
        pass

    @abc.abstractmethod
    def crop(self, begin: int, end: int) -> typing.Self:
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @abc.abstractmethod
    def get_padding(self, size: int) -> typing.Self:
        pass


class Batch(abc.ABC):
    # TODO: Relate to `BatchConfig`?
    @classmethod
    @abc.abstractmethod
    def from_samples(cls, samples: typing.Iterable[Sample]) -> typing.Self:
        pass

    @abc.abstractmethod
    def to_samples(self) -> list[Sample]:
        pass

    def crop(self, begin: int, end: int) -> typing.Self:
        return self.from_samples(sample.crop(begin, end) for sample in self.to_samples())

    def to_device_(self, device: "torch.device | str"):
        pass


class MemmapReader[ConfigType: MemmapReaderBaseConfig](Configurable[ConfigType]):
    def __init__(self, config: ConfigType, buffer: memoryview):
        super().__init__(config)
        self._buffer = buffer[self._config.begin : self._config.end]

    @abc.abstractmethod
    def get_document(self, index: int, begin: int, end: int) -> Sample:
        pass

    @classmethod
    @abc.abstractmethod
    def write(cls, documents: typing.Iterable[Sample], stream: io.BufferedWriter) -> MemmapReaderBaseConfig:
        pass


class MemmapIndexedDatasetReader[ConfigType: MemmapIndexDatasetReaderConfig](MemmapReader[ConfigType]):
    @abc.abstractmethod
    def get_document_sizes(self) -> "torch.Tensor":
        pass

    @abc.abstractmethod
    def get_document_size(self, index: int) -> int:
        pass
