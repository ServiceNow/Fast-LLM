import abc
import typing

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
