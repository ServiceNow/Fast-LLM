import abc
import dataclasses
import typing

if typing.TYPE_CHECKING:
    import torch


@dataclasses.dataclass(kw_only=True)
class Document(abc.ABC):
    pass


@dataclasses.dataclass(kw_only=True)
class Batch(Document):
    @abc.abstractmethod
    def crop(self, begin: int, end: int) -> typing.Self:
        pass

    def to_device(self, device: "torch.device | str") -> typing.Self:
        return self
