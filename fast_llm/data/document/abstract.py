import abc
import dataclasses


@dataclasses.dataclass(kw_only=True)
class Document(abc.ABC):
    pass


@dataclasses.dataclass(kw_only=True)
class Batch(Document):
    pass
    # @classmethod
    # @abc.abstractmethod
    # def from_documents(cls, documents: typing.Iterable[typing.Self]) -> typing.Self:
    #    pass

    # @abc.abstractmethod
    # def crop(self, begin: int, end: int) -> typing.Self:
    #    pass

    # def to_device_(self, device: "torch.device | str"):
    #    pass
