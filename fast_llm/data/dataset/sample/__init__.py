import abc
import typing

from fast_llm.data.dataset.sample.abstract import Batch, Sample


class LanguageModelSample(Sample):

    @classmethod
    @abc.abstractmethod
    def merge_documents(cls, documents: typing.Iterable[typing.Self]) -> typing.Self:
        pass

    @classmethod
    @abc.abstractmethod
    def merge_into_batch(cls, samples: typing.Iterable[typing.Self]) -> "LanguageModelBatch":
        pass

    @abc.abstractmethod
    def crop(self, offset: int = 0, length: int | None = None):
        pass


class LanguageModelBatch(Batch):
    pass
