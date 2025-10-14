import io
import typing

import numpy as np

from fast_llm.data.dataset.sample import Batch, Sample
from fast_llm.data.dataset.sample.abstract import MemmapIndexedDatasetReader
from fast_llm.data.dataset.sample.config import LanguageModelReaderConfig, NullReaderConfig
from fast_llm.data.dataset.sample.range import RangeBatch, RangeReader, RangeSample
from fast_llm.data.dataset.sample.token import TokenBatch, TokenReader, TokenSample
from fast_llm.utils import Assert, get_unique


class LanguageModelSample(Sample):
    def __init__(
        self,
        tokens: TokenSample,
        loss_masking_spans: RangeSample | None = None,
        preference_spans: RangeSample | None = None,
    ):
        self.tokens = tokens
        self.loss_masking_spans = loss_masking_spans
        self.preference_spans = preference_spans

    @classmethod
    def from_documents(cls, documents: typing.Iterable[typing.Self]) -> typing.Self:
        return cls(
            TokenSample.from_documents(document.tokens for document in documents),
            _merge_optional(RangeSample.from_documents, (document.loss_masking_spans for document in documents)),
            _merge_optional(RangeSample.from_documents, (document.preference_spans for document in documents)),
        )

    def crop(self, begin: int, end: int) -> typing.Self:
        return self.__class__(
            self.tokens.crop(begin, end),
            None if self.loss_masking_spans is None else self.loss_masking_spans.crop(begin, end),
            None if self.preference_spans is None else self.preference_spans.crop(begin, end),
        )

    def __len__(self) -> int:
        return len(self.tokens)


class LanguageModelBatch(Batch):
    def __init__(
        self,
        tokens: TokenBatch,
        loss_masking_spans: RangeBatch | None = None,
        preference_spans: RangeBatch | None = None,
    ):
        self.tokens = tokens
        self.loss_masking_spans = loss_masking_spans
        self.preference_spans = preference_spans

    @classmethod
    def from_samples(cls, samples: typing.Iterable[LanguageModelSample]) -> typing.Self:
        return cls(
            TokenBatch.from_samples(sample.tokens for sample in samples),
            _merge_optional(RangeBatch.from_samples, (sample.loss_masking_spans for sample in samples)),
            _merge_optional(RangeBatch.from_samples, (sample.preference_spans for sample in samples)),
        )


def _merge_optional[T](fn: typing.Callable[[typing.Iterable], T], args: typing.Iterable) -> T | None:
    return None if any(arg is None for arg in args) else fn(args)


class LanguageModelReader[ConfigType: LanguageModelReaderConfig](MemmapIndexedDatasetReader[ConfigType]):
    def __init__(self, config: ConfigType, buffer: memoryview):
        super().__init__(config, buffer)
        # Using `buffer` and not `self._buffer` because nested offsets (`begin`, `end`) are global.
        self._tokens = self._config.tokens.get_reader(buffer)
        self._loss_masking_spans = self._config.loss_masking_spans.get_reader(buffer)
        self._preference_spans = self._config.preference_spans.get_reader(buffer)

    def get_document(self, index: int, begin: int, end: int) -> Sample:
        return LanguageModelSample(
            self._tokens.get_document(index, begin, end),
            self._loss_masking_spans.get_document(index, begin, end),
            self._preference_spans.get_document(index, begin, end),
        )

    def get_document_sizes(self) -> "np.ndarray":
        return self._tokens.get_document_sizes()

    def get_document_size(self, index: int) -> int:
        return self._tokens.get_document_size(index)

    @classmethod
    def write(
        cls, documents: typing.Iterable[LanguageModelSample], stream: io.BufferedWriter
    ) -> LanguageModelReaderConfig:
        begin = stream.tell()
        tokens = TokenReader.write((document.tokens for document in documents), stream)

        # Ensure either all samples have loss masking spans or none of them do.
        if get_unique(document.loss_masking_spans is not None for document in documents):
            loss_masking_spans = RangeReader.write((document.loss_masking_spans for document in documents), stream)
        else:
            loss_masking_spans = NullReaderConfig()

        # If enabled, ensure all samples have exactly 2 spans
        num_preference_spans = get_unique(
            None if document.preference_spans is None else len(document.preference_spans.ranges)
            for document in documents
        )
        if num_preference_spans == 2:
            preference_spans = RangeReader.write((document.preference_spans for document in documents), stream)
        else:
            Assert.none(num_preference_spans)
            preference_spans = NullReaderConfig()

        return LanguageModelReaderConfig(
            begin=begin,
            end=stream.tell(),
            tokens=tokens,
            loss_masking_spans=loss_masking_spans,
            preference_spans=preference_spans,
        )
