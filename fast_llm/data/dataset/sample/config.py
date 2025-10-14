import abc
import pathlib
import typing

from fast_llm.config import Config, Field, FieldHint, config_class
from fast_llm.data.dataset.config import IndexedDatasetConfig, SampledDatasetConfig
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    from fast_llm.data.dataset.memmap import MemmapDataset
    from fast_llm.data.dataset.sample.abstract import MemmapIndexedDatasetReader, MemmapReader
    from fast_llm.data.dataset.sample.language_model import LanguageModelReader
    from fast_llm.data.dataset.sample.range import RangeReader
    from fast_llm.data.dataset.sample.token import TokenReader


@config_class(registry=True)
class MemmapReaderBaseConfig(Config):
    @classmethod
    def _from_dict(cls, default: dict[str, typing.Any], strict: bool = True) -> typing.Self:
        if cls is MemmapReaderBaseConfig and cls.get_subclass(default.get("type")) is None:
            # Default subclass, necessary for loading configs where some components could be absent.
            return NullReaderConfig._from_dict(default, strict)
        return super()._from_dict(default, strict=strict)

    @abc.abstractmethod
    def get_reader(self, buffer: memoryview) -> "MemmapReader|None":
        pass

    @property
    @abc.abstractmethod
    def expected_buffer_size(self) -> int:
        """
        The expected buffer size in bytes. Used for self-validation.
        """


@config_class(dynamic_type={MemmapReaderBaseConfig: "none"})
class NullReaderConfig(MemmapReaderBaseConfig):
    def get_reader(self, buffer: memoryview) -> None:
        return None

    @property
    def expected_buffer_size(self) -> int:
        return 0


@config_class(registry=True)
class MemmapReaderConfig(MemmapReaderBaseConfig):
    begin: int = Field()
    end: int = Field()

    @property
    def reader_class(self) -> "type[MemmapReader]":
        raise NotImplementedError()

    def get_reader(self, buffer: memoryview) -> "MemmapReader":
        return self.reader_class(self, buffer)

    def _validate(self):
        super()._validate()
        Assert.eq(self.end - self.begin, self.expected_buffer_size)


@config_class()
class MemmapIndexDatasetReaderConfig(MemmapReaderConfig):
    @property
    def reader_class(self) -> "type[MemmapIndexedDatasetReader]":
        raise NotImplementedError()

    #
    def get_reader(
        self,
        buffer: memoryview,
    ) -> "MemmapIndexedDatasetReader":
        return self.reader_class(self, buffer[self.begin : self.end])


@config_class(dynamic_type={MemmapReaderBaseConfig: "range"})
class RangeReaderConfig(MemmapReaderConfig):
    num_documents: int = Field()
    num_ranges: int = Field()

    @property
    def reader_class(self) -> "type[RangeReader]":
        from fast_llm.data.dataset.sample.range import RangeReader

        return RangeReader

    @property
    def expected_buffer_size(self) -> int:
        return (self.num_ranges + 1) * 4 * 2 + (self.num_documents + 1) * 4


@config_class(dynamic_type={MemmapReaderBaseConfig: "token"})
class TokenReaderConfig(MemmapIndexDatasetReaderConfig):
    num_documents: int = Field()
    num_tokens: int = Field()
    data_type: DataType = Field()

    @property
    def reader_class(self) -> "type[TokenReader]":
        from fast_llm.data.dataset.sample.token import TokenReader

        return TokenReader

    @property
    def expected_buffer_size(self) -> int:
        return self.num_tokens * self.data_type.numpy.itemsize + (self.num_documents + 1) * 8


@config_class(dynamic_type={MemmapReaderBaseConfig: "language_model"})
class LanguageModelReaderConfig(MemmapIndexDatasetReaderConfig):
    tokens: TokenReaderConfig = Field()
    # Using dynamic type for optional readers for enabling/disabling
    loss_masking_spans: MemmapReaderBaseConfig = Field()
    preference_spans: MemmapReaderBaseConfig = Field()

    @property
    def reader_class(self) -> "type[LanguageModelReader]":
        from fast_llm.data.dataset.sample.language_model import LanguageModelReader

        return LanguageModelReader

    @property
    def expected_buffer_size(self) -> int:
        return (
            self.tokens.expected_buffer_size
            + self.loss_masking_spans.expected_buffer_size
            + self.preference_spans.expected_buffer_size
        )


@config_class(dynamic_type={SampledDatasetConfig: "memmap"})
class MemmapDatasetConfig(IndexedDatasetConfig):
    _abstract: typing.ClassVar[bool] = False
    path: pathlib.Path = Field(
        default=None,
        desc="The path to the dataset, excluding the `.bin` or `.idx` suffix.",
        hint=FieldHint.core,
    )

    def build(self) -> "MemmapDataset":
        from fast_llm.data.dataset.memmap import MemmapDataset

        return MemmapDataset(str(self.path).replace("/", "__"), self.path)
