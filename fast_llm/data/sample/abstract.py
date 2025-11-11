import abc
import io
import pathlib
import typing

from fast_llm.config import Config, Configurable, Field, config_class
from fast_llm.utils import Assert

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
    def crop(self, begin: int, end: int) -> typing.Self:
        pass

    def to_device_(self, device: "torch.device | str"):
        pass


@config_class(registry=True)
class MemmapReaderBaseConfig(Config):
    """
    Configuration for a memmap reader or reader-like object.
    Note: `MemmapDataset` requires a `MemmapIndexedDatasetReader`.
      Other readers need to be nested within a `MemmapIndexedDatasetReader`
    Note: Reader configs are not typical configs, and do not need to be located in a separate `config.py` file.
    """

    _abstract = True

    @classmethod
    def _from_dict(cls, default: dict[str, typing.Any], strict: bool = True) -> typing.Self:
        if cls is MemmapReaderBaseConfig and cls.get_subclass(default.get("type")) is None:
            # Default subclass, necessary for loading configs where some components could be absent.
            return NullReaderConfig._from_dict(default, strict)
        return super()._from_dict(default, strict=strict)

    def get_reader(self, buffer: memoryview) -> "MemmapReader|None":
        raise NotImplementedError()

    @property
    def expected_buffer_size(self) -> int:
        """
        The expected buffer size in bytes, including header and footer. Used for self-validation.
        """
        raise NotImplementedError()


@config_class(dynamic_type={MemmapReaderBaseConfig: "none"})
class NullReaderConfig(MemmapReaderBaseConfig):
    """
    Configuration for a dynamically disabled reader.
    """

    _abstract = False

    def get_reader(self, buffer: memoryview) -> None:
        return None

    @property
    def expected_buffer_size(self) -> int:
        return 0


@config_class(registry=True)
class MemmapReaderConfig(MemmapReaderBaseConfig):
    """
    Configuration for a standard memmap reader.
    """

    # Data location in the file.
    begin: int = Field()
    end: int = Field()
    # Constant strings for alignment safety.
    header: typing.ClassVar[bytes]
    footer: typing.ClassVar[bytes]

    @property
    def reader_class(self) -> "type[MemmapReader]":
        raise NotImplementedError()

    def get_reader(self, buffer: memoryview) -> "MemmapReader":
        return self.reader_class(self, buffer)

    @property
    def expected_buffer_size(self) -> int:
        """
        The expected buffer size in bytes, including header and footer. Used for self-validation.
        """
        return self._expected_buffer_size + len(self.header) + len(self.footer)

    @property
    def _expected_buffer_size(self) -> int:
        """
        The expected buffer size in bytes, excluding header and footer. Used for self-validation.
        """
        raise NotImplementedError()

    @property
    def writer_class(self) -> "type[MemmapWriter]":
        raise NotImplementedError()

    def get_writer(self, stream: io.BufferedWriter) -> "MemmapWriter":
        return self.writer_class(stream)

    def _validate(self):
        super()._validate()
        Assert.eq(self.end - self.begin, self.expected_buffer_size)


@config_class()
class MemmapIndexDatasetReaderConfig(MemmapReaderConfig):
    """
    Configuration for a standard memmap reader matching the indexed dataset interface, i.e.,
    consisting of a list of documents of known lengths.
    """

    def __len__(self) -> int:
        raise NotImplementedError()

    @property
    def num_tokens(self) -> int:
        raise NotImplementedError()

    @property
    def reader_class(self) -> "type[MemmapIndexedDatasetReader]":
        raise NotImplementedError()

    def get_reader(
        self,
        buffer: memoryview,
    ) -> "MemmapIndexedDatasetReader":
        return self.reader_class(self, buffer)


class MemmapReader[ConfigType: MemmapReaderConfig](Configurable[ConfigType]):
    def __init__(self, config: ConfigType, buffer: memoryview):
        super().__init__(config)
        buffer_begin = self._config.begin + len(self._config.header)
        buffer_end = self._config.end - len(self._config.footer)
        Assert.eq(buffer[self._config.begin : buffer_begin].tobytes(), self._config.header)
        Assert.eq(buffer[buffer_end : self._config.end].tobytes(), self._config.footer)
        self._buffer = buffer[buffer_begin:buffer_end]

    @abc.abstractmethod
    def get_document(self, index: int, begin: int, end: int) -> Sample:
        pass


class MemmapIndexedDatasetReader[ConfigType: MemmapIndexDatasetReaderConfig](MemmapReader[ConfigType]):
    def __len__(self) -> int:
        return len(self._config)

    @property
    def num_tokens(self) -> int:
        return self._config.num_tokens

    @abc.abstractmethod
    def get_document_sizes(self) -> "torch.Tensor":
        pass

    @abc.abstractmethod
    def get_document_size(self, index: int) -> int:
        pass


class MemmapWriter(abc.ABC):
    def __init__(self, stream: io.BufferedWriter | pathlib.Path):
        self._owns_stream = isinstance(stream, pathlib.Path)
        if self._owns_stream:
            stream = stream.open("wb")
        self._stream = stream

    def __enter__(self):
        self._begin = self._stream.tell()
        self._stream.write(self._get_config_class().header)
        return self

    def write(self, document: Sample):
        assert hasattr(self, "_begin") and not hasattr(self, "_end")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self._stream.write(self._get_config_class().footer)
            self._end = self._stream.tell()
        if self._owns_stream:
            self._stream.close()

    @classmethod
    @abc.abstractmethod
    def _get_config_class(cls) -> type[MemmapReaderConfig]:
        pass

    def get_config(self, offset: int = 0) -> MemmapReaderConfig:
        assert hasattr(self, "_end")
        return self._get_config(self._begin + offset, self._end + offset)

    @abc.abstractmethod
    def _get_config(self, begin: int, end: int):
        pass

    @classmethod
    def write_dataset(cls, stream: io.BufferedWriter, documents: typing.Iterable[Sample]) -> MemmapReaderConfig:
        with cls(stream) as writer:
            for document in documents:
                writer.write(document)
        return writer.get_config()
