import abc
import io
import pathlib
import typing

import torch

from fast_llm.config import Configurable
from fast_llm.data.dataset.memmap.config import (
    MemmapIndexDatasetReaderConfig,
    MemmapReaderBaseConfig,
    MemmapReaderConfig,
    NullReaderConfig,
)
from fast_llm.data.document.abstract import Document
from fast_llm.data.preprocessing.abstract import NullPreprocessingConfig, PreprocessingConfig
from fast_llm.utils import Assert


class MemmapReaderBase[ConfigType: MemmapReaderBaseConfig](Configurable[ConfigType]):
    @abc.abstractmethod
    def get_document(self, index: int, begin: int, end: int) -> Document | None:
        pass


class NullMemmapReader[ConfigType: NullReaderConfig](MemmapReaderBase[ConfigType]):
    def get_document(self, index: int, begin: int, end: int) -> None:
        return None


class MemmapReader[ConfigType: MemmapReaderConfig](MemmapReaderBase[ConfigType]):
    def __init__(self, config: ConfigType, buffer: memoryview, model_preprocessing: PreprocessingConfig | None = None):
        super().__init__(config)
        # Note: This is the requirement at reading time (ex. from the model),
        # which may differ from how the dataset was actually preprocessed (`config.preprocessing`)
        # Compatibility checked in `MemmapDataset`.
        self._model_preprocessing = NullPreprocessingConfig if model_preprocessing is None else model_preprocessing
        buffer_begin = self._config.begin + len(self._config.header)
        buffer_end = self._config.end - len(self._config.footer)
        Assert.eq(buffer[self._config.begin : buffer_begin].tobytes(), self._config.header)
        Assert.eq(buffer[buffer_end : self._config.end].tobytes(), self._config.footer)
        self._buffer = buffer[buffer_begin:buffer_end]

    @abc.abstractmethod
    def get_document(self, index: int, begin: int, end: int) -> Document:
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

    def get_split(self, begin_ratio: float, end_ratio: float) -> tuple[int, int, dict[str, typing.Any]]:
        raise NotImplementedError()


class MemmapWriter(abc.ABC):
    def __init__(
        self, stream: io.BufferedWriter | pathlib.Path, preprocessing_config: PreprocessingConfig | None = None
    ):
        self._owns_stream = isinstance(stream, pathlib.Path)
        if self._owns_stream:
            stream = stream.open("wb")
        self._stream = stream
        self._preprocessing_config = (
            NullPreprocessingConfig() if preprocessing_config is None else preprocessing_config
        )

    def __enter__(self):
        self._begin = self._stream.tell()
        self._stream.write(self._get_config_class().header)
        return self

    def write(self, document: Document):
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
    def write_dataset(
        cls,
        stream: io.BufferedWriter,
        documents: typing.Iterable[Document],
        preprocessing_config: PreprocessingConfig | None = None,
    ) -> MemmapReaderConfig:
        with cls(stream, preprocessing_config) as writer:
            for document in documents:
                writer.write(document)
        return writer.get_config()
