import functools
import math
import typing

import numpy as np
import torch

from fast_llm.config import Field, config_class
from fast_llm.data.preprocessing.abstract import PreprocessingConfig
from fast_llm.data.sample.abstract import (
    Batch,
    MemmapReader,
    MemmapReaderBase,
    MemmapReaderBaseConfig,
    MemmapReaderConfig,
    MemmapWriter,
    Sample,
)
from fast_llm.data.sample.patch import PatchReaderBaseConfig
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.utils import Assert, get_unique


class TokenDataSample(Sample):
    """
    A reusable component holding tensor-valued data of fixed dtype and shape for each token.
    TODO: Use as base class for `TokenSample` and `PatchSample`?
    """

    def __init__(self, data: torch.Tensor):
        self.data = data

    @classmethod
    def from_documents(cls, documents: typing.Iterable[typing.Self]) -> typing.Self:
        return cls(torch.cat([document.data for document in documents]))

    def crop(self, begin: int, end: int) -> typing.Self:
        return self.__class__(self.data[begin:end])

    def __len__(self) -> int:
        return len(self.data)

    def get_padding(self, size: int) -> typing.Self:
        return self.__class__(torch.full([size], 0, dtype=self.data.dtype))


class TokenDataBatch(Batch):
    def __init__(self, data: torch.Tensor) -> None:
        self.data = data

    @classmethod
    def from_samples(cls, samples: typing.Iterable[TokenDataSample]) -> typing.Self:
        return cls(torch.stack([sample.data for sample in samples]))

    def crop(self, begin: int, end: int) -> typing.Self:
        return self.__class__(self.data[:, begin:end])

    def to_device_(self, device: "torch.device | str"):
        self.data = self.data.to(device, non_blocking=True)


@config_class(dynamic_type={MemmapReaderBaseConfig: "token_data"})
class TokenDataReaderConfig(MemmapReaderConfig):
    _abstract = False
    header: typing.ClassVar[bytes] = b"token data begin"
    footer: typing.ClassVar[bytes] = b"token data end"
    num_documents: int = Field()
    num_tokens: int = Field()
    shape: tuple[int, ...] = Field()
    data_type: DataType = Field()

    def __len__(self) -> int:
        return self.num_documents

    @functools.cached_property
    def size(self) -> int:
        return math.prod(self.shape)

    @property
    def reader_class(self) -> "type[TokenDataReader]":
        return TokenDataReader

    @property
    def writer_class(self) -> "type[TokenDataWriter]":
        return TokenDataWriter

    @property
    def _expected_buffer_size(self) -> int:
        return (
            self.num_tokens * self.data_type.torch.itemsize * self.size
            + (self.num_documents + 1) * torch.int64.itemsize
        )

    def get_metadata(self) -> dict[str, typing.Any]:
        return {
            "num_tokens": self.num_tokens,
            "num_documents": self.num_documents,
            "data_type": str(self.data_type),
            "shape": self.shape,
        }

    @classmethod
    def blend_metadata(cls, metadata: list[dict[str, typing.Any]]) -> dict[str, typing.Any]:
        return {
            "num_tokens": sum(metadata_["num_tokens"] for metadata_ in metadata),
            "num_documents": sum(metadata_["num_documents"] for metadata_ in metadata),
            "data_type": get_unique(metadata_["data_type"] for metadata_ in metadata),
            "shape": get_unique(metadata_["shape"] for metadata_ in metadata),
        }


class TokenDataReader[ConfigType: TokenDataReaderConfig](MemmapReader[ConfigType]):
    def __init__(self, config: ConfigType, buffer: memoryview, model_preprocessing: PreprocessingConfig | None = None):
        super().__init__(config, buffer, model_preprocessing)
        self._data = torch.frombuffer(
            self._buffer,
            dtype=self._config.data_type.torch,
            count=self._config.num_tokens * self._config.size,
        ).view(-1, *self._config.shape)
        self._size_cumsums = torch.frombuffer(
            self._buffer, dtype=torch.int64, count=self._config.num_documents + 1, offset=self._data.nbytes
        )

    def get_document(self, index: int, begin: int, end: int) -> Sample:
        begin_ = self._size_cumsums[index].item()
        return TokenDataSample(self._data[begin_ + begin : begin_ + end])

    def get_split(self, begin_index: int, end_index: int) -> dict[str, typing.Any]:
        Assert.custom(lambda x: x == sorted(x), [0, begin_index, end_index, self._config.num_documents])

        return {
            "num_tokens": self._size_cumsums[end_index].item() - self._size_cumsums[begin_index].item(),
            "num_documents": end_index - begin_index,
            "data_type": str(self._config.data_type),
            "shape": self._config.shape,
        }


class EmptyPatchReader[ConfigType: PatchReaderBaseConfig](MemmapReaderBase[ConfigType]):
    def get_document(self, index: int, begin: int, end: int) -> Sample:
        # TODO: Does this make sense?
        return TokenDataSample(torch.zeros(end - begin, *self._config.shape, dtype=self._config.data_type.torch))


def _get_nearest_split(cumsum: torch.Tensor, value: float) -> int:
    left = torch.searchsorted(cumsum, value, side="right")
    if left == len(cumsum):
        return left.item()
    return left.item() + 1 if (value - cumsum[left]) / (cumsum[left + 1] - cumsum[left]) > 0.5 else left.item()


class TokenDataWriter(MemmapWriter):
    def __enter__(self):
        super().__enter__()
        self._size_cumsum = [0]
        self._data_type = None
        self._shape = None
        return self

    def write(self, document: TokenDataSample):
        super().write(document)
        if self._data_type is None:
            self._data_type = document.data.dtype
            self._shape = document.data.shape[1:]
        else:
            Assert.eq(self._data_type, document.data.dtype)
            Assert.eq(self._shape, document.data.shape[1:])
        self._stream.write(document.data.numpy().tobytes())
        self._size_cumsum.append(self._size_cumsum[-1] + len(document.data))

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self._stream.write(np.array(self._size_cumsum, dtype=np.int64).tobytes(order="C"))
        super().__exit__(exc_type, exc_val, exc_tb)

    @classmethod
    def _get_config_class(cls) -> type[TokenDataReaderConfig]:
        return TokenDataReaderConfig

    def _get_config(self, begin: int, end: int):
        return TokenDataReaderConfig(
            begin=begin,
            end=end,
            num_documents=len(self._size_cumsum) - 1,
            num_tokens=self._size_cumsum[-1],
            shape=self._shape,
            data_type=DataType.from_torch(self._data_type),
            preprocessing=self._preprocessing_config,
        )
