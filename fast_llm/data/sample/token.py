import typing

import numpy as np
import torch

from fast_llm.config import Field, config_class
from fast_llm.data.preprocessing.abstract import PreprocessingConfig
from fast_llm.data.sample.abstract import (
    Batch,
    MemmapIndexedDatasetReader,
    MemmapReaderBaseConfig,
    MemmapReaderConfig,
    MemmapWriter,
    Sample,
)
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.utils import Assert, get_unique


def crop_lengths(lengths: list[int], begin: int, end: int) -> list[int]:
    if len(lengths) == 1:
        # Shortcut for the frequent case of a single document.
        return [end - begin]
    begin_ = 0
    lengths_ = []
    for length in lengths:
        end_ = begin_ + length
        cropped_length = min(end_, end) - max(begin_, begin)
        if cropped_length > 0:
            lengths_.append(cropped_length)
        if end_ > end:
            break
        begin_ = end_
    return lengths_


class TokenSample(Sample):
    def __init__(self, tokens: torch.Tensor, lengths: list[int] | None = None):
        self.tokens = tokens
        # Length of each document in the sample. TODO: Use cumsums instead?
        if lengths is None:
            lengths = [len(tokens)]
        else:
            Assert.eq(sum(lengths), len(tokens))
        self.lengths = lengths

    @classmethod
    def from_documents(cls, documents: typing.Iterable[typing.Self]) -> typing.Self:
        return cls(
            torch.cat([document.tokens for document in documents]),
            sum((document.lengths for document in documents), []),
        )

    def crop(self, begin: int, end: int) -> typing.Self:
        return self.__class__(self.tokens[begin:end], crop_lengths(self.lengths, begin, end))

    def __len__(self) -> int:
        return len(self.tokens)

    def get_padding(self, size: int) -> typing.Self:
        return self.__class__(torch.full([size], -100, dtype=self.tokens.dtype), [size])


class TokenBatch(Batch):
    def __init__(self, tokens: torch.Tensor, lengths: list[list[int]] | None) -> None:
        self.tokens = tokens
        if lengths is None:
            lengths = [[tokens.size(1)]] * tokens.size(0)
        self.lengths = lengths

    @classmethod
    def from_samples(cls, samples: typing.Iterable[TokenSample]) -> typing.Self:
        return cls(
            torch.stack([sample.tokens for sample in samples]),
            [sample.lengths for sample in samples],
        )

    def crop(self, begin: int, end: int) -> typing.Self:
        return self.__class__(
            self.tokens[:, begin:end],
            [crop_lengths(lengths, begin, end) for lengths in self.lengths],
        )

    def to_device_(self, device: "torch.device | str"):
        # Also standardize the dtype while we're here.
        self.tokens = self.tokens.to(device, dtype=torch.int64, non_blocking=True)


@config_class(dynamic_type={MemmapReaderBaseConfig: "token"})
class TokenReaderConfig(MemmapReaderConfig):
    _abstract = False
    header: typing.ClassVar[bytes] = b"token begin"
    footer: typing.ClassVar[bytes] = b"token end"
    num_documents: int = Field()
    num_tokens: int = Field()
    data_type: DataType = Field()

    def __len__(self) -> int:
        return self.num_documents

    @property
    def reader_class(self) -> "type[TokenReader]":
        return TokenReader

    @property
    def writer_class(self) -> "type[TokenWriter]":
        return TokenWriter

    @property
    def _expected_buffer_size(self) -> int:
        return self.num_tokens * self.data_type.torch.itemsize + (self.num_documents + 1) * torch.int64.itemsize

    def get_metadata(self) -> dict[str, typing.Any]:
        return {
            "num_tokens": self.num_tokens,
            "num_documents": self.num_documents,
            "data_type": str(self.data_type),
        }

    @classmethod
    def blend_metadata(cls, metadata: list[dict[str, typing.Any]]) -> dict[str, typing.Any]:
        return {
            "num_tokens": sum(metadata_["num_tokens"] for metadata_ in metadata),
            "num_documents": sum(metadata_["num_documents"] for metadata_ in metadata),
            "data_type": get_unique(metadata_["data_type"] for metadata_ in metadata),
        }


class TokenReader[ConfigType: TokenReaderConfig](MemmapIndexedDatasetReader[ConfigType]):
    def __init__(self, config: ConfigType, buffer: memoryview, model_preprocessing: PreprocessingConfig | None = None):
        super().__init__(config, buffer, model_preprocessing)
        self._tokens = torch.frombuffer(
            self._buffer,
            dtype=self._config.data_type.torch,
            count=self._config.num_tokens,
        )
        self._size_cumsums = torch.frombuffer(
            self._buffer, dtype=torch.int64, count=self._config.num_documents + 1, offset=self._tokens.nbytes
        )

    def get_document(self, index: int, begin: int, end: int) -> Sample:
        begin_ = self._size_cumsums[index].item()
        # Torch doesn't support type promotion between signed and unsigned types, so we convert here to avoid issues.
        # Convert begin and end to int to avoid numpy dtype overflow when adding to begin_
        return TokenSample(self._tokens[begin_ + begin : begin_ + end].to(torch.int64), [end - begin])

    def get_document_sizes(self) -> torch.Tensor:
        return self._size_cumsums[1:] - self._size_cumsums[:-1]

    def get_document_size(self, index: int) -> int:
        return self._size_cumsums[index + 1].item() - self._size_cumsums[index].item()

    def get_split(self, begin_ratio: float, end_ratio: float) -> tuple[int, int, dict[str, typing.Any]]:
        Assert.custom(lambda x: x == sorted(x), [0, begin_ratio, end_ratio, 1])
        begin_index = _get_nearest_split(self._size_cumsums[1:], begin_ratio * self.num_tokens)
        end_index = _get_nearest_split(self._size_cumsums[1:], end_ratio * self.num_tokens)

        return (
            begin_index,
            end_index,
            {
                "num_tokens": self._size_cumsums[end_index].item() - self._size_cumsums[begin_index].item(),
                "num_documents": end_index - begin_index,
                "data_type": str(self._config.data_type),
            },
        )


def _get_nearest_split(cumsum: torch.Tensor, value: float) -> int:
    left = torch.searchsorted(cumsum, value, side="right")
    if left == len(cumsum):
        return left.item()
    return left.item() + 1 if (value - cumsum[left]) / (cumsum[left + 1] - cumsum[left]) > 0.5 else left.item()


class TokenWriter(MemmapWriter):
    def __enter__(self):
        super().__enter__()
        self._size_cumsum = [0]
        self._data_type = None
        return self

    def write(self, document: TokenSample):
        super().write(document)
        if self._data_type is None:
            self._data_type = document.tokens.dtype
        else:
            Assert.eq(self._data_type, document.tokens.dtype)
        self._stream.write(document.tokens.numpy().tobytes())
        self._size_cumsum.append(self._size_cumsum[-1] + len(document.tokens))

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self._stream.write(np.array(self._size_cumsum, dtype=np.int64).tobytes(order="C"))
        super().__exit__(exc_type, exc_val, exc_tb)

    @classmethod
    def _get_config_class(cls) -> type[TokenReaderConfig]:
        return TokenReaderConfig

    def _get_config(self, begin: int, end: int):
        return TokenReaderConfig(
            begin=begin,
            end=end,
            num_documents=len(self._size_cumsum) - 1,
            num_tokens=self._size_cumsum[-1],
            data_type=DataType.from_torch(self._data_type),
            preprocessing=self._preprocessing_config,
        )
