import typing

import numpy as np
import torch

from fast_llm.config import Field, config_class
from fast_llm.data.sample.abstract import (
    Batch,
    MemmapIndexedDatasetReader,
    MemmapReaderBaseConfig,
    MemmapReaderConfig,
    MemmapWriter,
    Sample,
)
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.utils import Assert


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
        sample_size = end - begin
        if self.lengths == [len(self.tokens)]:
            # Shortcut for the frequent case of a single document.
            lengths = [sample_size]
        else:
            begin_ = 0
            lengths = []
            for length in self.lengths:
                end_ = begin_ + length
                cropped_length = min(end_, end) - max(begin_, begin)
                if cropped_length > 0:
                    lengths.append(cropped_length)
                if end_ > end:
                    break
                begin_ = end_
        return self.__class__(self.tokens[begin:end], lengths)

    def __len__(self) -> int:
        return len(self.tokens)

    def get_padding(self, size: int) -> typing.Self:
        return TokenSample(torch.full([size], -100, dtype=self.tokens.dtype), [size])


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

    def to_samples(self) -> list[TokenSample]:
        return [TokenSample(tokens, lengths) for tokens, lengths in zip(self.tokens, self.lengths, strict=True)]

    def crop(self, begin: int, end: int) -> typing.Self:
        return self.__class__(
            self.tokens[:, begin:end], [sample.crop(begin, end).lengths for sample in self.to_samples()]
        )

    def to_device_(self, device: "torch.device | str"):
        # Also standardize the dtype while we're here.
        self.tokens = self.tokens.to(device, dtype=torch.int64, non_blocking=True)


@config_class(dynamic_type={MemmapReaderBaseConfig: "token"})
class TokenReaderConfig(MemmapReaderConfig):
    _abstract = False
    num_documents: int = Field()
    num_tokens: int = Field()
    data_type: DataType = Field()
    header: typing.ClassVar[bytes] = b"token begin"
    footer: typing.ClassVar[bytes] = b"token end"

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


class TokenReader[ConfigType: TokenReaderConfig](MemmapIndexedDatasetReader[ConfigType]):
    def __init__(self, config: ConfigType, buffer: memoryview):
        super().__init__(config, buffer)
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
        return TokenSample(self._tokens[begin_ + begin : begin_ + end].to(torch.int64), [end - begin])

    def get_document_sizes(self) -> torch.Tensor:
        return self._size_cumsums[1:] - self._size_cumsums[:-1]

    def get_document_size(self, index: int) -> int:
        return self._size_cumsums[index + 1].item() - self._size_cumsums[index].item()


class TokenWriter(MemmapWriter):
    def __enter__(self):
        super().__enter__()
        self._size_cumsum = [0]
        self._data_type = None
        return self

    def write(self, document: TokenSample):
        # ====== TODO: Make sure input uses end = 1 past last index (currently use last index) ======
        super().write(document)
        if self._data_type is None:
            self._data_type = document.tokens.dtype
        else:
            Assert.eq(self._data_type, document.tokens.dtype)
        self._stream.write(document.tokens.numpy().tobytes())
        self._size_cumsum.append(self._size_cumsum[-1] + len(document.tokens))

    def __exit__(self, exc_type, exc_val, exc_tb):
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
        )
