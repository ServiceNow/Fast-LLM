import io
import typing

import numpy as np
import torch

from fast_llm.data.sample import Batch, Sample
from fast_llm.data.sample.abstract import MemmapIndexedDatasetReader
from fast_llm.data.sample.config import TokenReaderConfig
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.utils import Assert


class TokenSample(Sample):
    def __init__(self, tokens: torch.Tensor, lengths: list[int] | None = None):
        self.tokens = tokens
        # Length of each document in the sample. TODO: Use cumsums instead?
        if lengths is None:
            lengths = [len(tokens)]
        self.lengths = lengths

    @classmethod
    def from_documents(cls, documents: typing.Iterable[typing.Self]) -> typing.Self:
        return cls(
            torch.cat([document.tokens for document in documents]),
            sum((document.lengths for document in documents), []),
        )

    def crop(self, begin: int, end: int) -> typing.Self:
        # We only expect to crop documents, not samples. TODO: Support other cases?
        Assert.eq(self.lengths, [len(self.tokens)])
        return self.__class__(self.tokens[begin:end], [end - begin])

    def __len__(self) -> int:
        return len(self.tokens)


class TokenBatch(Batch):
    def __init__(self, tokens: torch.Tensor, lengths: list[list[int]]) -> None:
        self.tokens = tokens
        self.lengths = lengths

    @classmethod
    def from_samples(cls, samples: typing.Iterable[TokenSample]) -> typing.Self:
        return cls(
            torch.stack([sample.tokens for sample in samples]),
            [sample.lengths for sample in samples],
        )


class TokenReader[ConfigType: TokenReaderConfig](MemmapIndexedDatasetReader[ConfigType]):
    def __init__(self, config: ConfigType, buffer: memoryview):
        super().__init__(config, buffer)
        self._tokens = np.frombuffer(
            self._buffer,
            dtype=self._config.data_type.numpy,
            count=self._config.num_tokens,
        )
        self._size_cumsums = np.frombuffer(
            self._buffer, dtype=np.uint64, count=self._config.num_documents + 1, offset=self._tokens.nbytes
        )

    def get_document(self, index: int, begin: int, end: int) -> Sample:
        begin_ = self._size_cumsums[index].item()
        return TokenSample(torch.from_numpy(self._tokens[begin_ + begin : begin_ + end]), [end - begin])

    def get_document_sizes(self) -> torch.Tensor:
        return torch.from_numpy(self._size_cumsums[1:] - self._size_cumsums[:-1])

    def get_document_size(self, index: int) -> int:
        return self._size_cumsums[index + 1].item() - self._size_cumsums[index].item()

    @classmethod
    def write(cls, documents: typing.Iterable[TokenSample], stream: io.BufferedWriter) -> TokenReaderConfig:
        begin = stream.tell()
        size_cumsum = [0]
        data_type = None
        for document in documents:
            if data_type is None:
                data_type = document.tokens.dtype
            else:
                Assert.eq(data_type, document.tokens.dtype)
            stream.write(document.tokens.numpy().tobytes())
            size_cumsum.append(size_cumsum[-1] + len(document.tokens))

        # Write the cumsums (pointers and sizes)
        stream.write(np.array(size_cumsum, dtype=np.uint64).tobytes(order="C"))

        return TokenReaderConfig(
            begin=begin,
            end=stream.tell(),
            num_documents=len(size_cumsum) - 1,
            num_tokens=size_cumsum[-1],
            data_type=DataType.from_torch(data_type),
        )
