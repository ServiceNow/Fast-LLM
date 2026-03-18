import typing

import numpy as np
import torch

from fast_llm.data.dataset.memmap.abstract import MemmapIndexedDatasetReader, MemmapWriter
from fast_llm.data.dataset.memmap.config import TokenReaderConfig
from fast_llm.data.document.token import TokenDocument
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.utils import Assert


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

    def get_document(self, index: int, begin: int, end: int) -> TokenDocument:
        begin_ = self._size_cumsums[index].item()
        # Torch doesn't support type promotion between signed and unsigned types, so we convert here to avoid issues.
        # Convert begin and end to int to avoid numpy dtype overflow when adding to begin_
        return TokenDocument(tokens=self._tokens[begin_ + begin : begin_ + end].to(torch.int64))

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

    def write(self, document: TokenDocument):
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
        )
