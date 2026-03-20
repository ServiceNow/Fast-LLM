import typing

import numpy as np
import torch

from fast_llm.data.dataset.memmap.abstract import MemmapReader, MemmapWriter
from fast_llm.data.dataset.memmap.config import TokenDataReaderConfig
from fast_llm.data.document.token_data import TokenDataDocument
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.utils import Assert


class TokenDataReader[ConfigType: TokenDataReaderConfig](MemmapReader[ConfigType]):
    def __init__(self, config: ConfigType, buffer: memoryview):
        super().__init__(config, buffer)
        self._data = torch.frombuffer(
            self._buffer,
            dtype=self._config.data_type.torch,
            count=self._config.num_tokens * self._config.size,
        ).view(-1, *self._config.shape)
        self._size_cumsums = torch.frombuffer(
            self._buffer, dtype=torch.int64, count=self._config.num_documents + 1, offset=self._data.nbytes
        )

    def get_document(self, index: int, begin: int, end: int) -> TokenDataDocument:
        begin_ = self._size_cumsums[index].item()
        return TokenDataDocument(data=self._data[begin_ + begin : begin_ + end])

    def get_split(self, begin_index: int, end_index: int) -> dict[str, typing.Any]:
        Assert.custom(lambda x: x == sorted(x), [0, begin_index, end_index, self._config.num_documents])

        return {
            "num_tokens": self._size_cumsums[end_index].item() - self._size_cumsums[begin_index].item(),
            "num_documents": end_index - begin_index,
            "data_type": str(self._config.data_type),
            "shape": self._config.shape,
        }


class TokenDataWriter(MemmapWriter):
    def __enter__(self):
        super().__enter__()
        self._size_cumsum = [0]
        self._data_type = None
        self._shape = None
        return self

    def write(self, document: TokenDataDocument):
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
        )
