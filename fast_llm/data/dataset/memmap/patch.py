import typing

import numpy as np
import torch

from fast_llm.data.dataset.memmap.abstract import MemmapReader, MemmapWriter
from fast_llm.data.dataset.memmap.config import PatchReaderConfig
from fast_llm.data.document.abstract import Document
from fast_llm.data.document.patch import PatchDocument, filter_lengths
from fast_llm.data.preprocessing.abstract import PreprocessingConfig
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.utils import Assert


class PatchReader[ConfigType: PatchReaderConfig](MemmapReader[ConfigType]):
    def __init__(self, config: ConfigType, buffer: memoryview, model_preprocessing: PreprocessingConfig | None = None):
        super().__init__(config, buffer, model_preprocessing)
        self._patches = torch.frombuffer(
            self._buffer,
            dtype=self._config.data_type.torch,
            count=self._config.num_patches * self._config.patch_size,
        ).view(self._config.num_patches, *self._config.patch_shape)
        offset = self._patches.nbytes
        self._token_map = torch.frombuffer(
            self._buffer,
            dtype=torch.int32,
            count=self._config.num_patches,
            offset=offset,
        )
        offset += self._token_map.nbytes
        self._positions = torch.frombuffer(
            self._buffer,
            dtype=torch.int32,
            count=self._config.num_patches * self._config.grid_dims,
            offset=offset,
        ).view(self._config.num_patches, self._config.grid_dims)
        offset += self._positions.nbytes
        self._patch_count_cumsums = torch.frombuffer(
            self._buffer,
            dtype=torch.int32,
            count=self._config.num_documents + 1,
            offset=offset,
        )
        offset += self._patch_count_cumsums.nbytes
        self._group_lengths = torch.frombuffer(
            self._buffer,
            dtype=torch.int32,
            count=self._config.num_patch_groups,
            offset=offset,
        )
        offset += self._group_lengths.nbytes
        self._group_count_cumsums = torch.frombuffer(
            self._buffer,
            dtype=torch.int32,
            count=self._config.num_documents + 1,
            offset=offset,
        )

    def get_document(self, index: int, begin: int, end: int) -> Document:
        token_map = self._token_map[
            token_slice := slice(self._patch_count_cumsums[index], self._patch_count_cumsums[index + 1])
        ]
        patch_filter = (token_map >= begin) & (token_map < end)
        return PatchDocument(
            patches=self._patches[token_slice][patch_filter],
            token_map=token_map[patch_filter] - begin,
            positions=self._positions[token_slice][patch_filter],
            lengths=filter_lengths(
                self._group_lengths[self._group_count_cumsums[index] : self._group_count_cumsums[index + 1]].tolist(),
                patch_filter,
            ),
        )

    def get_split(self, begin_index: int, end_index: int) -> dict[str, typing.Any]:
        Assert.custom(lambda x: x == sorted(x), [0, begin_index, end_index, self._config.num_documents])
        num_patches = self._patch_count_cumsums[end_index].item() - self._patch_count_cumsums[begin_index].item()
        return {
            "num_documents": end_index - begin_index,
            "num_patches": num_patches,
            "num_patch_groups": self._group_count_cumsums[end_index].item()
            - self._group_count_cumsums[begin_index].item(),
            "num_pixels": self._config.patch_size * num_patches,
            "patch_shape": self._config.patch_shape,
            "data_type": str(self._config.data_type),
        }


class PatchWriter(MemmapWriter):
    def __enter__(self):
        super().__enter__()
        self._patch_count_cumsum = [0]
        self._group_count_cumsum = [0]
        self._token_map = []
        self._positions = []
        self._group_lengths = []
        self._data_type = None
        self._patch_shape = None
        return self

    def write(self, document: PatchDocument):
        super().write(document)
        if self._data_type is None:
            self._data_type = document.patches.dtype
        else:
            Assert.eq(self._data_type, document.patches.dtype)
        if self._patch_shape is None:
            self._patch_shape = tuple(document.patches.shape[1:])
        else:
            Assert.eq(self._patch_shape, document.patches.shape[1:])
        self._stream.write(document.patches.numpy().tobytes())
        self._token_map.extend(document.token_map)
        self._positions.extend(document.positions)
        self._patch_count_cumsum.append(self._patch_count_cumsum[-1] + len(document.patches))
        self._group_count_cumsum.append(self._group_count_cumsum[-1] + len(document.lengths))
        self._group_lengths.extend(document.lengths)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            Assert.lt(self._patch_count_cumsum[-1], np.iinfo(np.int32).max)
            self._stream.write(np.array(self._token_map, dtype=np.int32).tobytes(order="C"))
            self._stream.write(np.array(self._positions, dtype=np.int32).tobytes(order="C"))
            self._stream.write(np.array(self._patch_count_cumsum, dtype=np.int32).tobytes(order="C"))
            self._stream.write(np.array(self._group_lengths, dtype=np.int32).tobytes(order="C"))
            self._stream.write(np.array(self._group_count_cumsum, dtype=np.int32).tobytes(order="C"))
        super().__exit__(exc_type, exc_val, exc_tb)

    @classmethod
    def _get_config_class(cls) -> type[PatchReaderConfig]:
        return PatchReaderConfig

    def _get_config(self, begin: int, end: int):
        return PatchReaderConfig(
            begin=begin,
            end=end,
            num_documents=len(self._patch_count_cumsum) - 1,
            num_patches=self._patch_count_cumsum[-1],
            num_patch_groups=self._group_count_cumsum[-1],
            patch_shape=self._patch_shape,
            data_type=DataType.from_torch(self._data_type),
            preprocessing=self._preprocessing_config,
        )
