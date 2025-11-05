import math
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
from fast_llm.utils import Assert, get_unique


class PatchSample(Sample):
    """
    A reusable component holding a set of fixed-shape patches (ex. images, audio, video),
    each of which providing a single token embedding in a multimodal model.
    """

    def __init__(self, patches: torch.Tensor, token_map: torch.Tensor, position_ids: torch.Tensor, sample_size: int):
        # Tensor of dimensions (patch, *patch_shape)
        self.patches = patches
        # Mapping from patch to token index
        self.token_map = token_map
        # A position identifier for each patch.
        self.position_ids = position_ids
        # Number of tokens in the sample (not the number of patches)
        self.sample_size = sample_size

    @classmethod
    def from_documents(cls, documents: typing.Iterable[typing.Self]) -> typing.Self:
        total_size = 0
        embedding_maps = []
        for document in documents:
            embedding_maps.append(document.token_map + total_size)
            total_size += document.sample_size
        return cls(
            torch.cat([document.patches for document in documents]),
            torch.cat(embedding_maps),
            torch.cat([document.position_ids for document in documents]),
            total_size,
        )

    def crop(self, begin: int, end: int) -> typing.Self:
        sample_size = end - begin
        patch_filter = begin <= self.token_map < end
        return self.__class__(
            self.patches[patch_filter],
            self.token_map[patch_filter] - begin,
            self.position_ids[patch_filter],
            sample_size,
        )

    def __len__(self) -> int:
        return self.sample_size

    def get_padding(self, size: int) -> typing.Self:
        return PatchSample(
            self.patches.new_empty((0, *self.patches.shape[1:])),
            self.token_map.new_empty(0),
            self.position_ids.new_empty(0),
            size,
        )


class PatchBatch(Batch):
    def __init__(
        self,
        patches: torch.Tensor,
        sample_map: torch.Tensor,
        token_map: torch.Tensor,
        position_ids: torch.Tensor,
        num_samples: int,
        sample_size: int,
    ):
        # Concatenated along patch index rather than stacked since the lengths are not constant
        self.patches = patches
        # Mapping from patch to sample index
        self.sample_map = sample_map
        self.token_map = token_map
        self.position_ids = position_ids
        self.num_samples = num_samples
        self.sample_size = sample_size

    @classmethod
    def from_samples(cls, samples: typing.Sequence[PatchSample]) -> typing.Self:
        return cls(
            torch.cat([sample.patches for sample in samples]),
            torch.cat(
                [torch.full_like(sample.token_map, sample_index) for sample_index, sample in enumerate(samples)]
            ),
            torch.cat([sample.token_map for sample in samples]),
            torch.cat([sample.position_ids for sample in samples]),
            len(samples),
            get_unique(sample.sample_size for sample in samples),
        )

    def to_samples(self) -> list[PatchSample]:
        samples = []
        for sample_index in range(self.num_samples):
            patch_filter = self.sample_map == sample_index
            samples.append(
                PatchSample(
                    self.patches[patch_filter],
                    self.token_map[patch_filter],
                    self.position_ids[patch_filter],
                    self.sample_size,
                )
            )
        return samples

    def crop(self, begin: int, end: int) -> typing.Self:
        sample_size = end - begin
        patch_filter = begin <= self.token_map < end
        return self.__class__(
            self.patches[patch_filter],
            self.sample_map[patch_filter],
            self.token_map[patch_filter],
            self.position_ids[patch_filter],
            self.num_samples,
            sample_size,
        )

    def to_device_(self, device: "torch.device | str"):
        self.patches = self.patches.to(device, non_blocking=True)
        self.sample_map = self.sample_map.to(device, non_blocking=True)
        self.token_map = self.token_map.to(device, non_blocking=True)
        self.position_ids = self.position_ids.to(device, non_blocking=True)


@config_class(dynamic_type={MemmapReaderBaseConfig: "patch"})
class PatchReaderConfig(MemmapReaderConfig):
    _abstract = False
    header: typing.ClassVar[bytes] = b"patch begin"
    footer: typing.ClassVar[bytes] = b"patch end"
    num_documents: int = Field()
    num_patches: int = Field()
    patch_shape: tuple[int, ...] = Field()
    data_type: DataType = Field()

    def __len__(self) -> int:
        return self.num_documents

    @property
    def reader_class(self) -> "type[PatchReader]":
        return PatchReader

    @property
    def writer_class(self) -> "type[PatchWriter]":
        return PatchWriter

    @property
    def patch_size(self) -> int:
        return math.prod(self.patch_shape)

    @property
    def _expected_buffer_size(self) -> int:
        return (
            self.num_patches * self.patch_size * self.data_type.torch.itemsize
            + (self.num_patches + self.num_documents + 1) * torch.int32.itemsize
        )


class PatchReader[ConfigType: PatchReaderConfig](MemmapIndexedDatasetReader[ConfigType]):
    def __init__(self, config: ConfigType, buffer: memoryview):
        super().__init__(config, buffer)
        self._patches = torch.frombuffer(
            self._buffer,
            dtype=self._config.data_type.torch,
            count=self._config.num_patches * self._config.patch_size,
        ).view(self._config.num_patches, *self._config.patch_size)
        self._token_map = torch.frombuffer(
            self._buffer,
            dtype=torch.int32,
            count=self._config.num_patches,
            offset=self._patches.nbytes,
        )
        self._position_ids = torch.frombuffer(
            self._buffer,
            dtype=torch.int32,
            count=self._config.num_patches,
            offset=self._patches.nbytes + self._token_map.nbytes,
        )
        self._count_cumsums = torch.frombuffer(
            self._buffer,
            dtype=torch.int32,
            count=self._config.num_documents + 1,
            offset=self._patches.nbytes + self._token_map.nbytes + self._position_ids.nbytes,
        )

    def get_document(self, index: int, begin: int, end: int) -> Sample:
        token_map = self._token_map[token_slice := slice(self._count_cumsums[index], self._count_cumsums[index + 1])]
        patch_filter = begin <= token_map < end
        return PatchSample(
            self._patches[token_slice][patch_filter],
            token_map[patch_filter] - begin,
            self._position_ids[patch_filter],
            end - begin,
        )


class PatchWriter(MemmapWriter):
    def __enter__(self):
        super().__enter__()
        self._count_cumsum = [0]
        self._token_map = []
        self._position_ids = []
        self._data_type = None
        self._patch_shape = None
        return self

    def write(self, document: PatchSample):
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
        self._position_ids.extend(document.position_ids)
        self._count_cumsum.append(self._count_cumsum[-1] + len(document.patches))

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            Assert.lt(self._count_cumsum[-1], np.iinfo(np.int32).max)
            self._stream.write(np.array(self._token_map, dtype=np.int32).tobytes(order="C"))
            self._stream.write(np.array(self._position_ids, dtype=np.int32).tobytes(order="C"))
            self._stream.write(np.array(self._count_cumsum, dtype=np.int32).tobytes(order="C"))
        super().__exit__(exc_type, exc_val, exc_tb)

    @classmethod
    def _get_config_class(cls) -> type[PatchReaderConfig]:
        return PatchReaderConfig

    def _get_config(self, begin: int, end: int):
        return PatchReaderConfig(
            begin=begin,
            end=end,
            num_documents=len(self._count_cumsum) - 1,
            num_patches=self._count_cumsum[-1],
            patch_shape=self._patch_shape,
            data_type=DataType.from_torch(self._data_type),
        )
