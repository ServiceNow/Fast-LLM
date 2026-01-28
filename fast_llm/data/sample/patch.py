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
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.utils import Assert, get_unique, padded_cumsum


def filter_lengths(lengths: list[int], filter: torch.Tensor) -> list[int]:
    length_cumsum = padded_cumsum(lengths)
    filtered_lengths = (filter[begin:end].sum().item() for begin, end in zip(length_cumsum[:-1], length_cumsum[1:]))
    return [length for length in filtered_lengths if length > 0]


class PatchSample(Sample):
    """
    A reusable component holding a set of fixed-shape patches (ex. images, audio, video),
    each of which providing a single token embedding in a multimodal model.
    """

    def __init__(
        self,
        patches: torch.Tensor,
        token_map: torch.Tensor,
        positions: torch.Tensor,
        sample_size: int,
        lengths: list[int] | None = None,
    ):
        # Tensor of dimensions (patch, *patch_shape)
        self.patches = patches
        # Mapping from patch to token index
        self.token_map = token_map
        # A position identifier for each patch in the patch grid.
        Assert.eq(positions.shape, (self.patches.size(0), self.patches.ndim - 2))
        self.positions = positions
        # Number of tokens in the sample (not the number of patches)
        self.sample_size = sample_size
        # Length of each patch group (ex. image) in the sample. TODO: Use cumsums instead?
        if lengths is None:
            lengths = [len(patches)]
        else:
            Assert.eq(sum(lengths), len(patches))
        self.lengths = lengths

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
            torch.cat([document.positions for document in documents]),
            total_size,
            sum((document.lengths for document in documents), []),
        )

    def crop(self, begin: int, end: int) -> typing.Self:
        sample_size = end - begin
        patch_filter = (self.token_map >= begin) & (self.token_map < end)
        return self.__class__(
            self.patches[patch_filter],
            self.token_map[patch_filter] - begin,
            self.positions[patch_filter],
            sample_size,
            filter_lengths(self.lengths, patch_filter),
        )

    def __len__(self) -> int:
        return self.sample_size

    def get_padding(self, size: int) -> typing.Self:
        return self.__class__(
            self.patches.new_empty((0, *self.patches.shape[1:])),
            self.token_map.new_empty(0),
            self.positions.new_empty([0, self.patches.ndim - 2]),
            size,
            [],
        )


class PatchBatch(Batch):
    def __init__(
        self,
        patches: torch.Tensor,
        sample_map: torch.Tensor,
        token_map: torch.Tensor,
        positions: torch.Tensor,
        num_samples: int,
        sample_size: int,
        lengths: list[int],
    ):
        # Concatenated along patch index rather than stacked since the lengths are not constant
        self.patches = patches
        # Mapping from patch to sample index
        self.sample_map = sample_map
        self.token_map = token_map
        self.positions = positions
        self.num_samples = num_samples
        self.sample_size = sample_size
        self.lengths = lengths

    @classmethod
    def from_samples(cls, samples: typing.Sequence[PatchSample]) -> typing.Self:
        return cls(
            torch.cat([sample.patches for sample in samples]),
            torch.cat(
                [torch.full_like(sample.token_map, sample_index) for sample_index, sample in enumerate(samples)]
            ),
            torch.cat([sample.token_map for sample in samples]),
            torch.cat([sample.positions for sample in samples]),
            len(samples),
            get_unique(sample.sample_size for sample in samples),
            [length for sample in samples for length in sample.lengths],
        )

    def crop(self, begin: int, end: int) -> typing.Self:
        sample_size = end - begin
        patch_filter = (self.token_map >= begin) & (self.token_map < end)

        return self.__class__(
            self.patches[patch_filter],
            self.sample_map[patch_filter],
            self.token_map[patch_filter],
            self.positions[patch_filter],
            self.num_samples,
            sample_size,
            filter_lengths(self.lengths, patch_filter),
        )

    def to_device_(self, device: "torch.device | str"):
        self.patches = self.patches.to(device, non_blocking=True)
        self.sample_map = self.sample_map.to(device, non_blocking=True)
        self.token_map = self.token_map.to(device, non_blocking=True)
        self.positions = self.positions.to(device, non_blocking=True)


@config_class()
class PatchReaderBaseConfig(MemmapReaderBaseConfig):
    _abstract = False
    patch_shape: tuple[int, ...] = Field()
    data_type: DataType = Field()

    @property
    def patch_size(self) -> int:
        return math.prod(self.patch_shape)

    @property
    def grid_dims(self) -> int:
        return len(self.patch_shape) - 1


@config_class(dynamic_type={MemmapReaderBaseConfig: "patch"})
class PatchReaderConfig(PatchReaderBaseConfig, MemmapReaderConfig):
    header: typing.ClassVar[bytes] = b"patch begin"
    footer: typing.ClassVar[bytes] = b"patch end"
    num_documents: int = Field()
    num_patches: int = Field()
    num_patch_groups: int = Field()

    def __len__(self) -> int:
        return self.num_documents

    @property
    def reader_class(self) -> "type[PatchReader]":
        return PatchReader

    @property
    def writer_class(self) -> "type[PatchWriter]":
        return PatchWriter

    @property
    def _expected_buffer_size(self) -> int:
        return (
            self.num_patches * self.patch_size * self.data_type.torch.itemsize
            + ((1 + self.grid_dims) * self.num_patches + self.num_patch_groups + 2 * self.num_documents + 2)
            * torch.int32.itemsize
        )

    def get_metadata(self) -> dict[str, typing.Any]:
        return {
            "num_documents": self.num_documents,
            "num_patches": self.num_patches,
            "num_patch_groups": self.num_patch_groups,
            "num_pixels": self.patch_size * self.num_patches,
            "patch_shape": self.patch_shape,
            "data_type": str(self.data_type),
        }

    @classmethod
    def blend_metadata(cls, metadata: list[dict[str, typing.Any]]) -> dict[str, typing.Any]:
        return {
            "num_documents": sum(metadata_["num_documents"] for metadata_ in metadata),
            "num_patches": sum(metadata_["num_patches"] for metadata_ in metadata),
            "num_patch_groups": sum(metadata_["num_patch_groups"] for metadata_ in metadata),
            "num_pixels": sum(metadata_["num_pixels"] for metadata_ in metadata),
            "patch_shape": get_unique(metadata_["patch_shape"] for metadata_ in metadata),
            "data_type": get_unique(metadata_["data_type"] for metadata_ in metadata),
        }


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

    def get_document(self, index: int, begin: int, end: int) -> Sample:
        token_map = self._token_map[
            token_slice := slice(self._patch_count_cumsums[index], self._patch_count_cumsums[index + 1])
        ]
        patch_filter = (token_map >= begin) & (token_map < end)
        return PatchSample(
            self._patches[token_slice][patch_filter],
            token_map[patch_filter] - begin,
            self._positions[token_slice][patch_filter],
            end - begin,
            filter_lengths(
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


class EmptyPatchReader[ConfigType: PatchReaderBaseConfig](MemmapReaderBase[ConfigType]):
    def get_document(self, index: int, begin: int, end: int) -> Sample:
        return PatchSample(
            torch.empty(0, *self._config.patch_shape, dtype=self._config.data_type.torch),
            torch.empty(0, dtype=torch.int32),
            torch.empty(0, self._config.grid_dims, dtype=torch.int32),
            end - begin,
        )


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
