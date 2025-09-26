import functools
import json
import pathlib
import struct
import typing

import numpy as np

from fast_llm.data.dataset.gpt.components.config import GPTMemmapDatasetHeader
from fast_llm.data.dataset.gpt.components.images import GPTImageDatasetComponent
from fast_llm.data.dataset.gpt.components.spans import GPTSpansDatasetComponent
from fast_llm.data.dataset.gpt.components.tokens import GPTTokensDatasetComponent
from fast_llm.data.dataset.gpt.config import GPTSamplingParameters
from fast_llm.data.dataset.gpt.indexed import GPTIndexedDataset
from fast_llm.data.dataset.gpt.sampled import GPTSample
from fast_llm.data.preparator.gpt_memmap.config import MEMMAP_INDEX_HEADER
from fast_llm.utils import Assert, div


class BufferOffset:
    # This makes offsets mutable.
    def __init__(self, value: int):
        self.value: int = value


class ShiftMap:
    """
    A map between original and shifted token indices (i.e., accounting for extra content such as images).
    Also serves as a cache so we don't have to recompute positions and sizes every time.
    """

    def __init__(self, positions_and_sizes: list[tuple[int, int]]):
        self._positions_and_sizes = positions_and_sizes

    @functools.cached_property
    def shifted_positions(self) -> list[int]:
        return [self.shift(position) for position, _ in self._positions_and_sizes]

    def shift(self, index: int) -> int:
        return index + sum(size for position, size in self._positions_and_sizes if index > position)

    def unshift(self, index: int) -> int:
        return index - sum(
            size
            for shifted_position, (_, size) in zip(self.shifted_positions, self._positions_and_sizes, strict=True)
            if shifted_position < index
        )


class GPTMemmapDataset(GPTIndexedDataset):
    """
    A memory map dataset, which handles lazy loading of a pre-processed dataset in the Megatron-LM format,
    i.e. a pair of numpy file containing
    1. A data file (`{prefix}.bin`) containing a flat buffer containing the concatenated, tokenized documents.
    2. An index file (`{prefix}.idx`) containing a list of document sizes and pointers (start index) in the data file.
    See https://github.com/NVIDIA/Megatron-LM?tab=readme-ov-file#data-preprocessing for more details.
    """

    def __init__(
        self,
        name: str,
        prefix: pathlib.Path | str,
        num_documents: int | None = None,
        num_tokens: int | None = None,
        num_pixels: int | None = None,
    ):
        self._init(name, prefix, num_documents, num_tokens, num_pixels)

    def _init(
        self,
        name: str,
        prefix: pathlib.Path | str,
        num_documents: int | None = None,
        num_tokens: int | None = None,
        num_pixels: int | None = None,
    ) -> None:
        super().__init__()
        self._name = name
        self._prefix = pathlib.Path(prefix)

        with self._prefix.with_suffix(".idx").open("rb") as stream:
            Assert.eq(stream.read(9), MEMMAP_INDEX_HEADER, msg=f"File: {stream.name}")
            self._version = struct.unpack("<Q", stream.read(8))[0]
            assert self._version in [1, 2, 3, 4, 5], f"Unsupported version for gpt_memmap dataset: {self._version}."
            if self._version < 5:
                raise NotImplementedError()
                # TODO: Not backward compatible.
                # self._header = GPTMemmapDatasetHeader(
                #    has_spans = self._version >= 2 and bool(struct.unpack("<B", stream.read(1))[0]),
                #    has_preference_spans=self._version >= 3 and bool(struct.unpack("<B", stream.read(1))[0]),
                #    has_images=self._version >= 4 and bool(struct.unpack("<B", stream.read(1))[0]),
                #    token_data_type = MEMMAP_DTYPES[struct.unpack("<B", stream.read(1))[0]],
                #    num_documents = struct.unpack("<Q", stream.read(8))[0]
                # )
            else:
                header_length = struct.unpack("<Q", stream.read(8))[0]
                self._header = GPTMemmapDatasetHeader(**json.loads(stream.read(header_length).decode("utf-8")))

            offset = BufferOffset(stream.tell())

        if num_documents is not None and self._header.num_documents != num_documents:
            raise ValueError(
                f"Inconsistent num_documents for dataset {self.name} - {self._prefix}."
                f" Expected {num_documents}, got {self._header.num_documents}."
            )

        self._index_binary_buffer_mmap = np.memmap(self._prefix.with_suffix(".idx"), mode="r", order="C")
        self._index_binary_buffer = memoryview(self._index_binary_buffer_mmap)
        self._binary_buffer_mmap = np.memmap(self._prefix.with_suffix(".bin"), mode="r", order="C")
        self._binary_buffer = memoryview(self._binary_buffer_mmap)

        self._tokens = GPTTokensDatasetComponent(self._header, self._index_binary_buffer, self._binary_buffer, offset)

        # Read pointers to the beginning of each document
        self._buffer_offsets = np.frombuffer(
            self._index_binary_buffer,
            dtype=np.int64,
            count=self._header.num_documents,
            offset=offset.value,
        )
        offset.value += self._buffer_offsets.nbytes

        self._spans = (
            GPTSpansDatasetComponent(self._header, self._index_binary_buffer, self._binary_buffer, offset)
            if self._header.has_spans
            else None
        )
        self._images = (
            GPTImageDatasetComponent(self._header, self._index_binary_buffer, self._binary_buffer, offset)
            if self._header.has_images
            else None
        )

        if num_pixels is not None:
            Assert.eq(num_pixels, self._images.total_pixels)

        # TODO: Simplify.
        self._num_tokens = (
            self._binary_buffer_mmap.size
            if self._images is None
            else self._binary_buffer_mmap.size - self._images.total_pixels
        )
        if num_tokens is not None:
            Assert.eq(num_tokens, self._num_tokens)

    def __getstate__(self) -> tuple[str, pathlib.Path]:
        return (self._name, self._prefix)

    def __setstate__(self, state: tuple[str, pathlib.Path]):
        self._init(*state)

    def __del__(self):
        if hasattr(self, "_bin_buffer_mmap"):
            self._binary_buffer_mmap._mmap.close()  # noqa
            del self._binary_buffer_mmap
        if hasattr(self, "_index_bin_buffer"):
            self._index_binary_buffer_mmap._mmap.close()  # noqa
            del self._index_binary_buffer_mmap

    def get(
        self,
        index: int,
        start_offset: int = 0,
        end_offset: int | None = None,
        parameters: GPTSamplingParameters | None = None,
    ) -> GPTSample:

        if end_offset is None:
            end_offset = self.get_document_size(index, parameters)

        shift_map = ShiftMap(
            self._images.get_unshifted_positions_and_sizes(index, parameters) if parameters.use_images else []
        )

        buffer_offset = BufferOffset(self._buffer_offsets[index].item())
        sample = GPTSample(token_ids=self._tokens.get(index, start_offset, end_offset, shift_map, buffer_offset))

        if parameters.use_loss_masking_spans:
            sample.loss_masking_spans = self._spans.get(index, start_offset, end_offset, shift_map)

        if parameters.use_images:
            sample.images, sample.image_positions = self._images.get(
                index, start_offset, end_offset, shift_map, buffer_offset
            )

            start_pos = 0
            sample_token_ids = []
            for idx, im_position in enumerate(sample.image_positions):
                # add placeholder masked tokens for images
                # if image_break_token is set, it is appended after every row
                # if image_end_token is set, it is appended at the end of the image instead  of image_break_token
                text_part = sample.token_ids[start_pos:im_position]
                if parameters.image_break_token is not None:
                    height, width = resized_image_lengths[idx]
                    num_patches_h = div(height, parameters.patch_size)
                    num_patches_w = div(width, parameters.patch_size)
                    image_token_array = np.full((image_sizes[idx],), -100, dtype=np.int64)
                    # account for break tokens after each row
                    for row in range(num_patches_h - 1):
                        position = (row + 1) * num_patches_w + row
                        image_token_array[position] = parameters.image_break_token
                    # handle the last row separately
                    last_row_position = num_patches_h * num_patches_w + num_patches_h - 1
                    if parameters.image_end_token is not None:
                        image_token_array[last_row_position] = parameters.image_end_token
                    else:
                        image_token_array[last_row_position] = parameters.image_break_token
                else:
                    image_token_array = np.full((image_sizes[idx],), -100, dtype=np.int64)
                    if parameters.image_end_token is not None:
                        image_token_array[-1] = parameters.image_end_token
                sample_token_ids.append(np.concatenate([text_part, image_token_array], dtype=np.int64))
                text_tokens_added += len(text_part)
                image_positions.append(text_tokens_added + image_tokens_added)
                image_sizes[idx]
                start_pos = im_position

        return sample

    @property
    def name(self) -> str:
        return self._name

    def __len__(self) -> int:
        return self._header.num_documents

    def get_document_sizes(self, parameters: GPTSamplingParameters | None = None) -> np.ndarray:
        """
        The size of each document in the dataset.
        The resulting array could be very large, so this method should be called cautiously,
        and derived classes should try to avoid holding the whole array im memory.
        """
        if parameters is not None and parameters.use_images:
            # TODO: Optimize this.
            return np.array([self.get_document_size(index, parameters) for index in range(self._header.num_documents)])
        return self._tokens.sizes

    def get_document_size(self, index: int, parameters: GPTSamplingParameters | None = None) -> int:
        size = self._tokens.sizes[index].item()
        if parameters is not None and parameters.use_images:
            for _, size_ in self._images.get_positions_and_sizes(index, parameters):
                size += size_
        return size

    def _shift_offset(self, offset, index: int, parameters: GPTSamplingParameters | None = None) -> int:
        if parameters is not None and parameters.use_images:
            offset += sum(
                size for position, size in self._images.get_positions_and_sizes(index, parameters) if position < offset
            )
        return offset

    def _unshift_offset(self, offset, index: int, parameters: GPTSamplingParameters | None = None) -> int:
        unshifted_offset = offset
        if parameters is not None and parameters.use_images:
            for position, size in self._images.get_positions_and_sizes(index, parameters):
                shifted_position = self._shift_offset(position, index, parameters)
                if shifted_position < offset:
                    unshifted_offset -= size
        return unshifted_offset

    @classmethod
    def write_dataset(cls, prefix: pathlib.Path | str, documents: typing.Iterable[GPTSample]):
        buffer_offsets = []
        index_data = {}
        num_documents = 0
        component_classes = (GPTTokensDatasetComponent, GPTSpansDatasetComponent, GPTImageDatasetComponent)

        prefix = pathlib.Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        # Write the binary data file (.bin) lazily
        with prefix.with_suffix(".bin").open("wb") as binary_stream:

            for document in documents:
                buffer_offsets.append(binary_stream.tell())
                for component_class in component_classes:
                    component_class.write_document_and_gather_index(document, index_data, binary_stream)

                # TODO: Address
                assert document.chosen_span is None and document.rejected_span is None

                num_documents += 1

        # Write the index file (.idx)
        with prefix.with_suffix(".idx").open("wb") as index_stream:
            index_stream.write(MEMMAP_INDEX_HEADER)
            # Version.
            index_stream.write(struct.pack("<Q", 5))
            header = GPTMemmapDatasetHeader(
                num_documents=num_documents,
                token_data_type=index_data["token_data_type"],
                has_spans=index_data["has_spans"],
                has_images=index_data["has_images"],
            )
            header_binary = json.dumps(header).encode("utf-8")
            index_stream.write(struct.pack("<Q", len(header_binary)))
            index_stream.write(header_binary)

            # Document begin offsets in the binary file TODO: Address reordering.
            index_stream.write(np.array(buffer_offsets, dtype=np.int64).tobytes(order="C"))

            for component_class in component_classes:
                component_class.write_index(index_data, index_stream)
