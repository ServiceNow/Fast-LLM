import io
import math
import typing

import numpy as np
import PIL.Image

from fast_llm.data.dataset.gpt.components.config import GPTMemmapDatasetHeader
from fast_llm.data.dataset.gpt.config import GPTSamplingParameters
from fast_llm.data.dataset.gpt.memmap import BufferOffset, ShiftMap
from fast_llm.data.dataset.gpt.sampled import GPTSample
from fast_llm.utils import Assert, div


class GPTImageDatasetComponent:
    def __init__(
        self,
        header: GPTMemmapDatasetHeader,
        index_binary_buffer: memoryview,
        binary_buffer: memoryview,
        offset: BufferOffset,
    ):
        self._header = header
        self._index_binary_buffer = index_binary_buffer
        self._binary_buffer = binary_buffer

        self._count_cumsum = np.frombuffer(
            self._index_binary_buffer,
            dtype=np.int32,
            count=self._header.num_documents + 1,
            offset=offset.value,
        )
        offset.value += self._count_cumsum.nbytes
        self._sizes = np.frombuffer(
            self._index_binary_buffer,
            dtype=np.int32,
            count=self._count_cumsum[-1] * 2,
            offset=offset.value,
        ).reshape(-1, 2)
        offset.value += self._sizes.nbytes
        self._positions = np.frombuffer(
            self._index_binary_buffer,
            dtype=np.int32,
            count=self._count_cumsum[-1],
            offset=offset.value,
        ).reshape(-1, 2)
        offset.value += self._positions.nbytes

    def get(
        self,
        index: int,
        start_offset: int,
        end_offset: int,
        shift_map: ShiftMap,
        buffer_offset: BufferOffset,
        parameters: GPTSamplingParameters,
    ) -> tuple[list[np.ndarray] | None, np.ndarray | None]:
        # We get images from the document, discarding those outside the selected range.
        images = []
        positions = []
        for image_index in range(self._count_cumsum[index], self._count_cumsum[index + 1]):
            image_buffer_size = self._sizes[image_index].prod(initial=3)
            image_position = shift_map.shift(self._positions[image_index].item())
            if start_offset <= image_position < end_offset:
                images.append(
                    np.frombuffer(
                        self._binary_buffer,
                        dtype=np.dtype(np.uint8),
                        count=image_buffer_size,
                        offset=buffer_offset.value,
                    ).reshape(3, *self._sizes[image_index])
                )
                positions.append(self._positions[image_index])

            buffer_offset.value += image_buffer_size

    def _get_insert(self, image_index: int, parameters: GPTSamplingParameters):
        height, width = resized_image_length
        height_patches = div(height, parameters.patch_size)
        width_patches = div(width, parameters.patch_size)
        image_size = height_patches * width_patches
        if parameters.image_break_token is not None:
            image_size += height_patches
        elif parameters.image_end_token is not None:
            image_size += 1

        image_token_array = np.full((image_size,), -100, dtype=np.int64)
        if parameters.image_break_token is not None:
            for row in range(height_patches):
                position = (row + 1) * width_patches + row
                image_token_array[position] = parameters.image_break_token

        if parameters.image_end_token is not None:
            # Will override the last image_break_token.
            image_token_array[-1] = parameters.image_end_token

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

        resized_image_lengths = [
            get_resize_dims(
                *image_length,
                parameters.max_image_size,
                parameters.max_image_size,
                parameters.patch_size,
            )
            for image_length in image_lengths
        ]
        return images, positions

    @classmethod
    def write_document_and_gather_index(
        cls, document: GPTSample, index_data: dict[str, typing.Any], binary_stream: io.BufferedWriter
    ):
        has_images = document.images is not None
        if "has_images" in index_data:
            Assert.eq(index_data["has_images"], has_images)
        else:
            index_data["has_images"] = has_images
        if has_images:
            if "image_sizes" not in index_data:
                index_data["image_sizes"] = []
            if "image_positions" not in index_data:
                index_data["image_positions"] = []
            if "num_pixels" not in index_data:
                index_data["num_pixels"] = 0
            for image, image_position in zip(document.images, document.image_positions, strict=True):
                # assume 3 channels (RGB) for all images
                # TODO: Not consistent with GPTSample?
                with PIL.Image.open(io.BytesIO(image["bytes"])) as img:
                    if img.mode != "RGB":
                        # Convert all images to RGB
                        img = img.convert("RGB")
                    pixels = np.array(img).transpose(2, 0, 1)  # HWC to CHW
                    assert pixels.dtype == np.uint8, f"Expected uint8 pixels, got {pixels.dtype}."
                index_data["image_sizes"].append(np.array(pixels.shape[1:]))
                index_data["image_positions"].append(image_position)
                # TODO: Shouldn't pixel count exclude the channel dimension?
                index_data["num_pixels"] += pixels.size
                binary_stream.write(pixels.tobytes(order="C"))
            # Cumsum holds both image counts and buffer offsets.
            if "image_cumsum" not in index_data:
                index_data["image_cumsum"] = [0]
            index_data["image_cumsum"].append(len(index_data["image_sizes"]))

    @classmethod
    def write_index(self, index_data: dict[str, typing.Any], index_stream: io.BufferedWriter):
        if index_data["has_images"]:
            Assert.leq(index_data["image_cumsum"][-1], np.iinfo(np.int32).max)
            Assert.eq(len(index_data["image_cumsum"]), index_data["num_documents"] + 1)
            Assert.eq(len(index_data["image_sizes"]), index_data["image_cumsum"][-1])
            Assert.eq(len(index_data["image_positions"]), index_data["image_cumsum"][-1])
            index_stream.write(np.array(index_data["image_cumsum"], dtype=np.int32).tobytes(order="C"))
            # n_pixels * 3 per image
            index_stream.write(np.stack(index_data["image_sizes"], dtype=np.int32).tobytes(order="C"))
            # Position of each image in the document
            index_stream.write(np.array(index_data["image_positions"], dtype=np.int32).tobytes(order="C"))

    def get_sizes(self, index: int, parameters: GPTSamplingParameters) -> list[int]:
        return [
            get_num_image_tokens(
                *get_resize_dims(
                    *size.item(),
                    parameters.max_image_size,
                    parameters.max_image_size,
                    parameters.patch_size,
                ),
                parameters.patch_size,
                image_break=parameters.image_break_token is not None,
                image_end=parameters.image_end_token is not None,
            )
            for size in self._sizes[self._count_cumsum[index] : self._count_cumsum[index + 1]]
        ]

    def get_unshifted_positions_and_sizes(
        self, index: int, parameters: GPTSamplingParameters
    ) -> list[tuple[int, int]]:
        return [
            (position, size)
            for position, size in zip(
                self._positions[self._count_cumsum[index] : self._count_cumsum[index + 1]],
                self.get_sizes(index, parameters),
                strict=True,
            )
        ]


def get_num_image_tokens(height: int, width: int, patch_size: int, image_break: bool, image_end: bool) -> int:
    """
    Calculate the number of image tokens.
    If image_break is True, we consider 1 additional token after every row of patches.
    """
    height_patches = div(height, patch_size)
    width_patches = div(width, patch_size)
    num_tokens = height_patches * width_patches
    if image_break:
        num_tokens += height_patches
    elif image_end:
        num_tokens += 1
    return num_tokens


def get_resize_dims(height: int, width: int, max_height: int, max_width: int, patch_size: int) -> tuple[int, int]:
    """
    Calculate the new dimensions for resizing an image while maintaining the aspect ratio.
    If the image is larger than the max dimensions, it will be resized to fit within them.
    If the image is smaller, it will be resized to the nearest multiple of the patch size.
    """
    ratio = max(height / max_height, width / max_width)
    if ratio > 1:
        # Resize to fit within max dimensions
        height = int(height / ratio)
        width = int(width / ratio)
    return patch_size * math.ceil(height / patch_size), patch_size * math.ceil(width / patch_size)
