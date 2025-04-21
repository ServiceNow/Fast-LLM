import io
import pathlib
import struct
import typing

import numpy as np
import PIL.Image

from fast_llm.data.dataset.gpt.indexed import GPTIndexedDataset
from fast_llm.data.dataset.gpt.sampled import GPTSample
from fast_llm.data.preparator.gpt_memmap.config import MEMMAP_DTYPES, MEMMAP_DTYPES_INV, MEMMAP_INDEX_HEADER
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.utils import Assert, div


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
        num_documents: int | None,
        num_tokens: int | None,
        num_pixels: int | None,
    ) -> None:
        super().__init__()
        self._name = name
        self._prefix = pathlib.Path(prefix)
        self._has_spans = 0
        self._has_images = 0

        with self._prefix.with_suffix(".idx").open("rb") as stream:
            Assert.eq(stream.read(9), MEMMAP_INDEX_HEADER, msg=f"File: {stream.name}")
            self._version = struct.unpack("<Q", stream.read(8))[0]
            assert self._version in [1, 2, 3], f"Unsupported version for gpt_memmap dataset: {self._version}."
            if self._version >= 2:
                self._has_spans = struct.unpack("<B", stream.read(1))[0]

            if self._version >= 3:
                self._has_images = struct.unpack("<B", stream.read(1))[0]

            self._dtype = MEMMAP_DTYPES[struct.unpack("<B", stream.read(1))[0]].numpy
            self._num_documents = struct.unpack("<Q", stream.read(8))[0]
            _ = struct.unpack("<Q", stream.read(8))[0]
            offset = stream.tell()

        if num_documents is not None:
            assert self._num_documents == num_documents

        self._index_bin_buffer_mmap = np.memmap(self._prefix.with_suffix(".idx"), mode="r", order="C")
        self._index_bin_buffer = memoryview(self._index_bin_buffer_mmap)
        self._document_sizes = np.frombuffer(
            self._index_bin_buffer, dtype=np.int32, count=self._num_documents, offset=offset
        )
        self._pointers = np.frombuffer(
            self._index_bin_buffer,
            dtype=np.int64,
            count=self._num_documents,
            offset=offset + self._document_sizes.nbytes,
        )

        offset += self._document_sizes.nbytes + self._pointers.nbytes
        self._spans = None
        if self._has_spans and self._version >= 2:
            self._spans = []
            self._num_spans = np.frombuffer(
                self._index_bin_buffer,
                dtype=np.int32,
                count=self._num_documents,
                offset=offset,
            )
            self._num_spans_cumsum = np.r_[0, np.cumsum(self._num_spans[:-1], dtype=np.int64)]
            for idx in range(self._num_documents):
                self._spans.append(
                    np.frombuffer(
                        self._index_bin_buffer,
                        dtype=np.int32,
                        count=self._num_spans[idx] * 2,
                        offset=offset
                        + self._num_spans.nbytes
                        + self._num_spans_cumsum[idx] * 2 * np.dtype(np.int32).itemsize,
                    ).reshape(-1, 2)
                )
            offset += (
                self._num_spans.nbytes
                + self._num_spans.sum() * 2 * np.dtype(np.int32).itemsize
                + sum([x.nbytes for x in self._spans])
            )
        self._n_pixels = 0
        if self._has_images and self._version >= 3:
            self._n_images = np.frombuffer(
                self._index_bin_buffer, dtype=np.int32, count=self._num_documents, offset=offset
            )
            self._im_lengths = []
            self._im_positions = []
            images_seen = 0
            # TODO Soham: verify correctness, reshaping into width, height?
            for n_images in self._n_images:
                self._im_lengths.append(
                    np.frombuffer(
                        self._index_bin_buffer,
                        dtype=np.int32,
                        count=n_images * 2,
                        offset=offset + self._n_images.nbytes + 2 * images_seen * np.dtype(np.int32).itemsize,
                    ).reshape(-1, 2)
                )
                self._n_pixels += self._im_lengths[-1].prod(axis=1, initial=3).sum()
                self._im_positions.append(
                    np.frombuffer(
                        self._index_bin_buffer,
                        dtype=np.int32,
                        count=n_images,
                        offset=offset
                        + self._n_images.nbytes
                        + 2 * self._n_images.sum() * np.dtype(np.int32).itemsize
                        + images_seen * np.dtype(np.int32).itemsize,
                    )
                )
                images_seen += n_images

        self._bin_buffer_mmap = np.memmap(self._prefix.with_suffix(".bin"), mode="r", order="C")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

        # TODO Soham: fix num_tokens to include images. Get total number of image pixels from index file and assign
        # self._num_tokens = div(self._bin_buffer_mmap.size - n_pixels, np.dtype(self._dtype).itemsize)
        self._num_tokens = div(self._bin_buffer_mmap.size - self._n_pixels, np.dtype(self._dtype).itemsize)
        if num_pixels is not None:
            assert self._n_pixels == num_pixels
        if num_tokens is not None:
            assert self._num_tokens == num_tokens

    def __getstate__(self) -> tuple[str, pathlib.Path, int | None, int | None]:
        return (self._name, self._prefix, self._num_documents, self._num_tokens)

    def __setstate__(self, state: tuple[str, pathlib.Path, int | None, int | None]):
        self._init(*state)

    def __del__(self):
        if hasattr(self, "_bin_buffer_mmap"):
            self._bin_buffer_mmap._mmap.close()  # noqa
            del self._bin_buffer_mmap
        if hasattr(self, "_index_bin_buffer"):
            self._index_bin_buffer_mmap._mmap.close()  # noqa
            del self._index_bin_buffer_mmap

    # TODO Soham: get images
    def get(
        self,
        idx: int,
        offset: int = 0,
        length: int | None = None,
        use_loss_masking_spans: bool = False,
        # , patch_size: tuple(int), max_height: int, max_width: int
    ):
        # TODO Soham: Handle truncations?
        # if self._has_images:
        #     doc_size = self._document_sizes[idx]
        #     n_images = self._n_images[idx]
        #     image_positions = self._im_positions[idx]
        #     image_lengths = self._im_lengths[idx]
        #     image_tokens_seen = 0
        #     for idx in range(n_images):
        #         height, width = ImageProcessor.get_resize_dims(image_lengths[0], image_lengths[1], max_height, max_width)
        #         n_image_tokens = (height // patch_size[0]) * (width // patch_size[1])
        #         if (image_positions[idx] > offset + length) or (image_positions[idx] + n_tokens < offset):
        #             continue
        token_ids = np.frombuffer(
            self._bin_buffer,
            dtype=self._dtype,
            count=self._document_sizes[idx] - offset if length is None else length,
            offset=self._pointers[idx] + offset * np.dtype(self._dtype).itemsize,
        )
        if self._has_images:
            image_positions = self._im_positions[idx]
            images = np.frombuffer(
                self._bin_buffer,
                dtype=np.dtype(np.uint8).itemsize,
                count=self._image_lengths[idx][0] * self._image_lengths[idx][1] * 3,
                offset=self._pointers[idx] + self._document_sizes[idx] * np.dtype(self._dtype).itemsize,
            )
        return GPTSample(token_ids=token_ids, images=images, image_positions=image_positions)

    def get(
        self, idx: int, offset: int = 0, length: int | None = None, use_loss_masking_spans: bool = False
    ) -> GPTSample:
        token_ids = np.frombuffer(
            self._bin_buffer,
            dtype=self._dtype,
            count=self._document_sizes[idx] - offset if length is None else length,
            offset=self._pointers[idx] + offset * np.dtype(self._dtype).itemsize,
        )
        sample_spans = None
        if use_loss_masking_spans and self._spans is not None:
            sample_spans = self._spans[idx]
            # adjust the spans for the offset and length
            sample_spans = sample_spans[
                (sample_spans[:, 0] < offset + len(token_ids)) & (sample_spans[:, 1] >= offset)
            ]
            sample_spans[:, 0] = np.maximum(sample_spans[:, 0], offset) - offset
            sample_spans[:, 1] = np.minimum(sample_spans[:, 1], offset + len(token_ids) - 1) - offset
        return GPTSample(token_ids=token_ids, loss_masking_spans=sample_spans)

    @property
    def name(self) -> str:
        return self._name

    def __len__(self) -> int:
        return self._num_documents

    @property
    def num_tokens(self) -> int:
        return self._num_tokens

    @property
    def has_images(self) -> bool:
        return self._has_images

    # TODO: image sizes
    def get_document_sizes(self) -> np.ndarray:
        """
        The size of each document in the dataset.
        The resulting array could be very large, so this method should be called cautiously,
        and derived classes should try to avoid holding the whole array im memory.
        """
        return self._document_sizes, self._im_lengths

    def get_document_size(self, index: int, patch_size: list[int]) -> int:
        return self._document_sizes[index].item() + (
            sum((h // patch_size[0]) * (w // patch_size[1]) for h, w in self._image_lengths[index])
            if self._has_images
            else 0
        )

    @classmethod
    def write_dataset(cls, prefix: pathlib.Path | str, documents: typing.Iterable[GPTSample]):
        # Initialize metadata
        dtype = None
        num_documents = 0
        doc_lengths = []
        n_images = []
        im_lengths = []
        im_positions = []
        total_images = 0
        pointers = []
        offset = 0
        # number of spans for each document
        num_spans = []
        spans = []

        prefix = pathlib.Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)

        # Write the binary data file (.bin) lazily
        with prefix.with_suffix(".bin").open("wb") as bin_stream:
            for document in documents:
                # Infer dtype from the first document
                if dtype is None:
                    dtype = document.token_ids.dtype
                    assert dtype is not None, "Document dtype could not be inferred from the data."

                # Ensure all documents have the same dtype
                assert document.token_ids.dtype == dtype, f"Expected dtype {dtype}, got {document.token_ids.dtype}."

                # Write document to binary file
                bin_stream.write(document.token_ids.tobytes(order="C"))
                total_im_size = 0
                if document.images:
                    n_images.append(len(document.images))
                    total_images += len(document.images)
                    for image in document.images:
                        # assume 3 channels (RGB) for all images
                        with PIL.Image.open(io.BytesIO(image["bytes"])) as img:
                            pixels = np.array(img)
                        im_lengths.append(np.array(pixels.shape[:2]))
                        bin_stream.write(pixels.tobytes(order="C"))
                        total_im_size += pixels.size
                    im_positions.append(document.image_positions)

                # Update metadata
                doc_length = len(document.token_ids)
                doc_lengths.append(doc_length)
                pointers.append(offset)
                if document.loss_masking_spans is not None:
                    num_spans.append(len(document.loss_masking_spans))
                    spans.append(document.loss_masking_spans)
                offset += doc_length * np.dtype(dtype).itemsize + total_im_size * np.dtype(np.uint8).itemsize
                num_documents += 1

        # Finalize metadata arrays
        doc_lengths = np.array(doc_lengths, dtype=np.int32)
        pointers = np.array(pointers, dtype=np.int64)
        num_spans = np.array(num_spans, dtype=np.int32)
        if len(spans) > 0:
            spans = np.vstack(spans, dtype=np.int32)
        else:
            spans = np.array(spans, dtype=np.int32)

        if total_images:
            n_images = np.array(n_images, dtype=np.int32)
        else:
            n_images = np.array([])
        im_lengths = np.stack(im_lengths, dtype=np.int32)
        im_positions = np.array(im_positions, dtype=np.int32)

        # Write the index file (.idx)
        with prefix.with_suffix(".idx").open("wb") as idx_stream:
            idx_stream.write(MEMMAP_INDEX_HEADER)
            # Indicates the version
            # Version 2 onwards optionally add loss-masking spans
            # Version 3 onwards optionally add images
            idx_stream.write(struct.pack("<Q", 3))
            # Flag to indicate whether loss-masking spans are present
            idx_stream.write(struct.pack("<B", 1 if spans.size > 0 else 0))
            # Flag to indicate whether images are present
            idx_stream.write(struct.pack("<B", 1 if total_images > 0 else 0))
            # Data type
            idx_stream.write(struct.pack("<B", MEMMAP_DTYPES_INV[DataType.from_numpy(dtype.type)]))
            # "Number of sequences", same as documents in our case
            idx_stream.write(struct.pack("<Q", num_documents))
            # "Number of documents", needs a +1 for some reason
            idx_stream.write(struct.pack("<Q", num_documents + 1))
            # Sequence (document) doc_lengths
            idx_stream.write(doc_lengths.tobytes(order="C"))
            # Sequence (document) begin offsets in the bin file
            idx_stream.write(pointers.tobytes(order="C"))
            # Number of spans per document
            idx_stream.write(num_spans.tobytes(order="C"))
            # Span indices for each document
            idx_stream.write(spans.tobytes(order="C"))
            # Number of images per document
            idx_stream.write(n_images.tobytes(order="C"))
            # n_pixels * 3 per image
            idx_stream.write(im_lengths.tobytes(order="C"))
            # Position of each image in the document
            idx_stream.write(im_positions.tobytes(order="C"))
            # Document indices, unused but needed for compatibility with Megatron-LM
            idx_stream.write(np.arange(num_documents + 1, dtype=np.int64).tobytes(order="C"))
