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

        with self._prefix.with_suffix(".idx").open("rb") as stream:
            Assert.eq(stream.read(9), MEMMAP_INDEX_HEADER, msg=f"File: {stream.name}")
            self._version = struct.unpack("<Q", stream.read(8))[0]
            assert self._version in [1, 2, 3, 4], f"Unsupported version for gpt_memmap dataset: {self._version}."
            self._has_spans = bool(struct.unpack("<B", stream.read(1))[0]) if self._version >= 2 else False
            self._has_preference_spans = bool(struct.unpack("<B", stream.read(1))[0]) if self._version >= 3 else False
            self._has_images = bool(struct.unpack("<B", stream.read(1))[0]) if self._version >= 4 else False
            self._dtype = MEMMAP_DTYPES[struct.unpack("<B", stream.read(1))[0]].numpy
            self._num_documents = struct.unpack("<Q", stream.read(8))[0]
            _ = struct.unpack("<Q", stream.read(8))[0]
            offset = stream.tell()

        if num_documents is not None:
            assert self._num_documents == num_documents

        self._index_bin_buffer_mmap = np.memmap(self._prefix.with_suffix(".idx"), mode="r", order="C")
        self._index_bin_buffer = memoryview(self._index_bin_buffer_mmap)

        # read document sizes
        self._document_sizes = np.frombuffer(
            self._index_bin_buffer, dtype=np.int32, count=self._num_documents, offset=offset
        )

        # read pointers
        self._pointers = np.frombuffer(
            self._index_bin_buffer,
            dtype=np.int64,
            count=self._num_documents,
            offset=offset + self._document_sizes.nbytes,
        )

        offset += self._document_sizes.nbytes + self._pointers.nbytes
        # read spans
        if self._has_spans:
            offset = self._init_spans(offset)

        if self._has_preference_spans:
            offset = self._init_preference_spans(offset)

        total_pixels, _ = self._init_images(offset) if self._has_images else (0, offset)
        if num_pixels is not None:
            assert total_pixels == num_pixels

        self._bin_buffer_mmap = np.memmap(self._prefix.with_suffix(".bin"), mode="r", order="C")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

        self._num_tokens = div(self._bin_buffer_mmap.size - total_pixels, np.dtype(self._dtype).itemsize)
        if num_tokens is not None:
            assert self._num_tokens == num_tokens

    def _init_spans(self, offset: int) -> int:
        num_spans = np.frombuffer(
            self._index_bin_buffer,
            dtype=np.int32,
            count=self._num_documents,
            offset=offset,
        )
        num_spans_cumsum = np.r_[0, np.cumsum(num_spans[:-1], dtype=np.int64)]
        self._spans = [
            np.frombuffer(
                self._index_bin_buffer,
                dtype=np.int32,
                count=num_spans[idx] * 2,
                offset=offset + num_spans.nbytes + num_spans_cumsum[idx] * 2 * np.dtype(np.int32).itemsize,
            ).reshape(-1, 2)
            for idx in range(self._num_documents)
        ]
        return offset + num_spans.nbytes + num_spans.sum() * 2 * np.dtype(np.int32).itemsize

    def _init_preference_spans(self, offset: int) -> int:
        item_size = np.dtype(np.int32).itemsize
        self._chosen_spans = [
            np.frombuffer(
                self._index_bin_buffer,
                dtype=np.int32,
                count=2,
                offset=offset + 2 * idx * item_size,
            )
            for idx in range(self._num_documents)
        ]
        offset += 2 * item_size * self._num_documents
        self._rejected_spans = [
            np.frombuffer(
                self._index_bin_buffer,
                dtype=np.int32,
                count=2,
                offset=offset + 2 * idx * item_size,
            )
            for idx in range(self._num_documents)
        ]
        return offset + 2 * item_size * self._num_documents

    def _init_images(self, offset: int) -> tuple[int, int]:
        total_pixels = 0
        image_counts = np.frombuffer(self._index_bin_buffer, dtype=np.int32, count=self._num_documents, offset=offset)
        offset += image_counts.nbytes

        self._image_sizes = []
        self._image_positions = []
        item_size = np.dtype(np.int32).itemsize

        for image_count in image_counts:
            self._image_sizes.append(
                np.frombuffer(
                    self._index_bin_buffer,
                    dtype=np.int32,
                    count=image_count * 2,
                    offset=offset,
                ).reshape(-1, 2)
            )
            total_pixels += self._image_sizes[-1].prod(axis=1, initial=3).sum()
            offset += 2 * image_count * item_size

        for image_count in image_counts:
            self._image_positions.append(
                np.frombuffer(self._index_bin_buffer, dtype=np.int32, count=image_count, offset=offset)
            )
            offset += image_count * item_size
        return total_pixels, offset

    def __getstate__(self) -> tuple[str, pathlib.Path, int | None, int | None]:
        return (self._name, self._prefix, self._num_documents, self._num_tokens, self._num_pixels)

    def __setstate__(self, state: tuple[str, pathlib.Path, int | None, int | None]):
        self._init(*state)

    def __del__(self):
        if hasattr(self, "_bin_buffer_mmap"):
            self._bin_buffer_mmap._mmap.close()  # noqa
            del self._bin_buffer_mmap
        if hasattr(self, "_index_bin_buffer"):
            self._index_bin_buffer_mmap._mmap.close()  # noqa
            del self._index_bin_buffer_mmap

    def get(
        self,
        idx: int,
        offset: int = 0,
        length: int | None = None,
        use_loss_masking_spans: bool = False,
        use_preference_loss_spans: bool = False,
    ) -> GPTSample:
        token_ids = np.frombuffer(
            self._bin_buffer,
            dtype=self._dtype,
            count=self._document_sizes[idx] - offset if length is None else length,
            offset=self._pointers[idx] + offset * np.dtype(self._dtype).itemsize,
        )

        loss_masking_spans = self._get_loss_masking_spans(idx, offset, token_ids)
        chosen_span, rejected_span = (
            self._get_preference_spans(idx, offset, token_ids) if use_preference_loss_spans else (None, None)
        )
        images, image_positions = self._get_images(idx)

        return GPTSample(
            token_ids=token_ids,
            images=images,
            image_positions=image_positions,
            loss_masking_spans=loss_masking_spans,
            chosen_span=chosen_span,
            rejected_span=rejected_span,
        )

    def _get_loss_masking_spans(self, idx: int, offset: int, token_ids: np.ndarray) -> np.ndarray | None:
        if not self._has_spans:
            return None
        loss_masking_spans = self._spans[idx]

        # filter spans that are outside the range of the selected tokens in the document
        loss_masking_spans = loss_masking_spans[
            (loss_masking_spans[:, 0] < offset + len(token_ids)) & (loss_masking_spans[:, 1] >= offset)
        ]

        # subtract by offset to normalize span boundaries
        loss_masking_spans[:, 0] = np.maximum(loss_masking_spans[:, 0], offset) - offset  # offset
        loss_masking_spans[:, 1] = np.minimum(loss_masking_spans[:, 1], offset + len(token_ids) - 1) - offset
        return loss_masking_spans

    def _get_preference_spans(self, idx: int, offset: int, token_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self._has_preference_spans:
            raise ValueError(f"Dataset {self.name} doesn't have preference spans.")
        chosen_span = self._chosen_spans[idx]

        # filter spans that are outside the range of the selected tokens in the document
        chosen_span = chosen_span[(chosen_span[0] < offset + len(token_ids)) & (chosen_span[1] >= offset)][0]

        # subtract by offset to normalize span boundaries
        chosen_span[0] = np.maximum(chosen_span[0], offset) - offset  # offset
        chosen_span[1] = np.minimum(chosen_span[1], offset + len(token_ids) - 1) - offset

        rejected_span = self._rejected_spans[idx]

        # filter spans that are outside the range of the selected tokens in the document
        rejected_span = rejected_span[(rejected_span[0] < offset + len(token_ids)) & (rejected_span[1] >= offset)][0]

        # subtract by offset to normalize span boundaries
        rejected_span[0] = np.maximum(rejected_span[0], offset) - offset  # offset
        rejected_span[1] = np.minimum(rejected_span[1], offset + len(token_ids) - 1) - offset
        return chosen_span, rejected_span

    def _get_images(self, idx: int) -> tuple[list[np.ndarray] | None, np.ndarray | None]:
        if not self._has_images:
            return None, None
        # Truncations with images are not yet supported, so we get all images from the document
        pixels = np.frombuffer(
            self._bin_buffer,
            dtype=np.dtype(np.uint8),
            count=self._image_sizes[idx].prod(initial=3, axis=1).sum(),
            offset=self._pointers[idx] + self._document_sizes[idx] * np.dtype(self._dtype).itemsize,
        )
        images = []
        start = 0
        for image_size in self._image_sizes[idx]:
            n_pixels = image_size.prod(initial=3)
            images.append(pixels[start : start + n_pixels].reshape(3, image_size[0], image_size[1]))
            start += n_pixels
        return images, self._image_positions[idx]

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

    def get_document_sizes(self) -> np.ndarray:
        """
        The size of each document in the dataset.
        The resulting array could be very large, so this method should be called cautiously,
        and derived classes should try to avoid holding the whole array im memory.
        """
        return self._document_sizes

    def get_image_sizes(self) -> list[np.ndarray]:
        return self._image_sizes if self._has_images else [np.array([])] * self._num_documents

    def get_document_size(self, index: int) -> int:
        return self._document_sizes[index].item()

    def get_image_size(self, index: int) -> np.ndarray:
        return self._image_sizes[index] if self._has_images else []

    @classmethod
    def write_dataset(cls, prefix: pathlib.Path | str, documents: typing.Iterable[GPTSample]):
        # Initialize metadata
        dtype = None
        num_documents = 0
        doc_lengths = []
        n_images = []
        image_sizes = []
        image_positions = []
        has_images = False
        pointers = []
        offset = 0
        # number of spans for each document
        num_spans = []
        spans = []
        chosen_spans = []
        rejected_spans = []

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

                pointers.append(offset)
                doc_lengths.append(doc_length := len(document.token_ids))

                # Write document to binary file
                bin_stream.write(document.token_ids.tobytes(order="C"))
                offset += doc_length * np.dtype(dtype).itemsize

                if document.loss_masking_spans is not None:
                    num_spans.append(len(document.loss_masking_spans))
                    spans.append(document.loss_masking_spans)
                if document.chosen_span is not None:
                    chosen_spans.append(document.chosen_span)
                if document.rejected_span is not None:
                    rejected_spans.append(document.rejected_span)

                if document.images is not None:
                    n_images.append(len(document.images))
                    has_images = True
                    for image in document.images:
                        # assume 3 channels (RGB) for all images
                        with PIL.Image.open(io.BytesIO(image["bytes"])) as img:
                            if img.mode != "RGB":
                                # Convert all images to RGB
                                img = img.convert("RGB")
                            pixels = np.array(img).transpose(2, 0, 1)  # HWC to CHW
                            assert pixels.dtype == np.uint8, f"Expected uint8 pixels, got {pixels.dtype}."
                        image_sizes.append(np.array(pixels.shape[1:]))
                        bin_stream.write(pixels.tobytes(order="C"))
                        offset += pixels.size * np.dtype(np.uint8).itemsize
                    image_positions.extend(document.image_positions)
                else:
                    n_images.append(0)

                num_documents += 1

        # Finalize metadata arrays
        doc_lengths = np.array(doc_lengths, dtype=np.int32)
        pointers = np.array(pointers, dtype=np.int64)

        assert len(spans) == len(num_spans)
        if has_loss_masking_spans := len(spans) > 0:
            assert len(spans) == num_documents
            num_spans = np.array(num_spans, dtype=np.int32)
            spans = np.vstack(spans, dtype=np.int32)

        assert len(chosen_spans) == len(rejected_spans)
        if has_preference_spans := len(chosen_spans) > 0:
            assert len(chosen_spans) == num_documents
            chosen_spans = np.array(chosen_spans, dtype=np.int32).reshape(-1, 2)
            rejected_spans = np.array(rejected_spans, dtype=np.int32).reshape(-1, 2)

        if has_images:
            n_images = np.array(n_images, dtype=np.int32)
            image_sizes = np.stack(image_sizes, dtype=np.int32)
            image_positions = np.array(image_positions, dtype=np.int32)

        # Write the index file (.idx)
        with prefix.with_suffix(".idx").open("wb") as idx_stream:
            idx_stream.write(MEMMAP_INDEX_HEADER)
            # Indicates the version
            # Version 2 onwards supports loss-masking spans
            # Version 3 onwards supports preference spans
            # Version 4 onwards supports images
            idx_stream.write(struct.pack("<Q", 4))
            # Flag to indicate whether loss-masking spans are present
            idx_stream.write(struct.pack("<B", bool(has_loss_masking_spans)))
            # Flag to indicate whether preference loss-masking spans are present
            idx_stream.write(struct.pack("<B", bool(has_preference_spans)))
            # Flag to indicate whether images are present
            idx_stream.write(struct.pack("<B", int(has_images)))
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
            if has_loss_masking_spans:
                # Number of spans per document
                idx_stream.write(num_spans.tobytes(order="C"))
                # Span indices for each document
                idx_stream.write(spans.tobytes(order="C"))
                # Chosen indices for each document
            if has_preference_spans:
                idx_stream.write(chosen_spans.tobytes(order="C"))
                # Rejected indices for each document
                idx_stream.write(rejected_spans.tobytes(order="C"))
            if has_images:
                # Number of images per document
                idx_stream.write(n_images.tobytes(order="C"))
                # n_pixels * 3 per image
                idx_stream.write(image_sizes.tobytes(order="C"))
                # Position of each image in the document
                idx_stream.write(image_positions.tobytes(order="C"))
            # Document indices, unused but needed for compatibility with Megatron-LM
            idx_stream.write(np.arange(num_documents + 1, dtype=np.int64).tobytes(order="C"))
