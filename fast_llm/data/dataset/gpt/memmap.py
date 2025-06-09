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
        self._has_preference_spans = False
        
        with self._prefix.with_suffix(".idx").open("rb") as stream:
            Assert.eq(stream.read(9), MEMMAP_INDEX_HEADER, msg=f"File: {stream.name}")
            self._version = struct.unpack("<Q", stream.read(8))[0]
            assert self._version in [1, 2, 3, 4], f"Unsupported version for gpt_memmap dataset: {self._version}."
            if self._version >= 2:
                self._has_spans = struct.unpack("<B", stream.read(1))[0]
            if self._version >= 3:
                self._has_preference_spans = struct.unpack("<B", stream.read(1))[0]
            if self._version >= 4:
                self._has_images = struct.unpack("<B", stream.read(1))[0]
                # not sure of assignment, but has to read something here w.r.t preference loss masking spans
                self._has_preference_spans = struct.unpack("<B", stream.read(1))[0]

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
            )   
        # read preference spans
        self._chosen_spans = None
        self._rejected_spans = None
        if self._has_preference_spans and self._version >= 3:
            self._chosen_spans = []
            self._rejected_spans = []
            for idx in range(self._num_documents):
                self._chosen_spans.append(
                    np.frombuffer(
                        self._index_bin_buffer,
                        dtype=np.int32,
                        count=2,
                        offset=offset + idx * 2 * np.dtype(np.int32).itemsize,
                    )
                )

            rejected_span_offset = offset + np.array(self._chosen_spans).nbytes
            for idx in range(self._num_documents):
                self._rejected_spans.append(
                    np.frombuffer(
                        self._index_bin_buffer,
                        dtype=np.int32,
                        count=2,
                        offset=rejected_span_offset + idx * 2 * np.dtype(np.int32).itemsize,
                    )
                )
            offset += np.array(self._chosen_spans).nbytes + np.array(self._rejected_spans).nbytes

        self._num_pixels = 0
        self._image_lengths = None
        self._image_positions = None
        if self._has_images and self._version >= 4:
            # Read number of images per document
            self._n_images = np.frombuffer(
                self._index_bin_buffer, dtype=np.int32, count=self._num_documents, offset=offset
            )
            offset += self._n_images.nbytes
            # Read image dimensions
            total_images = self._n_images.sum()
            if total_images > 0:
                image_lengths_flat = np.frombuffer(
                    self._index_bin_buffer,
                    dtype=np.int32,
                    count=total_images * 2,
                    offset=offset
                ).reshape(-1, 2)
                offset += image_lengths_flat.nbytes
                
                # Split image lengths by document
                self._image_lengths = []
                img_start = 0
                for n_images in self._n_images:
                    if n_images > 0:
                        self._image_lengths.append(image_lengths_flat[img_start:img_start + n_images])
                        self._num_pixels += self._image_lengths[-1].prod(axis=1, initial=3).sum()
                        img_start += n_images
                    else:
                        self._image_lengths.append(np.array([], dtype=np.int32).reshape(0, 2))
                
                # Read padded image positions
                max_images_per_doc = self._n_images.max() if len(self._n_images) > 0 else 0
                if max_images_per_doc > 0:
                    padded_positions = np.frombuffer(
                        self._index_bin_buffer,
                        dtype=np.int32,
                        count=self._num_documents * max_images_per_doc,
                        offset=offset,
                    ).reshape(self._num_documents, max_images_per_doc)
                    
                    # Filter out padding (-1 values) to get actual positions
                    self._image_positions = []
                    for doc_idx, n_images in enumerate(self._n_images):
                        if n_images > 0:
                            actual_positions = padded_positions[doc_idx][:n_images]
                            # Remove any -1 padding that might exist
                            actual_positions = actual_positions[actual_positions != -1]
                            self._image_positions.append(actual_positions)
                        else:
                            self._image_positions.append(np.array([], dtype=np.int32))
                else:
                    self._image_positions = [np.array([], dtype=np.int32) for _ in range(self._num_documents)]
            else:
                self._image_lengths = [np.array([], dtype=np.int32).reshape(0, 2) for _ in range(self._num_documents)]
                self._image_positions = [np.array([], dtype=np.int32) for _ in range(self._num_documents)]

        self._bin_buffer_mmap = np.memmap(self._prefix.with_suffix(".bin"), mode="r", order="C")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

        self._num_tokens = div(self._bin_buffer_mmap.size - self._num_pixels, np.dtype(self._dtype).itemsize)
        if num_pixels is not None:
            assert self._num_pixels == num_pixels
        if num_tokens is not None:
            assert self._num_tokens == num_tokens

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
        images = None
        image_positions = None
        if self._has_images:
            image_positions = self._image_positions[idx]
            # Truncations with images are not yet supported, so we get all images from the document
            if len(self._image_lengths[idx]) > 0:
                total_pixels_needed = sum(
                    length[0] * length[1] * 3 for length in self._image_lengths[idx]
                )
                
                pixels = np.frombuffer(
                    self._bin_buffer,
                    dtype=np.dtype(np.uint8),
                    count=total_pixels_needed, 
                    offset=self._pointers[idx] + self._document_sizes[idx] * np.dtype(self._dtype).itemsize,
                )
                
                images = []
                start = 0
                for image_length in self._image_lengths[idx]:
                    height, width = image_length[0], image_length[1]
                    n_pixels = height * width * 3
                    image_data = pixels[start : start + n_pixels].reshape(3, height, width)
                    images.append(image_data)
                    start += n_pixels
            else:
                images = []
                
        sample_spans = None
        if use_loss_masking_spans and self._spans is not None:
            sample_spans = self._spans[idx]

            # filter spans that are outside the range of the selected tokens in the document
            sample_spans = sample_spans[
                (sample_spans[:, 0] < offset + len(token_ids)) & (sample_spans[:, 1] >= offset)
            ]

            # subtract by offset to normalize span boundaries
            sample_spans[:, 0] = np.maximum(sample_spans[:, 0], offset) - offset  # offset
            sample_spans[:, 1] = np.minimum(sample_spans[:, 1], offset + len(token_ids) - 1) - offset

        chosen_span = None
        rejected_span = None

        if use_preference_loss_spans:
            if not self._has_preference_spans:
                raise ValueError("No preference spans found in memmap dataset.")
            elif self._has_preference_spans and self._chosen_spans is None:
                raise ValueError("Failed to read chosen spans from memmap dataset.")
            elif self._has_preference_spans and self._rejected_spans is None:
                raise ValueError("Failed to read rejected spans from memmap dataset.")
            else:
                chosen_span = self._chosen_spans[idx]

                # filter spans that are outside the range of the selected tokens in the document
                chosen_span = chosen_span[(chosen_span[0] < offset + len(token_ids)) & (chosen_span[1] >= offset)][0]

                # subtract by offset to normalize span boundaries
                chosen_span[0] = np.maximum(chosen_span[0], offset) - offset  # offset
                chosen_span[1] = np.minimum(chosen_span[1], offset + len(token_ids) - 1) - offset

                rejected_span = self._rejected_spans[idx]

                # filter spans that are outside the range of the selected tokens in the document
                rejected_span = rejected_span[
                    (rejected_span[0] < offset + len(token_ids)) & (rejected_span[1] >= offset)
                ][0]

                # subtract by offset to normalize span boundaries
                rejected_span[0] = np.maximum(rejected_span[0], offset) - offset  # offset
                rejected_span[1] = np.minimum(rejected_span[1], offset + len(token_ids) - 1) - offset

        return GPTSample(
            token_ids=token_ids,
            images=images,
            image_positions=image_positions,
            loss_masking_spans=sample_spans,
            chosen_span=chosen_span,
            rejected_span=rejected_span,
        )

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

    def get_document_sizes(self) -> tuple[np.ndarray, np.ndarray]:
        """
        The size of each document in the dataset.
        The resulting array could be very large, so this method should be called cautiously,
        and derived classes should try to avoid holding the whole array im memory.
        """
        return self._document_sizes, self._image_lengths

    def get_document_size(self, index: int) -> int:
        return self._document_sizes[index].item(), self._image_lengths[index] if self._has_images else []

    @classmethod
    def write_dataset(cls, prefix: pathlib.Path | str, documents: typing.Iterable[GPTSample]):
        # Initialize metadata
        dtype = None
        num_documents = 0
        doc_lengths = []
        n_images = []
        image_lengths = []
        im_positions = []
        total_images = 0
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

                # Write document to binary file
                bin_stream.write(document.token_ids.tobytes(order="C"))
                total_im_size = 0
                if document.images:
                    n_images.append(len(document.images))
                    total_images += len(document.images)
                    for image in document.images:
                        # assume 3 channels (RGB) for all images
                        with PIL.Image.open(io.BytesIO(image["bytes"])) as img:
                            if img.mode != "RGB":
                                # Convert all images to RGB
                                img = img.convert("RGB")
                            pixels = np.array(img).transpose(2, 0, 1)  # HWC to CHW
                            assert pixels.dtype == np.uint8, f"Expected uint8 pixels, got {pixels.dtype}."
                        image_lengths.append(np.array(pixels.shape[1:]))
                        bin_stream.write(pixels.tobytes(order="C"))
                        total_im_size += pixels.size
                    im_positions.append(document.image_positions)
                else:
                    n_images.append(0)
                    im_positions.append([])

                # Update metadata
                doc_length = len(document.token_ids)
                doc_lengths.append(doc_length)
                pointers.append(offset)
                if document.loss_masking_spans is not None:
                    num_spans.append(len(document.loss_masking_spans))
                    spans.append(document.loss_masking_spans)
                if document.chosen_span is not None:
                    chosen_spans.append(document.chosen_span)
                if document.rejected_span is not None:
                    rejected_spans.append(document.rejected_span)
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
        chosen_spans = np.array(chosen_spans, dtype=np.int32).reshape(-1, 2)
        rejected_spans = np.array(rejected_spans, dtype=np.int32).reshape(-1, 2)

        if total_images:
            n_images = np.array(n_images, dtype=np.int32)
            image_lengths = np.stack(image_lengths, dtype=np.int32)
            
            # Pad im_positions to make them equal length
            max_images = max(len(pos_list) for pos_list in im_positions)
            padded_im_positions = []
            for pos_list in im_positions:
                padded_pos = pos_list + [-1] * (max_images - len(pos_list))
                padded_im_positions.append(padded_pos)
            im_positions = np.array(padded_im_positions, dtype=np.int32)
        else:
            n_images = np.array([])
            image_lengths = np.array([])
            im_positions = np.array([])

        # Write the index file (.idx)
        with prefix.with_suffix(".idx").open("wb") as idx_stream:
            idx_stream.write(MEMMAP_INDEX_HEADER)
            # Indicates the version
            # Version 2 onwards optionally add loss-masking spans
            # Version 3 optionally adds chosen/rejected spans
            # Version 4 onwards optionally add images
            idx_stream.write(struct.pack("<Q", 4))
            # Flag to indicate whether loss-masking spans are present
            idx_stream.write(struct.pack("<B", 1 if spans.size > 0 else 0))
            # Placeholder flag for preference spans
            idx_stream.write(struct.pack("<B", 0))
            # Flag to indicate whether images are present
            idx_stream.write(struct.pack("<B", 1 if total_images > 0 else 0))
            # Flag to indicate whether preference loss-masking spans are present
            idx_stream.write(struct.pack("<B", 1 if chosen_spans.size > 0 and rejected_spans.size > 0 else 0))
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
            # Chosen indices for each document
            idx_stream.write(chosen_spans.tobytes(order="C"))
            # Rejected indices for each document
            idx_stream.write(rejected_spans.tobytes(order="C"))
            # Number of images per document
            idx_stream.write(n_images.tobytes(order="C"))
            # n_pixels * 3 per image
            idx_stream.write(image_lengths.tobytes(order="C"))
            # Position of each image in the document
            idx_stream.write(im_positions.tobytes(order="C"))
            # Document indices, unused but needed for compatibility with Megatron-LM
            idx_stream.write(np.arange(num_documents + 1, dtype=np.int64).tobytes(order="C"))
