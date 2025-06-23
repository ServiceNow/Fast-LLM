import io
import pathlib
import struct
import typing

import numpy as np
import PIL.Image
import torchaudio
import soundfile as sf

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
            assert self._version in [1, 2, 3, 4, 5], f"Unsupported version for gpt_memmap dataset: {self._version}."
            if self._version >= 2:
                self._has_spans = struct.unpack("<B", stream.read(1))[0]

            if self._version >= 3:
                self._has_preference_spans = struct.unpack("<B", stream.read(1))[0]

            if self._version >= 4:
                self._has_images = struct.unpack("<B", stream.read(1))[0]

            if self._version >= 5:
                self._has_audio = struct.unpack("<B", stream.read(1))[0]

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
                # + sum([x.nbytes for x in self._spans])
            )
        self._num_pixels = 0
        self._image_lengths = []
        self._image_positions = []
        if self._has_images and self._version >= 4:
            self._n_images = np.frombuffer(
                self._index_bin_buffer, dtype=np.int32, count=self._num_documents, offset=offset
            )
            images_seen = 0
            for n_images in self._n_images:
                self._image_lengths.append(
                    np.frombuffer(
                        self._index_bin_buffer,
                        dtype=np.int32,
                        count=n_images * 2,
                        offset=offset + self._n_images.nbytes + 2 * images_seen * np.dtype(np.int32).itemsize,
                    ).reshape(-1, 2)
                )
                self._num_pixels += self._image_lengths[-1].prod(axis=1, initial=3).sum()
                self._image_positions.append(
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
            offset = offset + self._n_images.nbytes + 3 * self._n_images.sum() * np.dtype(np.int32).itemsize
        self._audio_lengths = []  # list of arrays
        self._audio_positions = []  # list of arrays
        if self._has_audio and self._version >= 5:
            self._n_audio = np.frombuffer(
                self._index_bin_buffer, dtype=np.int32, count=self._num_documents, offset=offset
            )
            audio_seen = 0

            offset = offset + self._n_audio.nbytes
            for n_audio in self._n_audio:
                self._audio_lengths.append(
                    np.frombuffer(
                        self._index_bin_buffer,
                        dtype=np.int32,
                        count=n_audio,
                        offset=offset + audio_seen * np.dtype(np.int32).itemsize,
                    )
                )
                # self._num_pixels += self._image_lengths[-1].prod(axis=1, initial=3).sum()
                self._audio_positions.append(
                    np.frombuffer(
                        self._index_bin_buffer,
                        dtype=np.int32,
                        count=n_audio,
                        offset=offset
                        + self._n_audio.sum() * np.dtype(np.int32).itemsize
                        + audio_seen * np.dtype(np.int32).itemsize,
                    )
                )
                audio_seen += n_audio

        self._bin_buffer_mmap = np.memmap(self._prefix.with_suffix(".bin"), mode="r", order="C")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

        # TODO Soham: fix num_tokens to include images. Get total number of image pixels from index file and assign
        # self._num_tokens = div(self._bin_buffer_mmap.size - n_pixels, np.dtype(self._dtype).itemsize)

        # TODO Toby: Add audio num tokens check
        self._num_tokens = div(self._bin_buffer_mmap.size - self._num_pixels, np.dtype(self._dtype).itemsize)
        # if num_pixels is not None:
        #     assert self._num_pixels == num_pixels
        # if num_tokens is not None:
        #     assert self._num_tokens == num_tokens

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

    # def get(
    #     self,
    #     idx: int,
    #     offset: int = 0,
    #     image_offset: int = 0,
    #     length: int | None = None,
    #     use_loss_masking_spans: bool = False,
    # ):
    #     token_ids = np.frombuffer(
    #         self._bin_buffer,
    #         dtype=self._dtype,
    #         count=self._document_sizes[idx] - offset if length is None else length,
    #         offset=self._pointers[idx] + offset * np.dtype(self._dtype).itemsize,
    #     )
    #     if self._has_images:
    #         image_positions = self._image_positions[idx]
    #         pixels = np.frombuffer(
    #             self._bin_buffer,
    #             dtype=np.dtype(np.uint8),
    #             count=self._image_lengths[idx].prod(initial=3),
    #             offset=self._pointers[idx] + self._document_sizes[idx] * np.dtype(self._dtype).itemsize,
    #         )
    #         images = []
    #         start = 0
    #         for image_length in self._image_lengths[idx]:
    #             n_pixels = image_length.prod(initial=3)
    #             images.append(pixels[start : start + n_pixels].reshape(3, image_length[0], image_length[1]))
    #             start += n_pixels
    #     return GPTSample(token_ids=token_ids, images=images, image_positions=image_positions)

    def get(
        self,
        idx: int,
        offset: int = 0,
        length: int | None = None,
        use_loss_masking_spans: bool = False,
        patch_size: int | None = None,
        image_size: int | None = None,
        image_break: bool = False,
        image_end: bool = False,
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
            # Truncations with images are not yet supported
            image_positions = self._image_positions[idx]
            pixels = np.frombuffer(
                self._bin_buffer,
                dtype=np.dtype(np.uint8),
                count=self._image_lengths[idx].prod(initial=3),
                offset=self._pointers[idx] + self._document_sizes[idx] * np.dtype(self._dtype).itemsize,
            )
            start = 0
            for image_length in self._image_lengths[idx]:
                n_pixels = image_length.prod(initial=3)
                images.append(pixels[start : start + n_pixels].reshape(3, image_length[0], image_length[1]))
                start += n_pixels

        audio = []
        audio_positions = None
        if self._has_audio:
            audio_positions = self._audio_positions[idx]
            # increment offset by documents and images
            aud_offset = (
                self._pointers[idx]
                + offset * np.dtype(self._dtype).itemsize
                + self._document_sizes[idx] * np.dtype(self._dtype).itemsize
            )

            if self._has_images and len(self._image_lengths) > 0:
                aud_offset += self._image_lengths[idx].prod(initial=3) * np.dtype(np.uint8).itemsize
            all_audio = np.frombuffer(
                self._bin_buffer,
                dtype=np.dtype(np.float32),
                count=self._audio_lengths[idx].sum(),
                offset=aud_offset,
            )
            start = 0
            for audio_length in self._audio_lengths[idx]:
                audio.append(all_audio[start : start + audio_length])
                start += audio_length

        # TODO Soham: return loss_masking_spans
        sample_spans = None
        if use_loss_masking_spans and self._spans is not None:
            sample_spans = self._spans[idx]
            sample_spans = sample_spans[
                (sample_spans[:, 0] < offset + len(token_ids)) & (sample_spans[:, 1] >= offset)
            ]
            sample_spans[:, 0] = np.maximum(sample_spans[:, 0], offset) - offset
            sample_spans[:, 1] = np.minimum(sample_spans[:, 1], offset + len(token_ids) - 1) - offset
            # if images:
            #     image_idx = 0
            #     for span in sample_spans:
            #         additional_tokens = 0
            #         image_position = image_positions[image_idx] if image_idx < len(image_positions) else float("inf")
            #         while image_position >= span[0] and image_position <= span[1]:
            #             image_tokens = get_num_image_tokens(
            #                 get_resize_dims(*self._image_lengths[idx][image_idx], image_size, image_size, patch_size),
            #                 patch_size,
            #                 image_break=image_break,
            #             )
            #             additional_tokens += image_tokens
            #             image_idx += 1
            #             image_position = (
            #                 image_positions[image_idx] if image_idx < len(image_positions) else float("inf")
            #             )
            #         span[1] += additional_tokens
            # if audio:
            #     audio_idx = 0
            #     for span in sample_spans:
            #         additional_tokens = 0
            #         audio_position = audio_positions[audio_idx] if audio_idx < len(audio_positions) else float("inf")
            #         while audio_position >= span[0] and audio_position <= span[1]:
            #             audio_tokens = ...
            #             additional_tokens += audio_tokens
            #             audio_idx += 1
            #             audio_position = (
            #                 audio_positions[audio_idx] if audio_idx < len(audio_positions) else float("inf")
            #             )
            #         span[1] += additional_tokens

        return GPTSample(
            token_ids=token_ids,
            images=images,
            image_positions=image_positions,
            audio=audio,
            audio_positions=audio_positions,
            loss_masking_spans=sample_spans,
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

    @property
    def has_audio(self) -> bool:
        return self._has_audio

    def get_document_sizes(self) -> tuple[np.ndarray, np.ndarray]:
        """
        The size of each document in the dataset.
        The resulting array could be very large, so this method should be called cautiously,
        and derived classes should try to avoid holding the whole array im memory.
        """
        return self._document_sizes, self._image_lengths, self._audio_lengths

    def get_document_size(self, index: int) -> int:
        # return self._document_sizes[index].item() + (
        #     sum((h // patch_size[0]) * (w // patch_size[1]) for h, w in self._image_lengths[index])
        #     if self._has_images
        #     else 0
        # )
        docsize = self._document_sizes[index].item()
        imagesize = self._image_lengths[index] if self._has_images else []
        audiosize = self._audio_lengths[index] if self._has_audio else []
        return docsize, imagesize, audiosize

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
        n_audio = []
        audio_lengths = []
        aud_positions = []
        total_audio = 0
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
                total_aud_size = 0
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
                if document.audio is not None:
                    num_audio = 0
                    for audio in document.audio:
                        # audio_arr, _ = torchaudio.load(io.BytesIO(audio["bytes"]))
                        audio_arr, _ = sf.read(io.BytesIO(audio["bytes"]))
                        audio_arr = audio_arr.astype(np.float32)
                        if len(audio_arr) > 0:
                            num_audio += 1
                            audio_lengths.append(len(audio_arr))
                            bin_stream.write(audio_arr.tobytes(order="C"))
                            total_aud_size += audio_arr.size
                    n_audio.append(num_audio)
                    total_audio += num_audio
                    if num_audio > 0:
                        aud_positions += document.audio_positions

                # Update metadata
                doc_length = len(document.token_ids)
                doc_lengths.append(doc_length)
                pointers.append(offset)
                if document.loss_masking_spans is not None:
                    num_spans.append(len(document.loss_masking_spans))
                    spans.append(document.loss_masking_spans)
                offset += (
                    doc_length * np.dtype(dtype).itemsize
                    + total_im_size * np.dtype(np.uint8).itemsize
                    + total_aud_size * np.dtype(np.float32).itemsize
                )
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
            image_lengths = np.stack(image_lengths, dtype=np.int32)
            im_positions = np.array(im_positions, dtype=np.int32)
        else:
            n_images = np.array([])
            image_lengths = np.array([])
            im_positions = np.array([])

        if total_audio:
            n_audio = np.array(n_audio, dtype=np.int32)
            audio_lengths = np.array(audio_lengths, dtype=np.int32)
            aud_positions = np.array(aud_positions, dtype=np.int32)
        else:
            n_audio = np.array([])
            audio_lengths = np.array([])
            aud_positions = np.array([])

        # Write the index file (.idx)
        with prefix.with_suffix(".idx").open("wb") as idx_stream:
            idx_stream.write(MEMMAP_INDEX_HEADER)
            # Indicates the version
            # Version 2 onwards optionally add loss-masking spans
            # Version 4 onwards optionally add images
            idx_stream.write(struct.pack("<Q", 5))
            # Flag to indicate whether loss-masking spans are present
            idx_stream.write(struct.pack("<B", 1 if spans.size > 0 else 0))
            # Placeholder flag for preference spans
            idx_stream.write(struct.pack("<B", 0))
            # Flag to indicate whether images are present
            idx_stream.write(struct.pack("<B", 1 if total_images > 0 else 0))
            # Flag to indicate whether audio is present
            idx_stream.write(struct.pack("<B", 1 if total_audio > 0 else 0))
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
            idx_stream.write(image_lengths.tobytes(order="C"))
            # Position of each image in the document
            idx_stream.write(im_positions.tobytes(order="C"))
            # Number of audio per document
            idx_stream.write(n_audio.tobytes(order="C"))
            # Audio lengths
            idx_stream.write(audio_lengths.tobytes(order="C"))
            # Position of each audio in the document
            idx_stream.write(aud_positions.tobytes(order="C"))
            # Document indices, unused but needed for compatibility with Megatron-LM
            idx_stream.write(np.arange(num_documents + 1, dtype=np.int64).tobytes(order="C"))
