import dataclasses
import logging
import math
import pathlib
import typing
import warnings

import numpy as np
import torch
import yaml

from fast_llm.data.dataset.abstract import SampledDataset
from fast_llm.data.dataset.gpt.config import GPTSamplingData, ShufflingType
from fast_llm.data.dataset.gpt.indexed import GPTIndexedDataset
from fast_llm.engine.config_utils.data_type import DataType, get_unsigned_integer_type
from fast_llm.engine.config_utils.run import log_main_rank
from fast_llm.layers.vision_encoder.preprocessing import get_num_image_tokens, get_resize_dims
from fast_llm.utils import Assert

try:
    from fast_llm.csrc.data import build_padded_token_cumsum, build_sample_idx  # noqa

    _extension_available = True
except ImportError:
    _extension_available = False

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class GPTSample:
    token_ids: np.ndarray
    loss_masking_spans: np.ndarray | None = None
    images: np.ndarray | None = None
    image_positions: np.ndarray | None = None
    audio: np.ndarray | None = None
    audio_positions: np.ndarray | None = None
    sequence_lengths: np.ndarray | None = None


class MemmapArray:
    """
    An array with lazy loading in memmap mode.
    """

    _array: np.ndarray | None

    def __init__(self, path: pathlib.Path | None = None):
        self._path = path
        self._array = None

    def exists(self):
        return self._array is not None if self._path is None else self._path.is_file()

    def save(self, array: np.ndarray | None):
        if self._path is None or array is None:
            self._array = array
        else:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            np.save(self._path, array)

    @property
    def array(self):
        self._lazy_load()
        return self._array

    def __getitem__(self, item: typing.Any) -> np.ndarray:
        self._lazy_load()
        return self._array[item]

    def _lazy_load(self):
        if self._array is None:
            assert self.exists(), self._path
            self._array = np.load(self._path, mmap_mode="r")


# TODO: Make configurable?
TOKEN_CUMSUM_RATE = 10


class GPTSampledIndexedDataset(SampledDataset):
    """
    A sampled GPT dataset.
    """

    def __init__(
        self,
        indexed_dataset: GPTIndexedDataset,
        sampling: GPTSamplingData,
    ):
        assert isinstance(sampling, GPTSamplingData)
        self._indexed_dataset = indexed_dataset
        self._config = sampling.config
        self._parameters = sampling.parameters
        self._truncate_documents = sampling.truncate_documents
        self._device = torch.device("cuda" if self._config.gpu else "cpu")

        if self._indexed_dataset.has_images and self._truncate_documents:
            raise RuntimeError(
                "Truncating documents with images is not yet supported. Please turn off truncation to use images."
            )
        if self._indexed_dataset.has_audio and self._truncate_documents:
            raise RuntimeError(
                "Truncating documents with audio is not supported. Please turn off truncation to use audio."
            )

        if sampling.cache_directory is None:
            self._document_shuffling = MemmapArray()
            self._token_cumsum_shuffled = MemmapArray()
            self._token_cumsum_unshuffled = MemmapArray()
            self._yaml_path = None
            log_main_rank(
                " > No dataset cache directory provided, building the index map on all ranks."
                "This may be very inefficient...",
                log_fn=logger.warning,
            )
            self._sample()
        else:
            base_path = (
                sampling.cache_directory
                / f"{self.name}_ns_{self._parameters.num_samples}_sl_{self._parameters.sequence_length}"
                f"_s_{self._config.seed}"
            )
            # TODO: Names are confusing
            self._document_shuffling = MemmapArray(base_path.with_name(base_path.name + "_shuffling.npy"))
            self._token_cumsum_shuffled = MemmapArray(base_path.with_name(base_path.name + "_shuffled_cumsum.npy"))
            self._token_cumsum_unshuffled = MemmapArray(base_path.with_name(base_path.name + "_unshuffled_cumsum.npy"))
            self._yaml_path = base_path.with_suffix(".yaml")
            # Sample or validate the dataset of a given rank.
            if sampling.distributed.config.rank == sampling.get_next_rank():
                self._sample()
            # No barrier yet to allow running in parallel.
            # There needs to be one before calling `__getitem__`, normally handled through `GPTData`.

    def _compute_audio_token_size(self, sizes):
        if len(sizes) == 0:  # sample has no audio
            return sizes, False
        to_filter = False
        # account for padding
        if self._parameters.aud_padding_duration > 0:
            raw_audio_seq_length = self._parameters.aud_padding_duration * self._parameters.aud_sampling_rate
            sizes = sizes.copy()  # original is read-only
            to_filter = bool(np.any(sizes > raw_audio_seq_length))  # filter sample where any audio is too long
            sizes.fill(raw_audio_seq_length)  # set all audio sizes to padded amount

        # account for mel spectogram, convolution, downsampling k
        audio_token_size_arr = sizes // 160  # default hop length TODO Toby: check divisible?
        audio_token_size_arr = audio_token_size_arr // (
            2 * self._parameters.aud_downsampling_k
        )  # convolution (2) * downsampling
        return audio_token_size_arr, to_filter

    def apply_audio_padding(self, audio):
        if len(audio) == 0:
            return audio
        # TODO Toby: check 2d
        padded_audio = []
        if self._parameters.aud_padding_duration > 0:
            raw_audio_seq_length = self._parameters.aud_padding_duration * self._parameters.aud_sampling_rate
            for aud in audio:
                padded = np.pad(aud, (0, raw_audio_seq_length - len(aud)), mode="constant", constant_values=0)
                padded_audio.append(padded)
            return padded_audio
        else:
            return audio

    def _sample(self) -> None:
        """
        Create a `GPTSampledDataset` with the requested parameters.
        """
        # Get the document sizes, the main information needed for sampling.
        document_sizes, image_sizes, audio_sizes = self._indexed_dataset.get_document_sizes()
        document_sizes = torch.from_numpy(document_sizes).to(self._device)
        image_token_sizes = torch.zeros_like(document_sizes).to(self._device)
        if image_sizes:
            image_token_sizes = []
            for i, sizes in enumerate(image_sizes):
                image_token_sizes.append(
                    sum(
                        get_num_image_tokens(
                            *get_resize_dims(
                                *size,
                                self._parameters.image_size,
                                self._parameters.image_size,
                                self._parameters.patch_size,
                            ),
                            self._parameters.patch_size,
                            image_break=self._parameters.image_break_token is not None,
                        )
                        for size in sizes
                    )
                )
            image_token_sizes = torch.tensor(image_token_sizes).to(self._device)
        else:
            image_token_sizes = torch.zeros_like(document_sizes)

        audio_token_sizes = torch.zeros_like(document_sizes).to(self._device)
        long_audio_filter = torch.zeros_like(document_sizes, dtype=torch.bool)  # longer than audio padding
        for i, sizes in enumerate(audio_sizes):
            audio_token_size_arr, to_filter = self._compute_audio_token_size(sizes)
            audio_token_sizes[i] = audio_token_size_arr.sum()
            long_audio_filter[i] = to_filter

        documents_per_epoch = document_sizes.numel()
        tokens_per_epoch = (
            document_sizes.sum().item() + image_token_sizes.sum().item() + audio_token_sizes.sum().item()
        )

        # Calculate basic stats.
        if not self._truncate_documents:
            assert _extension_available, (
                "The C++ extension for dataset sampling is missing."
                " Please make sure Fast-LLM is installed correctly."
            )
            long_docs_filter = (
                document_sizes + image_token_sizes + audio_token_sizes > self._parameters.sequence_length + 1
            )
            ignored_documents = sum(long_docs_filter)
            if ignored_documents:
                log_main_rank(
                    f" > {ignored_documents}/{documents_per_epoch} documents are longer than {self._parameters.sequence_length+1} tokens and will be ignored.",
                    log_fn=logger.warning,
                )
            ignored_audio_samples = sum(long_audio_filter)
            if ignored_audio_samples:
                log_main_rank(
                    f" > {ignored_audio_samples}/{documents_per_epoch} samples contain audio longer than {self._parameters.aud_padding_duration} seconds and will be ignored.",
                    log_fn=logger.warning,
                )
            long_docs_filter = long_docs_filter | long_audio_filter
            tokens_per_epoch = (
                (
                    document_sizes[~long_docs_filter]
                    + image_token_sizes[~long_docs_filter]
                    + audio_token_sizes[~long_docs_filter]
                )
                .sum()
                .item()
            )
            if tokens_per_epoch == 0:
                raise RuntimeError(
                    f" > No documents shorter than {self._parameters.sequence_length+1} tokens found in dataset {self._indexed_dataset.name}."
                )
        # We produce sequences of length `self._sequence_length + extra_tokens` so the last token has a label for all prediction heads,
        # but in case of truncations we also include those last labels in the following sample,
        # so we need `sequence_length * num_samples + extra_tokens` tokens in total.
        if self._truncate_documents:
            num_epochs = math.ceil(
                (self._parameters.sequence_length * self._parameters.num_samples + self._parameters.extra_tokens)
                / tokens_per_epoch
            )
        else:
            num_epochs = math.ceil(
                ((self._parameters.sequence_length + self._parameters.extra_tokens) * self._parameters.num_samples)
                / tokens_per_epoch
            )

        # Prepare for shuffling.
        generator = torch.Generator(device=self._device)
        if self._config.shuffle == ShufflingType.skip_first_epoch:
            shuffled_epochs = num_epochs - 1
        elif self._config.shuffle == ShufflingType.disabled:
            shuffled_epochs = 0
        else:
            shuffled_epochs = num_epochs
        shuffled_documents = documents_per_epoch * shuffled_epochs
        unshuffled_epochs = num_epochs - shuffled_epochs

        yaml_data = {
            "dataset": {
                "name": self._indexed_dataset.name,
                "documents_per_epoch": documents_per_epoch,
                "tokens_per_epoch": tokens_per_epoch,
            },
            "num_samples": self._parameters.num_samples,
            "unshuffled_epochs": unshuffled_epochs,
            "sequence_length": self._parameters.sequence_length,
            "patch_size": self._parameters.patch_size,
            "truncate_documents": self._truncate_documents,
            "image_break_token": self._parameters.image_break_token,
            "config": self._config.to_dict(),
        }
        if self._truncate_documents:
            yaml_data["unshuffled_tokens"] = tokens_per_epoch * unshuffled_epochs

        if self._yaml_path is not None and self._yaml_path.is_file():
            loaded_yaml_data = yaml.safe_load(self._yaml_path.open("r"))
            self._load_yaml_data(loaded_yaml_data)
            if not self._truncate_documents:
                del loaded_yaml_data["unshuffled_tokens"]

            if loaded_yaml_data != yaml_data:
                raise RuntimeError(
                    f"Invalid dataset cache for dataset {self.name}."
                    " If this is due to an intended configuration change,"
                    " please delete the cache before continuing."
                    f"\nCurrent config:\n{yaml.safe_dump(yaml_data)}"
                    f"\nCached config:\n{yaml.safe_dump(loaded_yaml_data)}"
                )
            # Dataset is already sampled, skip.
            logger.info(f"Using existing sampling for dataset {self.name}")
            return

        if shuffled_documents > 1e8:
            warnings.warn(
                f"Shuffling {shuffled_documents:.2e} documents for dataset {self._indexed_dataset.name}."
                f" This may take a while and/or use an excessive amount of memory."
            )
        elif documents_per_epoch > 1e8:
            # TODO: Most of the damage is already done in `get_document_sizes`. Find a way to warn earlier?
            warnings.warn(
                f"The dataset {self._indexed_dataset.name} contains {documents_per_epoch:.2e} documents."
                f" Sampling may take a while and/or use an excessive amount of memory."
            )

        # Use the smallest possible data type to save memory and disk usage.
        document_shuffling_dtype = get_unsigned_integer_type(documents_per_epoch).torch
        # Shuffle the dataset (documents)
        # This generates a document shuffling index `all_document_index`, the unshuffled part is trivial
        #   so we only evaluate and store the shuffled part `document_shuffling`.
        if self._config.shuffle == ShufflingType.full:
            generator.manual_seed(self._config.seed)
            # Equivalent to `shuffle(range(documents_per_epoch * num_epochs)) % documents_per_epoch`
            document_shuffling = (
                torch.randperm(
                    shuffled_documents,
                    generator=generator,
                    dtype=get_unsigned_integer_type(shuffled_documents).torch,
                    device=self._device,
                )
                .remainder_(documents_per_epoch)
                .to(dtype=document_shuffling_dtype)
            )
        elif self._config.shuffle in (ShufflingType.skip_first_epoch, ShufflingType.epoch):
            document_shuffling = torch.empty(
                shuffled_documents,
                dtype=document_shuffling_dtype,
                device=self._device,
            )
            for i in range(shuffled_epochs):
                generator.manual_seed(self._config.seed + i * 571)
                torch.randperm(
                    documents_per_epoch,
                    generator=generator,
                    out=document_shuffling[i * documents_per_epoch : (i + 1) * documents_per_epoch],
                )
        elif self._config.shuffle == ShufflingType.disabled:
            document_shuffling = None
        else:
            raise NotImplementedError(f"Unknown shuffling type: {self._config.shuffle}")

        # To get a sample on the fly we need to know where it begins,
        # and this is a non-trivial information because the documents have variable length.
        # The starting point `(document[idx], token[idx])` corresponds to the `(idx * sequence_length)` th token, i.e.
        # `document_sizes[all_document_index][:document[idx]].sum() + token[idx] == idx * sequence_length`.
        # This can be computed quickly provided we know a (partial) sum close to `(idx * sequence_length)`.
        # So it is enough to pre-compute the (zero-padded) token cumsum at regular intervals `TOKEN_CUMSUM_RATE`.
        # Using `TOKEN_CUMSUM_RATE > 1` reduces pre-computation overhead at the cost of runtime computation.
        # Equivalent to `torch.hstack((0, document_sizes[all_document_index].cumsum()[::TOKEN_CUMSUM_RATE]))`
        if unshuffled_epochs > 0:
            token_cumsum_unshuffled, unshuffled_tokens = self._get_token_cumsum(
                document_sizes + image_token_sizes + audio_token_sizes,
                offset=0,
                # TODO: Allowing for max 100% extra tokens for padding, is that enough?
                dtype=get_unsigned_integer_type((2 - self._truncate_documents) * tokens_per_epoch * num_epochs),
            )
            self._token_cumsum_unshuffled.save(token_cumsum_unshuffled)
        else:
            unshuffled_tokens = 0

        if not self._truncate_documents:
            yaml_data["unshuffled_tokens"] = unshuffled_tokens.item()
        self._load_yaml_data(yaml_data)
        if self._yaml_path is not None:
            self._yaml_path.parent.mkdir(parents=True, exist_ok=True)
            yaml.safe_dump(yaml_data, self._yaml_path.open("w"))

        if shuffled_epochs > 0:
            token_cumsum_shuffled, _ = self._get_token_cumsum(
                document_sizes[
                    # Torch indexing only works with int32 or int64
                    document_shuffling.to(
                        dtype=torch.int64 if document_shuffling.dtype == torch.int64 else torch.int32
                    )
                ]
                + image_token_sizes[
                    document_shuffling.to(
                        dtype=torch.int64 if document_shuffling.dtype == torch.int64 else torch.int32
                    )
                ]
                + audio_token_sizes[
                    document_shuffling.to(
                        dtype=torch.int64 if document_shuffling.dtype == torch.int64 else torch.int32
                    )
                ],
                offset=self._unshuffled_tokens,
                # TODO: Allowing for max 100% extra tokens for padding, is that enough?
                dtype=get_unsigned_integer_type((2 - self._truncate_documents) * tokens_per_epoch * num_epochs),
            )
            self._token_cumsum_shuffled.save(token_cumsum_shuffled)
            self._document_shuffling.save(
                document_shuffling[: (token_cumsum_shuffled.size + 1) * TOKEN_CUMSUM_RATE].numpy(
                    force=self._config.gpu
                )
            )
            # Free memory
            del document_shuffling

    def _get_token_cumsum(self, sizes: torch.Tensor, offset: int, dtype: DataType) -> tuple[np.ndarray, int | None]:
        if self._truncate_documents:
            # Create the output tensor.
            out = sizes.new_empty(sizes.numel() // TOKEN_CUMSUM_RATE + 1, dtype=dtype.torch)
            # Get partial sums for regular intervals, excluding the last incomplete interval.
            torch.sum(
                sizes[: sizes.numel() - sizes.numel() % TOKEN_CUMSUM_RATE].view(-1, TOKEN_CUMSUM_RATE),
                dim=1,
                out=out[1:],
            )
            # Pad with the begin offset
            out[0] = offset
            # Calculate the cumsum.
            out.cumsum_(0)
            # Crop unnecessary entries.
            out = out[
                : torch.clamp_min_(
                    torch.searchsorted(
                        out, self._parameters.num_samples * self._parameters.sequence_length, side="right"
                    ),
                    0,
                )
            ]
            return out.numpy(force=self._config.gpu), None
        else:
            # TODO: dynamically handle int64 or int32 in CPP
            out = build_padded_token_cumsum(
                sizes.cpu().numpy(), (self._parameters.sequence_length + 1), TOKEN_CUMSUM_RATE, offset
            )
            num_tokens = out[-1]
            out = out[:-1][
                : np.clip(
                    np.searchsorted(
                        out, self._parameters.num_samples * (self._parameters.sequence_length + 1), side="right"
                    ),
                    0,
                    None,
                )
            ]
            return out, num_tokens

    def __len__(self) -> int:
        return self._parameters.num_samples

    def __getitem__(self, index: int) -> typing.Any:
        """
        Get the sample, (fixed-length sequence of tokens holding one or more complete or partial documents)
        with the requested sampling index.
        The returned sample is ready to be concatenated, then fed to a `GPTModel` (see `GPTModel.preprocess`).
        """
        self._lazy_load()
        # tokens at the boundary are included in only one sample when we pack without truncations
        # in case of packing with truncations, the last token from the previous sample is also the first token of the next sample
        sample_length = (
            self._parameters.sequence_length
            if self._truncate_documents
            else self._parameters.sequence_length + self._parameters.extra_tokens
        )
        token_start = index * sample_length
        token_end = token_start + self._parameters.sequence_length + self._parameters.extra_tokens

        if token_start < self._unshuffled_tokens:
            token_start_array = self._token_cumsum_unshuffled.array
            token_start_array_document_offset = 0
        else:
            token_start_array = self._token_cumsum_shuffled.array
            token_start_array_document_offset = self._unshuffled_documents

        # Find the rightmost location `token_start_cumsum_index` in `token_cumsum` with `token_cumsum[token_start_cumsum_index] <= token_start`
        token_start_cumsum_index = np.searchsorted(token_start_array, token_start, side="right").item() - 1

        document_sampling_index = token_start_cumsum_index * TOKEN_CUMSUM_RATE + token_start_array_document_offset

        token_count = token_start_array[token_start_cumsum_index]

        token_ids = []
        loss_masking_spans = []
        images = []
        audio = []
        image_positions = []
        audio_positions = []
        mm_tokens_added = 0
        text_tokens_added = 0
        while token_count < token_end:
            # Find the document index in the dataset.
            if document_sampling_index < self._unshuffled_documents:
                document_index = document_sampling_index % self._documents_per_epoch
            else:
                document_index = self._document_shuffling[document_sampling_index - self._unshuffled_documents].item()

            text_size, image_lengths, audio_lengths = self._indexed_dataset.get_document_size(document_index)

            image_sizes = [
                get_num_image_tokens(
                    *get_resize_dims(
                        *image_length,
                        self._parameters.image_size,
                        self._parameters.image_size,
                        self._parameters.patch_size,
                    ),
                    self._parameters.patch_size,
                    image_break=self._parameters.image_break_token is not None,
                )
                for image_length in image_lengths
            ]
            image_tokens = sum(image_sizes)

            audio_token_size_arr, _ = self._compute_audio_token_size(audio_lengths)
            audio_tokens = audio_token_size_arr.sum()

            document_size = text_size + image_tokens + audio_tokens

            if not self._truncate_documents:
                # Document too long, ignore
                if document_size > self._parameters.sequence_length + 1:
                    document_sampling_index += 1
                    continue

                # Where are we currently in sample?
                tokens_in_sample = token_count % (self._parameters.sequence_length + 1)

                # Add padding
                if document_size + tokens_in_sample > self._parameters.sequence_length + 1:
                    # Document belongs to the next sample, need to account for padding.
                    padding_size = self._parameters.sequence_length + 1 - tokens_in_sample
                    if token_count > token_start:
                        # Add padding tokens to current sample
                        try:
                            token_ids.append(np.full((padding_size,), -100, dtype=np.int64))
                        except:
                            pass
                        Assert.eq(token_count + padding_size, token_end)
                        break
                    else:
                        # Move on to the next sample.
                        token_count += padding_size

            # Determine if the document belongs to the requested sample.
            if token_count + document_size >= token_start:
                # Determine which part of the document belong to the sample, and add it to the list.
                token_start_index_in_document = max(token_start - token_count, 0)
                token_end_index_in_document = min(token_end - token_count, text_size)
                sample = self._indexed_dataset.get(
                    document_index,
                    offset=token_start_index_in_document,
                    length=token_end_index_in_document - token_start_index_in_document,
                    use_loss_masking_spans=self._parameters.use_loss_masking_spans,
                )
                start_pos = 0

                # add tokens and multi modal padding placeholders
                # multimodal_positions = np.concatenate(
                #     [
                #         arr.astype(np.int32)
                #         for arr in (sample.image_positions, sample.audio_positions)
                #         if arr is not None
                #     ]
                # ) or np.array([], dtype=np.int32)
                # multimodal_positions.sort()

                multimodal_positions = []
                if sample.image_positions is not None:
                    multimodal_positions.extend(
                        [(pos, "image", idx) for idx, pos in enumerate(sample.image_positions)]
                    )
                if sample.audio_positions is not None:
                    multimodal_positions.extend(
                        [(pos, "audio", idx) for idx, pos in enumerate(sample.audio_positions)]
                    )

                multimodal_positions.sort(key=lambda x: x[0])
                for global_idx, (mm_position, mm_type, source_idx) in enumerate(multimodal_positions):
                    # Add placeholders for image and audio tokens tokens
                    token_ids.append(sample.token_ids[start_pos:mm_position])
                    if mm_type == "image":
                        text_tokens_added += len(token_ids[-1])
                        image_positions.append(text_tokens_added + mm_tokens_added)
                        # token_ids.append(np.full((image_sizes[idx],), -100, dtype=np.int64))
                        if self._parameters.image_break_token is not None:
                            # Calculate patch dimensions for the image
                            height, width = get_resize_dims(
                                image_lengths[source_idx][0],
                                image_lengths[source_idx][1],
                                self._parameters.image_size,
                                self._parameters.image_size,
                                self._parameters.patch_size,
                            )
                            num_patches_h = math.ceil(height / self._parameters.patch_size)
                            num_patches_w = math.ceil(width / self._parameters.patch_size)

                            # Calculate the token count considering break tokens
                            tokens_per_row = num_patches_w
                            resized_image_tokens = num_patches_h * tokens_per_row + (
                                num_patches_h - 1
                            )  # Add break tokens after each row except last

                            # Create image token placeholder array
                            image_token_array = np.full((resized_image_tokens,), -100, dtype=np.int64)

                            # Add break tokens after each row except the last row
                            for row in range(num_patches_h - 1):
                                position = (row + 1) * tokens_per_row + row
                                image_token_array[position] = self._parameters.image_break_token

                            token_ids.append(image_token_array)

                            # Update mm_tokens_added to reflect actual number of tokens added
                            mm_tokens_added += resized_image_tokens
                        else:
                            # Just add placeholders for all image tokens without break tokens
                            token_ids.append(np.full((image_sizes[source_idx],), -100, dtype=np.int64))
                            mm_tokens_added += image_sizes[source_idx]
                    elif mm_type == "audio":
                        audio_pos = sum(t.size for t in token_ids)  # includes mm tokens added already
                        audio_positions.append(audio_pos)
                        token_ids.append(
                            np.full((audio_token_size_arr[source_idx],), -100, dtype=np.int64)
                        )  # TODO Toby: index doesnt work here
                        mm_tokens_added += audio_tokens
                    start_pos = mm_position
                token_ids.append(sample.token_ids[start_pos:])

                # TODO Soham: add offsets for loss masking spans
                text_tokens_added += len(token_ids[-1])
                if sample.images:
                    images.append(sample.images)
                else:
                    images.append([])
                if sample.audio:
                    audio.append(self.apply_audio_padding(sample.audio))
                else:
                    audio.append([])

                if self._parameters.use_loss_masking_spans:
                    mm_idx = 0
                    total_mm_tokens = 0
                    for loss_masking_span in sample.loss_masking_spans:  # TODO: check these must be sorted
                        mm_tokens_in_span = 0
                        mm_position, mm_type, source_idx = (
                            multimodal_positions[mm_idx]
                            if mm_idx < len(multimodal_positions)
                            else (float("inf"), _, _)
                        )

                        # increment mm_idx until span is reached
                        while mm_position < loss_masking_span[0]:
                            if mm_type == "image":
                                num_mm_tokens = image_sizes[source_idx]
                            elif mm_type == "audio":
                                num_mm_tokens = audio_token_size_arr[source_idx]
                            total_mm_tokens += num_mm_tokens
                            mm_idx += 1
                            mm_position, mm_type, source_idx = (
                                multimodal_positions[mm_idx]
                                if mm_idx < len(multimodal_positions)
                                else (float("inf"), _, _)
                            )

                        # get all multimodal positions within span
                        while mm_position >= loss_masking_span[0] and mm_position <= loss_masking_span[1]:
                            if mm_type == "image":
                                num_mm_tokens = image_sizes[source_idx]
                            elif mm_type == "audio":
                                num_mm_tokens = audio_token_size_arr[source_idx]
                            mm_tokens_in_span += num_mm_tokens
                            mm_idx += 1
                            mm_position, mm_type, source_idx = (
                                multimodal_positions[mm_idx]
                                if mm_idx < len(multimodal_positions)
                                else (float("inf"), _, _)
                            )
                        loss_masking_span[0] += total_mm_tokens  # increment by all mm tokens before span
                        loss_masking_span[1] += total_mm_tokens + mm_tokens_in_span
                        total_mm_tokens += mm_tokens_in_span

                        span = np.clip(
                            loss_masking_span + token_count - token_start,
                            0,
                            self._parameters.sequence_length + self._parameters.extra_tokens,
                        )
                        if span[1] > span[0]:
                            loss_masking_spans.append(span)

            # Go to the next document.
            document_sampling_index += 1
            token_count += document_size

        sequence_lengths = (
            np.array([ids.size - (idx == len(token_ids) - 1) for idx, ids in enumerate(token_ids)], dtype=np.int32)
            if not self._parameters.cross_document_attention
            else None
        )
        token_ids = np.concatenate(token_ids, dtype=np.int64)
        loss_masking_spans = (
            (np.stack(loss_masking_spans, dtype=np.int32) if loss_masking_spans else np.array([]))
            if self._parameters.use_loss_masking_spans
            else None
        )

        images = [im for img_list in images for im in img_list] if images else None
        image_positions = np.array(image_positions) if image_positions else None

        audio = [aud for aud_list in audio for aud in aud_list] if audio else None  # flatten
        audio_positions = np.array(audio_positions) if audio_positions else None
        # Assert.eq(len(token_ids), self._parameters.sequence_length + self._parameters.extra_tokens)

        return GPTSample(
            token_ids=token_ids,
            loss_masking_spans=loss_masking_spans,
            sequence_lengths=sequence_lengths,
            images=images,
            image_positions=image_positions,
            audio=audio,
            audio_positions=audio_positions,
        )

    @property
    def name(self) -> str:
        return self._indexed_dataset.name

    def _lazy_load(self):
        if not hasattr(self, "_documents_per_epoch"):
            self._load_yaml_data(yaml.safe_load(self._yaml_path.open("r")))

    def _load_yaml_data(self, data: dict[str, typing.Any]) -> None:
        self._documents_per_epoch = data["dataset"]["documents_per_epoch"]

        if "unshuffled_tokens" not in data:
            # Backward compatibility
            # TODO v0.x: Remove
            assert self._truncate_documents
            data["unshuffled_tokens"] = data["tokens_per_epoch"] * data["unshuffled_epochs"]

        self._unshuffled_tokens = data["unshuffled_tokens"]
        self._unshuffled_documents = data["unshuffled_epochs"] * self._documents_per_epoch


class LegacyGPTSampledIndexedDataset(SampledDataset):
    """
    A GPT dataset augmented with a sampling, i.e.,
    a pre-computed, shuffled list of samples to be indexed sequentially (as-is) during training.
    The sampling exactly matches Megatron-LM with matching parameters.
    Supports optional post-processing with FIM.
    """

    def __init__(
        self,
        indexed_dataset: GPTIndexedDataset,
        sampling: GPTSamplingData,
    ):
        assert isinstance(sampling, GPTSamplingData)
        self._indexed_dataset = indexed_dataset
        if not sampling.truncate_documents:
            raise NotImplementedError(
                "Legacy sampling only supports document truncation. Please use the latest dataset format."
            )
        self._config = sampling.config
        self._parameters = sampling.parameters

        if sampling.cache_directory is None:
            log_main_rank(
                " > No dataset cache directory provided, building the index map on all ranks."
                "This may be very inefficient...",
                log_fn=logger.warning,
            )
            base_path = None
        else:
            base_path = (
                sampling.cache_directory
                / f"{self.name}_ns_{self._parameters.num_samples}_sl_{self._parameters.sequence_length}"
                f"_s_{self._config.seed}"
            )

        self._doc_idx = MemmapArray(
            None if base_path is None else base_path.with_name(base_path.name + "_doc_idx.npy")
        )
        self._sample_idx = MemmapArray(
            None if base_path is None else base_path.with_name(base_path.name + "_sample_idx.npy")
        )
        self._shuffle_idx = MemmapArray(
            None if base_path is None else base_path.with_name(base_path.name + "_shuffle_idx.npy")
        )

        # Build the indexed mapping if it doesn't exist.
        if base_path is None or (
            sampling.distributed.config.rank == sampling.get_next_rank()
            and not (self._doc_idx.exists() and self._sample_idx.exists() and self._shuffle_idx.exists())
        ):
            self._sample()

    def _sample(self) -> None:
        """
        Create a `GPTSampledDataset` with the requested parameters.
        """
        logger.info(f" > Sampling dataset {self._indexed_dataset.name} ...")
        document_sizes, _ = self._indexed_dataset.get_document_sizes()
        num_documents = len(document_sizes)
        num_tokens = document_sizes.sum()
        np_rng = np.random.RandomState(seed=self._config.seed)

        num_epochs = math.ceil((self._parameters.sequence_length * self._parameters.num_samples + 1) / num_tokens)
        main_epochs_samples = ((num_epochs - 1) * num_tokens - 1) // self._parameters.sequence_length
        last_epoch_samples = self._parameters.num_samples - main_epochs_samples
        samples_per_epoch = (num_tokens - 1) // self._parameters.sequence_length
        separate_last_epoch = num_epochs > 1 and last_epoch_samples < 0.8 * samples_per_epoch

        doc_idx = np.tile(np.arange(num_documents, dtype=np.int32), num_epochs)
        if separate_last_epoch:
            np_rng.shuffle(doc_idx[:-num_documents])
            np_rng.shuffle(doc_idx[-num_documents:])
        else:
            np_rng.shuffle(doc_idx)

        assert _extension_available, (
            "The C++ extension for dataset sampling is missing." " Please make sure Fast-LLM is installed correctly."
        )

        sample_idx = build_sample_idx(
            document_sizes,
            doc_idx,
            self._parameters.sequence_length,
            num_epochs,
            num_tokens,
            True,
        )

        total_size = sample_idx.shape[0] - 1
        shuffle_idx = np.arange(
            0, total_size, dtype=np.int64 if total_size >= (np.iinfo(np.uint32).max - 1) else np.uint32
        )
        if separate_last_epoch:
            np_rng.shuffle(shuffle_idx[:main_epochs_samples])
            np_rng.shuffle(shuffle_idx[main_epochs_samples:])
        else:
            np_rng.shuffle(shuffle_idx)

        Assert.geq(len(shuffle_idx), self._parameters.num_samples)
        self._doc_idx.save(doc_idx)
        self._sample_idx.save(sample_idx)
        self._shuffle_idx.save(shuffle_idx[: self._parameters.num_samples])

    def __len__(self) -> int:
        return self._parameters.num_samples

    def __getitem__(self, idx: int) -> typing.Any:
        """
        Get the sample, (fixed-length sequence of tokens holding one or more complete or partial documents)
        with the requested sampling index.
        The returned sample is ready to be concatenated, then fed to a `GPTModel` (see `GPTModel.preprocess`).
        """
        # Get the shuffled index.
        shuffled_idx = self._shuffle_idx[idx]
        # Start and end documents and offsets.
        doc_f, offset_f = self._sample_idx[shuffled_idx]
        doc_l, offset_l = self._sample_idx[shuffled_idx + 1]
        sample_list = [
            self._indexed_dataset.get(
                self._doc_idx[doc].item(),
                offset=(doc == doc_f) * offset_f,
                length=offset_l + 1 - (doc == doc_f) * offset_f if doc == doc_l else None,
                use_loss_masking_spans=self._parameters.use_loss_masking_spans,
            )
            for doc in range(doc_f, doc_l + 1)
        ]
        token_ids = np.concatenate([sample.token_ids for sample in sample_list], dtype=np.int64)
        Assert.eq(len(token_ids), self._parameters.sequence_length + 1)

        if self._parameters.use_loss_masking_spans:
            spans = []
            offset = 0
            for sample in sample_list:
                for span in sample.loss_masking_spans:
                    spans.append(span + offset)
                offset += len(sample.token_ids)
            spans = np.stack(spans, dtype=np.int32) if spans else np.array([])
        else:
            spans = None
        sequence_lengths = (
            np.array(
                [sample.token_ids.size - (idx == len(sample_list) - 1) for idx, sample in enumerate(sample_list)],
                dtype=np.int32,
            )
            if not self._parameters.cross_document_attention
            else None
        )
        return GPTSample(token_ids=token_ids, loss_masking_spans=spans, sequence_lengths=sequence_lengths)

    @property
    def name(self) -> str:
        return self._indexed_dataset.name
