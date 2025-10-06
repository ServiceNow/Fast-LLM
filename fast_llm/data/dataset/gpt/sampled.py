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
from fast_llm.layers.vision.preprocessing import get_num_image_tokens, get_resize_dims
from fast_llm.utils import Assert, div

try:
    from fast_llm.csrc.data import build_padded_token_cumsum, build_sample_idx  # noqa

    _extension_available = True
except ImportError:
    _extension_available = False

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class GPTSample:
    token_ids: np.ndarray
    images: list[np.ndarray] | None = None
    image_positions: np.ndarray | None = None
    loss_masking_spans: np.ndarray | None = None
    chosen_span: np.ndarray | None = None
    rejected_span: np.ndarray | None = None
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
        self._truncate_documents = sampling.parameters.truncate_documents
        self._device = torch.device("cuda" if self._config.gpu else "cpu")

        # TODO: address
        assert not self._parameters.use_preference_loss_spans

        if self._parameters.use_images:
            assert not self._truncate_documents, (
                "Truncating documents with images is not yet supported." " Please turn off truncation to use images."
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
            self._yaml_path = base_path.with_suffix(".yaml")

            self._token_cumsum_shuffled = MemmapArray(base_path.with_name(base_path.name + "_shuffled_cumsum.npy"))
            self._token_cumsum_unshuffled = MemmapArray(base_path.with_name(base_path.name + "_unshuffled_cumsum.npy"))

            # Sample or validate the dataset of a given rank.
            if sampling.distributed.config.rank == sampling.get_next_rank():
                self._sample()
            # No barrier yet to allow running in parallel.
            # There needs to be one before calling `__getitem__`, normally handled through `GPTData`.

    def _sample(self) -> None:
        """
        Create a `GPTSampledDataset` with the requested parameters.
        """
        # Get the size each document, the main information needed for sampling.
        # Note: "document" may refer to more than just text.
        document_sizes = torch.from_numpy(self._indexed_dataset.get_document_sizes(self._parameters)).to(self._device)

        documents_per_epoch, tokens_per_epoch, long_docs_filter = self._get_epoch_size(document_sizes)
        num_epochs, shuffled_epochs = self._get_epoch_count(documents_per_epoch, tokens_per_epoch)

        shuffled_documents = documents_per_epoch * shuffled_epochs
        unshuffled_epochs = num_epochs - shuffled_epochs

        yaml_data, cached = self._get_and_compare_yaml_data(documents_per_epoch, tokens_per_epoch, unshuffled_epochs)
        if cached:
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

        document_shuffling = self._get_document_shuffling(documents_per_epoch, shuffled_documents, shuffled_epochs)

        # To get a sample on the fly we need to know where it begins,
        # and this is a non-trivial information because the documents have variable length.
        # The starting point `(document[idx], token[idx])` corresponds to the `(idx * sequence_length)` th token, i.e.
        # `document_sizes[all_document_index][:document[idx]].sum() + token[idx] == idx * sequence_length`.
        # This can be computed quickly provided we know a (partial) sum close to `(idx * sequence_length)`.
        # So it is enough to pre-compute the (zero-padded) token cumsum at regular intervals `TOKEN_CUMSUM_RATE`.
        # Using `TOKEN_CUMSUM_RATE > 1` reduces pre-computation overhead at the cost of runtime computation.
        # Equivalent to `torch.hstack((0, document_sizes[all_document_index].cumsum()[::TOKEN_CUMSUM_RATE]))`

        # TODO: Allowing for max 100% extra tokens for padding, is that enough?
        cumsum_dtype = get_unsigned_integer_type((2 - self._truncate_documents) * tokens_per_epoch * num_epochs)
        if unshuffled_epochs > 0:
            token_cumsum_unshuffled, unshuffled_tokens = self._get_token_cumsum(document_sizes, 0, cumsum_dtype)
            self._token_cumsum_unshuffled.save(token_cumsum_unshuffled)
        else:
            unshuffled_tokens = 0

        if shuffled_epochs > 0:
            token_cumsum_shuffled, _ = self._get_token_cumsum(
                document_sizes[
                    # Torch indexing only works with int32 or int64
                    document_shuffling.to(
                        dtype=torch.int64 if document_shuffling.dtype == torch.int64 else torch.int32
                    )
                ],
                self._unshuffled_tokens,
                cumsum_dtype,
            )
            self._token_cumsum_shuffled.save(token_cumsum_shuffled)
            self._document_shuffling.save(
                document_shuffling[: (token_cumsum_shuffled.size + 1) * TOKEN_CUMSUM_RATE].numpy(force=True)
            )

        yaml_data["unshuffled_tokens"] = unshuffled_tokens
        self._load_yaml_data(yaml_data)
        if self._yaml_path is not None:
            self._yaml_path.parent.mkdir(parents=True, exist_ok=True)
            yaml.safe_dump(yaml_data, self._yaml_path.open("w"))

    def _get_epoch_size(self, document_sizes: torch.Tensor) -> tuple[int, int, torch.Tensor | None]:
        documents_per_epoch = document_sizes.numel()
        if self._truncate_documents:
            tokens_per_epoch = document_sizes.sum().item()
            long_docs_filter = None
        else:
            assert _extension_available, (
                "The C++ extension for dataset sampling is missing."
                " Please make sure Fast-LLM is installed correctly."
            )
            long_docs_filter = document_sizes <= self._parameters.sequence_length + 1
            documents_per_epoch_filtered = long_docs_filter.sum().item()
            if ignored_documents := documents_per_epoch_filtered - documents_per_epoch:
                log_main_rank(
                    f" > {ignored_documents}/{documents_per_epoch} documents"
                    f" are longer than {self._parameters.sequence_length+1} tokens and will be ignored.",
                    log_fn=logger.warning,
                )
            tokens_per_epoch = document_sizes[long_docs_filter].sum().item()
            if tokens_per_epoch == 0:
                raise RuntimeError(
                    f" > No documents shorter than {self._parameters.sequence_length+1}"
                    f" tokens found in dataset {self._indexed_dataset.name}."
                )
        return documents_per_epoch, tokens_per_epoch, long_docs_filter

    def _get_epoch_count(self, documents_per_epoch: int, tokens_per_epoch: int) -> tuple[int, int]:
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
        if self._config.shuffle == ShufflingType.skip_first_epoch:
            shuffled_epochs = num_epochs - 1
        elif self._config.shuffle == ShufflingType.disabled:
            shuffled_epochs = 0
        else:
            shuffled_epochs = num_epochs
        return num_epochs, shuffled_epochs

    def _get_and_compare_yaml_data(
        self,
        documents_per_epoch: int,
        tokens_per_epoch: int,
        unshuffled_epochs: int,
    ) -> tuple[dict[str, typing.Any], bool]:
        yaml_data = {
            "dataset": {
                "name": self._indexed_dataset.name,
                "documents_per_epoch": documents_per_epoch,
                "tokens_per_epoch": tokens_per_epoch,
            },
            "sampling": self._parameters.__dict__,
            "unshuffled_epochs": unshuffled_epochs,
            "config": self._config.to_dict(),
        }
        if self._truncate_documents:
            yaml_data["unshuffled_tokens"] = tokens_per_epoch * unshuffled_epochs

        if cached := (self._yaml_path is not None and self._yaml_path.is_file()):
            loaded_yaml_data = yaml.safe_load(self._yaml_path.open("r"))
            # Hack to make sure unshuffled tokens are loaded
            if not self._truncate_documents:
                yaml_data["unshuffled_tokens"] = loaded_yaml_data["unshuffled_tokens"]
            self._load_yaml_data(yaml_data)

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

        return yaml_data, cached

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

    def _get_document_shuffling(
        self,
        documents_per_epoch: int,
        shuffled_documents: int,
        shuffled_epochs: int,
    ) -> torch.Tensor | None:
        generator = torch.Generator(device=self._device)
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
        return document_shuffling

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
        if self._parameters.use_loss_masking_spans:
            loss_masking_spans = []
        if self._parameters.use_images:
            images = []
            image_positions = []
            image_tokens_added = 0
        text_tokens_added = 0
        while token_count < token_end:
            # Find the document index in the dataset.
            if document_sampling_index < self._unshuffled_documents:
                document_index = document_sampling_index % self._documents_per_epoch
            else:
                document_index = self._document_shuffling[document_sampling_index - self._unshuffled_documents].item()

            document_size = self._indexed_dataset.get_document_size(document_index, self._parameters)

            if not self._truncate_documents:
                if document_size > self._parameters.sequence_length + 1:
                    # Document too long, ignore
                    document_sampling_index += 1
                    continue
                tokens_in_sample = token_count % (self._parameters.sequence_length + 1)
                if document_size + tokens_in_sample > self._parameters.sequence_length + 1:
                    # Document belongs to the next sample, need to account for padding.
                    padding_size = self._parameters.sequence_length + 1 - tokens_in_sample
                    if token_count > token_start:
                        # Add padding tokens to current sample
                        token_ids.append(np.full((padding_size,), -100, dtype=np.int64))
                        Assert.eq(token_count + padding_size, token_end)
                        break
                    else:
                        # Move on to the next sample.
                        token_count += padding_size
                        continue
                elif document_size + tokens_in_sample == self._parameters.sequence_length + 1:
                    if token_count + document_size == token_start:
                        token_count += document_size
                        document_sampling_index += 1
                        continue

            # Determine if the document belongs to the requested sample.
            if token_count + document_size > token_start:
                # Determine which part of the document belong to the sample, and add it to the list.
                token_start_index_in_document = max(token_start - token_count, 0)
                token_end_index_in_document = min(token_end - token_count, document_size)
                sample: GPTSample = self._indexed_dataset.get(
                    document_index,
                    offset=token_start_index_in_document,
                    length=token_end_index_in_document - token_start_index_in_document,
                    use_loss_masking_spans=self._parameters.use_loss_masking_spans,
                )
                if self._parameters.use_images:
                    start_pos = 0
                    sample_token_ids = []
                    for idx, im_position in enumerate(sample.image_positions):
                        # add placeholder masked tokens for images
                        # if image_break_token is set, it is appended after every row
                        # if image_end_token is set, it is appended at the end of the image instead  of image_break_token
                        text_part = sample.token_ids[start_pos:im_position]
                        if self._parameters.image_break_token is not None:
                            height, width = resized_image_lengths[idx]
                            num_patches_h = div(height, self._parameters.patch_size)
                            num_patches_w = div(width, self._parameters.patch_size)
                            image_token_array = np.full((image_sizes[idx],), -100, dtype=np.int64)
                            # account for break tokens after each row
                            for row in range(num_patches_h - 1):
                                position = (row + 1) * num_patches_w + row
                                image_token_array[position] = self._parameters.image_break_token
                            # handle the last row separately
                            last_row_position = num_patches_h * num_patches_w + num_patches_h - 1
                            if self._parameters.image_end_token is not None:
                                image_token_array[last_row_position] = self._parameters.image_end_token
                            else:
                                image_token_array[last_row_position] = self._parameters.image_break_token
                        else:
                            image_token_array = np.full((image_sizes[idx],), -100, dtype=np.int64)
                            if self._parameters.image_end_token is not None:
                                image_token_array[-1] = self._parameters.image_end_token
                        sample_token_ids.append(np.concatenate([text_part, image_token_array], dtype=np.int64))
                        text_tokens_added += len(text_part)
                        image_positions.append(text_tokens_added + image_tokens_added)
                        image_tokens_added += image_sizes[idx]
                        start_pos = im_position
                    # Add the last text segment after the last image
                    sample_token_ids.append(sample.token_ids[start_pos:])
                    text_tokens_added += len(sample_token_ids[-1])
                    token_ids.append(np.concatenate(sample_token_ids))
                    images.append(sample.images)
                else:
                    token_ids.append(sample.token_ids)
                    text_tokens_added += len(token_ids[-1])
                if self._parameters.use_loss_masking_spans:
                    for loss_masking_span in sample.loss_masking_spans:
                        if self._parameters.use_images:
                            # Shift the spans to account for the images.
                            loss_masking_span[0] += sum(
                                image_size
                                for image_size, image_position in zip(image_sizes, sample.image_positions)
                                if image_position < loss_masking_span[0]
                            )
                            loss_masking_span[1] += sum(
                                image_size
                                for image_size, image_position in zip(image_sizes, sample.image_positions)
                                if image_position < loss_masking_span[1]
                            )

                        span = np.clip(
                            loss_masking_span + token_count - token_start,
                            0,
                            self._parameters.sequence_length + self._parameters.extra_tokens,
                        )
                        if span[1] >= span[0]:
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
        images = [im for img_list in images for im in img_list] if self._parameters.use_images else None
        image_positions = np.array(image_positions) if self._parameters.use_images else None
        Assert.eq(len(token_ids), self._parameters.sequence_length + self._parameters.extra_tokens)

        return GPTSample(
            token_ids=token_ids,
            loss_masking_spans=loss_masking_spans,
            sequence_lengths=sequence_lengths,
            images=images,
            image_positions=image_positions,
        )

    @property
    def name(self) -> str:
        return self._indexed_dataset.name

    def _get_image_sizes(self, document_index: int):
        # TODO: Duplicate of _get_document_sizes
        image_lengths = self._indexed_dataset.get_image_size(document_index)

        resized_image_lengths = [
            get_resize_dims(
                *image_length,
                self._parameters.max_image_size,
                self._parameters.max_image_size,
                self._parameters.patch_size,
            )
            for image_length in image_lengths
        ]
        image_sizes = [
            get_num_image_tokens(
                *image_length,
                self._parameters.patch_size,
                image_break=self._parameters.image_break_token is not None,
                image_end=self._parameters.image_end_token is not None,
            )
            for image_length in resized_image_lengths
        ]
        image_tokens = sum(image_sizes)
        return resized_image_lengths, image_sizes, image_tokens

    def _lazy_load(self):
        if not hasattr(self, "_documents_per_epoch"):
            self._load_yaml_data(yaml.safe_load(self._yaml_path.open("r")))

    def _load_yaml_data(self, data: dict[str, typing.Any]) -> None:
        self._documents_per_epoch = data["dataset"]["documents_per_epoch"]
        self._unshuffled_tokens = data["unshuffled_tokens"]
        self._unshuffled_documents = data["unshuffled_epochs"] * self._documents_per_epoch
