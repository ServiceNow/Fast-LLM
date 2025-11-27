import logging
import math
import pathlib
import typing
import warnings

import numpy as np
import torch
import yaml

from fast_llm.data.dataset.abstract import SamplableIterableDataset, SampledDataset, SampledIterableDataset
from fast_llm.data.dataset.config import SamplingData, ShufflingType
from fast_llm.data.dataset.indexed import IndexedDataset
from fast_llm.data.sample.abstract import Sample
from fast_llm.engine.config_utils.data_type import DataType, get_unsigned_integer_type
from fast_llm.engine.config_utils.run import log_main_rank
from fast_llm.utils import Assert

try:
    from fast_llm.csrc.data import build_padded_token_cumsum  # noqa

    _extension_available = True
except ImportError:
    _extension_available = False

logger = logging.getLogger(__name__)


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


class SampledIndexedDataset[SampleType: Sample](SampledDataset[SampleType]):
    """
    A sampled dataset.
    """

    def __init__(
        self,
        indexed_dataset: IndexedDataset[SampleType],
        sampling: SamplingData,
    ):
        self._indexed_dataset = indexed_dataset
        self._config = sampling.config
        self._parameters = sampling.parameters
        self._truncate_documents = sampling.parameters.truncate_documents
        self._device = torch.device("cuda" if self._config.gpu else "cpu")

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
            # There needs to be one before calling `__getitem__`, normally handled through `Data`.

    def _sample(self) -> None:
        """
        Create a `SampledDataset` with the requested parameters.
        """
        # Get the document sizes, the main information needed for sampling.
        document_sizes = self._indexed_dataset.get_document_sizes().to(self._device)
        documents_per_epoch = document_sizes.numel()
        tokens_per_epoch = document_sizes.sum().item()

        # Calculate basic stats.
        if not self._truncate_documents:
            assert _extension_available, (
                "The C++ extension for dataset sampling is missing."
                " Please make sure Fast-LLM is installed correctly."
            )
            long_docs_filter = document_sizes > self._parameters.sequence_length + 1
            ignored_documents = long_docs_filter.sum().item()
            if ignored_documents:
                log_main_rank(
                    f" > {ignored_documents}/{documents_per_epoch} documents are longer than {self._parameters.sequence_length+1} tokens and will be ignored.",
                    log_fn=logger.warning,
                )
            tokens_per_epoch = document_sizes[~long_docs_filter].sum().item()
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
            "truncate_documents": self._truncate_documents,
            "config": self._config.to_dict(),
        }
        if self._truncate_documents:
            yaml_data["unshuffled_tokens"] = tokens_per_epoch * unshuffled_epochs

        if self._yaml_path is not None and self._yaml_path.is_file():
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
                document_sizes,
                offset=0,
                # TODO: Allowing for max 100% extra tokens for padding, is that enough?
                dtype=get_unsigned_integer_type((2 - self._truncate_documents) * tokens_per_epoch * num_epochs),
            )
            self._token_cumsum_unshuffled.save(token_cumsum_unshuffled)
        else:
            unshuffled_tokens = 0

        if not self._truncate_documents:
            yaml_data["unshuffled_tokens"] = unshuffled_tokens
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

    def __getitem__(self, index: int) -> SampleType:
        """
        Get the sample, (fixed-length sequence of tokens holding one or more complete or partial documents)
        with the requested sampling index.
        The returned sample is ready to be concatenated, then fed to a `Model`.
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

        documents: list[SampleType] = []
        while token_count < token_end:
            # Find the document index in the dataset.
            if document_sampling_index < self._unshuffled_documents:
                document_index = document_sampling_index % self._documents_per_epoch
            else:
                document_index = self._document_shuffling[document_sampling_index - self._unshuffled_documents].item()

            document_size = self._indexed_dataset.get_document_size(document_index)

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
                        documents.append(documents[-1].get_padding(padding_size))
                        Assert.eq(token_count + padding_size, token_end)
                        break
                    else:
                        # Move on to the next sample.
                        token_count += padding_size

            # Determine if the document belongs to the requested sample.
            if token_count + document_size > token_start:
                # Determine which part of the document belong to the sample, and add it to the list.
                token_start_index_in_document = max(token_start - token_count, 0)
                token_end_index_in_document = min(token_end - token_count, document_size)
                documents.append(
                    self._indexed_dataset.get_document(
                        document_index,
                        begin=token_start_index_in_document,
                        end=token_end_index_in_document,
                        parameters=self._parameters,
                    )
                )

            # Go to the next document.
            document_sampling_index += 1
            token_count += document_size

        return documents[0].from_documents(documents)

    @property
    def name(self) -> str:
        return self._indexed_dataset.name

    def _lazy_load(self):
        if not hasattr(self, "_documents_per_epoch"):
            self._load_yaml_data(yaml.safe_load(self._yaml_path.open("r")))

    def _load_yaml_data(self, data: dict[str, typing.Any]) -> None:
        self._documents_per_epoch = data["dataset"]["documents_per_epoch"]

        self._unshuffled_tokens = data["unshuffled_tokens"]
        self._unshuffled_documents = data["unshuffled_epochs"] * self._documents_per_epoch


class NaiveSampledIterableDataset[SampleType: Sample](SampledIterableDataset[SampleType]):
    def __init__(
        self,
        iterable_dataset: SamplableIterableDataset[SampleType],
        sampling: SamplingData,
    ):
        self._dataset = iterable_dataset
        self._config = sampling.config
        self._parameters = sampling.parameters

        assert self._parameters.truncate_documents == False
        assert self._config.shuffle == ShufflingType.disabled

    def __iter__(self) -> typing.Iterator[SampleType]:
        sample_length = self._parameters.sequence_length + self._parameters.extra_tokens
        current_sample_length = 0
        documents: list[SampleType] = []
        for doc in self._dataset:
            if len(doc) > sample_length:
                logging.warning(f"Dropping doc with length {len(doc)} higher then sample_length {sample_length}")
                continue
            if current_sample_length + len(doc) > sample_length:
                padding_length = sample_length - current_sample_length
                assert padding_length > 0
                documents.append(documents[-1].get_padding(padding_length))

                yield documents[0].from_documents(documents)

                documents = [doc]
                current_sample_length = len(doc)
            else:
                documents.append(doc)
                current_sample_length += len(doc)

            if current_sample_length == sample_length:
                yield documents[0].from_documents(documents)

                documents = []
                current_sample_length = 0

        if current_sample_length > 0:
            padding_length = sample_length - current_sample_length
            assert padding_length > 0
            documents.append(documents[-1].get_padding(padding_length))

            yield documents[0].from_documents(documents)

    @property
    def name(self) -> str:
        return self._dataset.name
