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
from fast_llm.engine.config_utils.data_type import get_unsigned_integer_type
from fast_llm.engine.config_utils.run import log_main_rank
from fast_llm.utils import Assert

try:
    from fast_llm.csrc.data import build_sample_idx  # noqa

    _extension_available = True
except ImportError:
    _extension_available = False

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class GPTSample:
    token_ids: np.ndarray
    loss_masking_spans: np.ndarray | None = None


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
            assert self.exists()
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
        self._num_samples = sampling.num_samples
        self._sequence_length = sampling.sequence_length
        self._config = sampling.config
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
                sampling.cache_directory / f"{self.name}_ns_{self._num_samples}_sl_{self._sequence_length}"
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

    def _sample(self) -> None:
        """
        Create a `GPTSampledDataset` with the requested parameters.
        """
        # Get the document sizes, the main information needed for sampling.
        document_sizes = torch.from_numpy(self._indexed_dataset.get_document_sizes()).to(self._device)

        # Calculate basic stats.
        documents_per_epoch = document_sizes.numel()
        tokens_per_epoch = document_sizes.sum().item()
        # We produce sequences of length `self._sequence_length + 1` so the last token has a label,
        # but we also include that last label in the following sample,
        # so we need `sequence_length * num_samples + 1` tokens in total.
        num_epochs = math.ceil((self._sequence_length * self._num_samples + 1) / tokens_per_epoch)

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
            "num_samples": self._num_samples,
            "unshuffled_epochs": unshuffled_epochs,
            "sequence_length": self._sequence_length,
            "config": self._config.to_serialized(),
        }
        self._load_yaml_data(yaml_data)
        if self._yaml_path is not None:
            if self._yaml_path.is_file():
                Assert.eq(yaml.safe_load(self._yaml_path.open("r")), yaml_data)
                # Dataset is already sampled, skip.
                return
            else:
                self._yaml_path.parent.mkdir(parents=True, exist_ok=True)
                yaml.safe_dump(yaml_data, self._yaml_path.open("w"))

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
        if shuffled_epochs > 0:
            token_cumsum_shuffled = self._get_token_cumsum(
                document_sizes[
                    # Torch indexing only works with int32 or int64
                    document_shuffling.to(
                        dtype=torch.int64 if document_shuffling.dtype == torch.int64 else torch.int32
                    )
                ],
                offset=unshuffled_epochs * tokens_per_epoch,
                dtype=get_unsigned_integer_type(tokens_per_epoch * num_epochs).torch,
            )
            self._token_cumsum_shuffled.save(token_cumsum_shuffled.numpy(force=self._config.gpu))
            self._document_shuffling.save(
                document_shuffling[: (token_cumsum_shuffled.numel() + 1) * TOKEN_CUMSUM_RATE].numpy(
                    force=self._config.gpu
                )
            )
            # Free memory
            del token_cumsum_shuffled
            del document_shuffling

        if unshuffled_epochs > 0:
            token_cumsum_unshuffled = self._get_token_cumsum(
                document_sizes, offset=0, dtype=get_unsigned_integer_type(tokens_per_epoch * num_epochs).torch
            )
            self._token_cumsum_unshuffled.save(token_cumsum_unshuffled.numpy(force=self._config.gpu))

    def _get_token_cumsum(self, sizes: torch.Tensor, offset: int, dtype: torch.dtype) -> torch.Tensor:
        # Create the output tensor.
        out = sizes.new_empty(sizes.numel() // TOKEN_CUMSUM_RATE + 1, dtype=dtype)
        # Get partial sums for regular intervals, excluding the last incomplete interval.
        torch.sum(
            sizes[: sizes.numel() - sizes.numel() % TOKEN_CUMSUM_RATE].view(-1, TOKEN_CUMSUM_RATE), dim=1, out=out[1:]
        )
        # Pad with the begin offset
        out[0] = offset
        # Calculate the cumsum.
        out.cumsum_(0)
        # Crop unnecessary entries.
        return out[
            : torch.clamp_min_(
                torch.searchsorted(out, self._num_samples * self._sequence_length, side="right"),
                0,
            )
        ]

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, index: int) -> typing.Any:
        """
        Get the sample, (fixed-length sequence of tokens holding one or more complete or partial documents)
        with the requested sampling index.
        The returned sample is ready to be concatenated, then fed to a `GPTModel` (see `GPTModel.preprocess`).
        """
        self._lazy_load()
        token_start = index * self._sequence_length
        token_end = token_start + self._sequence_length + 1

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
        while token_count < token_end:
            # Find the document index in the dataset.
            if document_sampling_index < self._unshuffled_documents:
                document_index = document_sampling_index % self._documents_per_epoch
            else:
                document_index = self._document_shuffling[document_sampling_index - self._unshuffled_documents].item()

            document_size = self._indexed_dataset.get_document_size(document_index)
            # Determine if the document belongs to the requested sample.
            if token_count + document_size >= token_start:
                # Determine which part of the document belong to the sample, and add it to the list.
                token_start_index_in_document = max(token_start - token_count, 0)
                token_end_index_in_document = min(token_end - token_count, document_size)
                sample = self._indexed_dataset.get(
                    document_index,
                    offset=token_start_index_in_document,
                    length=token_end_index_in_document - token_start_index_in_document,
                    use_loss_masking_spans=self._config.use_loss_masking_spans,
                )
                token_ids.append(sample.token_ids)
                if self._config.use_loss_masking_spans:
                    for loss_masking_span in sample.loss_masking_spans:
                        span = np.clip(loss_masking_span + token_count - token_start, 0, self._sequence_length + 1)
                        if span[1] > span[0]:
                            loss_masking_spans.append(span)

            # Go to the next document.
            document_sampling_index += 1
            token_count += document_size

        token_ids = np.concatenate(token_ids, dtype=np.int64)
        loss_masking_spans = (
            np.stack(loss_masking_spans, dtype=np.int32) if self._config.use_loss_masking_spans else None
        )
        Assert.eq(len(token_ids), self._sequence_length + 1)

        return GPTSample(token_ids=token_ids, loss_masking_spans=loss_masking_spans)

    @property
    def name(self) -> str:
        return self._indexed_dataset.name

    def _lazy_load(self):
        if not hasattr(self, "_documents_per_epoch"):
            self._load_yaml_data(yaml.safe_load(self._yaml_path.open("r")))

    def _load_yaml_data(self, data: dict[str, typing.Any]):
        self._documents_per_epoch = data["dataset"]["documents_per_epoch"]
        self._unshuffled_tokens = data["unshuffled_epochs"] * data["dataset"]["tokens_per_epoch"]
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
        self._num_samples = sampling.num_samples
        self._sequence_length = sampling.sequence_length
        self._config = sampling.config

        if sampling.cache_directory is None:
            log_main_rank(
                " > No dataset cache directory provided, building the index map on all ranks."
                "This may be very inefficient...",
                log_fn=logger.warning,
            )
            base_path = None
        else:
            base_path = (
                sampling.cache_directory / f"{self.name}_ns_{self._num_samples}_sl_{self._sequence_length}"
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
        document_sizes = self._indexed_dataset.get_document_sizes()
        num_documents = len(document_sizes)
        num_tokens = document_sizes.sum()
        np_rng = np.random.RandomState(seed=self._config.seed)

        num_epochs = math.ceil((self._sequence_length * self._num_samples + 1) / num_tokens)
        main_epochs_samples = ((num_epochs - 1) * num_tokens - 1) // self._sequence_length
        last_epoch_samples = self._num_samples - main_epochs_samples
        samples_per_epoch = (num_tokens - 1) // self._sequence_length
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
            self._sequence_length,
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

        Assert.geq(len(shuffle_idx), self._num_samples)
        self._doc_idx.save(doc_idx)
        self._sample_idx.save(sample_idx)
        self._shuffle_idx.save(shuffle_idx[: self._num_samples])

    def __len__(self) -> int:
        return self._num_samples

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
                use_loss_masking_spans=self._config.use_loss_masking_spans,
            )
            for doc in range(doc_f, doc_l + 1)
        ]
        token_ids = np.concatenate([sample.token_ids for sample in sample_list], dtype=np.int64)
        Assert.eq(len(token_ids), self._sequence_length + 1)

        if self._config.use_loss_masking_spans:
            spans = []
            offset = 0
            for sample in sample_list:
                for span in sample.loss_masking_spans:
                    spans.append(span + offset)
                offset += len(sample.token_ids)
            spans = np.stack(spans, dtype=np.int32)
        else:
            spans = None
        return GPTSample(token_ids=token_ids, loss_masking_spans=spans)

    @property
    def name(self) -> str:
        return self._indexed_dataset.name
