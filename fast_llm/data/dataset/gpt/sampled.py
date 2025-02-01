import logging
import math
import pathlib
import typing
import warnings

import numpy as np
import torch

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


class MemmapArray:
    """
    An array with lazy loading in memmap mode.
    """

    _array: np.ndarray | None

    def __init__(self, path: pathlib.Path | None):
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

    def __getitem__(self, item: typing.Any) -> np.ndarray:
        if self._array is None:
            assert self.exists()
            self._array = np.load(self._path, mmap_mode="r")
        return self._array[item]


# TODO: Make configurable?
TOKEN_CUMSUM_RATE = 10


class GPTSampledIndexedDataset(SampledDataset):
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
        self._device = torch.device("cuda" if self._config.gpu else "cpu")

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

        # TODO: Names are confusing
        self._document_index = MemmapArray(
            None if base_path is None else base_path.with_name(base_path.name + "_document_index.npy")
        )
        self._token_cumsum_shuffled = MemmapArray(
            None if base_path is None else base_path.with_name(base_path.name + "_token_cumsum_shuffled.npy")
        )
        self._token_cumsum_unshuffled = MemmapArray(
            None if base_path is None else base_path.with_name(base_path.name + "_token_cumsum_unshuffled.npy")
        )

        if sampling.cache_directory is None:
            log_main_rank(
                " > No dataset cache directory provided, sampling on all ranks." "This may be very inefficient...",
                log_fn=logger.warning,
            )
            self._doc_idx, self._sample_idx, self._shuffle_idx = self._sample()
        else:
            cache_prefix = f"{self.name}_ns_{self._num_samples}_sl_{self._sequence_length}" f"_s_{self._config.seed}"
            self._doc_idx = MemmapArray(sampling.cache_directory / (cache_prefix + "_doc_idx.npy"))
            self._sample_idx = MemmapArray(sampling.cache_directory / (cache_prefix + "_sample_idx.npy"))
            self._shuffle_idx = MemmapArray(sampling.cache_directory / (cache_prefix + "_shuffle_idx.npy"))

            # Build the indexed mapping if it doesn't exist.
            # TODO: This only works if the dataset location is accessible by all job.
            if sampling.distributed.config.rank == sampling.get_next_rank() and not (
                self._doc_idx.exists() and self._sample_idx.exists() and self._shuffle_idx.exists()
            ):
                logger.info(f" > Sampling dataset {self._indexed_dataset.name} ...")
                doc_idx, sample_idx, shuffle_idx = self._sample()
                sampling.cache_directory.mkdir(parents=True, exist_ok=True)
                self._doc_idx.save(doc_idx)
                self._sample_idx.save(sample_idx)
                self._shuffle_idx.save(shuffle_idx)

    def _sample(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create a `GPTSampledDataset` with the requested parameters.
        """
        # Get the document sizes, the main information needed for sampling.
        document_sizes = torch.from_numpy(self._indexed_dataset.get_document_sizes()).to(self._device)

        # Calculate basic stats.
        documents_per_epoch = document_sizes.numel()
        tokens_per_epoch = document_sizes.sum().item()
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

        # Shuffle the dataset (documents)
        # This generates a document shuffling index `all_document_index`, the unshuffled part is trivial
        #   so we only evaluate and store the shuffled part `document_index`.
        document_index_dtype = get_unsigned_integer_type(documents_per_epoch).torch
        if self._config.shuffle == ShufflingType.full:
            generator.manual_seed(self._config.seed)
            # Equivalent to `shuffle(range(documents_per_epoch * num_epochs)) % documents_per_epoch`
            document_index = (
                torch.randperm(
                    shuffled_documents,
                    generator=generator,
                    dtype=get_unsigned_integer_type(shuffled_documents).torch,
                    device=self._device,
                )
                .remainder_(documents_per_epoch)
                .to(dtype=document_index_dtype)
            )
        elif self._config.shuffle in (ShufflingType.skip_first_epoch, ShufflingType.epoch):
            document_index = torch.empty(
                shuffled_documents,
                dtype=document_index_dtype,
                device=self._device,
            )
            for i in range(shuffled_epochs):
                generator.manual_seed(self._config.seed + i * 571)
                torch.randperm(
                    documents_per_epoch,
                    generator=generator,
                    out=document_index[i * documents_per_epoch : (i + 1) * documents_per_epoch],
                )
        elif self._config.shuffle == ShufflingType.disabled:
            document_index = None
        else:
            raise NotImplementedError(f"Unknown shuffling type: {self._config.shuffle}")

        # To get a sample on the fly we need to know where it begins,
        # and this is a non-trivial information because the documents have variable length.
        # The starting point `(document[idx], token[idx])` corresponds to the `(idx * sequence_length)` th token, i.e.
        # `document_sizes[all_document_index][:document[idx]].sum() + token[idx] == idx * sequence_length`.
        # This can be computed quickly provided we know a (partial) sum close to `(idx * sequence_length)`.
        # So it is enough to pre-compute `document_sizes[all_document_index].cumsum()[::TOKEN_CUMSUM_RATE]`.
        # Using `TOKEN_CUMSUM_RATE > 1` reduces pre-computation overhead at the cost of runtime computation.
        if shuffled_epochs > 0:
            # Equivalent to `document_sizes[all_document_index].cumsum()[::TOKEN_CUMSUM_RATE]`
            token_cumsum_shuffled = (
                document_sizes[document_index][: document_index.numel() - document_index.numel() % TOKEN_CUMSUM_RATE]
                .view(-1, TOKEN_CUMSUM_RATE)
                .sum(1)
                .cumsum(0, dtype=get_unsigned_integer_type(tokens_per_epoch * num_epochs).torch)
            )
            if unshuffled_tokens > 0:
                token_cumsum_shuffled.add_(unshuffled_tokens)
            # Crop surplus samples from the incomplete last epoch.
            crop = torch.clamp_min_(
                torch.searchsorted(token_cumsum_shuffled, self._num_samples * self._sequence_length, side="right") - 1,
                0,
            )
            self._token_cumsum_shuffled.save(token_cumsum_shuffled[:crop].numpy(force=self._config.gpu))
            self._document_index.save(document_index[: crop * TOKEN_CUMSUM_RATE].numpy(force=self._config.gpu))
            # Free memory
            del token_cumsum_shuffled
            del document_index

        if shuffled_epochs < num_epochs:
            token_cumsum_unshuffled = (
                document_sizes[: document_sizes.numel() - document_sizes.numel() % TOKEN_CUMSUM_RATE]
                .view(-1, TOKEN_CUMSUM_RATE)
                .sum(1)
                .cumsum(0, dtype=get_unsigned_integer_type(tokens_per_epoch * num_epochs).torch)
            )
            if shuffled_epochs == 0:
                crop = torch.clamp_min_(
                    torch.searchsorted(
                        token_cumsum_unshuffled, self._num_samples * self._sequence_length, side="right"
                    )
                    - 1,
                    0,
                )
                token_cumsum_unshuffled = token_cumsum_unshuffled[:crop]
            self._token_cumsum_unshuffled.save(token_cumsum_unshuffled.numpy(force=self._config.gpu))

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> typing.Any:
        """
        Get the sample, (fixed-length sequence of tokens holding one or more complete or partial documents)
        with the requested sampling index.
        The returned sample is ready to be concatenated, then fed to a `GPTModel` (see `GPTModel.preprocess`).
        """
        # TODO: Names are confusing
        token_start_index = idx * self._sequence_length
        token_end_index = token_start_index + self._sequence_length
        if token_start_index < self._unshuffled_tokens:
            array = self._token_cumsum_unshuffled
            shift = 0
        else:
            array = self._token_cumsum_shuffled
            shift = self._unshuffled_documents

        # Find the rightmost location `cumsum_index` in `token_cumsum` with `token_cumsum[cumsum_index] <= token_start_index`
        cumsum_index = np.searchsorted(array, token_start_index, side="right").item() - 1
        if cumsum_index < 0:
            document_index, token_count = shift, 0
        else:
            document_index = cumsum_index * TOKEN_CUMSUM_RATE + shift
            token_count = array[cumsum_index]

        sample_list = []
        while token_count < token_end_index:
            # Find the document index.
            if document_index < self._unshuffled_documents:
                document = document_index % self._documents_per_epoch
            else:
                document = self._document_index[document_index - self._unshuffled_documents]

            document_size = self._indexed_dataset.get_document_size(document)
            if token_count + document_size >= token_start_index:
                sample_list.append(
                    self._indexed_dataset.get(
                        document,
                        offset=max(token_start_index - token_count, 0),
                        length=min(token_end_index - token_count, document_size),
                    )
                )
            document_index += 1
            token_count += document_size

        sample = np.concatenate(
            sample_list,
            dtype=np.int64,
        )
        return sample

    @property
    def name(self) -> str:
        return self._indexed_dataset.name


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
            and not (self._doc_idx.exists())
            and self._sample_idx.exists()
            and self._shuffle_idx.exists()
        ):
            logger.info(f" > Sampling dataset {self._indexed_dataset.name} ...")
            doc_idx, sample_idx, shuffle_idx = self._sample()
            self._doc_idx.save(doc_idx)
            self._sample_idx.save(sample_idx)
            self._shuffle_idx.save(shuffle_idx)

    def _sample(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create a `GPTSampledDataset` with the requested parameters.
        """
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

        # shuffle-idx.
        # -1 is due to data structure used to retrieve the index:
        #    sample i --> [sample_idx[i], sample_idx[i+1])
        total_size = sample_idx.shape[0] - 1
        # TODO: Isn't the dataset already shuffled above?
        shuffle_idx = np.arange(
            0, total_size, dtype=np.int64 if total_size >= (np.iinfo(np.uint32).max - 1) else np.uint32
        )
        if separate_last_epoch:
            np_rng.shuffle(shuffle_idx[:main_epochs_samples])
            np_rng.shuffle(shuffle_idx[main_epochs_samples:])
        else:
            np_rng.shuffle(shuffle_idx)

        Assert.geq(len(shuffle_idx), self._num_samples)
        # TODO: The doc and sample idx are way bigger than needed when sampling for << 1 epoch.
        return doc_idx, sample_idx, shuffle_idx[: self._num_samples]

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
            )
            for doc in range(doc_f, doc_l + 1)
        ]
        sample = np.concatenate(
            sample_list,
            dtype=np.int64,
        )
        return sample

    @property
    def name(self) -> str:
        return self._indexed_dataset.name
