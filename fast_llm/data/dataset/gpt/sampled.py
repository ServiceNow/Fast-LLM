import logging
import math

import numpy as np

from fast_llm.data.data.gpt.data import GPTData
from fast_llm.data.dataset.abstract import SampledDataset
from fast_llm.data.dataset.gpt.config import GPTSamplingConfig
from fast_llm.data.dataset.gpt.fim.fim import Fim
from fast_llm.data.dataset.gpt.indexed import GPTIndexedDataset
from fast_llm.engine.config_utils.run import log_main_rank
from fast_llm.engine.distributed.config import MAX_SEED

logger = logging.getLogger(__name__)

try:
    from fast_llm.csrc.data import build_sample_idx  # noqa

    _extension_available = True
except ImportError:
    _extension_available = False


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
        sampling_config: GPTSamplingConfig,
        data: GPTData,
    ):
        assert isinstance(sampling_config, GPTSamplingConfig)
        assert isinstance(data, GPTData)
        self._indexed_dataset = indexed_dataset
        self._sampling_config = sampling_config
        self._shuffle_epochs = data.config.shuffle_epochs

        if data.config.fim.rate > 0:
            assert data.tokenizer is not None
            self._fim = Fim(data.config.fim, data.tokenizer)
        else:
            self._fim = None

        cache_prefix = f"{self.name}_ns_{self._sampling_config.num_samples}_sl_{self._sampling_config.sequence_length}_s_{self._sampling_config.seed}"
        # TODO: Any way to combine into a single file? (Memmap is harder)
        self._doc_idx_filename = self._sampling_config.cache_directory / (cache_prefix + "_doc_idx.npy")
        self._sample_idx_filename = self._sampling_config.cache_directory / (cache_prefix + "_sample_idx.npy")
        self._shuffle_idx_filename = self._sampling_config.cache_directory / (cache_prefix + "_shuffle_idx.npy")

        group = data.distributed.world_group
        # Build the indexed mapping if it doesn't exist.
        # TODO: This only works if the dataset location is accessible by all job.

        rank, verbose = data.get_next_sampling_rank_and_verbose()
        if (group is None or group.rank() == rank) and not (
            self._doc_idx_filename.is_file()
            and self._sample_idx_filename.is_file()
            and self._shuffle_idx_filename.is_file()
        ):
            if verbose:
                logger.info(f" > Building the index map on rank {rank} ...")
            doc_idx, sample_idx, shuffle_idx = self._sample()
            self._sampling_config.cache_directory.mkdir(parents=True, exist_ok=True)
            np.save(self._doc_idx_filename, doc_idx)
            np.save(self._sample_idx_filename, sample_idx)
            np.save(self._shuffle_idx_filename, shuffle_idx)

    def _sample(self, verbose: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create a `GPTSampledDataset` with the requested parameters.
        """
        document_sizes = self._indexed_dataset.get_document_sizes()
        documents_per_epoch = len(document_sizes)
        tokens_per_epoch = document_sizes.sum()
        np_rng = np.random.RandomState(seed=self._sampling_config.seed)

        total_tokens = self._sampling_config.sequence_length * self._sampling_config.num_samples
        num_epochs = math.ceil((total_tokens + 1) / tokens_per_epoch)
        epoch_begins_in_sample_index = (
            np.arange(num_epochs) * tokens_per_epoch - 1
        ) // self._sampling_config.sequence_length

        # Treat the last epoch differently if we use less than 80% of it.
        # Necessary to match the behavior of Megatron-LM.
        last_epoch_samples = self._sampling_config.num_samples - epoch_begins_in_sample_index[-1]
        samples_per_epoch = (tokens_per_epoch - 1) // self._sampling_config.sequence_length
        separate_last_epoch = num_epochs > 1 and last_epoch_samples < 0.8 * samples_per_epoch

        # Shuffle documents.
        doc_idx = np.tile(np.arange(documents_per_epoch, dtype=np.int32), num_epochs)
        if self._shuffle_epochs:
            if separate_last_epoch:
                np_rng.shuffle(doc_idx[:-documents_per_epoch])
                np_rng.shuffle(doc_idx[-documents_per_epoch:])
            else:
                np_rng.shuffle(doc_idx)
        else:
            for epoch in range(num_epochs):
                # Reseed each epoch to make sampling reproducible with a different number of epochs.
                np.random.RandomState(seed=self._sampling_config.seed + 738741 * epoch + 90823).shuffle(
                    doc_idx[epoch * documents_per_epoch : (epoch + 1) * documents_per_epoch]
                )

        if self._fast_sampling:
            # TODO: Crop token_cumsum and doc_idx to num samples
            token_cumsum = document_sizes[doc_idx].cumsum(dtype=np.int64)
            # TODO: Verify.
            # Trim the unused part of the last epoch.
            total_documents = np.searchsorted(token_cumsum, total_tokens)
            token_cumsum = token_cumsum[:total_documents]
            doc_idx = token_cumsum[:doc_idx]

            # Shuffle samples.
            sample_idx = np.arange(self._sampling_config.num_samples)
            if self._shuffle_epochs:
                if separate_last_epoch:
                    np_rng.shuffle(sample_idx[: epoch_begins_in_sample_index[-1]])
                    np_rng.shuffle(sample_idx[epoch_begins_in_sample_index[-1] :])
                else:
                    np_rng.shuffle(sample_idx)
            else:
                for epoch in range(num_epochs):
                    # Shuffle samples within an epoch, excluding the first one which may span two epochs.
                    # TODO: Include the first one if it's entirely in the epoch.
                    # Reseed each epoch to make sampling reproducible with a different number of epochs.
                    np.random.RandomState(seed=self._sampling_config.seed + 36478921 * epoch + 587469).shuffle(
                        sample_idx[epoch_begins_in_sample_index[epoch] + 1 : epoch_begins_in_sample_index[epoch + 1]]
                    )
            return doc_idx, sample_idx, token_cumsum

        assert (
            _extension_available
        ), "The C++ extension for dataset sampling is missing. Please make sure Fast-LLM is installed correctly."
        if verbose:
            logger.info(f" > Building sample index for {self._sampling_config.num_samples} samples ...")

        sample_idx = build_sample_idx(
            document_sizes,
            doc_idx,
            self._sampling_config.sequence_length,
            self._sampling_config.num_samples,
        )
        # shuffle-idx.
        # -1 is due to data structure used to retrieve the index:
        #    sample i --> [sample_idx[i], sample_idx[i+1])
        total_size = sample_idx.shape[0] - 1
        shuffle_idx = np.arange(
            0, total_size, dtype=np.int64 if total_size >= (np.iinfo(np.uint32).max - 1) else np.uint32
        )
        if separate_last_epoch:
            np_rng.shuffle(shuffle_idx[: epoch_begins_in_sample_index[-1]])
            np_rng.shuffle(shuffle_idx[epoch_begins_in_sample_index[-1] :])
        else:
            np_rng.shuffle(shuffle_idx)

        # TODO: The doc and sample idx are way bigger than needed when sampling for << 1 epoch.
        return doc_idx, sample_idx, shuffle_idx[: self._sampling_config.num_samples]

    def __getstate__(self):
        return (
            self._indexed_dataset,
            self._fim,
            self._sampling_config.to_serialized(),
            self._doc_idx_filename,
            self._sample_idx_filename,
            self._shuffle_idx_filename,
        )

    def __setstate__(self, state):
        (
            self._indexed_dataset,
            self._fim,
            sampling_config,
            self._doc_idx_filename,
            self._sample_idx_filename,
            self._shuffle_idx_filename,
        ) = state
        self._sampling_config = GPTSamplingConfig.from_dict(sampling_config)

    def _load_mappings(self, verbose=False):
        if hasattr(self, "_doc_idx"):
            return
        if verbose:
            log_main_rank(lambda: f" > loading doc-idx mapping from {self._doc_idx_filename}")
        self._doc_idx = np.load(self._doc_idx_filename, mmap_mode="r")
        if verbose:
            log_main_rank(lambda: f" > loading sample-idx mapping from {self._sample_idx_filename}")
        self._sample_idx = np.load(self._sample_idx_filename, mmap_mode="r")
        if verbose:
            log_main_rank(lambda: f" > loading shuffle-idx mapping from {self._shuffle_idx_filename}")
        self._shuffle_idx = np.load(self._shuffle_idx_filename, mmap_mode="r")
        if verbose:
            log_main_rank(lambda: f"  loaded dataset with {len(self)} samples.")

    def __len__(self):
        # -1 is due to data structure used to retrieve the index:
        #    sample i --> [sample_idx[i], sample_idx[i+1])
        return self._shuffle_idx.shape[0]

    def __getitem__(self, idx):
        """
        Get the sample, (fixed-length sequence of tokens holding one or more complete or partial documents)
        with the requested sampling index.
        The returned sample is ready to be concatenated, then fed to a `GPTModel` (see `GPTModel.preprocess`).
        """
        # Lazy load mappings
        self._load_mappings()

        if self._fast_sampling:
            token_cumsum = self._shuffle_idx
            token_start_index = idx * self._sampling_config.sequence_length
            doc_begin = np.searchsorted(token_cumsum, token_start_index)

            sample_list = []
            current_doc = doc_begin
            remaining_tokens = self._sampling_config.sequence_length + 1
            while remaining_tokens > 0:
                offset = token_start_index - token_cumsum[current_doc] if current_doc == doc_begin else 0
                # TODO: Boundary
                document_size = token_cumsum[current_doc] - token_cumsum[current_doc - 1]
                length = min(document_size - offset, remaining_tokens)
                sample_list.append(
                    self._indexed_dataset.get(
                        self._doc_idx[current_doc],
                        offset=offset,
                        length=length,
                    )
                )
                remaining_tokens -= length
                current_doc += 1

        else:
            # Get the shuffled index.
            shuffled_idx = self._shuffle_idx[idx]
            # Start and end documents and offsets.
            doc_f, offset_f = self._sample_idx[shuffled_idx]
            doc_l, offset_l = self._sample_idx[shuffled_idx + 1]
            sample_list = [
                self._indexed_dataset.get(
                    self._doc_idx[doc],
                    offset=(doc == doc_f) * offset_f,
                    length=offset_l + 1 - (doc == doc_f) * offset_f if doc == doc_l else None,
                )
                for doc in range(doc_f, doc_l + 1)
            ]

        sample = np.concatenate(
            sample_list,
            dtype=np.int64,
        )
        if self._fim is not None:
            sample = self._fim(sample, np.random.RandomState(seed=(self._sampling_config.seed + idx) % MAX_SEED))

        return sample

    @property
    def name(self):
        return self._indexed_dataset.name
