import math
import pathlib
import typing

import numpy as np

from fast_llm.core.distributed import safe_barrier
from fast_llm.data.dataset.abstract import SampledDataset
from fast_llm.data.dataset.gpt.config import GPTSamplingConfig
from fast_llm.data.dataset.gpt.indexed import GPTIndexedDataset
from fast_llm.engine.config_utils.run import log_main_rank
from fast_llm.utils import Assert

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
    ):
        assert isinstance(sampling_config, GPTSamplingConfig)
        self._indexed_dataset = indexed_dataset

        cache_prefix = (
            f"{self.name}_ns_{sampling_config.num_samples}_sl_{sampling_config.sequence_length}"
            f"_s_{sampling_config.seed}"
        )
        # TODO: Any way to combine into a single file? (Memmap is harder)
        self._doc_idx_filename = sampling_config.cache_directory / (cache_prefix + "_doc_idx.npy")
        self._sample_idx_filename = sampling_config.cache_directory / (cache_prefix + "_sample_idx.npy")
        self._shuffle_idx_filename = sampling_config.cache_directory / (cache_prefix + "_shuffle_idx.npy")

        group = sampling_config.distributed.world_group
        # Build the indexed mapping if it doesn't exist.
        # TODO: This only works if the dataset location is accessible by all job.
        if (group is None or group.rank() == 0) and not (
            self._doc_idx_filename.is_file()
            and self._sample_idx_filename.is_file()
            and self._shuffle_idx_filename.is_file()
        ):
            if sampling_config.verbose:
                log_main_rank(" > Building the index map on rank 0 ...")
            doc_idx, sample_idx, shuffle_idx = self._sample(sampling_config)
            sampling_config.cache_directory.mkdir(parents=True, exist_ok=True)
            np.save(self._doc_idx_filename, doc_idx)
            np.save(self._sample_idx_filename, sample_idx)
            np.save(self._shuffle_idx_filename, shuffle_idx)

        safe_barrier(group, self._indexed_dataset.name)
        self._load_mappings(sampling_config.verbose)

    def _sample(self, sampling_config: GPTSamplingConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create a `GPTSampledDataset` with the requested parameters.
        """
        document_sizes = self._indexed_dataset.get_document_sizes()
        num_documents = len(document_sizes)
        num_tokens = document_sizes.sum()
        np_rng = np.random.RandomState(seed=sampling_config.seed)

        num_epochs = math.ceil((sampling_config.sequence_length * sampling_config.num_samples + 1) / num_tokens)
        # For the last epoch, decide whether include the entire epoch
        # in the global shuffle or not.
        # Get the number of samples for the last epoch
        main_epochs_samples = ((num_epochs - 1) * num_tokens - 1) // sampling_config.sequence_length
        last_epoch_samples = sampling_config.num_samples - main_epochs_samples
        samples_per_epoch = (num_tokens - 1) // sampling_config.sequence_length
        # If we have less than 80% of the samples for the last epoch, separate out the epoch and treat it differently.
        # Note: the 80% number is just based on common sense and can be adjusted if needed.
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
            sampling_config.sequence_length,
            num_epochs,
            num_tokens,
            sampling_config.verbose,
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

        Assert.geq(len(shuffle_idx), sampling_config.num_samples)
        # TODO: The doc and sample idx are way bigger than needed when sampling for << 1 epoch.
        return doc_idx, sample_idx, shuffle_idx[: sampling_config.num_samples]

    def __getstate__(self) -> tuple[GPTIndexedDataset, pathlib.Path, pathlib.Path, pathlib.Path]:
        return (
            self._indexed_dataset,
            self._doc_idx_filename,
            self._sample_idx_filename,
            self._shuffle_idx_filename,
        )

    def __setstate__(self, state: tuple[GPTIndexedDataset, pathlib.Path, pathlib.Path, pathlib.Path]) -> None:
        (
            self._indexed_dataset,
            self._doc_idx_filename,
            self._sample_idx_filename,
            self._shuffle_idx_filename,
        ) = state
        self._load_mappings(False)

    def _load_mappings(self, verbose: bool) -> None:
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

    def __len__(self) -> int:
        # -1 is due to data structure used to retrieve the index:
        #    sample i --> [sample_idx[i], sample_idx[i+1])
        return self._shuffle_idx.shape[0]

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
        return sample

    @property
    def name(self) -> str:
        return self._indexed_dataset.name
