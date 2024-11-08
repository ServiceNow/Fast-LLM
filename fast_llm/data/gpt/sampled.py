import pathlib

import numpy as np
from torch._C._distributed_c10d import ProcessGroup

from fast_llm.core.distributed import safe_barrier
from fast_llm.data.config import SampledDataset
from fast_llm.data.fim import Fim
from fast_llm.data.gpt.config import DataConfig
from fast_llm.data.gpt.dataset import GPTIndexedDataset
from fast_llm.data.tokenizer import Tokenizer
from fast_llm.engine.config_utils.run import log_main_rank
from fast_llm.engine.distributed.config import MAX_SEED


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
        num_samples: int,
        sequence_length: int,
        seed: int,
        group: ProcessGroup | None,
        config: DataConfig,
        tokenizer: Tokenizer | None,
        cache_directory: pathlib.Path,
        verbose: bool = True,
    ):
        self._indexed_dataset = indexed_dataset

        if config.fim.rate > 0:
            assert tokenizer is not None
            self._fim = Fim(config.fim, tokenizer)
        else:
            self._fim = None

        self._seed = seed
        # rng state
        np_rng = np.random.RandomState(seed=self._seed)

        cache_prefix = f"{self.name}_ns_{num_samples}_sl_{sequence_length}_s_{seed}"
        # TODO: Any way to combine into a single file? (Memmap is harder)
        self._doc_idx_filename = cache_directory / (cache_prefix + "_doc_idx.npy")
        self._sample_idx_filename = cache_directory / (cache_prefix + "_sample_idx.npy")
        self._shuffle_idx_filename = cache_directory / (cache_prefix + "_shuffle_idx.npy")

        # Build the indexed mapping if it doesn't exist.
        # TODO: This only works if the dataset location is accessible by all job.
        if (group is None or group.rank() == 0) and not (
            self._doc_idx_filename.is_file()
            and self._sample_idx_filename.is_file()
            and self._shuffle_idx_filename.is_file()
        ):
            if verbose:
                log_main_rank(" > Building the index map on rank 0 ...")
            doc_idx, sample_idx, shuffle_idx = self._indexed_dataset.sample(
                num_samples, sequence_length, np_rng, verbose
            )
            cache_directory.mkdir(parents=True, exist_ok=True)
            np.save(self._doc_idx_filename, doc_idx)
            np.save(self._sample_idx_filename, sample_idx)
            np.save(self._shuffle_idx_filename, shuffle_idx)

        safe_barrier(group, self._indexed_dataset.name)
        self._load_mappings(verbose)

    def __getstate__(self):
        return (
            self._indexed_dataset,
            self._fim,
            self._seed,
            self._doc_idx_filename,
            self._sample_idx_filename,
            self._shuffle_idx_filename,
        )

    def __setstate__(self, state):
        (
            self._indexed_dataset,
            self._fim,
            self._seed,
            self._doc_idx_filename,
            self._sample_idx_filename,
            self._shuffle_idx_filename,
        ) = state
        self._load_mappings(False)

    def _load_mappings(self, verbose):
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
            sample = self._fim(sample, np.random.RandomState(seed=(self._seed + idx) % MAX_SEED))

        return sample

    @property
    def name(self):
        return self._indexed_dataset.name
