import math
import pathlib

import numpy as np
import numpy.random
from torch._C._distributed_c10d import ProcessGroup

from fast_llm.core.distributed import safe_barrier
from fast_llm.data.config import SampledDataset
from fast_llm.data.fim import Fim
from fast_llm.data.gpt.config import GPTDataConfig
from fast_llm.data.gpt.dataset import GPTIndexedDataset
from fast_llm.data.tokenizer import Tokenizer
from fast_llm.engine.config_utils.run import log_main_rank
from fast_llm.engine.distributed.config import MAX_SEED
from fast_llm.utils import Assert

try:
    from fast_llm.csrc.data import build_sample_idx  # noqa

    _extension_available = True
except ImportError:
    _extension_available = False


class GPTSampledDataset(SampledDataset):
    """
    A GPT dataset augmented with a sampling, i.e.,
    a pre-computed, shuffled list of samples to be indexed sequentially (as-is) during training.
    The sampling exactly matches Megatron-LM with matching parameters.
    Supports optional post-processing with FIM.
    """

    def __init__(
        self,
        dataset: GPTIndexedDataset,
        num_samples: int,
        sequence_length: int,
        seed: int,
        group: ProcessGroup | None,
        config: GPTDataConfig,
        tokenizer: Tokenizer | None,
        cache_dir: pathlib.Path,
        verbose: bool = True,
    ):
        self._dataset = dataset

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
        self._doc_idx_filename = cache_dir / (cache_prefix + "_doc_idx.npy")
        self._sample_idx_filename = cache_dir / (cache_prefix + "_sample_idx.npy")
        self._shuffle_idx_filename = cache_dir / (cache_prefix + "_shuffle_idx.npy")

        # Build the indexed mapping if it doesn't exist.
        # TODO: This only works if the dataset location is accessible by all job.
        if (group is None or group.rank() == 0) and not (
            self._doc_idx_filename.is_file()
            and self._sample_idx_filename.is_file()
            and self._shuffle_idx_filename.is_file()
        ):
            if verbose:
                log_main_rank(" > Building the index map on rank 0 ...")
            doc_idx, sample_idx, shuffle_idx = self._sample(num_samples, sequence_length, np_rng, verbose)
            np.save(self._doc_idx_filename, doc_idx)
            np.save(self._sample_idx_filename, sample_idx)
            np.save(self._shuffle_idx_filename, shuffle_idx)

        safe_barrier(group, self._dataset.name)
        self._load_mappings(verbose)

    def _sample(self, num_samples: int, sequence_length: int, np_rng: numpy.random.RandomState, verbose: bool):
        """
        Create a `GPTSampledDataset` with the requested parameters.
        """
        tokens_per_epoch = self._dataset.num_tokens
        num_epochs = math.ceil((sequence_length * num_samples + 1) / tokens_per_epoch)
        # For the last epoch, decide whether include the entire epoch
        # in the global shuffle or not.
        # Get the number of samples for the last epoch
        main_epochs_samples = ((num_epochs - 1) * tokens_per_epoch - 1) // sequence_length
        last_epoch_samples = num_samples - main_epochs_samples
        samples_per_epoch = (tokens_per_epoch - 1) // sequence_length
        # If we have less than 80% of the samples for the last epoch, separate out the epoch and treat it differently.
        # Note: the 80% number is just based on common sense and can be adjusted if needed.
        separate_last_epoch = num_epochs > 1 and last_epoch_samples < 0.8 * samples_per_epoch

        doc_idx = np.tile(np.arange(len(self._dataset), dtype=np.int32), num_epochs)
        if separate_last_epoch:
            np_rng.shuffle(doc_idx[: -len(self._dataset)])
            np_rng.shuffle(doc_idx[-len(self._dataset) :])
        else:
            np_rng.shuffle(doc_idx)

        assert _extension_available, "Please run `make -C ./fast_llm/csrc/` first."
        sample_idx = build_sample_idx(
            self._dataset.document_sizes, doc_idx, sequence_length, num_epochs, tokens_per_epoch, verbose
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

        Assert.geq(len(shuffle_idx), num_samples)
        # TODO: The doc and sample idx are way bigger than needed when sampling for << 1 epoch.
        return doc_idx, sample_idx, shuffle_idx[:num_samples]

    def __getstate__(self):
        return (
            self._dataset,
            self._fim,
            self._seed,
            self._doc_idx_filename,
            self._sample_idx_filename,
            self._shuffle_idx_filename,
        )

    def __setstate__(self, state):
        (
            self._dataset,
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
            self._dataset.get(
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
        return self._dataset.name
