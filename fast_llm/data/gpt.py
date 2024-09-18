import math
import pathlib

import numpy as np
import numpy.random

from fast_llm.core.distributed import ProcessGroup, safe_barrier
from fast_llm.data.config import DataConfig
from fast_llm.data.dataset import RawDataset, SampledDataset
from fast_llm.data.fim import Fim
from fast_llm.data.mmap import MMapIndexedDataset
from fast_llm.data.tokenizer import Tokenizer
from fast_llm.distributed import MAX_SEED, PhaseType
from fast_llm.run import log_main_rank
from fast_llm.utils import Assert, padded_cumsum

try:
    from fast_llm.csrc.data import build_sample_idx  # noqa

    _extension_available = True
except ImportError:
    _extension_available = False


class GPTDataset(RawDataset):
    """
    A GPT dataset, which reads samples from (a split of) a `MMapIndexedDataset` pointing to a GPT dataset.
    """

    def __init__(
        self,
        name: str,
        indexed_dataset: MMapIndexedDataset,
        split_begin: int | None = None,
        split_end: int | None = None,
    ):
        self._name = name
        self._indexed_dataset = indexed_dataset

        self._split_begin = 0 if split_begin is None else split_begin
        self._split_end = len(indexed_dataset) if split_end is None else split_end

        # Checks
        try:
            Assert.geq(self._split_begin, 0)
            Assert.in_range_incl(self._split_end, self._split_begin + 1, len(indexed_dataset))
        except Exception as e:
            raise AssertionError(
                f"Invalid document indices for dataset {name} with length {len(indexed_dataset)}"
            ) from e

    def __len__(self):
        return self._split_end - self._split_begin

    def __getitem__(self, index: int):
        """
        Get the sample (document) with the given index (in the split dataset).
        """
        return self.get(index)

    def get(self, idx, offset=0, length=None):
        """
        Get the sample (document) with the given index (in the split dataset),
        optionally sub-sampled to a specific offset (starting point) and maximum length
        (end = min(offset + length, sample_length).
        """
        return self._indexed_dataset.get(idx, offset, length)

    @property
    def name(self):
        return self._name

    @classmethod
    def from_splits(cls, name: str, indexed_dataset: MMapIndexedDataset, phase_split: dict[PhaseType, float]):
        """
        Create a set of GPT datasets from a MMapIndexedDataset,
        each containing approximately the requested proportion of the total tokens.
        """
        split_probs = list(phase_split.values())
        Assert.eq(sum(split_probs), 1)
        num_documents = indexed_dataset.sizes.shape[0]
        splits = [round(x) for x in padded_cumsum(split_probs) * num_documents]
        return {
            phase: GPTDataset(f"{name}_{phase.value}", indexed_dataset, split_begin, split_end)
            for phase, split_begin, split_end in zip(phase_split, splits[:-1], splits[1:])
        }

    def sample(self, num_samples: int, sequence_length: int, np_rng: numpy.random.RandomState, verbose: bool):
        """
        Create a `GPTSampledDataset` with the requested parameters.
        """
        tokens_per_epoch = np.sum(self._indexed_dataset.sizes[self._split_begin : self._split_end])
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

        doc_idx = np.tile(np.arange(self._split_begin, self._split_end, dtype=np.int32), num_epochs)
        if separate_last_epoch:
            np_rng.shuffle(doc_idx[: -len(self)])
            np_rng.shuffle(doc_idx[-len(self) :])
        else:
            np_rng.shuffle(doc_idx)

        assert _extension_available, "Please run `make -C ./fast_llm/csrc/` first."
        sample_idx = build_sample_idx(
            self._indexed_dataset.sizes, doc_idx, sequence_length, num_epochs, tokens_per_epoch, verbose
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


class GPTSampledDataset(SampledDataset):
    """
    A GPT dataset augmented with a sampling, i.e.,
    a pre-computed, shuffled list of samples to be indexed sequentially (as-is) during training.
    The sampling exactly matches Megatron-LM with matching parameters.
    Supports optional post-processing with FIM.
    """

    def __init__(
        self,
        dataset: GPTDataset,
        num_samples: int,
        sequence_length: int,
        seed: int,
        group: ProcessGroup | None,
        config: DataConfig,
        tokenizer: Tokenizer | None,
        cache_dir: pathlib.Path,
        verbose: bool = True,
    ):
        self._dataset = dataset

        if config.fim.fim_rate > 0:
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
            doc_idx, sample_idx, shuffle_idx = self._dataset.sample(num_samples, sequence_length, np_rng, verbose)
            np.save(self._doc_idx_filename, doc_idx)
            np.save(self._sample_idx_filename, sample_idx)
            np.save(self._shuffle_idx_filename, shuffle_idx)

        safe_barrier(group, self._dataset.name)
        self._load_mappings(verbose)

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


class DummyGPTDataset(SampledDataset):
    """
    A dummy dataset that always returns the same sample, for debugging purposes.
    The sample can be purely random, or read from a file to allow reproducing in other runs.
    """

    def __init__(
        self, prefix: pathlib.Path | None, num_samples: int, sequence_length: int, vocab_size: int, name: str = "dummy"
    ):
        self._num_samples = num_samples
        if prefix is None:
            self._dummy_sample = np.random.randint(0, vocab_size, size=(sequence_length + 1,), dtype=np.int64)
        else:
            log_main_rank(f"> Loading dummy dataset from file {prefix}")
            self._dummy_sample = np.load(prefix, allow_pickle=True)[: sequence_length + 1]
            Assert.eq(self._dummy_sample.shape, (sequence_length + 1,))
            Assert.eq(self._dummy_sample.dtype, np.int64)
            Assert.lt(self._dummy_sample.max(), vocab_size)
            Assert.geq(self._dummy_sample.min(), 0)
        self._name = name

    def __len__(self):
        return self._num_samples

    def __getitem__(self, idx):
        return self._dummy_sample

    @property
    def name(self):
        return self._name
