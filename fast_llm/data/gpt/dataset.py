import math

import numpy as np
import numpy.random

from fast_llm.data.config import RawDataset
from fast_llm.data.gpt.memmap import GPTMemmapDataset
from fast_llm.engine.distributed.config import PhaseType
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
        indexed_dataset: GPTMemmapDataset,
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
    def from_splits(cls, name: str, indexed_dataset: GPTMemmapDataset, phase_split: dict[PhaseType, float]):
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

        assert _extension_available, (
            "The C++ extension for dataset sampling is missing." " Please make sure Fast-LLM is installed correctly."
        )

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
