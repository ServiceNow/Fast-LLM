import abc
import typing

import numpy as np

from fast_llm.data.dataset.abstract import SamplableSplitDataset
from fast_llm.data.dataset.gpt.config import GPTSamplingConfig
from fast_llm.data.dataset.indexed import ConcatenatedDataset, DatasetSlice, IndexedDataset
from fast_llm.engine.distributed.config import PhaseType
from fast_llm.utils import normalize_probabilities, padded_cumsum

if typing.TYPE_CHECKING:
    from fast_llm.data.dataset.gpt.sampled import GPTSampledIndexedDataset


class GPTIndexedDataset(IndexedDataset):
    @abc.abstractmethod
    def get_document_sizes(self) -> np.ndarray:
        """
        The size of each document in the dataset.
        The resulting array could be very large, so this method should be called cautiously,
        and derived classes should try to avoid holding the whole array im memory.
        """

    def sample(self, config: GPTSamplingConfig) -> "GPTSampledIndexedDataset":
        from fast_llm.data.dataset.gpt.sampled import GPTSampledIndexedDataset

        return GPTSampledIndexedDataset(self, config)


class GPTDatasetSlice(DatasetSlice, GPTIndexedDataset):
    """
    A GPT dataset, which reads samples from (a split of) a `MMapIndexedDataset` pointing to a GPT dataset.
    """

    _dataset: GPTIndexedDataset

    def get_document_sizes(self) -> np.ndarray:
        # TODO: This can be really big.
        return self._dataset.get_document_sizes()[self._begin : self._end]

    @classmethod
    def from_splits(cls, dataset: GPTIndexedDataset, phase_split: dict[PhaseType, float]):
        """
        Create a set of GPT datasets from a MMapIndexedDataset,
        each containing approximately the requested proportion of the total tokens.
        """
        probabilities = normalize_probabilities(list(phase_split.values()))
        splits = [round(x) for x in padded_cumsum(probabilities) * len(dataset)]
        return SamplableSplitDataset[GPTDatasetSlice](
            f"{dataset.name}_split",
            {
                phase: GPTDatasetSlice(f"{dataset.name}_{phase.value}", dataset, split_begin, split_end)
                for phase, split_begin, split_end in zip(phase_split, splits[:-1], splits[1:])
            },
        )


class GPTConcatenatedDataset(ConcatenatedDataset, GPTIndexedDataset):
    _datasets: list[GPTIndexedDataset]

    def get_document_sizes(self) -> np.ndarray:
        # TODO: This can be really big.
        return np.concatenate([dataset.get_document_sizes() for dataset in self._datasets])
