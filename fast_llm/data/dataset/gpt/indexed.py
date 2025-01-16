import abc
import typing

import numpy as np

from fast_llm.data.dataset.gpt.config import GPTSamplingConfig
from fast_llm.data.dataset.indexed import ConcatenatedDataset, DatasetSlice, IndexedDataset

if typing.TYPE_CHECKING:
    from fast_llm.data.dataset.gpt.sampled import GPTSampledIndexedDataset


class GPTIndexedDataset(IndexedDataset):
    """
    A GPT dataset containing a list of samples.
    """

    # def get(self, index: int, offset: int = 0, length: int | None = None):
    #    pass

    # def __len__(self) -> int:
    #    """
    #    Number of documents in the dataset.
    #    Can be calculated from document sizes but may be overridden if there is a better method.
    #    """
    #    return len(self.get_document_sizes())

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


class GPTConcatenatedDataset(ConcatenatedDataset, GPTIndexedDataset):
    _datasets: list[GPTIndexedDataset]

    def get_document_sizes(self) -> np.ndarray:
        # TODO: This can be really big.
        return np.concatenate([dataset.get_document_sizes() for dataset in self._datasets])
