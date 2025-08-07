import abc
import typing

import numpy as np

from fast_llm.data.dataset.gpt.config import GPTSamplingData, ShufflingType
from fast_llm.data.dataset.indexed import ConcatenatedDataset, DatasetSlice, IndexedDataset

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

    @abc.abstractmethod
    def get_document_size(self, index: int) -> int:
        """
        The size of a document in the dataset.
        """

    def sample(self, sampling: GPTSamplingData) -> "GPTSampledIndexedDataset":
        from fast_llm.data.dataset.gpt.sampled import GPTSampledIndexedDataset, LegacyGPTSampledIndexedDataset

        return (
            LegacyGPTSampledIndexedDataset(self, sampling)
            if sampling.config.shuffle == ShufflingType.legacy
            else GPTSampledIndexedDataset(self, sampling)
        )

    @property
    @abc.abstractmethod
    def has_images(self) -> bool:
        """
        Whether the dataset contains images.
        This is used to determine whether to use image-related fields in the sampled data.
        """


class GPTDatasetSlice[IndexedDatasetType: GPTIndexedDataset](DatasetSlice[IndexedDatasetType], GPTIndexedDataset):
    """
    A GPT dataset, which reads samples from (a split of) a `MMapIndexedDataset` pointing to a GPT dataset.
    """

    _dataset: GPTIndexedDataset

    def get_document_sizes(self) -> np.ndarray:
        # TODO: This can be really big.
        doc_sizes, im_sizes = self._dataset.get_document_sizes()
        return doc_sizes[self._begin : self._end], im_sizes[self._begin : self._end] if im_sizes else np.array([])

    def get_document_size(self, index: int) -> int:
        return self._dataset.get_document_size(self._begin + index)

    @property
    def has_images(self) -> bool:
        return self._dataset.has_images


class GPTConcatenatedDataset[IndexedDatasetType: GPTIndexedDataset](
    ConcatenatedDataset[IndexedDatasetType], GPTIndexedDataset
):
    _datasets: list[GPTIndexedDataset]

    def get_document_sizes(self) -> np.ndarray:
        # TODO: This can be really big.
        # return np.concatenate([dataset.get_document_sizes() for dataset in self._datasets])
        sizes = [dataset.get_document_sizes() for dataset in self._datasets]
        return (
            np.concatenate([size[0] for size in sizes]),
            np.concatenate([size[1] for size in sizes]) if sizes[0][1] is not None else np.array([]),
        )

    def get_document_size(self, index: int) -> int:
        dataset = np.searchsorted(self._dataset_splits[1:], index, side="right")
        return self._datasets[dataset].get_document_size(index - self._dataset_splits[dataset].item())

    @property
    def has_images(self) -> bool:
        return any(dataset.has_images for dataset in self._datasets)
