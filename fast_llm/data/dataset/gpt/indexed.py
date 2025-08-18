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

    def get_image_sizes(self) -> list[np.ndarray]:
        """
        The size of each image in the dataset.
        The resulting array could be very large, so this method should be called cautiously,
        and derived classes should try to avoid holding the whole array im memory.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_document_size(self, index: int) -> int:
        """
        The size of a document in the dataset.
        """

    def get_image_size(self, index: int) -> np.ndarray:
        """
        The size of an image in the dataset.
        """
        raise NotImplementedError()

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
        return self._dataset.get_document_sizes()[self._begin : self._end]

    def get_image_sizes(self) -> list[np.ndarray]:
        # TODO: This can be really big.
        return self._dataset.get_image_sizes()[self._begin : self._end]

    def get_document_size(self, index: int) -> int:
        return self._dataset.get_document_size(self._begin + index)

    def get_image_size(self, index: int) -> np.ndarray:
        return self._dataset.get_image_size(self._begin + index)

    @property
    def has_images(self) -> bool:
        return self._dataset.has_images


class GPTConcatenatedDataset[IndexedDatasetType: GPTIndexedDataset](
    ConcatenatedDataset[IndexedDatasetType], GPTIndexedDataset
):
    _datasets: list[GPTIndexedDataset]

    def get_document_sizes(self) -> np.ndarray:
        # TODO: This can be really big.
        return np.concatenate([dataset.get_document_sizes() for dataset in self._datasets])

    def get_image_sizes(self) -> list[np.ndarray]:
        # TODO: This can be really big.
        return sum([dataset.get_image_sizes() for dataset in self._datasets], [])

    def get_document_size(self, index: int) -> int:
        dataset = np.searchsorted(self._dataset_splits[1:], index, side="right")
        return self._datasets[dataset].get_document_size(index - self._dataset_splits[dataset].item())

    def get_image_size(self, index: int) -> np.ndarray:
        dataset = np.searchsorted(self._dataset_splits[1:], index, side="right")
        return self._datasets[dataset].get_image_size(index - self._dataset_splits[dataset].item())

    @property
    def has_images(self) -> bool:
        return any(dataset.has_images for dataset in self._datasets)
