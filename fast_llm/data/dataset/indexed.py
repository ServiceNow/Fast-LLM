import abc

import torch

from fast_llm.data.dataset.abstract import SamplableDataset
from fast_llm.data.dataset.config import SamplingData, SamplingParameters
from fast_llm.data.sample.abstract import Sample
from fast_llm.utils import Assert, padded_cumsum


class IndexedDataset[SampleType: Sample](SamplableDataset[SampleType]):
    """
    A dataset containing a list of samples.
    TODO: Move sampling responsibility here?
    """

    @abc.abstractmethod
    def get_document_sizes(self) -> torch.Tensor:
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

    @abc.abstractmethod
    def get_document(
        self, index: int, begin: int = 0, end: int | None = None, parameters: SamplingParameters | None = None
    ) -> SampleType:
        pass

    def __len__(self) -> int:
        """
        Number of documents in the dataset.
        Note: this default implementation is slow and should be overridden when possible.
        """
        return len(self.get_document_sizes())

    @property
    def num_tokens(self) -> int:
        """
        Number of tokens in the dataset.
        Note: this default implementation is slow and should be overridden when possible.
        """
        return self.get_document_sizes().sum().item()

    def sample(self, sampling: SamplingData) -> "GPTSampledIndexedDataset":
        from fast_llm.data.dataset.sampled import SampledIndexedDataset

        return SampledIndexedDataset(self, sampling)


class DatasetSlice[SampleType: Sample](IndexedDataset[SampleType]):

    def __init__(
        self,
        name: str,
        dataset: IndexedDataset[SampleType],
        begin: int | None = None,
        end: int | None = None,
    ):
        self._name = name
        self._dataset = dataset
        self._begin = 0 if begin is None else begin
        num_samples = len(dataset)
        self._end = num_samples if end is None else end

        # Checks
        try:
            Assert.geq(self._begin, 0)
            Assert.in_range_incl(self._end, self._begin + 1, num_samples)
        except Exception as e:
            raise AssertionError(f"Invalid document indices for dataset {name} with length {num_samples}") from e

    def get_document_sizes(self) -> torch.Tensor:
        # TODO: This can be really big.
        return self._dataset.get_document_sizes()[self._begin : self._end]

    def get_document_size(self, index: int) -> int:
        return self._dataset.get_document_size(self._begin + index)

    def get_document(
        self, index: int, begin: int = 0, end: int | None = None, parameters: SamplingParameters | None = None
    ) -> SampleType:
        """
        Get the sample (document) with the given index (in the dataset slice),
        optionally subsampled to a specific offset (starting point) and maximum length
        (end = min(offset + length, sample_length).
        """
        return self._dataset.get_document(index + self._begin, begin, end, parameters)

    def __len__(self) -> int:
        return self._end - self._begin

    @property
    def name(self) -> str:
        return self._name


class ConcatenatedDataset[SampleType: Sample](IndexedDataset[SampleType]):

    def __init__(
        self,
        name: str,
        datasets: list[IndexedDataset[SampleType]],
    ):
        self._name = name
        self._datasets = datasets
        sizes = [len(dataset) for dataset in self._datasets]
        self._dataset_splits = torch.from_numpy(padded_cumsum(sizes))

    def __len__(self) -> int:
        return self._dataset_splits[-1].item()

    def num_tokens(self) -> int:
        """
        Number of tokens in the dataset.
        """
        return sum(len(dataset) for dataset in self._datasets)

    def get_document_sizes(self) -> torch.Tensor:
        # TODO: This can be really big.
        return torch.cat([dataset.get_document_sizes() for dataset in self._datasets])

    def get_document_size(self, index: int) -> int:
        dataset = torch.searchsorted(self._dataset_splits[1:], index, side="right")
        return self._datasets[dataset].get_document_size(index - self._dataset_splits[dataset].item())

    def get_document(
        self, index: int, begin: int = 0, end: int | None = None, parameters: SamplingParameters | None = None
    ) -> SampleType:
        dataset = torch.searchsorted(self._dataset_splits[1:], index, side="right")
        return self._datasets[dataset].get_document(
            index - self._dataset_splits[dataset].item(), begin, end, parameters
        )

    @property
    def name(self) -> str:
        return self._name
