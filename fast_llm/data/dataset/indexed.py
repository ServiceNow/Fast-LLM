import abc
import typing

import numpy as np

from fast_llm.data.dataset.abstract import SamplableDataset
from fast_llm.utils import Assert, padded_cumsum


class IndexedDataset(SamplableDataset):
    """
    A dataset containing a list of samples.
    TODO: Move sampling responsibility here?
    """

    @abc.abstractmethod
    def get(self, index: int, *args, **kwargs) -> typing.Any:
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        """
        Number of samples in the dataset.
        """


class DatasetSlice[IndexedDatasetType: IndexedDataset](IndexedDataset):

    def __init__(
        self,
        name: str,
        dataset: IndexedDataset,
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

    def get(
        self, document: int, offset: int = 0, length: int | None = None, use_loss_masking_spans: bool = False
    ) -> typing.Any:
        """
        Get the sample (document) with the given index (in the dataset slice),
        optionally sub-sampled to a specific offset (starting point) and maximum length
        (end = min(offset + length, sample_length).
        """
        return self._dataset.get(document + self._begin, offset, length, use_loss_masking_spans)

    def __len__(self) -> int:
        return self._end - self._begin

    @property
    def name(self) -> str:
        return self._name


class ConcatenatedDataset[IndexedDatasetType: IndexedDataset](IndexedDataset):

    def __init__(
        self,
        name: str,
        datasets: list[IndexedDataset],
    ):
        self._name = name
        self._datasets = datasets
        sizes = [len(dataset) for dataset in self._datasets]
        self._dataset_splits = padded_cumsum(sizes)

    def __len__(self) -> int:
        return self._dataset_splits[-1].item()

    def get(self, index: int, *args, **kwargs):
        dataset = np.searchsorted(self._dataset_splits[1:], index, side="right")
        return self._datasets[dataset].get(index - self._dataset_splits[dataset].item(), *args, **kwargs)

    @property
    def name(self) -> str:
        return self._name
