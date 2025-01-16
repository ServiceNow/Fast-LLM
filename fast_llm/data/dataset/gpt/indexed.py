import abc
import typing

import numpy as np

from fast_llm.data.dataset.abstract import SamplableDataset, SamplableSplitDataset
from fast_llm.data.dataset.gpt.config import GPTSamplingConfig
from fast_llm.engine.distributed.config import PhaseType
from fast_llm.utils import Assert, normalize_probabilities, padded_cumsum

if typing.TYPE_CHECKING:
    from fast_llm.data.dataset.gpt.sampled import GPTSampledIndexedDataset


try:
    from fast_llm.csrc.data import build_sample_idx  # noqa

    _extension_available = True
except ImportError:
    _extension_available = False


class GPTIndexedDataset(SamplableDataset):
    """
    A GPT dataset containing a list of unsampled, unprocessed samples.
    TODO: Move sampling responsibility here?
    """

    def get(self, document: int, offset: int = 0, length: int | None = None):
        pass

    @property
    def __len__(self) -> int:
        """
        Number of documents in the dataset.
        Can be calculated from document sizes but may be overridden if there is a better method.
        """
        return len(self.get_document_sizes())

    @property
    def num_tokens(self) -> int:
        """
        Number of tokens in the dataset.
        Can be calculated from document sizes but may be overridden if there is a better method.
        """
        return self.get_document_sizes().sum()

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


class GPTDatasetSlice(GPTIndexedDataset):
    """
    A GPT dataset, which reads samples from (a split of) a `MMapIndexedDataset` pointing to a GPT dataset.
    """

    def __init__(
        self,
        name: str,
        dataset: GPTIndexedDataset,
        begin: int | None = None,
        end: int | None = None,
    ):
        self._name = name
        self._dataset = dataset
        self._begin = 0 if begin is None else begin
        dataset_documents = len(dataset)
        self._end = dataset_documents if end is None else end

        # Checks
        try:
            Assert.geq(self._begin, 0)
            Assert.in_range_incl(self._end, self._begin + 1, dataset_documents)
        except Exception as e:
            raise AssertionError(f"Invalid document indices for dataset {name} with length {dataset_documents}") from e

    def __getitem__(self, index: int):
        """
        Get the sample (document) with the given index (in the split dataset).
        """
        return self.get(index)

    def get(self, document: int, offset: int = 0, length: int | None = None):
        """
        Get the sample (document) with the given index (in the dataset slice),
        optionally sub-sampled to a specific offset (starting point) and maximum length
        (end = min(offset + length, sample_length).
        """
        return self._dataset.get(document + self._begin, offset, length)

    @property
    def __len__(self):
        return self._end - self._begin

    def get_document_sizes(self):
        # TODO: This can be really big.
        return self._dataset.get_document_sizes()[self._begin : self._end]

    @property
    def name(self):
        return self._name

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


class GPTConcatenatedDataset(GPTIndexedDataset):

    def __init__(
        self,
        name: str,
        datasets: list[GPTIndexedDataset],
    ):
        self._name = name
        self._datasets = datasets
        sizes = [len(dataset) for dataset in self._datasets]
        self._dataset_splits = padded_cumsum(sizes)
        self._num_documents = sum(sizes)

    @property
    def num_tokens(self) -> int:
        return sum(dataset.num_tokens for dataset in self._datasets)

    def __len__(self) -> int:
        return sum(len(dataset) for dataset in self._datasets)

    def get_document_sizes(self) -> np.ndarray:
        # TODO: This can be really big.
        return np.concatenate([dataset.get_document_sizes() for dataset in self._datasets])

    def get(self, document: int, offset: int = 0, length: int | None = None):
        """
        Get the sample (document) with the given index (in the dataset slice),
        optionally sub-sampled to a specific offset (starting point) and maximum length
        (end = min(offset + length, sample_length).
        """
        dataset = np.searchsorted(self._dataset_splits[1:], document, side="right")
        return self._datasets[dataset].get((document - self._dataset_splits[dataset]).item(), offset, length)

    @property
    def name(self) -> str:
        return self._name
