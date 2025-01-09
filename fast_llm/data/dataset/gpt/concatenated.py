import numpy as np

from fast_llm.data.dataset.gpt.abstract import GPTIndexedDataset
from fast_llm.utils import padded_cumsum


class GPTConcatenatedDataset(GPTIndexedDataset):

    def __init__(
        self,
        name: str,
        datasets: list[GPTIndexedDataset],
    ):
        self._name = name
        self._datasets = datasets
        sizes = [dataset.num_documents for dataset in self._datasets]
        self._dataset_splits = padded_cumsum(sizes)
        self._num_documents = sum(sizes)

    @property
    def num_tokens(self) -> int:
        return sum(dataset.num_tokens for dataset in self._datasets)

    def num_documents(self) -> int:
        return sum(dataset.num_documents for dataset in self._datasets)

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
        return self._datasets[dataset].get(document - self._dataset_splits[dataset], offset, length)

    @property
    def name(self) -> str:
        return self._name
