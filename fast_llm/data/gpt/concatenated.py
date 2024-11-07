import numpy as np

from fast_llm.data.gpt.config import GPTConcatenatedDatasetConfig, GPTRawDataset
from fast_llm.utils import padded_cumsum


class GPTConcatenatedDataset(GPTRawDataset):

    def __init__(
        self,
        config: GPTConcatenatedDatasetConfig,
        datasets: list[GPTRawDataset],
    ):
        self._config = config
        self._datasets = datasets
        sizes = [dataset.num_documents for dataset in self._datasets]
        self._dataset_splits = padded_cumsum(sizes)
        self._num_documents = sum(sizes)

    @property
    def num_tokens(self):
        return sum(dataset.num_tokens for dataset in self._datasets)

    def num_documents(self):
        return self._num_documents

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
        dataset = np.searchsorted(self._dataset_splits[1:], document, side="right")
        return self._datasets[dataset].get(document - self._dataset_splits[dataset], offset, length)

    @property
    def name(self):
        return self._config.name