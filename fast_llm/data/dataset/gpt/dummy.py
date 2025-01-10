import numpy as np

from fast_llm.data.dataset.abstract import SamplableDataset, SampledDataset
from fast_llm.data.dataset.gpt.config import GPTSamplingConfig


class GPTDummyDataset(SamplableDataset):
    """
    A dummy dataset that always returns the same random sample, for debugging purposes.
    """

    def __init__(self, name: str, sequence_length: int, vocab_size: int):
        self._dummy_sample = np.random.randint(0, vocab_size, size=(sequence_length + 1,), dtype=np.int64)
        self._name = name

    def sample(self, config: GPTSamplingConfig):
        return GPTDummySampledDataset(self, config)

    def get(self):
        return self._dummy_sample

    @property
    def name(self):
        return self._name


class GPTDummySampledDataset(SampledDataset):
    def __init__(self, dataset: GPTDummyDataset, config: GPTSamplingConfig):
        self._config = config
        self._dataset = dataset

    def __len__(self):
        return self._config.num_samples

    def __getitem__(self, idx):
        return self._dataset.get()

    @property
    def name(self):
        return self._dataset.name
