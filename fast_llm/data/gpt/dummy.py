import typing

import numpy as np

from fast_llm.data.config import SamplableDataset, SampledDataset
from fast_llm.data.gpt.config import GPTSamplingConfig

if typing.TYPE_CHECKING:
    from fast_llm.data.gpt.data import GPTData


class DummyGPTDataset(SamplableDataset):
    """
    A dummy dataset that always returns the same sample, for debugging purposes.
    The sample can be purely random, or read from a file to allow reproducing in other runs.
    """

    def __init__(self, name: str, sequence_length: int, vocab_size: int):
        self._dummy_sample = np.random.randint(0, vocab_size, size=(sequence_length + 1,), dtype=np.int64)
        self._name = name

    def sample(self, config: GPTSamplingConfig, data: "GPTData"):
        return DummyGPTSampledDataset(self, config)

    def get(self):
        return self._dummy_sample

    @property
    def name(self):
        return self._name


class DummyGPTSampledDataset(SampledDataset):
    """
    A dummy dataset that always returns the same sample, for debugging purposes.
    The sample can be purely random, or read from a file to allow reproducing in other runs.
    """

    def __init__(self, dataset: DummyGPTDataset, config: GPTSamplingConfig):
        self._config = config
        self._dataset = dataset

    def __len__(self):
        return self._config.num_samples

    def __getitem__(self, idx):
        return self._dataset.get()

    @property
    def name(self):
        return self._dataset.name
