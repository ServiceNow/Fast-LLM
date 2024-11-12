import numpy as np

from fast_llm.data.config import SampledDataset


class DummyGPTDataset(SampledDataset):
    """
    A dummy dataset that always returns the same sample, for debugging purposes.
    The sample can be purely random, or read from a file to allow reproducing in other runs.
    """

    def __init__(self, num_samples: int, sequence_length: int, vocab_size: int, name: str = "dummy"):
        self._num_samples = num_samples
        self._dummy_sample = np.random.randint(0, vocab_size, size=(sequence_length + 1,), dtype=np.int64)
        self._name = name

    def __len__(self):
        return self._num_samples

    def __getitem__(self, idx):
        return self._dummy_sample

    @property
    def name(self):
        return self._name
