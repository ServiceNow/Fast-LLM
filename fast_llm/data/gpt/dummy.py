import pathlib

import numpy as np

from fast_llm.data.config import SampledDataset
from fast_llm.engine.config_utils.run import log_main_rank
from fast_llm.utils import Assert


class DummyGPTDataset(SampledDataset):
    """
    A dummy dataset that always returns the same sample, for debugging purposes.
    The sample can be purely random, or read from a file to allow reproducing in other runs.
    """

    def __init__(
        self, prefix: pathlib.Path | None, num_samples: int, sequence_length: int, vocab_size: int, name: str = "dummy"
    ):
        self._num_samples = num_samples
        if prefix is None:
            self._dummy_sample = np.random.randint(0, vocab_size, size=(sequence_length + 1,), dtype=np.int64)
        else:
            log_main_rank(f"> Loading dummy dataset from file {prefix}")
            self._dummy_sample = np.load(prefix, allow_pickle=True)[: sequence_length + 1]
            Assert.eq(self._dummy_sample.shape, (sequence_length + 1,))
            Assert.eq(self._dummy_sample.dtype, np.int64)
            Assert.lt(self._dummy_sample.max(), vocab_size)
            Assert.geq(self._dummy_sample.min(), 0)
        self._name = name

    def __len__(self):
        return self._num_samples

    def __getitem__(self, idx):
        return self._dummy_sample

    @property
    def name(self):
        return self._name
