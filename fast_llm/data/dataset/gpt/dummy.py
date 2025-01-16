import numpy as np

from fast_llm.data.dataset.abstract import SamplableDataset, SampledDataset
from fast_llm.data.dataset.gpt.config import GPTSamplingConfig


class GPTDummyDataset(SamplableDataset):
    """
    A dummy dataset that always returns the same random sample, for debugging purposes.
    """

    def __init__(self, name: str):
        self._name = name

    def sample(self, config: GPTSamplingConfig) -> "GPTDummySampledDataset":
        return GPTDummySampledDataset(f"{self.name}_sampled", config)

    @property
    def name(self) -> str:
        return self._name


class GPTDummySampledDataset(SampledDataset):
    def __init__(self, name: str, config: GPTSamplingConfig):
        self._name = name
        self._config = config

    def __len__(self) -> int:
        return self._config.num_samples

    def __getitem__(self, idx) -> np.ndarray:
        return np.random.RandomState(self._config.seed + 4857643).randint(
            0, self._config.vocab_size, size=(self._config.sequence_length + 1,), dtype=np.int64
        )

    @property
    def name(self) -> str:
        return self._name
