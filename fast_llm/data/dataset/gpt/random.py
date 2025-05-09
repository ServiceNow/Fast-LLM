import numpy as np

from fast_llm.data.dataset.abstract import SamplableDataset, SampledDataset
from fast_llm.data.dataset.gpt.config import GPTSamplingData
from fast_llm.data.dataset.gpt.sampled import GPTSample


class GPTRandomDataset(SamplableDataset):
    """
    A dummy dataset that always returns the same random sample, for debugging purposes.
    """

    def __init__(self, name: str):
        self._name = name

    def sample(self, sampling: GPTSamplingData) -> "GPTRandomSampledDataset":
        return GPTRandomSampledDataset(sampling, f"{self.name}_sampled")

    @property
    def name(self) -> str:
        return self._name


class GPTRandomSampledDataset(SampledDataset):
    def __init__(self, sampling: GPTSamplingData, name: str):
        self._name = name
        self._seed = sampling.config.seed
        self._sequence_length = sampling.parameters.sequence_length
        self._vocab_size = sampling.parameters.vocab_size
        self._num_samples = sampling.parameters.num_samples

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx) -> np.ndarray:
        return GPTSample(
            np.random.RandomState(self._seed + 48576439 + 74593 * idx).randint(
                0, self._vocab_size, size=(self._sequence_length + 1,), dtype=np.int64
            )
        )

    @property
    def name(self) -> str:
        return self._name
