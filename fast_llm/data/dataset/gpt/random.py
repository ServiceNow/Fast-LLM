import numpy as np

from fast_llm.data.dataset.abstract import SamplableDataset, SampledDataset
from fast_llm.data.dataset.gpt.config import GPTSamplingConfig
from fast_llm.data.dataset.gpt.sampled import GPTSample


class GPTRandomDataset(SamplableDataset):
    """
    A dummy dataset that always returns the same random sample, for debugging purposes.
    """

    def __init__(self, name: str):
        self._name = name

    def sample(self, config: GPTSamplingConfig) -> "GPTRandomSampledDataset":
        return GPTRandomSampledDataset(config, f"{self.name}_sampled")

    @property
    def name(self) -> str:
        return self._name


class GPTRandomSampledDataset(SampledDataset):
    def __init__(self, config: GPTSamplingConfig, name: str):
        self._name = name
        self._seed = config.seed
        self._sequence_length = config.sequence_length
        self._vocab_size = config.vocab_size
        self._num_samples = config.num_samples

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx) -> GPTSample:
        np_seed = self._seed + 48576439 + 74593 * idx
        ids = np.random.RandomState(np_seed).randint(
            0, self._vocab_size, size=(self._sequence_length + 1,), dtype=np.int64
        )
        n_spans = np.random.RandomState(np_seed).randint(0, 3)
        spans = []
        prev_end = -1
        for _ in range(n_spans):
            if prev_end >= len(ids) - 1:
                break
            start = np.random.RandomState(np_seed).randint(prev_end + 1, len(ids))
            end = np.random.RandomState(np_seed).randint(start, len(ids))
            spans.append([start, end])
            prev_end = end
        return GPTSample(token_ids=ids, ignore_loss_spans=np.array(spans, dtype=np.int32).reshape(-1, 2))

    @property
    def name(self) -> str:
        return self._name
