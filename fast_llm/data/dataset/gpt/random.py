import numpy as np
import torch

from fast_llm.data.dataset.abstract import SamplableDataset, SampledDataset
from fast_llm.data.dataset.gpt.config import GPTSamplingData
from fast_llm.data.sample.language_model import LanguageModelSample
from fast_llm.data.sample.token import TokenSample


class GPTRandomDataset[SampleType: LanguageModelSample](SamplableDataset[SampleType]):
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


class GPTRandomSampledDataset[SampleType: LanguageModelSample](SampledDataset[SampleType]):
    def __init__(self, sampling: GPTSamplingData, name: str):
        self._name = name
        self._seed = sampling.config.seed
        self._parameters = sampling.parameters
        # TODO: Support?
        assert not self._parameters.use_loss_masking_spans
        assert not self._parameters.use_preference_loss_spans

    def __len__(self) -> int:
        return self._parameters.num_samples

    def __getitem__(self, index: int) -> SampleType:
        return LanguageModelSample(
            TokenSample(
                torch.from_numpy(
                    np.random.RandomState(self._seed + 48576439 + 74593 * index).randint(
                        0,
                        self._parameters.vocab_size,
                        size=(self._parameters.sequence_length + self._parameters.extra_tokens,),
                        dtype=np.int64,
                    )
                )
            )
        )

    @property
    def name(self) -> str:
        return self._name
