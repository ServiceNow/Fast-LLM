import numpy as np
import torch

from fast_llm.data.dataset.abstract import SampledDataset
from fast_llm.data.dataset.gpt.config import GPTSamplingData
from fast_llm.data.preprocessing.language_model import LanguageModelPreprocessingConfig
from fast_llm.data.sample.language_model import LanguageModelSample
from fast_llm.data.sample.token import TokenSample
from fast_llm.engine.config_utils.data_type import get_integer_type


class GPTRandomSampledDataset[SampleType: LanguageModelSample](SampledDataset[SampleType]):
    def __init__(self, sampling: GPTSamplingData, name: str):
        self._name = name
        self._seed = sampling.config.seed
        self._parameters = sampling.parameters

        assert isinstance(sampling.preprocessing, LanguageModelPreprocessingConfig)
        assert not sampling.preprocessing.use_loss_masking_spans
        assert not sampling.preprocessing.use_preference_spans
        assert not sampling.preprocessing.use_image_patches
        self._vocab_size = sampling.preprocessing.vocab_size

        self._dtype = get_integer_type(self._vocab_size).torch

    def __len__(self) -> int:
        return self._parameters.num_samples

    def __getitem__(self, index: int) -> SampleType:
        # TODO: Sample in self._dtype (breaking)
        return LanguageModelSample(
            TokenSample(
                torch.from_numpy(
                    np.random.RandomState(self._seed + 48576439 + 74593 * index).randint(
                        0,
                        self._vocab_size,
                        size=(self._parameters.sequence_length + self._parameters.extra_tokens,),
                    )
                ).to(self._dtype),
            )
        )

    @property
    def name(self) -> str:
        return self._name
