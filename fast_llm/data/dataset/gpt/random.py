import numpy as np
import torch

from fast_llm.data.dataset.abstract import SampledDataset
from fast_llm.data.dataset.gpt.config import GPTSamplingData
from fast_llm.data.document.language_model import LanguageModelDocument
from fast_llm.data.document.token import TokenDocument
from fast_llm.data.preprocessing.language_model import LanguageModelPreprocessingConfig
from fast_llm.engine.config_utils.data_type import get_unsigned_integer_type


class GPTRandomSampledDataset[DocumentType: LanguageModelDocument](SampledDataset[DocumentType]):
    def __init__(self, sampling: GPTSamplingData, name: str):
        self._name = name
        self._seed = sampling.config.seed
        self._parameters = sampling.parameters

        assert isinstance(sampling.preprocessing, LanguageModelPreprocessingConfig)
        self._vocab_size = sampling.preprocessing.vocab_size

        self._dtype = get_unsigned_integer_type(self._vocab_size).torch

    def __len__(self) -> int:
        return self._parameters.num_samples

    def __getitem__(self, index: int) -> list[DocumentType]:
        # TODO: Sample in self._dtype (breaking)
        return [
            LanguageModelDocument(
                tokens=TokenDocument(
                    tokens=torch.from_numpy(
                        np.random.RandomState(self._seed + 48576439 + 74593 * index).randint(
                            0,
                            self._vocab_size,
                            size=(self._parameters.sequence_length + self._parameters.extra_tokens,),
                        )
                    ).to(self._dtype),
                )
            )
        ]

    @property
    def name(self) -> str:
        return self._name
