import numpy as np
import torch

from fast_llm.data.dataset.abstract import SampledDataset
from fast_llm.data.dataset.gpt.config import GPTSamplingConfig
from fast_llm.data.document.language_model import LanguageModelDocument
from fast_llm.engine.config_utils.data_type import get_unsigned_integer_type


class GPTRandomSampledDataset[DocumentType: LanguageModelDocument](SampledDataset[DocumentType]):
    def __init__(self, sampling: GPTSamplingConfig, name: str, num_samples: int, seed: int):
        self._name = name
        self._seed = seed
        self._config = sampling
        self._num_samples = num_samples

        self._vocab_size = sampling.preprocessing.vocab_size

        self._dtype = get_unsigned_integer_type(self._vocab_size).torch

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, index: int) -> list[DocumentType]:
        # TODO: Sample in self._dtype (breaking)
        return [
            LanguageModelDocument(
                tokens=torch.from_numpy(
                    np.random.RandomState(self._seed + 48576439 + 74593 * index).randint(
                        0,
                        self._vocab_size,
                        size=(self._config.sample_size,),
                    )
                ).to(self._dtype),
            )
        ]

    @property
    def name(self) -> str:
        return self._name
