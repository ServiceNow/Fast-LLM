import logging
import typing

import numpy as np

from fast_llm.data.dataset.abstract import SampledDataset
from fast_llm.data.dataset.config import SamplingData
from fast_llm.utils import Assert, normalize_probabilities

logger = logging.getLogger(__name__)


class BlendedDataset(SampledDataset):
    """
    A blended sampling of multiple sampled datasets, where each dataset is sampled with the provided probability.
    The sampling order of each dataset is respected, but there is no strict guarantee
    on the total number of samples from each dataset.
    The sampling exactly matches Megatron-LM with matching parameters.
    """

    def __init__(
        self,
        name: str,
        datasets: list[SampledDataset],
        weights: list[float],
        sampling_config: SamplingData,
    ):
        self._name = name
        assert len(datasets) > 0
        Assert.eq(len(datasets), len(weights))
        self._datasets = datasets
        self._weights = np.array(normalize_probabilities(weights))
        self._num_samples = sampling_config.parameters.num_samples

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> typing.Any:
        """
        Blending is typically done in one of the following iterative way (ex. in Megatron datasets):
        ```python
        dataset_index=np.zeros(num_samples)
        sample_index=np.zeros(num_samples)
        sampled=np.zeros(len(weights))
        for idx in range(num_samples):
            error = weights * (idx + 1) - sampled
            dataset_index_ = np.argmax(error)
            dataset_index[idx] = dataset_index_
            sample_index[idx] = sampled[dataset_index_]
            sampled[dataset_index_] +=1
        ```
        I.e. it iteratively picks samples to minimize the error `weights * sum(sampled) - sampled`.
        This implementation computes values on the fly instead of pre-computing them all.
        """
        # We find the number of samples taken from each dataset prior to this point.
        sampled = self._get_sampled(idx)
        # Then get the present sample.
        dataset_index = self._get_next_dataset(idx, sampled)
        return self._datasets[dataset_index][sampled[dataset_index]]

    def _get_sampled(self, num_samples: int):
        # First we determine a lower bound.
        # This is indeed a lower bound because a lower value for one dataset would involve more sampling below,
        # and it would be from that same dataset because it would have the highest error,
        sampled = np.floor(self._weights * num_samples).astype(int)
        # Then we sample until we reach the target number of samples.
        # This may not match the actual sampling order, but the final value of `sampled` is correct.
        for idx in range(sampled.sum(), num_samples):
            dataset_index = self._get_next_dataset(idx, sampled)
            sampled[dataset_index] += 1
        return sampled

    def _get_next_dataset(self, idx, sampled):
        # The next sample is the one with the highest error.
        return (self._weights * (idx + 1) - sampled).argmax()

    @property
    def name(self):
        return self._name
