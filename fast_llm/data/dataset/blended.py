import logging
import typing

import numpy as np

from fast_llm.data.dataset.abstract import SampledDataset
from fast_llm.data.dataset.config import SamplingConfig
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
        sampling_config: SamplingConfig,
    ):
        self._name = name
        assert len(datasets) > 0
        Assert.eq(len(datasets), len(weights))
        self._datasets = datasets
        self._weights = np.array(normalize_probabilities(weights))
        self._num_samples = sampling_config.num_samples

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> typing.Any:
        """
        Blending is typically done in one of the following way (ex. in Megatron datasets):
        ```python
        dataset_index=np.zeros(num_samples)
        sample_index=np.zeros(num_samples)
        samples_per_dataset=np.zeros(len(weights))
        for idx in range(num_samples):
            error = weights * idx - samples_per_dataset
            dataset_index_ = np.argmax(error)
            dataset_index[idx] =dataset_index_
            sample_index[idx] =samples_per_dataset[dataset_index_]
            samples_per_dataset[dataset_index_] +=1
        ```
        Here we provide an implementation which allows computing values on the fly instead pre-computing them all.
        """
        # We have target sampling amount in floating point,
        # and we want to find a set of integers as close as possible to this value.
        # First, we determine the target prior to the present sample.
        target = self._weights * idx
        # Then we determine a lower bound for the number previous samples from each dataset by rounding down the target.
        # This is indeed a lower bound because a lower value for one dataset would involve more sampling below,
        # and it would be from that dataset because it would have the highest error,
        # i.e., the dataset would have to be sampled again before `idx`.
        sampled = np.floor(target).astype(int)
        # Then we calculate the error between the target and lower bound, which we'll try to minimize below.
        error = target - sampled
        # Now we're ready to start sampling, and since we calculated a lower bound,
        # we may need to sample a few more until we reach the current sample,
        # each time by taking the dataset with the highest error (argmax).
        # By construction each dataset will be sampled at most once, so
        dataset_index = error.argsort(stable=True)[::-1][idx - sampled.sum()].item()
        return self._datasets[dataset_index.item()][sampled[dataset_index].item()]

    @property
    def name(self):
        return self._name
