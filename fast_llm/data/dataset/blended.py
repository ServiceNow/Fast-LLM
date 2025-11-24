import logging

import torch

from fast_llm.data.dataset.abstract import SampledDataset
from fast_llm.data.dataset.config import SamplingData
from fast_llm.data.sample.abstract import Sample
from fast_llm.utils import Assert, normalize_probabilities

logger = logging.getLogger(__name__)


class BlendedDataset[SampleType: Sample](SampledDataset[SampleType]):
    """
    A blended sampling of multiple sampled datasets, where each dataset is sampled with the provided probability.
    The sampling order of each dataset is respected, but there is no strict guarantee
    on the total number of samples from each dataset.
    The sampling exactly matches Megatron-LM with matching parameters.
    """

    def __init__(
        self,
        name: str,
        datasets: list[SampledDataset[SampleType]],
        weights: list[float],
        sampling_config: SamplingData,
    ):
        self._name = name
        assert len(datasets) > 0
        Assert.eq(len(datasets), len(weights))
        self._datasets = datasets
        self._weights = torch.from_numpy(normalize_probabilities(weights, return_array=True))
        self._num_samples = sampling_config.parameters.num_samples

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, index: int) -> SampleType:
        """
        Blending is typically done in one of the following iterative way (ex. in Megatron datasets):
        ```python
        dataset_index=np.zeros(num_samples)
        sample_index=np.zeros(num_samples)
        sampled=np.zeros(len(weights))
        for index in range(num_samples):
            error = weights * (index + 1) - sampled
            dataset_index_ = np.argmax(error)
            dataset_index[index] = dataset_index_
            sample_index[index] = sampled[dataset_index_]
            sampled[dataset_index_] +=1
        ```
        I.e. it iteratively picks samples to minimize the error `weights * sum(sampled) - sampled`.
        This implementation computes values on the fly instead of pre-computing them all.
        """
        # We find the number of samples taken from each dataset prior to this point.
        sampled = self._get_sampled(index)
        # Then get the present sample.
        dataset_index = self._get_next_dataset(index, sampled)
        return self._datasets[dataset_index][sampled[dataset_index].item()]

    def _get_sampled(self, num_samples: int) -> torch.Tensor:
        # First we determine a lower bound.
        # This is indeed a lower bound because a lower value for one dataset would involve more sampling below,
        # and it would be from that same dataset because it would have the highest error,

        sampled = (self._weights * num_samples).to(torch.int64)
        # Then we sample until we reach the target number of samples.
        # This may not match the actual sampling order, but the final value of `sampled` is correct.
        for index in range(sampled.sum().item(), num_samples):
            dataset_index = self._get_next_dataset(index, sampled)
            sampled[dataset_index] += 1
        return sampled

    def _get_next_dataset(self, index: int, sampled: torch.Tensor) -> int:
        # The next sample is the one with the highest error.
        return (self._weights * (index + 1) - sampled).argmax().item()

    @property
    def name(self) -> str:
        return self._name
