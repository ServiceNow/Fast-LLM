import abc
import typing

import torch.utils.data

from fast_llm.data.sample.abstract import Sample

if typing.TYPE_CHECKING:
    from fast_llm.data.dataset.config import SamplingData


# NOTE: We need to inherit from IterableDataset otherwise torch data loader can not detect it properly
class SampledIterableDataset[SampleType: Sample](torch.utils.data.IterableDataset[SampleType]):
    """
    A sampled dataset class that provides an iterator over samples.
    """

    # NOTE: We add name here so it is compatible with Fast-LLM Dataset
    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        A name for the dataset to facilitate identification and debugging.
        """


class SamplableIterableDataset[SampleType: Sample](SampledIterableDataset[SampleType]):
    @abc.abstractmethod
    def sample(self, config: "SamplingData") -> SampledIterableDataset[SampleType]:
        pass
