import abc
import typing

import torch.utils.data.dataset

from fast_llm.data.sample.abstract import Sample

if typing.TYPE_CHECKING:
    from fast_llm.data.dataset.config import SamplingData


class Dataset[SampleType: Sample](abc.ABC):
    """
    A generic dataset class compatible with torch.utils.data.Dataset but with a slightly different signature.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        A name for the dataset to facilitate identification and debugging.
        """

    def __getstate__(self):
        state = super().__getstate__()
        # Pickling sometimes fails with bound `SampleType`.
        # This is not needed at runtime, so we just drop it.
        if "__orig_class__" in state:
            del state["__orig_class__"]
        return state


class SampledDataset[SampleType: Sample](Dataset[SampleType]):
    """
    A sampled dataset class containing a prepared list of samples to be indexed sequentially (as-is) during training.
    (See the `Sampler` class below.)
    """

    @abc.abstractmethod
    def __getitem__(self, index: int) -> SampleType:
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        pass


# NOTE: We need to inherit from IterableDataset overwise torch data loader can not detect it properly
class SampledIterableDataset[SampleType: Sample](torch.utils.data.dataset.IterableDataset[SampleType]):
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


class SamplableDataset[SampleType: Sample](Dataset[SampleType]):
    @abc.abstractmethod
    def sample(self, config: "SamplingData") -> SampledDataset[SampleType]:
        pass


class SamplableIterableDataset[SampleType: Sample](SampledIterableDataset[SampleType]):
    @abc.abstractmethod
    def sample(self, config: "SamplingData") -> SampledIterableDataset[SampleType]:
        pass
