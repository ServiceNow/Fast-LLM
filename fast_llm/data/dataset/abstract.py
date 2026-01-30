import abc
import typing

from fast_llm.data.sample.abstract import Sample

if typing.TYPE_CHECKING:
    from fast_llm.data.dataset.config import SamplingData
    from fast_llm.data.dataset.sampled import SampledIterableDataset


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

    @property
    def requires_broadcast(self) -> bool:
        """
        Some dataset schemes load the dataset on a batch-data-parallel group leaders,
        then broadcast to the other devices.
        """
        return False


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


class SamplableDataset[SampleType: Sample](Dataset[SampleType]):

    @abc.abstractmethod
    def sample(self, config: "SamplingData") -> SampledDataset[SampleType]:
        pass


class SamplableIterableDataset[SampleType: Sample](SamplableDataset[SampleType]):
    @abc.abstractmethod
    def iterate(self, sampling: "SamplingData") -> typing.Iterator[SampleType]:
        pass

    def sample(self, config: "SamplingData") -> "SampledIterableDataset[SampleType]":
        from fast_llm.data.dataset.sampled import SampledIterableDataset

        return SampledIterableDataset(self, config)
