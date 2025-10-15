import abc
import typing

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
