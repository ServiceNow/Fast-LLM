import abc
import typing

if typing.TYPE_CHECKING:
    from fast_llm.data.dataset.config import SamplingConfig


class Dataset(abc.ABC):
    """
    A generic dataset class compatible with torch.utils.data.Dataset but with a slightly different signature.
    """

    @property
    @abc.abstractmethod
    def name(self):
        """
        A name for the dataset to facilitate identification and debugging.
        """


class SampledDataset(Dataset):
    """
    A sampled dataset class containing a prepared list of samples to be indexed sequentially (as-is) during training.
    (See the `Sampler` class below.)
    """

    @abc.abstractmethod
    def __getitem__(self, index: int):
        pass

    @abc.abstractmethod
    def __len__(self):
        pass


class SamplableDataset(Dataset):

    @abc.abstractmethod
    def sample(self, config: "SamplingConfig") -> SampledDataset:
        pass
