import abc
import typing

if typing.TYPE_CHECKING:
    from fast_llm.data.dataset.config import SamplingData


class Dataset(abc.ABC):
    """
    A generic dataset class compatible with torch.utils.data.Dataset but with a slightly different signature.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        A name for the dataset to facilitate identification and debugging.
        """


class SampledDataset(Dataset):
    """
    A sampled dataset class containing a prepared list of samples to be indexed sequentially (as-is) during training.
    (See the `Sampler` class below.)
    """

    @abc.abstractmethod
    def __getitem__(self, index: int) -> typing.Any:
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        pass


class SamplableDataset(Dataset):

    @abc.abstractmethod
    def sample(self, config: "SamplingData") -> SampledDataset:
        pass
