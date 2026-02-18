import abc
import typing

from fast_llm.data.document.abstract import Document

if typing.TYPE_CHECKING:
    from fast_llm.data.dataset.config import SamplingData


class Dataset[DocumentType: Document](abc.ABC):
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
        # Pickling sometimes fails with bound `DocumentType`.
        # This is not needed at runtime, so we just drop it.
        if "__orig_class__" in state:
            del state["__orig_class__"]
        return state


class SampledDataset[DocumentType: Document](Dataset[DocumentType]):
    """
    A sampled dataset class containing a prepared list of samples to be indexed sequentially (as-is) during training.
    (See the `Sampler` class below.)
    """

    @abc.abstractmethod
    def __getitem__(self, index: int) -> list[DocumentType]:
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        pass


class SamplableDataset[DocumentType: Document](Dataset[DocumentType]):

    @abc.abstractmethod
    def sample(self, config: "SamplingData") -> SampledDataset[DocumentType]:
        pass
