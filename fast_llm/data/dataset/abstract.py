import abc
import typing

from fast_llm.engine.distributed.config import PhaseType

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

    @abc.abstractmethod
    def as_split(self, default_phase: PhaseType = PhaseType.training):
        pass


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

    def as_split(self, default_phase: PhaseType = PhaseType.training):
        return SplitDataset(self.name, {default_phase: self})


class SamplableDataset(Dataset):

    @abc.abstractmethod
    def sample(self, config: "SamplingConfig") -> SampledDataset:
        pass

    def as_split(self, default_phase: PhaseType = PhaseType.training) -> "SplitDataset":
        return SplitDataset(self.name, {default_phase: self})


_SplittableType = typing.TypeVar("_SplittableType")
_DatasetType = typing.TypeVar("_DatasetType", bound=Dataset)
_SampledDatasetType = typing.TypeVar("_SampledDatasetType", bound=SampledDataset)
_SamplableDatasetType = typing.TypeVar("_SamplableDatasetType", bound=SamplableDataset)


class PhaseSplits(dict[PhaseType, _SplittableType], typing.Generic[_SplittableType]):
    pass


class SplitDataset(Dataset, PhaseSplits[_DatasetType], typing.Generic[_DatasetType]):
    def __init__(self, name: str, datasets: dict[PhaseType, _DatasetType]):
        super().__init__(datasets)
        self._name = name

    def as_split(self, default_phase: PhaseType = PhaseType.training):
        return self

    @property
    def name(self):
        return self._name


class SampledSplitDataset(SplitDataset[_SampledDatasetType], typing.Generic[_SampledDatasetType]):
    pass


class SamplableSplitDataset(SplitDataset[_SamplableDatasetType], typing.Generic[_SamplableDatasetType]):
    def sample(self, sampling_configs: "PhaseSplits[SamplingConfig]"):
        return SampledSplitDataset(
            f"{self.name}_sampled",
            {phase: self[phase].sample(sampling_config) for phase, sampling_config in sampling_configs.items()},
        )


class CopySplitDataset(SamplableSplitDataset):
    def __init__(self, name: str, dataset: _SplittableType, phases: list[PhaseType]):
        super().__init__(name, {phase: dataset for phase in phases})
