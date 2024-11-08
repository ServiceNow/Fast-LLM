import abc
import enum
import logging
import pathlib
import typing

from fast_llm.config import Config, Field, FieldHint, check_field, config_class
from fast_llm.data.config import (
    DataConfig,
    FimConfig,
    MultiprocessingContext,
    RawDataset,
    SampledDataset,
    TokenizerConfig,
)
from fast_llm.engine.distributed.config import PhaseType
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    from fast_llm.data.gpt.data import GPTData

logger = logging.getLogger(__name__)


class GPTRawDataset(RawDataset):
    def __len__(self):
        return self.num_documents

    def get(self, document: int, offset: int = 0, length: int | None = None):
        pass

    def __getitem__(self, index: int):
        """
        Get the sample (document) with the given index (in the split dataset).
        """
        return self.get(index)

    @property
    @abc.abstractmethod
    def num_documents(self):
        pass

    @property
    @abc.abstractmethod
    def num_tokens(self):
        pass

    @property
    @abc.abstractmethod
    def document_sizes(self):
        pass


class GPTDatasetConfigType(str, enum.Enum):
    split = "split"
    splits = "splits"
    concatenated = "concatenated"
    blended = "blended"
    memmap = "random"


@config_class()
class GPTDatasetConfig(Config):

    _abstract = True
    type: GPTDatasetConfigType = Field(
        desc="Format for the dataset definition.",
        hint=FieldHint.core,
    )
    prefix = Field(
        desc="A prefix for the dataset name, set by wrapping datasets.",
        init=False,
    )

    def _validate(self):
        assert hasattr(self, "prefix")
        super()._validate()

    @classmethod
    def from_dict(
        cls,
        default: typing.Union["Config", dict[str, typing.Any]],
        *updates: typing.Union["Config", dict[str | tuple[str, ...], typing.Any]],
        strict: bool = True,
    ):
        if cls.type == GPTDatasetConfigType.split:
            cls_ = GPTSplitDatasetConfig
        elif cls.type == GPTDatasetConfigType.splits:
            cls_ = GPTDatasetSplitsConfig
        elif cls.type == GPTDatasetConfigType.concatenated:
            cls_ = GPTConcatenatedDatasetConfig
        elif cls.type == GPTDatasetConfigType.blended:
            cls_ = GPTBlendedDatasetConfig
        elif cls.type == GPTDatasetConfigType.memmap:
            cls_ = GPTMemmapDatasetConfig
        else:
            raise NotImplementedError(cls.type)
        return cls_.from_dict(default, *updates, strict=strict)

    @property
    def sampled(self) -> bool:
        raise NotImplementedError()

    @property
    def split(self) -> bool:
        raise NotImplementedError()

    def build_split_sampled(self, data: "GPTData") -> dict[PhaseType, SampledDataset]:
        if self.split:
            return self._build_split_sampled(data)
        else:
            return {PhaseType.training: self.build_unsplit_sampled(data)}

    def build_unsplit_sampled(self, data: "GPTData") -> SampledDataset:
        assert not self.split
        if self.sampled:
            return self._build_unsplit_sampled(data)
        else:
            return self._sample(self.build_unsplit_unsampled(), data)

    def build_split_unsampled(self) -> dict[PhaseType, GPTRawDataset]:
        assert not self.sampled
        if self.split:
            return self._build_split_unsampled()
        else:
            return {PhaseType.training: self.build_unsplit_unsampled()}

    def build_unsplit_unsampled(self) -> GPTRawDataset:
        assert not self.split
        assert not self.sampled
        return self._build_unsplit_unsampled()

    def _build(self) -> GPTRawDataset | SampledDataset | dict[PhaseType, GPTRawDataset | SampledDataset]:
        raise NotImplementedError()

    def _build_split_sampled(self, data: "GPTData") -> dict[PhaseType, SampledDataset]:
        raise NotImplementedError()

    def _build_unsplit_sampled(self, data: "GPTData") -> SampledDataset:
        raise NotImplementedError()

    def _build_split_unsampled(self) -> dict[PhaseType, GPTRawDataset]:
        raise NotImplementedError()

    def _build_unsplit_unsampled(self) -> GPTRawDataset:
        raise NotImplementedError()

    @property
    def full_name(self) -> str:
        return f"{self.prefix}{self._base_name}"

    @property
    def _base_name(self) -> str:
        raise NotImplementedError()


@config_class()
class GPTMemmapDatasetConfig(GPTDatasetConfig):
    # Path -> (unsampled, unsplit)
    _abstract = False
    path: pathlib.Path = Field(
        desc="The path to the dataset, excluding the `.bin` or `.idx` suffix.",
        hint=FieldHint.core,
    )

    @property
    def split(self) -> bool:
        return False

    @property
    def sampled(self) -> bool:
        return False

    def _build_unsplit_unsampled(self) -> GPTRawDataset:
        from fast_llm.data.gpt.memmap import GPTMemmapDataset

        return GPTMemmapDataset(self)

    @property
    def _base_name(self) -> str:
        return self.path.stem


@config_class()
class GPTConcatenatedDatasetConfig(GPTDatasetConfig):
    """
    Concatenate multiple datasets as if they were one.
    Must be done before sampling and splitting.
    TODO: OK after sampling (staged training?) or splitting (Equal split for each sub-dataset, probably better?
    [(unsampled, unsplit)] -> (unsampled, unsplit)
    """

    _abstract = False
    name: str = Field(
        default="concatenated",
        desc="The name of the dataset.",
        hint=FieldHint.core,
    )
    datasets: list[GPTDatasetConfig] = Field(
        desc="The datasets to concatenate.",
        hint=FieldHint.core,
    )

    def _validate(self):
        if not hasattr(self, "prefix"):
            self.prefix = ""
        for i, dataset in enumerate(self.datasets):
            # The phase name will also appear in the phase suffix,
            # but we still need this to disambiguate the non-suffixed name.
            dataset.prefix = f"{self.prefix}{self._base_name}{i}_"
        super()._validate()
        for dataset in self.datasets:
            assert not dataset.split
        assert not any(dataset.sampled for dataset in self.datasets)

    @property
    def split(self) -> bool:
        return False

    @property
    def sampled(self) -> bool:
        return False

    def _build_unsplit_unsampled(self) -> GPTRawDataset:
        from fast_llm.data.gpt.concatenated import GPTConcatenatedDataset

        return GPTConcatenatedDataset(self, [dataset.build_unsplit_unsampled() for dataset in self.datasets])

    @property
    def _base_name(self) -> str:
        return f"{self.name}/"


@config_class()
class GPTSplitDatasetConfig(GPTDatasetConfig):
    """
    Split a single dataset into multiple phases.
    Must be done before sampling.
    TODO: Ok after sampling?
    (unsampled, unsplit) -> (unsampled, split)
    """

    _abstract = False
    dataset: GPTDatasetConfig = Field(
        desc="The dataset to split.",
        hint=FieldHint.core,
    )
    ratios: dict[PhaseType, float] = Field(
        desc="The split ratio for each phase",
        hint=FieldHint.core,
    )
    name: str = Field(
        default="split",
        desc="The name of the dataset.",
        hint=FieldHint.core,
    )

    def _validate(self):
        if not hasattr(self, "prefix"):
            self.prefix = ""
        self.dataset.prefix = f"{self.prefix}{self._base_name}"
        super()._validate()
        assert not self.dataset.split
        assert not self.dataset.sampled

    @property
    def sampled(self) -> bool:
        return False

    @property
    def split(self) -> bool:
        return True

    def _build_split_unsampled(self) -> dict[PhaseType, GPTRawDataset]:
        from fast_llm.data.gpt.slice import GPTDatasetSlice

        return GPTDatasetSlice.from_splits(self.dataset.build_unsplit_unsampled(), self.ratios)

    @property
    def _base_name(self) -> str:
        return f"{self.name}/"


@config_class()
class GPTDatasetSplitsConfig(GPTDatasetConfig):
    """
    Create a separate dataset for each phase.
    May be done before or after sampling.
    {phase:(?sampled, unsplit)} -> (?sampled, split)
    """

    _abstract = False
    datasets: dict[PhaseType, GPTDatasetConfig] = Field(
        desc="The dataset to split.",
        hint=FieldHint.core,
    )

    def _validate(self):
        if not hasattr(self, "prefix"):
            self.prefix = ""
        for phase, dataset in self.datasets.items():
            # The phase name will also appear in the phase suffix,
            # but we still need this to disambiguate the non-suffixed name.
            dataset.prefix = f"{self.prefix}{phase.value}/"
        super()._validate()
        for phase, dataset in self.datasets.items():
            assert not dataset.split, phase
        _ = self.sampled

    @property
    def split(self) -> bool:
        return True

    @property
    def sampled(self) -> bool:
        sampled = {dataset.sampled for dataset in self.datasets.values()}
        assert len(sampled) == 1, sampled
        return sampled.pop()

    def _build_split_sampled(self, data: "GPTData") -> dict[PhaseType, SampledDataset]:
        return {phase: dataset.build_unsplit_sampled(data) for phase, dataset in self.datasets.items()}

    def _build_split_unsampled(self) -> dict[PhaseType, GPTRawDataset]:
        return {phase: dataset.build_unsplit_unsampled() for phase, dataset in self.datasets.items()}

    @property
    def _base_name(self) -> str:
        return ""


@config_class()
class GPTBlendedDatasetConfig(GPTDatasetConfig):
    # [(?sampled, ?split)] -> (sampled, split)
    _abstract = False
    datasets: list[GPTDatasetConfig] = Field(
        desc="The datasets to concatenate.",
        hint=FieldHint.core,
    )
    weights: list[float] = Field(
        desc="The blending weight of each dataset.",
        hint=FieldHint.core,
    )

    def _validate(self):
        super()._validate()
        for dataset in self.datasets:
            assert not dataset.split
        assert not any(dataset.sampled for dataset in self.datasets)

    @property
    def split(self) -> bool:
        return True

    @property
    def sampled(self) -> bool:
        return True

    def _build_split_sampled(self, data: "GPTData") -> dict[PhaseType, SampledDataset]:
        from fast_llm.data.blended import BlendedDataset

        datasets = {}
        for dataset in self.datasets:
            dataset_split = dataset.build_split_sampled(data)
            if datasets:
                Assert.eq(set(datasets), set(dataset_split))
            else:
                datasets = {phase: [] for phase in dataset_split}
            for phase, phase_datasets in datasets.items():
                phase_datasets.append(dataset_split[phase])
        return {
            phase: BlendedDataset(phase_datasets, self.weights, data) for phase, phase_datasets in datasets.items()
        }


@config_class()
class GPTDataConfig(DataConfig):
    """
    Configuration for the dataset(s), split and sampling.
    Currently hard-coded to a GPT dataset.
    TODO: Extract generalizable content.
    """

    _abstract = False

    dataset: GPTDatasetConfig = Field(
        # TODO: Dummy default?
        default_factory=GPTDatasetConfig,
        desc="Configuration for the dataset.",
        hint=FieldHint.core,
    )
    tokenizer: TokenizerConfig = Field(
        default_factory=TokenizerConfig,
        desc="Configuration for the tokenizer (for FIM).",
        hint=FieldHint.feature,
    )
    fim: FimConfig = Field(
        default_factory=FimConfig,
        desc="Configuration for Fill In the Middle (FIM).",
        hint=FieldHint.feature,
    )
    # TODO: set default to [1,0,0]?
    # split: list[float] = Field(
    #     default_factory=lambda: [969, 30, 1],
    #     desc="Split ratio for train, valid and test datasets.",
    #     hint=FieldHint.core,
    #     valid=_validate_split,
    # )
    # format: DatasetSource = Field(
    #     default=DatasetSource.list,
    #     desc="Format for the dataset definition.",
    #     hint=FieldHint.core,
    # )
    # path: list[str] = Field(
    #     default_factory=list,
    #     desc="Path or list of paths and weights.",
    #     hint=FieldHint.core,
    #     valid=_validate_path,
    # )
    data_sample_warn_time_ms: float = Field(
        default=1000,
        desc="Warn if a sample takes too long to load.",
        hint=FieldHint.feature,
        valid=check_field(Assert.gt, 0),
    )
    multiprocessing_context: MultiprocessingContext = Field(
        default=MultiprocessingContext.spawn,
        desc="Multiprocessing context. Do not touch.",
        hint=FieldHint.expert,
    )
