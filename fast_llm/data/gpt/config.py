import pathlib
import typing

from fast_llm.config import Config, Field, FieldHint, check_field, config_class
from fast_llm.data.config import (
    Data,
    DataConfig,
    DatasetSource,
    FimConfig,
    MultiprocessingContext,
    SamplableDataset,
    SampledDataset,
    SamplingConfig,
    TokenizerConfig,
    _validate_path,
    _validate_split,
)
from fast_llm.engine.distributed.config import PhaseType
from fast_llm.utils import Assert, Registry

if typing.TYPE_CHECKING:
    from fast_llm.data.gpt.data import GPTData
    from fast_llm.data.gpt.dataset import GPTIndexedDataset, GPTSamplingConfig

dataset_registry = Registry("dataset")


@config_class()
class DatasetConfig(Config):
    _abstract = True
    type: str = Field(
        desc="Format for the dataset definition.",
        hint=FieldHint.core,
    )

    @classmethod
    def from_dict(
        cls,
        default: Config | dict[str, typing.Any],
        *updates: Config | dict[str | tuple[str, ...], typing.Any],
        strict: bool = True,
    ):
        if cls.type == 1:
            cls_ = GPTSplitDatasetConfig
        elif cls.type == 1:
            cls_ = GPTConcatenatedDatasetConfig
        elif cls.type == 1:
            cls_ = GPTBlendedDatasetConfig
        elif cls.type == 1:
            cls_ = GPTMemmapDatasetConfig
        else:
            raise NotImplementedError(cls.type)
        Assert.custom(issubclass, cls_, cls)
        return cls_.from_dict(default, *updates, strict=strict)

    def build(self, config: "SamplingConfig", data: "Data") -> dict[PhaseType, SampledDataset]:
        raise NotImplementedError()


@config_class()
class SamplableDatasetConfig(DatasetConfig):
    def build(self, config: "GPTSamplingConfig", data: "GPTData") -> dict[PhaseType, SampledDataset]:
        return {phase: dataset.sample(config, data) for phase, dataset in self.build_unsampled().items()}

    def build_unsampled(self) -> dict[PhaseType, SamplableDataset]:
        raise NotImplementedError()


@config_class()
class SplittableDatasetConfig(DatasetConfig):
    def build_unsampled(self) -> dict[PhaseType, SamplableDataset]:
        return {PhaseType.training: self.build_unsplit()}

    def build_unsplit(self) -> SamplableDataset:
        raise NotImplementedError()


@config_class()
class GPTIndexedDatasetConfig(SplittableDatasetConfig):
    def build_unsplit(self) -> GPTIndexedDataset:
        raise NotImplementedError()


@config_class()
class GPTMemmapDatasetConfig(GPTIndexedDatasetConfig):
    # Path -> (unsampled, unsplit)
    _abstract = False
    path: pathlib.Path = Field(
        desc="The path to the dataset, excluding the `.bin` or `.idx` suffix.",
        hint=FieldHint.core,
    )

    def build_unsplit(self) -> SamplableDataset:
        from fast_llm.data.gpt.memmap import GPTMemmapDataset

        return GPTMemmapDataset(self)


@config_class()
class GPTConcatenatedDatasetConfig(GPTIndexedDatasetConfig):
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
    datasets: list[GPTIndexedDatasetConfig] = Field(
        desc="The datasets to concatenate.",
        hint=FieldHint.core,
    )

    def build_unsplit(self) -> SamplableDataset:
        from fast_llm.data.gpt.concatenated import GPTConcatenatedDataset

        return GPTConcatenatedDataset(self, [dataset.build_unsplit() for dataset in self.datasets])


@config_class()
class GPTSplitDatasetConfig(SamplableDatasetConfig):
    """
    Split a single dataset into multiple phases.
    Must be done before sampling.
    TODO: Ok after sampling?
    (unsampled, unsplit) -> (unsampled, split)
    """

    _abstract = False
    dataset: GPTIndexedDatasetConfig = Field(
        desc="The dataset to split.",
        hint=FieldHint.core,
    )
    ratios: dict[PhaseType, float] = Field(
        desc="The split ratio for each phase",
        hint=FieldHint.core,
    )

    def build_unsampled(self) -> dict[PhaseType, SamplableDataset]:
        from fast_llm.data.gpt.slice import GPTDatasetSlice

        return GPTDatasetSlice.from_splits(self)


@config_class()
class GPTBlendedDatasetConfig(DatasetConfig):
    # [(?sampled, ?split)] -> (sampled, split)
    _abstract = False
    datasets: list[DatasetConfig] = Field(
        desc="The datasets to concatenate.",
        hint=FieldHint.core,
    )
    weights: list[float] = Field(
        desc="The blending weight of each dataset.",
        hint=FieldHint.core,
    )

    @property
    def split(self) -> bool:
        return True

    @property
    def sampled(self) -> bool:
        return True

    def build(self, config: "GPTSamplingConfig", data: "GPTData") -> dict[PhaseType, SampledDataset]:
        from fast_llm.data.blended import BlendedDataset

        datasets = {}
        for dataset in self.datasets:
            dataset_split = dataset.build(data)
            if datasets:
                Assert.eq(set(datasets), set(dataset_split))
            else:
                datasets = {phase: [] for phase in dataset_split}
            for phase, phase_datasets in datasets.items():
                phase_datasets.append(dataset_split[phase])

        BlendedDataset(
            list(datasets.values()),
            weights=[self._dataset_weights[name] for name in datasets],
            name=phase.value,
            num_samples=self._samples_per_phase[phase],
            cache_directory=self._cache_directory,
            group=self._distributed.world_group,
            verbose=run.is_main_rank,
            data_sample_warn_time_ms=self._config.data_sample_warn_time_ms,
        )
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
    split: list[float] = Field(
        default_factory=lambda: [969, 30, 1],
        desc="Split ratio for train, valid and test datasets.",
        hint=FieldHint.core,
        valid=_validate_split,
    )
    format: DatasetSource = Field(
        default=DatasetSource.list,
        desc="Format for the dataset definition.",
        hint=FieldHint.core,
    )
    path: list[str] = Field(
        default_factory=list,
        desc="Path or list of paths and weights.",
        hint=FieldHint.core,
        valid=_validate_path,
    )
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
