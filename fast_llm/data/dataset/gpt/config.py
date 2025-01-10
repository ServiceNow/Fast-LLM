import dataclasses
import enum
import functools
import json
import pathlib
import typing

from fast_llm.config import Config, Field, FieldHint, FieldUpdate, check_field, config_class, skip_valid_if_none
from fast_llm.data.dataset.abstract import PhaseSplits, SamplableSplitDataset, SampledDataset, SampledSplitDataset
from fast_llm.data.dataset.config import (
    BlendedDatasetConfig,
    DatasetConfig,
    SamplableDatasetConfig,
    SamplableSplitDatasetConfig,
    SampledDatasetConfig,
    SampledSplitDatasetConfig,
    SamplingConfig,
)
from fast_llm.engine.distributed.config import PhaseType
from fast_llm.utils import Assert, Registry, normalize_probabilities

if typing.TYPE_CHECKING:
    from fast_llm.data.dataset.gpt.dummy import GPTDummySampledDataset
    from fast_llm.data.dataset.gpt.indexed import GPTConcatenatedDataset, GPTDatasetSlice, GPTIndexedDataset
    from fast_llm.data.dataset.gpt.memmap import GPTMemmapDataset


@dataclasses.dataclass
class GPTSamplingConfig(SamplingConfig):
    # TODO: Sort these out
    sequence_length: int | None = None
    vocab_size: int | None = None
    tokenizer: typing.Any = None


@config_class()
class GPTDatasetConfig(DatasetConfig):
    # TODO: Generalize dynamic types?
    _registry: typing.ClassVar[Registry[str, type["GPTDatasetConfig"]]] = Registry[str, type["GPTDatasetConfig"]](
        "gpt_dataset_class", {}
    )
    type_: typing.ClassVar[type["GPTDatasetConfig"] | None] = None
    type: str | None = Field(
        default=None,
        desc="The type of dataset.",
        hint=FieldHint.core,
    )

    def _validate(self):
        if self.type is not None:
            # Should be handled in `from_dict`, but can fail if instantiating directly.
            Assert.eq(self.type, self.type_)
        super()._validate()

    @classmethod
    def _from_dict(
        cls,
        default: dict[str, typing.Any],
        strict: bool = True,
        flat: bool = False,
    ):
        type_ = default.get("type")
        if type_ is None:
            actual_cls = cls
        else:
            actual_cls = cls._registry[type_]
            Assert.custom(issubclass, actual_cls, cls)
        if actual_cls == cls:
            return super()._from_dict(default, strict=strict, flat=flat)
        else:
            return actual_cls._from_dict(default, strict=strict, flat=flat)

    def __init_subclass__(cls, type_: str | None = None, **kwargs):
        if type_ is not None:
            GPTDatasetConfig._registry[type_] = cls
        cls.type_ = type_
        super().__init_subclass__()


@config_class()
class GPTSampledSplitDatasetConfig(SampledSplitDatasetConfig, GPTDatasetConfig):
    pass


@config_class()
class GPTSampledDatasetConfig(SampledDatasetConfig, GPTSampledSplitDatasetConfig):
    pass


@config_class()
class GPTSamplableSplitDatasetConfig(SamplableSplitDatasetConfig, GPTSampledSplitDatasetConfig):
    pass


@config_class()
class GPTSamplableDatasetConfig(SamplableDatasetConfig, GPTSampledDatasetConfig, GPTSamplableSplitDatasetConfig):
    pass


@config_class()
class GPTIndexedDatasetConfig(GPTSamplableDatasetConfig):
    def build(self) -> "GPTIndexedDataset":
        raise NotImplementedError()


@config_class()
class GPTDummyDatasetConfig(GPTSampledSplitDatasetConfig, type_="dummy"):
    # NA -> (unsampled, unsplit)
    _abstract = False
    name: str = Field(
        default="dummy",
        desc="The name of the dataset.",
        hint=FieldHint.core,
    )

    def build_split_sample(
        self,
        config: PhaseSplits[GPTSamplingConfig],
        default_phase: PhaseType = PhaseType.training,
    ) -> "SampledSplitDataset[GPTDummySampledDataset]":
        from fast_llm.data.dataset.gpt.dummy import GPTDummyDataset, GPTDummySampledDataset

        return SampledSplitDataset[GPTDummySampledDataset](
            self.name,
            {
                phase: GPTDummyDataset(
                    f"{self.name}_{phase.value}", phase_config.sequence_length, phase_config.vocab_size
                ).sample(phase_config)
                for phase, phase_config in config.items()
            },
        )


@config_class()
class GPTMemmapDatasetConfig(GPTIndexedDatasetConfig, type_="memmap"):
    # Path -> (unsampled, unsplit)
    _abstract = False
    path: pathlib.Path = Field(
        default=None,
        desc="The path to the dataset, excluding the `.bin` or `.idx` suffix.",
        hint=FieldHint.core,
    )

    def build(self) -> "GPTMemmapDataset":
        from fast_llm.data.dataset.gpt.memmap import GPTMemmapDataset

        return GPTMemmapDataset(str(self.path).replace("/", "__"), self.path)


@config_class()
class GPTConcatenatedDatasetConfig(GPTDatasetConfig, SamplableDatasetConfig, type_="concatenated"):
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
        default_factory=list,
        desc="The datasets to concatenate.",
        hint=FieldHint.core,
        valid=check_field(functools.partial(Assert.custom, lambda x: len(x) > 0)),
    )

    def build(self) -> "GPTConcatenatedDataset":
        from fast_llm.data.dataset.gpt.indexed import GPTConcatenatedDataset

        return GPTConcatenatedDataset(self.name, [dataset.build() for dataset in self.datasets])


@config_class()
class GPTSplitDatasetConfig(GPTSamplableSplitDatasetConfig, type_="split"):
    """
    Split a single dataset into multiple phases.
    Must be done before sampling.
    TODO: Ok after sampling?
    (unsampled, unsplit) -> (unsampled, split)
    """

    _abstract = False
    dataset: GPTIndexedDatasetConfig = Field(
        default=None,
        desc="The dataset to split.",
        hint=FieldHint.core,
    )
    ratios: dict[PhaseType, float] = Field(
        default=None,
        desc="The split ratio for each phase",
        hint=FieldHint.core,
    )

    def build_split(
        self,
        default_phase: PhaseType = PhaseType.training,
    ) -> "SamplableSplitDataset[GPTDatasetSlice]":
        from fast_llm.data.dataset.gpt.indexed import GPTDatasetSlice

        return GPTDatasetSlice.from_splits(self.dataset.build(), self.ratios)


@config_class()
class GPTBlendedDatasetConfig(BlendedDatasetConfig, GPTSampledDatasetConfig, type_="blended"):
    _abstract = False
    datasets: list[GPTSampledDatasetConfig] = FieldUpdate()


class LegacyDatasetSource(str, enum.Enum):
    """
    An enum for the different ways to load datasets.
    """

    list = "list"
    file = "file"
    random = "random"


def _validate_split(value):
    Assert.leq(len(value), 3)
    return value + [0] * (len(value) - 3)


def _validate_path(value):
    return [value] if isinstance(value, str) else value


@config_class()
class FimConfig(Config):
    """
    Configuration for FIM.
    """

    dataset: GPTSampledDatasetConfig = Field(
        default=None,
        desc="The dataset to wrap with fim.",
        hint=FieldHint.core,
    )
    rate: float = Field(
        default=0.5,
        desc="FIM rate for each sample.",
        hint=FieldHint.core,
        valid=check_field(Assert.in_range_incl, 0, 1),
    )
    max_middle_len: int | None = Field(
        default=None,
        desc="Maximum length of the middle segment in FIM.",
        hint=FieldHint.feature,
        valid=skip_valid_if_none(check_field(Assert.gt, 0)),
    )
    split_sample: str | None = Field(
        default=None,
        desc="Split samples on this token and permute each fragment separately.",
        hint=FieldHint.feature,
    )
    fragment_rate: float = Field(
        default=0.0,
        desc="FIM rate for each fragment when using fim_split_sample.",
        hint=FieldHint.feature,
        valid=check_field(Assert.in_range_incl, 0, 1),
    )
    ignore_prefix: str | None = Field(
        default=None,
        desc="Do not apply FIM to fragments that start with this prefix.",
        hint=FieldHint.feature,
    )
    spm_rate: float = Field(
        default=0.5,
        desc="TODO.",
        hint=FieldHint.feature,
        valid=check_field(Assert.in_range_incl, 0, 1),
    )
    truncate_or_pad: bool = Field(
        default=False,
        desc="TODO.",
        hint=FieldHint.feature,
    )


@config_class()
class FimSampledDatasetConfig(GPTSampledDatasetConfig, FimConfig, type_="fim"):
    """
    Configuration for FIM.
    """

    dataset: GPTSampledDatasetConfig = Field(
        default=None,
        desc="The dataset to wrap with fim.",
        hint=FieldHint.core,
    )

    @property
    def split(self):
        return self.dataset.split

    def build_sample(
        self,
        config: GPTSamplingConfig,
    ) -> SampledDataset:
        from fast_llm.data.dataset.gpt.fim import FimDataset

        assert not self.split
        return FimDataset(self, self.dataset.build_sample(config), config)

    def build_split_sample(
        self,
        config: PhaseSplits[GPTSamplingConfig],
        default_phase: PhaseType = PhaseType.training,
    ) -> SampledSplitDataset:
        from fast_llm.data.dataset.gpt.fim import FimDataset

        if not self.split:
            # Take the base class shortcut using build_sample if it's available.
            return super().build_split_sample(config, default_phase)

        # Build, sample and split the datasets.
        sampled_datasets = self.dataset.build_split_sample(
            # Blending is deterministic and the error will never be higher than 1.
            PhaseSplits[SamplingConfig]({phase: phase_config for phase, phase_config in config.items()}),
            default_phase,
        )

        # Blend the datasets for each phase.
        return SampledSplitDataset[FimDataset](
            # TODO: Name
            "fim",
            {phase: FimDataset(self, sampled_datasets[phase], phase_config) for phase, phase_config in config.items()},
        )


@config_class()
class GPTLegacyConfig(Config):
    @classmethod
    def _from_dict(
        cls,
        default: dict[str, typing.Any],
        strict: bool = True,
        flat: bool = False,
    ):
        # TODO v0.3: Remove.
        cls._handle_renamed_field(default, "split", "ratio")
        return super()._from_dict(default, strict, flat)

    ratio: list[float] = Field(
        default_factory=lambda: [969, 30, 1],
        desc="Split ratio for train, valid and test datasets.",
        hint=FieldHint.deprecated,
        valid=_validate_split,
    )
    format: LegacyDatasetSource = Field(
        default=LegacyDatasetSource.list,
        desc="Format for the dataset definition.",
        hint=FieldHint.deprecated,
    )
    path: list[str] = Field(
        default_factory=list,
        desc="Path or list of paths and weights.",
        hint=FieldHint.deprecated,
        valid=_validate_path,
    )
    fim: FimConfig = Field(
        default_factory=FimConfig,
        desc="Configuration for Fill In the Middle (FIM).",
        hint=FieldHint.feature,
    )


@config_class()
class GPTLegacyDatasetConfig(GPTSampledSplitDatasetConfig, GPTLegacyConfig, type_="legacy"):
    _abstract = False

    def build_split_sample(
        self,
        config: PhaseSplits[GPTSamplingConfig],
        default_phase: PhaseType = PhaseType.training,
    ) -> SampledSplitDataset:

        if self.format == LegacyDatasetSource.random:
            Assert.eq(len(self.path), 0)
            # TODO: Multiple phase.
            dataset_config = GPTDummyDatasetConfig()
        else:
            if self.format == LegacyDatasetSource.file:
                Assert.eq(len(self.path), 1)
                data_path = pathlib.Path(self.path[0])
                dataset_defs = json.load(data_path.open("r"))
                data_base_path = data_path.parent
                dataset_prefixes = [
                    (data_base_path / dataset_def["prefix"]).resolve() for dataset_def in dataset_defs["datasets"]
                ]
                dataset_weights = normalize_probabilities(
                    [dataset_def["weight"] for dataset_def in dataset_defs["datasets"]]
                )
            elif self.format == LegacyDatasetSource.list:
                Assert.geq(len(self.path), 1)
                if len(self.path) == 1:
                    dataset_prefixes, dataset_weights = [self.path[0].strip()], [1.0]
                else:
                    Assert.custom(lambda x: x % 2 == 0, len(self.path))
                    dataset_prefixes = [pathlib.Path(x.strip()).resolve() for x in self.path[1::2]]
                    assert len(dataset_prefixes) == len(set(dataset_prefixes))
                    dataset_weights = normalize_probabilities([float(x) for x in self.path[::2]])
            else:
                raise NotImplementedError(self.format)

            dataset_configs = [
                GPTSplitDatasetConfig(
                    dataset=GPTMemmapDatasetConfig(path=prefix),
                    ratios={
                        PhaseType.training: self.ratio[0],
                        PhaseType.validation: self.ratio[1],
                        PhaseType.test: self.ratio[2],
                    },
                )
                for prefix in dataset_prefixes
            ]
            dataset_config = (
                GPTBlendedDatasetConfig(
                    name="blended",
                    datasets=dataset_configs,
                    weights=dataset_weights,
                )
                if len(dataset_configs) > 1
                else dataset_configs[0]
            )

        return dataset_config.build_split_sample(config, default_phase)
