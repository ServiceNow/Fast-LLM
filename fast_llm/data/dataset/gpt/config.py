import enum
import functools
import json
import pathlib
import typing

from fast_llm.config import Config, Field, FieldHint, FieldUpdate, check_field, config_class
from fast_llm.data.data.config import SamplingConfig
from fast_llm.data.dataset.abstract import PhaseSplits, SamplableSplitDataset, SampledSplitDataset
from fast_llm.data.dataset.config import (
    BlendedDatasetConfig,
    DatasetConfig,
    SamplableDatasetConfig,
    SamplableSplitDatasetConfig,
    SampledDatasetConfig,
    SampledSplitDatasetConfig,
)
from fast_llm.engine.distributed.config import PhaseType
from fast_llm.utils import Assert, Registry, normalize_probabilities

if typing.TYPE_CHECKING:
    from fast_llm.data.data.gpt.data import GPTData
    from fast_llm.data.dataset.gpt.dummy import GPTDummySampledDataset
    from fast_llm.data.dataset.gpt.indexed import GPTConcatenatedDataset, GPTDatasetSlice, GPTIndexedDataset
    from fast_llm.data.dataset.gpt.memmap import GPTMemmapDataset


@config_class
class GPTSamplingConfig(SamplingConfig):
    sequence_length: int = Field(default=None, desc="Number of token in each sample.")


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
    def build(self, data: "GPTData") -> "GPTIndexedDataset":
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
        data: "GPTData",
        config: PhaseSplits[GPTSamplingConfig],
        default_phase: PhaseType = PhaseType.training,
    ) -> "SampledSplitDataset[GPTDummySampledDataset]":
        from fast_llm.data.dataset.gpt.dummy import GPTDummyDataset, GPTDummySampledDataset

        return SampledSplitDataset[GPTDummySampledDataset](
            self.name,
            {
                phase: GPTDummyDataset(f"{self.name}_{phase.value}", data.max_sequence_length, data.vocab_size).sample(
                    phase_config, data
                )
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

    def build(self, data: "GPTData") -> "GPTMemmapDataset":
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

    def build(self, data: "GPTData") -> "GPTConcatenatedDataset":
        from fast_llm.data.dataset.gpt.indexed import GPTConcatenatedDataset

        return GPTConcatenatedDataset(self.name, [dataset.build(data) for dataset in self.datasets])


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
        data: "GPTData",
        default_phase: PhaseType = PhaseType.training,
    ) -> "SamplableSplitDataset[GPTDatasetSlice]":
        from fast_llm.data.dataset.gpt.indexed import GPTDatasetSlice

        return GPTDatasetSlice.from_splits(self.dataset.build(data), self.ratios)


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


@config_class()
class GPTLegacyDatasetConfig(GPTSampledSplitDatasetConfig, GPTLegacyConfig, type_="legacy"):
    _abstract = False

    def build_split_sample(
        self,
        data: "GPTData",
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

        return dataset_config.build_split_sample(data, config, default_phase)
