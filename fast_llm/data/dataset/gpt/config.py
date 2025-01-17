import dataclasses
import enum
import json
import pathlib
import typing

from fast_llm.config import Config, Field, FieldHint, FieldUpdate, check_field, config_class, skip_valid_if_none
from fast_llm.data.dataset.abstract import SampledDataset
from fast_llm.data.dataset.config import (
    BlendedDatasetConfig,
    ConcatenatedDatasetConfig,
    DatasetSliceConfig,
    IndexedDatasetConfig,
    SamplableDatasetConfig,
    SampledDatasetConfig,
    SamplingConfig,
)
from fast_llm.engine.distributed.config import PhaseType
from fast_llm.utils import Assert, Registry, normalize_probabilities, padded_cumsum

if typing.TYPE_CHECKING:
    from fast_llm.data.dataset.gpt.dummy import GPTDummyDataset
    from fast_llm.data.dataset.gpt.indexed import GPTConcatenatedDataset, GPTDatasetSlice, GPTIndexedDataset
    from fast_llm.data.dataset.gpt.memmap import GPTMemmapDataset
    from fast_llm.data.tokenizer import Tokenizer


@dataclasses.dataclass
class GPTSamplingConfig(SamplingConfig):
    # TODO: Sort these out
    sequence_length: int
    vocab_size: int
    tokenizer: "Tokenizer"


@config_class()
class GPTSampledDatasetConfig(SampledDatasetConfig):

    # TODO: Generalize dynamic types?
    _registry: typing.ClassVar[Registry[str, type["GPTSampledDatasetConfig"]]] = Registry[
        str, type["GPTDatasetConfig"]
    ]("gpt_dataset_class", {})
    type_: typing.ClassVar[type["GPTSampledDatasetConfig"] | None] = None
    type: str | None = Field(
        default=None,
        desc="The type of dataset.",
        hint=FieldHint.core,
    )

    def _validate(self) -> None:
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
    ) -> typing.Self:
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

    def __init_subclass__(cls, type_: str | None = None, **kwargs) -> None:
        if type_ is not None:
            GPTSampledDatasetConfig._registry[type_] = cls
        cls.type_ = type_
        super().__init_subclass__()


@config_class()
class GPTSamplableDatasetConfig(SamplableDatasetConfig, GPTSampledDatasetConfig):
    pass


@config_class()
class GPTIndexedDatasetConfig(GPTSamplableDatasetConfig, IndexedDatasetConfig):
    def build(self) -> "GPTIndexedDataset":
        raise NotImplementedError()


@config_class()
class GPTDummyDatasetConfig(GPTSamplableDatasetConfig, type_="dummy"):
    _abstract = False
    name: str = Field(
        default="dummy",
        desc="The name of the dataset.",
        hint=FieldHint.core,
    )

    def build(self) -> "GPTDummyDataset":
        from fast_llm.data.dataset.gpt.dummy import GPTDummyDataset

        return GPTDummyDataset(self.name)


@config_class()
class GPTMemmapDatasetConfig(GPTIndexedDatasetConfig, type_="memmap"):
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
class GPTConcatenatedDatasetConfig(ConcatenatedDatasetConfig, GPTIndexedDatasetConfig, type_="concatenated"):
    _abstract = False
    datasets: list[GPTIndexedDatasetConfig] = FieldUpdate()

    def build(self) -> "GPTConcatenatedDataset":
        from fast_llm.data.dataset.gpt.indexed import GPTConcatenatedDataset

        return self._build(GPTConcatenatedDataset)


@config_class()
class GPTDatasetSliceConfig(DatasetSliceConfig, GPTIndexedDatasetConfig, type_="slice"):
    _abstract = False
    dataset: GPTIndexedDatasetConfig = FieldUpdate()

    def build(self) -> "GPTDatasetSlice":
        from fast_llm.data.dataset.gpt.indexed import GPTDatasetSlice

        return self._build(GPTDatasetSlice)


@config_class()
class GPTBlendedDatasetConfig(BlendedDatasetConfig, GPTSampledDatasetConfig, type_="blended"):
    _abstract = False
    datasets: list[GPTSampledDatasetConfig] = FieldUpdate()


@config_class()
class FimConfig(Config):
    """
    Configuration for FIM.
    """

    rate: float = Field(
        # TODO: Use meaningful default now that fim is a wrapper? (bad for legacy config)
        default=0.0,
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

    def build_and_sample(
        self,
        config: GPTSamplingConfig,
    ) -> SampledDataset:
        from fast_llm.data.dataset.gpt.fim import FimDataset

        return FimDataset(self, self.dataset.build_and_sample(config), config)


class LegacyDatasetSource(str, enum.Enum):
    """
    An enum for the different ways to load datasets.
    """

    list = "list"
    file = "file"
    random = "random"


def _validate_split(value: list[int]) -> list[int]:
    Assert.leq(len(value), 3)
    return value + [0] * (len(value) - 3)


def _validate_path(value: str | list[str]) -> list[str]:
    return [value] if isinstance(value, str) else value


@config_class()
class GPTLegacyConfig(Config):
    split: list[float] = Field(
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
class GPTLegacyDatasetConfig(GPTSampledDatasetConfig, GPTLegacyConfig, type_="legacy"):
    _abstract = False

    def build_and_sample(self, config: GPTSamplingConfig) -> SampledDataset:

        if self.format == LegacyDatasetSource.random:
            Assert.eq(len(self.path), 0)
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

            phase_splits = padded_cumsum(self.ratio)
            phase_index = {
                PhaseType.training: 0,
                PhaseType.validation: 1,
                PhaseType.test: 2,
            }[config.phase]

            dataset_configs = [
                GPTDatasetSliceConfig(
                    # TODO: this duplicates memmap datasets for each phase.
                    dataset=GPTMemmapDatasetConfig(path=prefix),
                    begin=phase_splits[phase_index],
                    end=phase_splits[phase_index + 1],
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
            if self.fim.rate > 0:
                dataset_config = FimSampledDatasetConfig.from_dict(
                    self.fim,
                    {"dataset": dataset_config},
                )

        return dataset_config.build_and_sample(config)
