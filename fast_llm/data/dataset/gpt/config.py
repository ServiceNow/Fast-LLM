import dataclasses
import enum
import json
import pathlib
import time
import typing
import warnings

import yaml

from fast_llm.config import Config, Field, FieldHint, FieldUpdate, check_field, config_class, skip_valid_if_none
from fast_llm.data.dataset.abstract import SamplableDataset, SampledDataset
from fast_llm.data.dataset.config import (
    BlendedDatasetConfig,
    ConcatenatedDatasetConfig,
    DatasetSliceConfig,
    IndexedDatasetConfig,
    SamplableDatasetConfig,
    SampledDatasetConfig,
    SampledDatasetUpdateConfig,
    SamplingConfig,
    SamplingData,
    SamplingParameters,
)
from fast_llm.engine.distributed.config import PhaseType
from fast_llm.utils import Assert, normalize_probabilities, padded_cumsum

if typing.TYPE_CHECKING:
    from fast_llm.data.dataset.gpt.indexed import GPTConcatenatedDataset, GPTDatasetSlice, GPTIndexedDataset
    from fast_llm.data.dataset.gpt.memmap import GPTMemmapDataset
    from fast_llm.data.dataset.gpt.random import GPTRandomDataset
    from fast_llm.data.tokenizer import Tokenizer


class ShufflingType(str, enum.Enum):
    # Shuffle all epochs together. Not extendable.
    full = "full"
    # Shuffle all epochs separately. Default mode, recommended if the dataset doesn't come pre-shuffled.
    epoch = "epoch"
    # Shuffle all epochs except the first one. Recommended for pre-shuffled datasets, especially big ones.
    skip_first_epoch = "skip_first_epoch"
    # Disable shuffling entirely.
    disabled = "disabled"
    legacy = "legacy"


@config_class()
class GPTSamplingConfig(SamplingConfig):
    """
    A dataset-dependent configuration for sampling.
    """

    gpu: bool = Field(
        default=True,
        desc="Enable fast sampling on GPU."
        " Note that random sampling works differently on GPU,"
        " so the sample won't match the CPU equivalent.",
        hint=FieldHint.feature,
    )
    shuffle: ShufflingType = Field(
        default=ShufflingType.epoch,
        desc="Shuffling strategy.",
        hint=FieldHint.feature,
    )


@dataclasses.dataclass(kw_only=True)
class GPTSamplingParameters(SamplingParameters):
    """
    Sampling parameters set externally to the dataset and data, ex. determined by the trainer or model.
    """

    sequence_length: int
    vocab_size: int
    use_loss_masking_spans: bool = False
    use_preference_loss_spans: bool = False
    cross_document_attention: bool = True
    # How many extra tokens to add to the sequence length.
    # This is used to provide labels even for the last tokens in the sequence.
    extra_tokens: int = 1


@dataclasses.dataclass(kw_only=True)
class GPTSamplingData(SamplingData):
    """
    Holds all the necessary information for sampling, including dataset-dependent ones (`GPTSamplingConfig`),
    usage-dependent ones (`GPTSamplingParameters`), and others set by the `Data`.
    """

    config: GPTSamplingConfig
    parameters: GPTSamplingParameters
    tokenizer: "Tokenizer"
    truncate_documents: bool = True


@config_class(registry=True)
class GPTSampledDatasetConfig(SampledDatasetConfig):
    pass


@config_class()
class GPTSamplableDatasetConfig(SamplableDatasetConfig, GPTSampledDatasetConfig):
    pass


@config_class()
class GPTIndexedDatasetConfig(GPTSamplableDatasetConfig, IndexedDatasetConfig):
    def build(self) -> "GPTIndexedDataset":
        raise NotImplementedError()


@config_class(dynamic_type={GPTSampledDatasetConfig: "random"})
class GPTRandomDatasetConfig(GPTSamplableDatasetConfig):
    _abstract: typing.ClassVar[bool] = False
    name: str = Field(
        default="dummy",
        desc="The name of the dataset.",
        hint=FieldHint.core,
    )

    def build(self) -> "GPTRandomDataset":
        from fast_llm.data.dataset.gpt.random import GPTRandomDataset

        return GPTRandomDataset(self.name)


@config_class(dynamic_type={GPTSampledDatasetConfig: "memmap"})
class GPTMemmapDatasetConfig(GPTIndexedDatasetConfig):
    _abstract: typing.ClassVar[bool] = False
    path: pathlib.Path = Field(
        default=None,
        desc="The path to the dataset, excluding the `.bin` or `.idx` suffix.",
        hint=FieldHint.core,
    )
    num_documents: int | None = Field(
        default=None,
        desc="Expected number of documents in the dataset.",
        hint=FieldHint.optional,
    )
    num_tokens: int | None = Field(
        default=None,
        desc="Expected number of tokens in the dataset.",
        hint=FieldHint.optional,
    )

    def build(self) -> "GPTMemmapDataset":
        from fast_llm.data.dataset.gpt.memmap import GPTMemmapDataset

        return GPTMemmapDataset(str(self.path).replace("/", "__"), self.path, self.num_documents, self.num_tokens)


@config_class(dynamic_type={GPTSampledDatasetConfig: "concatenated"})
class GPTConcatenatedDatasetConfig(ConcatenatedDatasetConfig, GPTIndexedDatasetConfig):
    _abstract: typing.ClassVar[bool] = False
    datasets: list[GPTIndexedDatasetConfig] = FieldUpdate()

    def build(self) -> "GPTConcatenatedDataset":
        from fast_llm.data.dataset.gpt.indexed import GPTConcatenatedDataset

        return self._build(GPTConcatenatedDataset)


@config_class(dynamic_type={GPTSampledDatasetConfig: "slice"})
class GPTDatasetSliceConfig(DatasetSliceConfig, GPTIndexedDatasetConfig):
    _abstract: typing.ClassVar[bool] = False
    dataset: GPTIndexedDatasetConfig = FieldUpdate()

    def build(self) -> "GPTDatasetSlice":
        from fast_llm.data.dataset.gpt.indexed import GPTDatasetSlice

        return self._build(GPTDatasetSlice)


@config_class(dynamic_type={GPTSampledDatasetConfig: "sampled"})
class GPTSampledDatasetUpdateConfig(SampledDatasetUpdateConfig, GPTSampledDatasetConfig):
    _abstract = False
    sampling: GPTSamplingConfig = FieldUpdate()
    dataset: GPTSampledDatasetConfig = FieldUpdate()


@config_class(dynamic_type={GPTSampledDatasetConfig: "blended"})
class GPTBlendedDatasetConfig(BlendedDatasetConfig, GPTSampledDatasetConfig):
    _abstract: typing.ClassVar[bool] = False
    datasets: list[GPTSampledDatasetConfig] = FieldUpdate()


@config_class(dynamic_type={GPTSampledDatasetConfig: "file"})
class GPTDatasetFromFileConfig(GPTSamplableDatasetConfig):
    _abstract: typing.ClassVar[bool] = False
    path: pathlib.Path = Field(
        default=None,
        desc="The path to a dataset config file.",
        hint=FieldHint.core,
    )

    def build_and_sample(self, sampling: SamplingData) -> SampledDataset:
        config = self._load_config()
        return config.build_and_sample(sampling)

    def build(self) -> SamplableDataset:
        config = self._load_config()
        assert isinstance(config, GPTSamplableDatasetConfig)
        return config.build()

    def _load_config(self):
        assert self.path.is_file(), f"File {self.path} does not exist."
        return GPTSampledDatasetConfig.from_dict(self._convert_paths(yaml.safe_load(self.path.open("r"))))

    def _convert_paths(self, config):
        # Recursively convert paths relative to `self.path.parent` to make them relative to cwd.
        # Assuming all path are in a field named "path"
        # TODO: Find a more generic way
        if isinstance(config, dict):
            for key, value in config.items():
                self._convert_paths(value)
            if "path" in config:
                assert isinstance(config["path"], (str, pathlib.Path))
                config["path"] = self.path.parent / config["path"]
        elif isinstance(config, list):
            for value in config:
                self._convert_paths(value)
        return config


# Add user-friendly names for the configs.
@config_class(dynamic_type={GPTSampledDatasetConfig: "concatenated_memmap"})
class GPTConcatenatedMemmapConfig(GPTIndexedDatasetConfig):
    # TODO v0.3: Remove.
    _abstract: typing.ClassVar[bool] = False
    path: pathlib.Path = Field(
        default=None,
        desc="The path to a dataset directory.",
        hint=FieldHint.core,
    )

    def _validate(self) -> None:
        warnings.warn("`concatenated_memmap` dataset is deprecated. Use `file` instead.", DeprecationWarning)
        super()._validate()

    def build(self) -> "GPTConcatenatedDataset":

        assert self.path.is_dir()
        index_path = self.path / "index.txt"

        if index_path.is_file():
            prefixes = [self.path / line.strip() for line in index_path.open("r").readlines()]
        else:
            warnings.warn(
                f"The dataset path {self.path} points to a directory."
                " The dataset will be indexed automatically, which may be unsafe."
                " We recommend using an index file instead."
            )
            prefixes = [
                path.with_suffix("")
                for path in self.path.iterdir()
                if path.suffix == ".idx" and path.is_file() and path.with_suffix(".bin").is_file()
            ]
        dataset_config = GPTConcatenatedDatasetConfig.from_dict(
            {"datasets": [{"type": "memmap", "path": prefix} for prefix in prefixes]}
        )
        return dataset_config.build()


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
    prefix_token: str = Field(
        default="<fim_prefix>",
        desc="TODO.",
        hint=FieldHint.feature,
    )
    middle_token: str = Field(
        default="<fim_middle>",
        desc="TODO.",
        hint=FieldHint.feature,
    )
    pad_token: str = Field(
        default="<fim_pad>",
        desc="TODO.",
        hint=FieldHint.feature,
    )
    suffix_token: str = Field(
        default="<fim_suffix>",
        desc="TODO.",
        hint=FieldHint.feature,
    )


@config_class(dynamic_type={GPTSampledDatasetConfig: "fim"})
class GPTFimSampledDatasetConfig(GPTSampledDatasetConfig, FimConfig):
    """
    Configuration for FIM.
    """

    _abstract: typing.ClassVar[bool] = False

    dataset: GPTSampledDatasetConfig = Field(
        default=None,
        desc="The dataset to wrap with fim.",
        hint=FieldHint.core,
    )

    def build_and_sample(
        self,
        sampling: GPTSamplingData,
    ) -> SampledDataset:
        from fast_llm.data.dataset.gpt.fim import GPTFimDataset

        return GPTFimDataset(self, self.dataset.build_and_sample(sampling), sampling)


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
        desc="Configuration for Fill In the Middle (FIM).",
        hint=FieldHint.feature,
    )


@config_class(dynamic_type={GPTSampledDatasetConfig: "legacy"})
class GPTLegacyDatasetConfig(GPTSampledDatasetConfig, GPTLegacyConfig):
    _abstract: typing.ClassVar[bool] = False

    def build_and_sample(self, sampling: GPTSamplingData) -> SampledDataset:

        if self.format == LegacyDatasetSource.random:
            Assert.eq(len(self.path), 0)
            dataset_config = GPTRandomDatasetConfig()
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

            phase_splits = padded_cumsum(normalize_probabilities(self.split))

            phase_index = {
                PhaseType.training.value.lower(): 0,
                PhaseType.validation.value.lower(): 1,
                PhaseType.test.value.lower(): 2,
            }[sampling.dataset_name]

            dataset_configs = [
                {
                    "type": "slice",
                    # TODO: this duplicates memmap datasets for each phase.
                    "dataset": {"type": "memmap", "path": prefix},
                    "begin": float(phase_splits[phase_index]),
                    "end": float(phase_splits[phase_index + 1]),
                }
                for prefix in dataset_prefixes
            ]
            dataset_config = (
                {
                    "type": "blended",
                    "name": "blended",
                    "datasets": dataset_configs,
                    "weights": dataset_weights,
                    "legacy": True,
                }
                if len(dataset_configs) > 1
                else dataset_configs[0]
            )
            if self.fim.rate > 0:
                dataset_config = {
                    "type": "fim",
                    "dataset": dataset_config,
                    **self.fim.to_dict(),
                }
        # Legacy sampling config
        dataset_config = {
            "type": "sampled",
            "dataset": dataset_config,
            "sampling": {
                "seed": sampling.distributed.config.seed,
                "shuffle": "legacy",
            },
        }

        return GPTSampledDatasetConfig.from_dict(dataset_config).build_and_sample(sampling)


@config_class(dynamic_type={GPTSampledDatasetConfig: "test_slow"})
class GPTTestSlowDatasetConfig(GPTSampledDatasetConfig):
    """
    A mock dataset that mimics a slow dataset creation on one rank, which may trigger a timeout.
    """

    # TODO: This belongs to a testing plugin.
    _abstract: typing.ClassVar[bool] = False
    sleep: float = Field(
        default=1,
        desc="Sleep time during build, in seconds.",
        hint=FieldHint.core,
    )

    def build_and_sample(self, sampling: SamplingData) -> SampledDataset:
        assert sampling.distributed.config.world_size > 1
        if sampling.distributed.config.rank == 0:
            time.sleep(self.sleep)
        return GPTRandomDatasetConfig().build_and_sample(sampling)
