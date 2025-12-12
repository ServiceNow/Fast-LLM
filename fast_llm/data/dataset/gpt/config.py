import dataclasses
import pathlib
import time
import typing

import yaml

from fast_llm.config import Config, Field, FieldHint, check_field, config_class, skip_valid_if_none
from fast_llm.data.dataset.abstract import SamplableDataset, SampledDataset
from fast_llm.data.dataset.config import SamplableDatasetConfig, SampledDatasetConfig, SamplingData
from fast_llm.data.preprocessing.abstract import PreprocessingConfig
from fast_llm.data.preprocessing.language_model import LanguageModelPreprocessingConfig
from fast_llm.data.preprocessing.tokenizer import TokenizerConfig
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    from fast_llm.data.dataset.gpt.fim import GPTFimDataset
    from fast_llm.data.dataset.gpt.random import GPTRandomSampledDataset
    from fast_llm.data.sample.language_model import LanguageModelSample


@dataclasses.dataclass(kw_only=True)
class GPTSamplingData(SamplingData):
    """
    Holds all the necessary information for sampling, including dataset-dependent ones (`GPTSamplingConfig`),
    usage-dependent ones (`GPTSamplingParameters`), and others set by the `Data`.
    """

    preprocessing: LanguageModelPreprocessingConfig


@config_class(dynamic_type={SampledDatasetConfig: "random"})
class GPTRandomDatasetConfig[SampleType: LanguageModelSample](SampledDatasetConfig[SampleType]):
    _abstract: typing.ClassVar[bool] = False
    name: str = Field(
        default="dummy",
        desc="The name of the dataset.",
        hint=FieldHint.core,
    )

    def build_and_sample(self, sampling: GPTSamplingData) -> "GPTRandomSampledDataset[SampleType]":
        from fast_llm.data.dataset.gpt.random import GPTRandomSampledDataset

        return GPTRandomSampledDataset[SampleType](sampling, self.name)


@config_class(dynamic_type={SampledDatasetConfig: "file"})
class GPTDatasetFromFileConfig[SampleType: LanguageModelSample](SamplableDatasetConfig[SampleType]):
    _abstract: typing.ClassVar[bool] = False
    path: pathlib.Path = Field(
        default=None,
        desc="The path to a dataset config file.",
        hint=FieldHint.core,
    )

    def build_and_sample(self, sampling: SamplingData) -> SampledDataset[SampleType]:
        config = self._load_config()
        return config.build_and_sample(sampling)

    def build(self, preprocessing: PreprocessingConfig) -> SamplableDataset[SampleType]:
        config = self._load_config()
        assert isinstance(config, SamplableDatasetConfig)
        return config.build(preprocessing)

    def _load_config(self) -> SampledDatasetConfig[SampleType]:
        assert self.path.is_file(), f"File {self.path} does not exist."
        config = yaml.safe_load(self.path.open("r"))
        Assert.eq(config.keys(), {"config", "metadata"})
        if config.keys() == {"config", "metadata"}:
            # Newer format with metadata
            config = config["config"]
        return SampledDatasetConfig[SampleType].from_dict(self._convert_paths(config))

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


@config_class()
class FimConfig(Config):
    """
    Configuration for FIM.
    """

    tokenizer: TokenizerConfig = Field(
        desc="Configuration for the tokenizer.",
        hint=FieldHint.feature,
    )
    rate: float = Field(
        # TODO: Use meaningful default now that fim is a wrapper?
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


@config_class(dynamic_type={SampledDatasetConfig: "fim"})
class GPTFimSampledDatasetConfig[SampleType: LanguageModelSample](SampledDatasetConfig[SampleType], FimConfig):
    """
    Configuration for FIM.
    """

    _abstract: typing.ClassVar[bool] = False

    dataset: SampledDatasetConfig[SampleType] = Field(
        default=None,
        desc="The dataset to wrap with fim.",
        hint=FieldHint.core,
    )

    def build_and_sample(
        self,
        sampling: GPTSamplingData,
    ) -> "GPTFimDataset[SampleType]":
        from fast_llm.data.dataset.gpt.fim import GPTFimDataset

        return GPTFimDataset[SampleType](self, self.dataset.build_and_sample(sampling), sampling)


@config_class(dynamic_type={SampledDatasetConfig: "test_slow"})
class GPTTestSlowDatasetConfig[SampleType: LanguageModelSample](SampledDatasetConfig[SampleType]):
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

    def build_and_sample(self, sampling: SamplingData) -> SampledDataset[SampleType]:
        assert sampling.distributed.config.world_size > 1
        if sampling.distributed.config.rank == 0:
            time.sleep(self.sleep)
        return GPTRandomDatasetConfig[SampleType]().build_and_sample(sampling)
