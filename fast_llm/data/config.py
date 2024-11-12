import abc
import argparse
import enum
import os
import pathlib
import typing

from fast_llm.config import Config, Field, FieldHint, check_field, config_class, skip_valid_if_none
from fast_llm.engine.config_utils.runnable import RunnableConfig
from fast_llm.engine.distributed.config import PhaseType
from fast_llm.engine.schedule.config import BatchConfig
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    from fast_llm.engine.distributed.distributed import Distributed


class DatasetSource(str, enum.Enum):
    """
    An enum for the different ways to load datasets.
    TODO: Reduce the diversity?
    TODO: Is this specific to GPT data?
    """

    list = "list"
    file = "file"
    sample = "sample"
    random = "random"


class MultiprocessingContext(str, enum.Enum):
    # Fast but risk of segfaults due to interactions with triton
    # (for example https://github.com/openai/triton/issues/2088).
    fork = "fork"
    # Safe but much slower.
    spawn = "spawn"


def _validate_split(value):
    Assert.leq(len(value), 3)
    return value + [0] * (len(value) - 3)


def _validate_path(value):
    return [value] if isinstance(value, str) else value


FIM_PREFIX = "<fim_prefix>"
FIM_MIDDLE = "<fim_middle>"
FIM_PAD = "<fim_pad>"
FIM_SUFFIX = "<fim_suffix>"


@config_class()
class FimConfig(Config):
    """
    Configuration for FIM.
    """

    rate: float = Field(
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

    def _validate(self):
        super()._validate()
        Assert.in_range_incl(self.rate, 0, 1)


EOD = "<|endoftext|>"
TokenizerFromFile = "TokenizerFromFile"


@config_class()
class TokenizerConfig(Config):
    """
    Configuration for the tokenizer.
    The tokenizer is needed for FIM and dataset preparation.
    """

    format: str = Field(
        default="TokenizerFromFile",
        desc="Unused.",
        hint=FieldHint.deprecated,
        valid=check_field(Assert.eq, TokenizerFromFile),
    )
    path: str | None = Field(
        default=None,
        desc="Path to the tokenizer file.",
        hint=FieldHint.core,
    )


@config_class()
class DataConfig(Config):
    _abstract = True


class Data(abc.ABC):
    # TODO: Improve interface
    @abc.abstractmethod
    def setup(self, distributed: "Distributed", samples_per_phase: dict[PhaseType, int]):
        pass

    @abc.abstractmethod
    def get_iterator(
        self,
        batch_config: BatchConfig,
        phase: PhaseType,
        *,
        consumed_samples: int,
        num_workers: int,
        prefetch_factor: int | None = None,
    ):
        pass


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


@config_class
class SamplingConfig(Config):
    num_samples: int = Field(default=1, desc="Number of samples to generate.")
    seed: int = Field(default=0, desc="Random seed.")
    cache_directory: pathlib.Path | None = Field(default=None, desc="Path to the sampling cache directory.")
    verbose: bool = Field(default=True, desc="Log sampling progress.")


class SamplableDataset(Dataset):
    def sample(self, config: SamplingConfig, data: Data):
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


@config_class
class _DistributedConfig(Config):
    # TODO: Unify with fast_llm.engine.distributed.config.DistributedConfig

    default_world_size: typing.ClassVar[int] = int(os.environ.get("WORLD_SIZE", 1))
    default_rank: typing.ClassVar[int] = int(os.environ.get("RANK", 0))
    world_size: int = Field(
        default=None,
        desc="Size of the world group. Typically provided by torchrun or equivalent through the `WORLD_SIZE` environment variable.",
        hint=FieldHint.expert,
        valid=check_field(Assert.gt, 0),
    )
    rank: int = Field(
        default=None,
        desc="Rank of the local process. Typically provided by torchrun or equivalent through the `RANK` environment variable.",
        hint=FieldHint.expert,
        valid=check_field(Assert.geq, 0),
    )
    backend: str = Field(
        default="gloo",
        desc="Distributed backend to use.",
        hint=FieldHint.optional,
    )

    def _validate(self):
        if self.world_size is None:
            self.world_size = self.default_world_size
        if self.rank is None:
            self.rank = self.default_rank
        super()._validate()
        Assert.in_range(self.rank, 0, self.world_size)


@config_class()
class DatasetPreparatorConfig(RunnableConfig):
    preparator_name: typing.ClassVar[str]

    output_path: pathlib.Path = Field(
        desc="Output directory for the processed dataset.",
        hint=FieldHint.core,
    )
    distributed: _DistributedConfig = Field(
        default_factory=_DistributedConfig,
        desc="Configuration for distributed processing.",
        hint=FieldHint.feature,
    )

    @classmethod
    def get_dataset_preparator_class(cls) -> typing.Type["DatasetPreparator"]:
        raise NotImplementedError

    def _get_runnable(self, parsed: argparse.Namespace) -> typing.Callable[[], None]:
        dataset_preparator = self.get_dataset_preparator_class()(config=self)
        return dataset_preparator.run


class DatasetPreparator(abc.ABC):
    _config: DatasetPreparatorConfig
    config_class: typing.ClassVar[type[DatasetPreparatorConfig]] = DatasetPreparatorConfig

    def __init__(self, config: DatasetPreparatorConfig) -> None:
        Assert.custom(isinstance, config, self.config_class)
        config.validate()
        self._config = config

    @abc.abstractmethod
    def run(self) -> None:
        raise NotImplementedError
