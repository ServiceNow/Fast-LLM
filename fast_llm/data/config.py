import enum

from fast_llm.config import Config, Field, FieldHint, check_field, config_class, skip_valid_if_none
from fast_llm.utils import Assert


class DatasetType(str, enum.Enum):
    """
    Placeholder for future generalization to other data types.
    """

    gpt = "gpt"


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
    if isinstance(value, str):
        # This happens with yaml serialization
        value = [float(x) for x in value.strip("[]").split(",")]
    Assert.leq(len(value), 3)
    return value + [0] * (len(value) - 3)


def _validate_path(value):
    if isinstance(value, str):
        # This happens with yaml serialization
        return [x.strip() for x in value.strip("[]").split(",")]
    elif value is None:
        return []
    else:
        return value


FIM_PREFIX = "<fim_prefix>"
FIM_MIDDLE = "<fim_middle>"
FIM_PAD = "<fim_pad>"
FIM_SUFFIX = "<fim_suffix>"


@config_class()
class FimConfig(Config):
    """
    Configuration for FIM.
    """

    fim_rate: float = Field(
        default=0.0,
        desc="FIM rate for each sample.",
        hint=FieldHint.core,
        valid=check_field(Assert.in_range_incl, 0, 1),
    )
    fim_max_middle_len: int | None = Field(
        default=None,
        desc="Maximum length of the middle segment in FIM.",
        hint=FieldHint.feature,
        valid=skip_valid_if_none(check_field(Assert.gt, 0)),
    )
    fim_split_sample: str | None = Field(
        default=None,
        desc="Split samples on this token and permute each fragment separately.",
        hint=FieldHint.feature,
    )
    fim_fragment_rate: float = Field(
        default=0.0,
        desc="FIM rate for each fragment when using fim_split_sample.",
        hint=FieldHint.feature,
        valid=check_field(Assert.in_range_incl, 0, 1),
    )
    fim_ignore_prefix: str | None = Field(
        default=None,
        desc="Do not apply FIM to fragments that start with this prefix.",
        hint=FieldHint.feature,
    )
    fim_spm_rate: float = Field(
        default=0.5,
        desc="TODO.",
        hint=FieldHint.feature,
        valid=check_field(Assert.in_range_incl, 0, 1),
    )
    fim_truncate_or_pad: bool = Field(
        default=False,
        desc="TODO.",
        hint=FieldHint.feature,
    )

    def _validate(self):
        super()._validate()
        Assert.in_range_incl(self.fim_rate, 0, 1)


EOD = "<|endoftext|>"
TokenizerFromFile = "TokenizerFromFile"


@config_class()
class TokenizerConfig(Config):
    """
    Configuration for the tokenizer.
    Currently, the tokenizer is only needed for FIM.
    """

    tokenizer_type: str = Field(
        default="TokenizerFromFile",
        desc="Unused.",
        hint=FieldHint.deprecated,
        valid=check_field(Assert.eq, TokenizerFromFile),
    )
    tokenizer_file: str | None = Field(
        default=None,
        desc="Path to the tokenizer file.",
        hint=FieldHint.core,
    )


@config_class()
class DataConfig(Config):
    """
    Configuration for the dataset(s), split and sampling.
    Currently hard-coded to a GPT dataset.
    TODO: Separate generic and GPT classes.
    """

    __argparse_map__ = {
        "data_path": {"nargs": "+"},
        "split": {"type": str, "default": [969, 30, 1]},
    }

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
    dataset_type: DatasetType = Field(
        default=DatasetType.gpt,
        desc="Unused.",
        hint=FieldHint.wip,
    )
    dataset_source: DatasetSource = Field(
        default=DatasetSource.list,
        desc="Format for the dataset definition.",
        hint=FieldHint.core,
    )
    data_path: list[str] = Field(
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
