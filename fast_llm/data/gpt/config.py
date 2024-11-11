from fast_llm.config import Field, FieldHint, check_field, config_class
from fast_llm.data.config import (
    DataConfig,
    DatasetSource,
    FimConfig,
    MultiprocessingContext,
    TokenizerConfig,
    _validate_path,
    _validate_split,
)
from fast_llm.utils import Assert


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
