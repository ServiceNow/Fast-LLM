from fast_llm.config import Field, FieldHint, check_field, config_class
from fast_llm.data.config import MultiprocessingContext, TokenizerConfig
from fast_llm.data.data.config import DataConfig
from fast_llm.data.dataset.gpt.config import GPTLegacyConfig
from fast_llm.utils import Assert


@config_class()
class GPTDataConfig(DataConfig, GPTLegacyConfig):
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
