import logging

from fast_llm.config import Field, FieldHint, check_field, config_class
from fast_llm.data.config import MultiprocessingContext, TokenizerConfig
from fast_llm.data.data.config import DataConfig
from fast_llm.data.dataset.gpt.config import GPTLegacyConfig, GPTLegacyDatasetConfig, GPTSampledSplitDatasetConfig
from fast_llm.data.dataset.gpt.fim.config import FimConfig
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)


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
    dataset: GPTSampledSplitDatasetConfig = Field(
        default=None,
        desc="Configuration for the dataset(s).",
        hint=FieldHint.core,
    )
    fim: FimConfig = Field(
        default_factory=FimConfig,
        desc="Configuration for Fill In the Middle (FIM).",
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

    def __post_init__(self):
        if self.dataset is None:
            logger.warning("Using the legacy dataset definition format." " Specify it through `data.dataset` instead.")
            self.dataset = GPTLegacyDatasetConfig(
                split=self.split,
                format=self.format,
                path=self.path,
            )
