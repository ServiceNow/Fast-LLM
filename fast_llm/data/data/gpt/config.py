import logging
import typing

from fast_llm.config import Field, FieldHint, check_field, config_class
from fast_llm.data.config import MultiprocessingContext
from fast_llm.data.data.config import DataConfig
from fast_llm.data.dataset.config import SampledDatasetConfig, SamplingConfigBase
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    from fast_llm.data.document.language_model import LanguageModelDocument
logger = logging.getLogger(__name__)


@config_class()
class GPTDataConfig(DataConfig, SamplingConfigBase):
    """
    Configuration for the dataset(s) and its sampling.
    """

    _abstract = False

    datasets: dict[str, SampledDatasetConfig["LanguageModelDocument"]] = Field(
        default_factory=dict,
        desc="Configuration for the dataset(s).",
        hint=FieldHint.core,
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
    seed: int = Field(
        default=784569,
        desc="Seed for random sampling.",
        hint=FieldHint.feature,
    )
