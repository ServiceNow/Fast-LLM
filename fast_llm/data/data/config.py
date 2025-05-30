import typing

from fast_llm.config import Config, Field, config_class
from fast_llm.data.dataset.config import SamplingConfig, SamplingData


@config_class()
class DataConfig(Config):
    _abstract = True
    _sampling_config_class: typing.ClassVar[type[SamplingData]]

    sampling: SamplingConfig = Field(desc="Default configuration for dataset sampling.")
