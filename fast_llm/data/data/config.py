import typing

from fast_llm.config import Config, config_class
from fast_llm.data.dataset.config import SamplingConfig


@config_class()
class DataConfig(Config):
    _abstract = True
    _sampling_config_class: typing.ClassVar[type[SamplingConfig]]
