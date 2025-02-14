import typing

from fast_llm.config import Config, Field, FieldHint, FieldUpdate, config_class
from fast_llm.data.dataset.config import SamplingConfig, SamplingData


@config_class()
class SamplingDefaultConfig(SamplingConfig):
    seed: int = FieldUpdate(
        default=784569,
        desc="Seed for random sampling.",
        hint=FieldHint.feature,
    )


@config_class()
class DataConfig(Config):
    _abstract = True
    _sampling_config_class: typing.ClassVar[type[SamplingData]]

    sampling: SamplingConfig = Field(
        default_factory=SamplingConfig, desc="Default configuration for dataset sampling."
    )
