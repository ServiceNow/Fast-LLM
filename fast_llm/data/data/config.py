import pathlib
import typing

from fast_llm.config import Config, Field, check_field, config_class
from fast_llm.utils import Assert


@config_class
class SamplingConfig(Config):
    num_samples: int = Field(default=1, desc="Number of samples to generate.", valid=check_field(Assert.gt, 0))
    seed: int = Field(default=0, desc="Random seed.")
    cache_directory: pathlib.Path | None = Field(default=None, desc="Path to the sampling cache directory.")
    verbose: bool = Field(default=True, desc="Log sampling progress.")


@config_class()
class DataConfig(Config):
    _abstract = True
    _sampling_config_class: typing.ClassVar[type[SamplingConfig]]
