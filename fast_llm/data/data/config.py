import pathlib
import typing

from fast_llm.config import Config, Field, config_class


@config_class
class SamplingConfig(Config):
    num_samples: int = Field(default=1, desc="Number of samples to generate.")
    seed: int = Field(default=0, desc="Random seed.")
    cache_directory: pathlib.Path | None = Field(default=None, desc="Path to the sampling cache directory.")
    verbose: bool = Field(default=True, desc="Log sampling progress.")


@config_class()
class DataConfig(Config):
    _abstract = True
    _sampling_config_class: typing.ClassVar[type[SamplingConfig]]
