import abc
import typing
import datasets

from fast_llm.data.preparator.hf_processors.configs.base import Applicable, ShardProcessorConfig
from fast_llm.config import Field, config_class


@config_class
class DocLengthFilterConfig(ShardProcessorConfig):
    _abstract: typing.ClassVar[bool] = False
    type_: typing.ClassVar[str | None] = "length_filter"

    field: str = Field(default='text')
    min_length_chars: int = Field(default=0)
    max_length_chars: int = Field(default=1_000_000) 

    def apply(self, dataset: datasets.Dataset) -> datasets.Dataset:
        from fast_llm.data.preparator.hf_processors.implementations.doc_length_filter import apply
        return apply(self, dataset)


