import datasets

from fast_llm.data.preparator.hf_processors.configs.base import Applicable, ShardProcessorConfig
from fast_llm.config import Field, Config, config_class

from fast_llm.data.preparator.hf_processors.configs.doc_length_filter import DocLengthFilterConfig

def default_processors():
    """Default processors to apply"""
    return [DocLengthFilterConfig()]


@config_class
class AgregatorConfig(Config, Applicable):
    steps: list[ShardProcessorConfig] = Field(default_factory=default_processors)

    def apply(self, dataset: datasets.Dataset) -> datasets.Dataset:
        from fast_llm.data.preparator.hf_processors.implementations.agregator import apply
        return apply(self, dataset)

    
