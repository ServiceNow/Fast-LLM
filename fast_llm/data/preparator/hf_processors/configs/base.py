import abc
import typing
import datasets

from fast_llm.config import TypeableConfig, config_class
from fast_llm.utils import Registry


class Applicable:
    @abc.abstractmethod
    def apply(self, dataset: datasets.Dataset) -> datasets.Dataset:
        raise NotImplementedError


@config_class()
class ShardProcessorConfig(TypeableConfig, Applicable):
    _registry: typing.ClassVar[Registry[str, type["ShardProcessorConfig"]] | None] = Registry(
        "ShardProcessorConfig", {}
    )
