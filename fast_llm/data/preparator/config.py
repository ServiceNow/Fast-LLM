import abc
import argparse
import typing

from fast_llm.config import config_class
from fast_llm.engine.config_utils.runnable import RunnableConfig
from fast_llm.utils import Assert


@config_class()
class DatasetPreparatorConfig(RunnableConfig):
    preparator_name: typing.ClassVar[str]

    @classmethod
    def get_dataset_preparator_class(cls) -> type["DatasetPreparator"]:
        raise NotImplementedError

    def _get_runnable(self, parsed: argparse.Namespace) -> typing.Callable[[], None]:
        dataset_preparator = self.get_dataset_preparator_class()(config=self)
        return dataset_preparator.run


class DatasetPreparator(abc.ABC):
    _config: DatasetPreparatorConfig
    config_class: typing.ClassVar[type[DatasetPreparatorConfig]] = DatasetPreparatorConfig

    def __init__(self, config: DatasetPreparatorConfig) -> None:
        Assert.custom(isinstance, config, self.config_class)
        config.validate()
        self._config = config

    @abc.abstractmethod
    def run(self) -> None:
        raise NotImplementedError
