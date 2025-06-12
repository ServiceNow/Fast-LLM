import abc
import typing

from fast_llm.config import Configurable, config_class
from fast_llm.engine.config_utils.runnable import RunnableConfig


@config_class(registry=True, dynamic_type={RunnableConfig: "prepare"})
class DatasetPreparatorConfig(RunnableConfig):
    preparator_name: typing.ClassVar[str]

    @classmethod
    def get_dataset_preparator_class(cls) -> type["DatasetPreparator"]:
        raise NotImplementedError

    def _get_runnable(self) -> typing.Callable[[], None]:
        dataset_preparator = self.get_dataset_preparator_class()(config=self)
        return dataset_preparator.run


class DatasetPreparator[ConfigType: DatasetPreparatorConfig](Configurable[ConfigType], abc.ABC):
    config_class: typing.ClassVar[type[DatasetPreparatorConfig]] = DatasetPreparatorConfig

    @abc.abstractmethod
    def run(self) -> None:
        raise NotImplementedError
