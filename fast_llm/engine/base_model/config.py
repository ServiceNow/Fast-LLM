import abc
import typing

from fast_llm.config import Config, config_class

if typing.TYPE_CHECKING:
    from fast_llm.engine.config_utils.tensor_space import TensorSpace


@config_class()
class BaseModelArchitectureConfig(Config):
    """
    Abstract config class for a base model architecture.
    # TODO: Find better name?
    """

    _abstract = True

    def setup_tensor_space(self, tensor_space: "TensorSpace") -> None:
        raise NotImplementedError()

    def get_architecture[T: BaseModelArchitectureConfig](self: T) -> T:
        return self

    def compare_architecture(
        self,
        model_config: "BaseModelArchitectureConfig",
        log_fn: typing.Union[type[BaseException], typing.Callable] = ValueError,
    ):
        return self.get_architecture().compare(model_config.get_architecture(), log_fn)


@config_class()
class BaseModelConfig(BaseModelArchitectureConfig):
    """
    Abstract config class for a base model.
    # TODO: Find better name?
    """

    architecture_class: typing.ClassVar[type[BaseModelArchitectureConfig]]

    def get_architecture(self) -> BaseModelArchitectureConfig:
        return self.architecture_class.from_dict(self, strict=False)


class Preprocessor(abc.ABC):
    def preprocess_meta(self, kwargs: dict[str, typing.Any]) -> None:
        pass

    @abc.abstractmethod
    def preprocess(self, batch, kwargs: dict[str, typing.Any]) -> None:
        pass
