import typing

from fast_llm.config import Config, config_class

if typing.TYPE_CHECKING:
    from fast_llm.engine.checkpoint.external import ExternalStateDictConverter
    from fast_llm.engine.config_utils.tensor_space import TensorSpace


@config_class()
class BaseModelArchitectureConfig(Config):
    """
    Abstract config class for a base model architecture.
    # TODO: Find better name?
    """

    _abstract = True

    def setup_tensor_space(self, tensor_space: "TensorSpace"):
        raise NotImplementedError()

    def get_architecture(self):
        return self

    def compare_architecture(
        self,
        model_config: "BaseModelArchitectureConfig",
        log_fn: typing.Union[BaseException, typing.Callable] = ValueError,
    ):
        return self.get_architecture().compare(model_config.get_architecture(), log_fn)

    @classmethod
    def get_converter_class(cls, model_type: str | None = None) -> type["ExternalStateDictConverter"]:
        raise NotImplementedError()


@config_class()
class BaseModelConfig(BaseModelArchitectureConfig):
    """
    Abstract config class for a base model.
    # TODO: Find better name?
    """

    architecture_cls: typing.ClassVar[type[BaseModelArchitectureConfig]]

    def get_architecture(self):
        return self.architecture_cls.from_dict(self, strict=False)
