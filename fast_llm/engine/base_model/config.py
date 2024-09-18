import typing

from fast_llm.config import Config, config_class
from fast_llm.tensor import TensorSpace

if typing.TYPE_CHECKING:
    from fast_llm.engine.multi_stage.conversion import ModelConverter


@config_class()
class BaseModelArchitectureConfig(Config):
    """
    Abstract config class for a base model architecture.
    # TODO: Find better name?
    """

    _abstract = True

    def setup_tensor_space(self, tensor_space: TensorSpace):
        raise NotImplementedError()

    def get_architecture(self):
        return self

    @classmethod
    def get_converter_class(cls, model_type: str | None = None) -> type["ModelConverter"]:
        raise NotImplementedError()


@config_class()
class BaseModelConfig(BaseModelArchitectureConfig):
    """
    Abstract config class for a base model.
    # TODO: Find better name?
    """

    architecture_cls: typing.ClassVar[type[BaseModelArchitectureConfig]]

    def get_architecture(self):
        return self.architecture_cls.from_other(self, strict=False, strict_cls=True)
