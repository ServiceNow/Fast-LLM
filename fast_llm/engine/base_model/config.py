import abc
import dataclasses
import typing

from fast_llm.config import MISSING, Config, Field, FieldHint, FieldVerboseLevel, config_class
from fast_llm.utils import compare_nested, log

if typing.TYPE_CHECKING:
    import torch


@config_class()
class BaseModelConfig(Config):
    """
    Abstract config class for a base model.
    # TODO: Find better name?
    """

    _abstract = True

    def compare_architecture(
        self,
        model_config: typing.Self,
        log_fn: typing.Union[type[BaseException], typing.Callable] = ValueError,
    ):
        errors = compare_nested(self._get_architecture(), model_config._get_architecture())
        if errors:
            return log(
                f"Config comparison errors:\n  " + "\n".join(errors),
                log_fn=log_fn,
            )
        return None

    def _get_architecture(self) -> dict[str, typing.Any]:
        architecture = {}
        for name, field in self.fields():
            if not field.init or field._field_type == dataclasses._FIELD_CLASSVAR:
                continue
            assert isinstance(field, Field), f"{name}, {field}"
            if field.hint == FieldHint.architecture:
                architecture[name] = self._serialize_architecture_field(getattr(self, name, MISSING))
        return architecture

    def _serialize_architecture_field(self, value: typing.Any) -> typing.Any:
        if isinstance(value, BaseModelConfig):
            # TODO: Make sure all nested configs have an architecture type hint?
            return value._get_architecture()
        elif isinstance(value, Config):
            # TODO: Explicitly prevent this case?
            return value.to_dict(verbose=FieldVerboseLevel.everything)
        elif isinstance(value, (list, tuple, set)):
            return [self._serialize_architecture_field(value_) for value_ in value]
        elif isinstance(value, dict):
            return {self._serialize_architecture_field(value_) for name, value_ in value.items()}
        else:
            return self._serialize_value(value)


class Preprocessor(abc.ABC):
    def preprocess_meta(self, kwargs: dict[str, typing.Any]) -> None:
        pass

    @abc.abstractmethod
    def preprocess(self, batch: "torch.Tensor", kwargs: dict[str, typing.Any]) -> None:
        pass


@dataclasses.dataclass
class ResourceUsageConfig:
    # Disable to get usage for current GPU only
    global_: bool = True
    # Enable to get hardware compute, i.e. include redundant computations.
    hardware: bool = False
    # Number of backward passes. Typically 1, may be 2 with full activation recomputation.
    forward: int = 1
    # Number of backward passes. Typically 1 for training, 0 for inference.
    backward: int = 1
