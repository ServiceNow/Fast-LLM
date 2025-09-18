import abc
import typing

from fast_llm.config import Config, Field, FieldHint, check_field, config_class
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    import torch

    from fast_llm.tensor import ParameterMeta


class Initialization(abc.ABC):
    """
    A common base class for initializations and initialization configs so both may be used interchangeably.
    """

    @abc.abstractmethod
    def get_initializer(self) -> "Initializer":
        pass


@config_class(registry=True)
class InitializationConfig(Config, Initialization):
    _abstract = True
    is_default: typing.ClassVar[bool] = False

    @classmethod
    def _from_dict(cls, default: dict[str, typing.Any], strict: bool = True) -> typing.Self:
        if cls is InitializationConfig and cls.get_subclass(default.get("type")) is None:
            # Default subclass.
            return DefaultInitializationConfig._from_dict(default, strict)
        return super()._from_dict(default, strict=strict)


@config_class()
class DefaultInitializationConfig(InitializationConfig):
    # A placeholder indicating that the class default should be used instead.
    _abstract = False
    is_default = True

    def get_initializer(self) -> "Initializer":
        raise NotImplementedError()


@config_class(dynamic_type={InitializationConfig: "fill"})
class FillInitializationConfig(InitializationConfig):
    """
    Normal initialization: normal(mean, std).clamp(min,max)
    """

    _abstract = False

    value: float = Field(
        default=1,
        desc="Initialization value.",
        hint=FieldHint.optional,
        valid=check_field(Assert.geq, 0),
    )

    def get_initializer(self) -> "Initializer":
        return init_fill_(self.value)


@config_class(dynamic_type={InitializationConfig: "normal"})
class NormalInitializationConfig(InitializationConfig):
    """
    Normal initialization: normal(mean, std).clamp(min,max)
    """

    _abstract = False

    std: float = Field(
        default=1,
        desc="Standard deviation for normal initialization.",
        hint=FieldHint.optional,
        valid=check_field(Assert.geq, 0),
    )
    mean: float = Field(
        default=0,
        desc="Mean for normal initialization.",
        hint=FieldHint.optional,
    )
    min: float | None = Field(
        default=None,
        desc="Min value for initialization clamping.",
        hint=FieldHint.optional,
    )
    max: float | None = Field(
        default=None,
        desc="Min value for initialization clamping.",
        hint=FieldHint.optional,
    )

    def get_initializer(self) -> "Initializer":
        return init_normal_(self.mean, self.std, self.min, self.max)


@config_class(dynamic_type={InitializationConfig: "uniform"})
class UniformInitializationConfig(InitializationConfig):
    """
    Uniform initialization: uniform(mean - scale, mean + scale).clamp(min,max)
    """

    _abstract = False

    scale: float = Field(
        default=None,
        desc="Initialization scale.",
        hint=FieldHint.optional,
        valid=check_field(Assert.geq, 0),
    )
    mean: float = Field(
        default=None,
        desc="Initialization mean.",
        hint=FieldHint.optional,
        valid=check_field(Assert.geq, 0),
    )

    def get_initializer(self) -> "Initializer":
        return init_uniform_centered_(self.scale, self.mean)


class Initializer(Initialization):
    @abc.abstractmethod
    def __call__(self, meta: "ParameterMeta", tensor: "torch.Tensor", generator: "torch.Generator") -> None:
        pass

    def get_initializer(self) -> "Initializer":
        return self

    requires_global_initialization = False


class LambdaInitializer(Initializer):
    def __init__(
        self,
        init_method: typing.Callable[["ParameterMeta", "torch.Tensor", "torch.Generator"], None],
        requires_global_initialization: bool = False,
    ) -> None:
        self._init_method = init_method
        self.requires_global_initialization = requires_global_initialization

    def __call__(self, meta: "ParameterMeta", tensor: "torch.Tensor", generator: "torch.Generator") -> None:
        return self._init_method(meta, tensor, generator)


def init_fill_(value: float) -> LambdaInitializer:
    def init_(meta: "ParameterMeta", tensor: "torch.Tensor", generator: "torch.Generator") -> None:  # noqa
        tensor.fill_(value)

    return LambdaInitializer(init_)


init_zeros_ = init_fill_(0.0)
init_ones_ = init_fill_(1.0)


def init_normal_(
    mean: float = 0.0, std: float = 1.0, min_val: float | None = None, max_val: float | None = None
) -> LambdaInitializer:
    def init_(meta: "ParameterMeta", tensor: "torch.Tensor", generator: "torch.Generator") -> None:  # noqa
        tensor = tensor.normal_(mean, std, generator=generator)
        if min_val is not None or max_val is not None:
            tensor.clamp_(min=min_val, max=max_val)

    return LambdaInitializer(init_)


def init_uniform_centered_(scale: float, mean: float = 0.0) -> LambdaInitializer:
    def init_(meta: "ParameterMeta", tensor: "torch.Tensor", generator: "torch.Generator") -> None:  # noqa
        tensor.uniform_(mean - scale, mean + scale, generator=generator)

    return LambdaInitializer(init_)
