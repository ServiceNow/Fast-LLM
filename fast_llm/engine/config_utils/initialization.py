import abc
import typing

if typing.TYPE_CHECKING:
    import torch

    from fast_llm.tensor import ParameterMeta


class Initializer(abc.ABC):
    @abc.abstractmethod
    def __call__(self, meta: "ParameterMeta", tensor: "torch.Tensor", generator: "torch.Generator") -> None:
        pass

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
