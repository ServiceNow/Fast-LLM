import itertools
import logging
import math
import typing

if typing.TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


def header(title: str | None = None, width: int = 60, fill_char: str = "-"):
    if title is None:
        return fill_char * width
    title_width = len(title) + 2
    left = (width - title_width) // 2
    right = width - left - title_width
    return fill_char * left + f" {title} " + fill_char * right


def get_type_name(type_):
    if isinstance(type_, type):
        module = type_.__module__
        return type_.__qualname__ if module == "builtins" else f"{module}.{type_.__qualname__}"
    # Happens for aliases, None and invalid types.
    return type_


def div(x, y):
    """
    Ensure that numerator is divisible by the denominator and return
    the division value.
    """
    if x % y != 0:
        raise ValueError(f"{x}%{y}!=0")
    return x // y


def get_unique(values: typing.Iterable):
    value = set(values)
    Assert.custom(lambda x: len(x) == 1, value)
    return value.pop()


def format_number(x, prec=4, exp_threshold=3):
    digits = 0 if x == 0 else math.log10(abs(x))
    if math.isfinite(digits) and -exp_threshold < math.floor(digits) < prec + exp_threshold:
        return f"{x:.{prec}f}"
    else:
        return f"{x:.{prec-1}e}"


def padded_cumsum(x):
    import numpy as np

    y = np.hstack((0, x))
    return y.cumsum(out=y)


def clamp(x, x_min, x_max):
    return min(max(x, x_min), x_max)


def rms_diff(x: "torch.Tensor", y: "torch.Tensor"):
    import torch

    return torch.norm(x - y, 2, dtype=torch.float32) / x.numel() ** 0.5  # noqa


class Tag:
    def __init__(self, value: str):
        self.value = value

    def __repr__(self):
        return self.value


class Assert:
    """
    A bunch of assertions that print relevant information on failure, packed into a namespace to simplify usage
    """

    @staticmethod
    def eq(x, *args):
        for arg in args:
            assert x == arg, f"{x} != {arg}"

    @staticmethod
    def is_(x, y):
        assert x is y, f"{x} is not {y}"

    @staticmethod
    def geq(x, y):
        assert x >= y, f"{x} not >= {y}"

    @staticmethod
    def leq(x, y):
        assert x <= y, f"{x} not <= {y}"

    @staticmethod
    def gt(x, y):
        assert x > y, f"{x} not > {y}"

    @staticmethod
    def lt(x, y):
        assert x < y, f"{x} not < {y}"

    @staticmethod
    def in_range(x, low, high):
        assert low <= x < high, f"x not in range({low}, {high})"

    @staticmethod
    def in_range_incl(x, low, high):
        assert low <= x <= high, f"{x} not in (inclusive) range({low}, {high})"

    @staticmethod
    def none(x):
        assert x is None, f"Object of type {type(x)} is not None ({str(x)})"

    @staticmethod
    def empty(x):
        assert len(x) == 0, f"Not empty (len={len(x)}), {x}"

    @staticmethod
    def incl(x, y):
        assert x in y, f"{x} not in {list(y)}"

    @staticmethod
    def not_incl(x, y):
        assert x not in y, f"{x} in {y}"

    @staticmethod
    def multiple(x, y):
        assert x % y == 0, f"{x} not a multiple of {y}"

    @staticmethod
    def rms_close(x, y, threshold):
        rms = rms_diff(x, y).item()
        assert rms <= threshold, f"Rms diff too big ({rms} > {threshold}) between tensors {x} and {y}"

    @staticmethod
    def all_equal(x, y):
        import torch

        neq = x != y
        if neq.any().item():  # noqa
            index = torch.where(neq)  # noqa
            raise AssertionError(
                f"Tensors have {index[0].numel()} different entries out of "
                f"{x.numel()}: {x[index]} != {y[index]} at index {torch.stack(index, -1)}"
            )

    @staticmethod
    def all_different(x, y):
        import torch

        eq = x == y
        if eq.any().item():  # noqa
            index = torch.where(eq)  # noqa
            raise AssertionError(
                f"Tensors have {index[0].numel()} unexpected matching entries out of "
                f"{x.numel()}: {x[index]} != {y[index]} at index {torch.stack(index, -1)}"
            )

    @staticmethod
    def custom(fn, *args, **kwargs):
        assert fn(
            *args, **kwargs
        ), f"Assertion failed: fn({', '.join(itertools.chain((str(x) for x in args),(f'{str(k)}={str(v)}' for k,v in kwargs.items())))})"

    @staticmethod
    def not_custom(fn, *args, **kwargs):
        assert not fn(
            *args, **kwargs
        ), f"Assertion failed: not fn({', '.join(itertools.chain((str(x) for x in args),(f'{str(k)}={str(v)}' for k,v in kwargs.items())))})"


_KeyType = typing.TypeVar("_KeyType")
_ValueType = typing.TypeVar("_ValueType")


class Registry(typing.Generic[_KeyType, _ValueType]):
    def __init__(self, name: str, data: dict[_KeyType, _ValueType]):
        self._name = name
        self._data = data.copy()

    def __getitem__(self, key):
        if key not in self:
            raise KeyError(f"Entry {key} not found in {self._name} registry")
        return self._data[key]

    def __setitem__(self, key, value):
        if key in self:
            raise KeyError(f"Entry {key} already in {self._name} registry")
        self._data[key] = value

    def keys(self):
        return list(self._data)

    def __contains__(self, item):
        return item in self._data


class LazyRegistry(Registry):
    def __getitem__(self, key):
        return super().__getitem__(key)()


def log(*message, log_fn: typing.Union[BaseException, typing.Callable] = logger.info, join: str = ", "):
    message = join.join([str(m() if callable(m) else m) for m in message])
    if isinstance(log_fn, BaseException):
        raise log_fn(message)
    else:
        return log_fn(message)


def normalize_probabilities(p: list[float]) -> list[float]:
    import numpy as np

    p = np.array(p)
    Assert.custom(lambda x: np.all(x >= 0), p)
    p_sum = p.sum()
    Assert.gt(p_sum, 0)
    return (p / p_sum).tolist()
