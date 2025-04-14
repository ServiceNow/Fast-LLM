import itertools
import logging
import math
import typing
from typing import Callable

if typing.TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch

logger = logging.getLogger(__name__)


def header(title: str | None = None, width: int = 60, fill_char: str = "-") -> str:
    if title is None:
        return fill_char * width
    title_width = len(title) + 2
    left = (width - title_width) // 2
    right = width - left - title_width
    return fill_char * left + f" {title} " + fill_char * right


def get_type_name(type_: typing.Any) -> str:
    if isinstance(type_, type):
        module = type_.__module__
        return type_.__qualname__ if module == "builtins" else f"{module}.{type_.__qualname__}"
    # Happens for aliases, None and invalid types.
    return type_


def div[T](x: T, y: T) -> T:
    """
    Ensure that numerator is divisible by the denominator and return
    the division value.
    """
    if x % y != 0:
        raise ValueError(f"{x}%{y}!=0")
    return x // y


def get_unique[T](values: typing.Iterable[T]) -> T:
    value = set(values)
    Assert.custom(lambda x: len(x) == 1, value)
    return value.pop()


def format_number(x: float | int, prec=4, exp_threshold=3) -> str:
    digits = 0 if x == 0 else math.log10(abs(x))
    if math.isfinite(digits) and -exp_threshold < math.floor(digits) < prec + exp_threshold:
        return f"{x:.{prec}f}"
    else:
        return f"{x:.{prec-1}e}"


def padded_cumsum(x: "npt.ArrayLike") -> "np.ndarray":
    import numpy as np

    y = np.hstack((0, x))
    return y.cumsum(out=y)


def clamp[T](x: T, x_min: T, x_max: T) -> T:
    return min(max(x, x_min), x_max)


def rms_diff(x: "torch.Tensor", y: "torch.Tensor") -> "torch.Tensor":
    import torch

    return torch.norm(x - y, 2, dtype=torch.float32) / x.numel() ** 0.5  # noqa


class Tag:
    __slots__ = ("value",)

    def __init__(self, value: str):
        self.value = value

    def __repr__(self) -> str:
        return self.value

    def __deepcopy__(self, memodict: dict[str, typing.Any]) -> typing.Self:
        return self


class Assert:
    """
    A bunch of assertions that print relevant information on failure, packed into a namespace to simplify usage
    """

    @staticmethod
    def eq(x, *args, msg=None):
        for arg in args:
            assert x == arg, f"{x} != {arg} " + (f"| {msg}" if msg else "")

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

        # Make it work for lists and numpy arrays.
        x = torch.as_tensor(x)
        y = torch.as_tensor(y)

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

        # Make it work for numpy arrays.
        x = torch.as_tensor(x)
        y = torch.as_tensor(y)

        eq = x == y
        if eq.any().item():  # noqa
            index = torch.where(torch.as_tensor(eq))  # noqa
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


class Registry[KeyType, ValueType]:
    # TODO: Inherit from dict instead?
    def __init__(self, name: str, data: dict[KeyType, ValueType]):
        self._name = name
        self._data = data.copy()

    def __getitem__(self, key: KeyType) -> ValueType:
        if key not in self:
            raise KeyError(f"Entry {key} not found in {self._name} registry")
        return self._data[key]

    def __setitem__(self, key: KeyType, value: ValueType):
        if key in self:
            raise KeyError(f"Entry {key} already in {self._name} registry")
        self._data[key] = value

    def keys(self) -> list[KeyType]:
        return list(self._data)

    def __contains__(self, key: KeyType) -> bool:
        return key in self._data

    def __iter__(self) -> typing.Iterator[KeyType]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def items(self):
        return self._data.items()

    @property
    def name(self) -> str:
        return self._name


class LazyRegistry[KeyType, ValueType](Registry[KeyType, ValueType]):
    def __getitem__(self, key: KeyType) -> ValueType:
        return super().__getitem__(key)()


def log[
    T
](*message: typing.Any, log_fn: type[BaseException] | typing.Callable[[str], T] = logger.info, join: str = ", ") -> T:
    message = join.join([str(m() if callable(m) else m) for m in message])
    logged = log_fn(message)
    if isinstance(logged, BaseException):
        raise logged
    else:
        return logged


def normalize_probabilities(p: "npt.ArrayLike", return_array: bool = False) -> "list[float] | np.ndarray":
    import numpy as np

    p = np.array(p)
    Assert.custom(lambda x: np.all(x >= 0), p)
    p_sum = p.sum()
    Assert.gt(p_sum, 0)
    out = p / p_sum
    return out if return_array else out.tolist()


class InvalidObject:
    """
    Store an error and raise it if accessed.
    Intended for missing optional imports, so that the actual import error is raised on access.
    """

    def __init__(self, error: Exception):
        self._error = error.__class__(*error.args)

    def __getattr__(self, item):
        raise self._error

    def __getitem__(self, item):

        raise self._error

    def __setitem__(self, key, value):
        raise self._error

    def __call__(self, *args, **kwargs):
        raise self._error


def try_decorate(get_decorator: Callable, _return_decorator: bool = True) -> Callable:
    """
    Try to decorate an object, but ignore the error until the object is actualy used.
    The wrapped decorator should always be instantiated before calling,
    i.e.. called as `@decorator()` rather than `@decorator`.
    """

    def new_decorator(*args, **kwargs):
        try:
            out = get_decorator()(*args, **kwargs)
        except Exception as e:
            out = InvalidObject(e)
        if _return_decorator:
            return try_decorate(lambda: out, _return_decorator=False)
        return out

    return new_decorator


def compare_nested(config_a, config_b, errors: list | None = None, prefix: tuple = ()):
    if errors is None:
        errors = []
    # Check for equality of both values and types.
    if type(config_a) != type(config_b):
        errors.append(f"Type mismatch for key `{".".join(prefix)}`: {type(config_a)} != {type(config_b)}")
    if isinstance(config_a, dict):
        for key in config_a.keys() | config_b.keys():
            key_ = prefix + (key,)
            if key not in config_a:
                errors.append(f"Key `{".".join(key_)}` missing in lhs.")
            elif key not in config_b:
                errors.append(f"Key `{".".join(key_)}` missing in rhs.")
            else:
                compare_nested(config_a[key], config_b[key], errors, key_)
    elif isinstance(config_a, (list, tuple, set)):
        if len(config_a) != len(config_b):
            errors.append(f"Length mismatch for key `{".".join(prefix)}`: {len(config_a)} != {len(config_b)}.")
        else:
            for i in range(len(config_a)):
                compare_nested(config_a[i], config_b[i], errors, prefix + (str(i),))
    elif config_a != config_b and config_a is not config_b:
        # `is not` needed for special cases like `math.nan`
        errors.append(f"Different value for key `{".".join(prefix)}`: {config_a} != {config_b}.")
    return errors


def check_equal_nested(config_a, config_b):
    if errors := compare_nested(config_a, config_b):
        raise ValueError("\n".join(errors))
