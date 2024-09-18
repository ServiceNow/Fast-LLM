import itertools
import logging
import math
import typing

import numpy as np
import torch

logger = logging.getLogger(__name__)


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
    y = np.hstack((0, x))
    return y.cumsum(out=y)


def clamp(x, x_min, x_max):
    return min(max(x, x_min), x_max)


def rms_diff(x: torch.Tensor, y: torch.Tensor):
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
        neq = x != y
        if neq.any().item():  # noqa
            index = torch.where(neq)  # noqa
            raise AssertionError(
                f"Tensors have {index[0].numel()} different entries out of "
                f"{x.numel()}: {x[index]} != {y[index]} at index {torch.stack(index, -1)}"
            )

    @staticmethod
    def all_different(x, y):
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


class Registry:
    def __init__(self, name, data: dict):
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
