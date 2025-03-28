import math
import pathlib

import numpy
import pytest

from fast_llm.config import FieldVerboseLevel, ValidationError
from fast_llm.utils import Assert
from tests.config.common import TestConfig, TestEnum


def check_config(internal_config, *alternate, serialized_config=None):
    serialized_config = serialized_config if serialized_config else alternate[0] if alternate else internal_config
    for init_config in (internal_config, *alternate):
        config = TestConfig.from_dict(init_config)
        Assert.eq(config.to_serialized(), serialized_config)
        Assert.eq(config._to_dict(), internal_config)


def check_invalid_config(config):
    with pytest.raises(ValidationError):
        TestConfig.from_dict(config)


def test_create_and_serialize_config():
    Assert.eq(TestConfig.from_dict({}).to_serialized(), {})


@pytest.mark.parametrize("value", (0, -6, 3, True))
def test_int_field(value):
    check_config({"int_field": value})


@pytest.mark.parametrize("value", (4.0, math.inf, "1", None, [4]))
def test_int_field_invalid(value):
    check_invalid_config({"int_field": value})


@pytest.mark.parametrize("value", (True, False))
def test_bool_field(value):
    check_config({"bool_field": value})


@pytest.mark.parametrize("value", (1, "True", None, [True]))
def test_bool_field_invalid(value):
    check_invalid_config({"bool_field": value})


@pytest.mark.parametrize("value", ("", "text", "1", TestEnum.a))
def test_str_field(value):
    check_config({"str_field": value})


@pytest.mark.parametrize("value", (1, True, None, ["text"], pathlib.Path("a")))
def test_str_field_invalid(value):
    check_invalid_config({"str_field": value})


@pytest.mark.parametrize("value", (".", "text", "/a/b/c.d"))
def test_path_field(value):
    check_config({"path_field": pathlib.Path(value)}, {"path_field": value})


@pytest.mark.parametrize("value", (1, True, None, [pathlib.Path("a")]))
def test_path_field_invalid(value):
    check_invalid_config({"path_field": value})


@pytest.mark.parametrize("value", (4.0, math.pi, math.inf, 3, True, numpy.float64(3), math.nan))
def test_float_field(value):
    check_config({"float_field": value})


@pytest.mark.parametrize("value", (None, [4.7], "0.0"))
def test_float_field_invalid(value):
    check_invalid_config({"float_field": value})


@pytest.mark.parametrize("value", ("", None, "text"))
def test_optional_field(value):
    check_config({"optional_field": value})


@pytest.mark.parametrize("value", (True, 6, [None]))
def test_optional_field_invalid(value):
    check_invalid_config({"optional": value})


@pytest.mark.parametrize("value", ("", 0, True, "text", 7))
def test_union_field(value):
    check_config({"union_field": value})


@pytest.mark.parametrize("value", (6.0, [""]))
def test_union_field_invalid(value):
    check_invalid_config({"optional": value})


@pytest.mark.parametrize("value", ("implicit", "", "text"))
def test_implicit_field(value):
    check_config({"implicit_field": value})


TUPLE_VALUES = ((), (1,), (3, 4, 6), (4, 5, 4))


@pytest.mark.parametrize("value", TUPLE_VALUES)
def test_list_field(value):
    check_config(
        {"list_field": list(value)},
        {"list_field": value},
        serialized_config={"list_field": list(value)},
    )


@pytest.mark.parametrize("value", TUPLE_VALUES)
def test_tuple_field(value):
    check_config(
        {"tuple_field": list(value)},
        {"tuple_field": value},
        serialized_config={"tuple_field": list(value)},
    )


@pytest.mark.parametrize("value", TUPLE_VALUES)
def test_set_field(value):
    check_config(
        {"set_field": list(set(value))},
        {"set_field": set(value)},
        {"set_field": list(value)},
        {"set_field": tuple(value)},
        serialized_config={"set_field": list(set(value))},
    )


# @pytest.mark.parametrize("value", ((0, ""), (5, "text"), (True, "True")))
# def test_tuple_fixed_length_field(value):
#    expected_config = {"tuple_variable_length_field": value}
#    Assert.eq(TestConfig.from_dict(expected_config).to_serialized(), expected_config)
#    Assert.eq(TestConfig.from_dict({"tuple_variable_length_field": list(value)}).to_serialized(), expected_config)
#    Assert.eq(TestConfig.from_dict({"tuple_variable_length_field": set(value)}).to_serialized(), {"tuple_variable_length_field": tuple(set(value))})


@pytest.mark.parametrize("value", ({}, {True: 2}, {1: 2, 3: 4}))
def test_dict_field(value):
    check_config({"dict_field": value})


class IntClass(int):
    pass


@pytest.mark.parametrize("value", (int, bool, IntClass))
def test_type_field(value):
    check_config({"type_field": value}, serialized_config={"type_field": str(value)})


@pytest.mark.parametrize("value", (TestEnum.a, TestEnum.b, TestEnum.c))
def test_enum_field(value):
    check_config({"enum_field": value}, {"enum_field": value.value})


def test_core_field():
    Assert.eq(TestConfig.from_dict({}).to_serialized(verbose=FieldVerboseLevel.core), {"core_field": 4})


@pytest.mark.parametrize(
    "value",
    (
        {},
        {3: None, "text": [], False: [["", 3], ["a", -7]]},
        {0: [[".", 8]]},
    ),
)
def test_complex_field(value):
    check_config({"complex_field": value})
