import math
import pathlib

import numpy
import pytest

from fast_llm.config import FieldVerboseLevel
from fast_llm.utils import Assert, check_equal_nested
from tests.config.common import ExampleConfig, ExampleEnum, ExampleVerboseConfig, check_config, check_invalid_config


def test_create_and_serialize_config():
    Assert.eq(ExampleConfig.from_dict({}).to_dict(), {})


@pytest.mark.parametrize("value", (0, -6, 3))
def test_int_field(value):
    check_config({"int_field": value})


@pytest.mark.parametrize("value", (4.0, math.inf, "1", None, [4], True))
def test_int_field_invalid(value):
    check_invalid_config({"int_field": value})


@pytest.mark.parametrize("value", (True, False))
def test_bool_field(value):
    check_config({"bool_field": value})


@pytest.mark.parametrize("value", (1, "True", None, [True]))
def test_bool_field_invalid(value):
    check_invalid_config({"bool_field": value})


@pytest.mark.parametrize("value", ("", "text", "1"))
def test_str_field(value):
    check_config({"str_field": str(value)}, {"str_field": value})


@pytest.mark.parametrize("value", (1, True, None, ["text"], pathlib.Path("a"), ExampleEnum.a))
def test_str_field_invalid(value):
    check_invalid_config({"str_field": value})


@pytest.mark.parametrize("value", (".", "text", "/a/b/c.d"))
def test_path_field(value):
    check_config({"path_field": pathlib.Path(value)}, {"path_field": value})


@pytest.mark.parametrize("value", (1, True, None, [pathlib.Path("a")]))
def test_path_field_invalid(value):
    check_invalid_config({"path_field": value})


@pytest.mark.parametrize("value", (4.0, math.pi, math.inf, 3, math.nan))
def test_float_field(value):
    check_config(
        {"float_field": float(value)}, {"float_field": value}, serialized_config={"float_field": float(value)}
    )


@pytest.mark.parametrize("value", (None, [4.7], "0.0", True, numpy.float64(3)))
def test_float_field_invalid(value):
    check_invalid_config({"float_field": value})


@pytest.mark.parametrize("value", ("", None, "text"))
def test_optional_field(value):
    check_config({"optional_field": value})


@pytest.mark.parametrize("value", (True, 6, [None]))
def test_optional_field_invalid(value):
    check_invalid_config({"optional": value})


@pytest.mark.parametrize("value", ("", 0, "text", 7))
def test_union_field(value):
    check_config({"union_field": value})


@pytest.mark.parametrize("value", (6.0, [""], True))
def test_union_field_invalid(value):
    check_invalid_config({"optional": value})


def test_implicit_field_value():
    Assert.eq(ExampleConfig.from_dict({}).implicit_field, "implicit")


@pytest.mark.parametrize("value", ("implicit", "", "text"))
def test_implicit_field(value):
    check_config({"implicit_field": value})


ARRAY_VALUES = ((), (1,), (3, 4, 6), (4, 5, 4))
ARRAY_VALUES_INVALID = (6.0, {}, True, "text")


@pytest.mark.parametrize("value", ARRAY_VALUES)
def test_list_field(value):
    check_config(
        {"list_field": list(value)},
        {"list_field": value},
        serialized_config={"list_field": list(value)},
    )


@pytest.mark.parametrize("value", ARRAY_VALUES_INVALID)
def test_list_field_invalid(value):
    check_invalid_config({"list_field": value})


@pytest.mark.parametrize("value", ARRAY_VALUES)
def test_tuple_field(value):
    check_config(
        {"tuple_field": list(value)},
        {"tuple_field": value},
        serialized_config={"tuple_field": list(value)},
    )


@pytest.mark.parametrize("value", ARRAY_VALUES_INVALID)
def test_tuple_field_invalid(value):
    check_invalid_config({"tuple_field": value})


@pytest.mark.parametrize("value", ARRAY_VALUES)
def test_set_field(value):
    check_config(
        {"set_field": list(set(value))},
        {"set_field": set(value)},
        {"set_field": list(value)},
        {"set_field": tuple(value)},
        serialized_config={"set_field": list(set(value))},
    )


@pytest.mark.parametrize("value", ARRAY_VALUES_INVALID)
def test_tuple_field_invalid(value):
    check_invalid_config({"set_field": value})


@pytest.mark.parametrize("value", ({}, {1: 2, 3: 4}))
def test_dict_field(value):
    check_config({"dict_field": value})


@pytest.mark.parametrize("value", ({True: 2}, {4: "3"}, {4: {1: 4}}, None, 4, {1}, [5, 7], "text"))
def test_dict_field_invalid(value):
    check_invalid_config({"dict_field": value})


class IntClass(int):
    pass


@pytest.mark.parametrize("value", (int, bool, IntClass))
def test_type_field(value):
    check_config({"type_field": value}, serialized_config={"type_field": str(value)})


@pytest.mark.parametrize("value", (5, None, [], "text"))
def test_type_field_invalid(value):
    check_invalid_config({"type_field": value})


@pytest.mark.parametrize("value", (ExampleEnum.a, ExampleEnum.b, ExampleEnum.c))
def test_enum_field(value):
    check_config({"enum_field": value}, {"enum_field": str(value)})


@pytest.mark.parametrize("value", (5, None, [], "text"))
def test_enum_field_invalid(value):
    check_invalid_config({"type_field": value})


def test_core_field():
    Assert.eq(ExampleConfig.from_dict({}).to_dict(verbose=FieldVerboseLevel.core), {"core_field": 4})


@pytest.mark.parametrize(
    "value",
    (
        {},
        {3: None, "text": [], 0: [["", 3], ["a", -7]]},
        {0: [[".", 8]]},
    ),
)
def test_complex_field(value):
    check_config({"complex_field": value})


@pytest.mark.parametrize(
    "value",
    ({3: None, "text": [], False: [["", 3], ["a", -7]]},),
)
def test_complex_field_invalid(value):
    check_invalid_config({"complex_field": value})


def test_verbose_config_default():
    default_values = {
        "list_default_field": [0],
        "tuple_default_field": [0, 1],
        "tuple_fixed_length_field": [5, "text"],
        "set_default_field": [0, 1, 2],
        "dict_default_field": {"0": 0, "1": 1},
        "explicit_field": "explicit",
    }
    config = ExampleVerboseConfig.from_dict({})
    check_equal_nested(config.to_dict(), default_values)
    check_equal_nested(config.to_dict(serialized=False), default_values)


@pytest.mark.parametrize("value", ((0, ""), (5, "text"), (7, "True")))
def test_tuple_fixed_length_field(value):
    check_config(
        {"tuple_fixed_length_field": list(value)},
        {"tuple_fixed_length_field": value},
        serialized_config={"tuple_fixed_length_field": list(value)},
        cls=ExampleVerboseConfig,
        fields=["tuple_fixed_length_field"],
    )


@pytest.mark.parametrize("value", ((), (5,), ("", 0), ("0", "True"), (0, "", "text")))
def test_tuple_fixed_length_field_invalid(value):
    check_invalid_config({"tuple_fixed_length_field": value}, cls=ExampleVerboseConfig)


# TODO: Test other fields with defaults.
# TODO: Test nested fields.
