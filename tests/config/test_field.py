import dataclasses
import functools
import math
import pathlib
from typing import Any

import numpy
import pytest

from fast_llm.config import (
    Config,
    Field,
    FieldHint,
    FieldOverride,
    FieldVerboseLevel,
    check_field,
    config_class,
    process_field,
    skip_valid_if_none,
)
from fast_llm.utils import Assert, check_equal_nested
from tests.config.common import (
    ExampleConfig,
    ExampleEnum,
    ExampleNestedConfig,
    ExampleVerboseConfig,
    check_config,
    check_invalid_config,
)


class IntSubclass(int):
    pass


# --- Validator configs (referenced in _FIELD_TEST_CASES) ---


@config_class()
class ExampleCheckFieldConfig(Config):
    positive_field: int = Field(default=0, hint=FieldHint.optional, valid=check_field(Assert.geq, 0))


@config_class()
class ExampleSkipIfNoneConfig(Config):
    optional_positive_field: int | None = Field(
        default=None,
        hint=FieldHint.optional,
        valid=skip_valid_if_none(check_field(Assert.geq, 0)),
    )


@config_class()
class ExampleProcessFieldConfig(Config):
    doubled_field: int = Field(default=0, hint=FieldHint.optional, valid=process_field(lambda value: value * 2))


# --- FieldOverride configs ---


@config_class()
class ExampleUpdatedDefaultConfig(ExampleConfig):
    int_field = FieldOverride(default=42)


@config_class()
class ExampleUpdatedHintConfig(ExampleConfig):
    # Promote str_field from optional to core so it appears at verbose=core.
    str_field = FieldOverride(hint=FieldHint.core)


@dataclasses.dataclass
class ValidCase:
    # Canonical Python-side value. Used as input to from_dict() and as expected to_dict(serialized=False) result.
    internal: Any
    # Expected to_dict() result. Defaults to internal.
    serialized: Any = None
    # Other input values that should produce the same internal/serialized result.
    alternates: tuple = ()

    def __post_init__(self):
        if self.serialized is None:
            self.serialized = self.internal


@dataclasses.dataclass
class FieldTestCase:
    field_name: str
    valid: list[ValidCase]
    invalid: list[Any]
    cls: type = ExampleConfig
    # When the config class has other fields with non-empty defaults, check only this field.
    fields: list[str] | None = None

    @functools.cached_property
    def params(self) -> list:
        return [
            *(
                pytest.param(
                    self.field_name,
                    self.cls,
                    valid_case.internal,
                    valid_case,
                    self.fields,
                    id=f"{self.field_name}-{valid_case.internal!r}",
                )
                for valid_case in self.valid
            ),
            *(
                pytest.param(
                    self.field_name,
                    self.cls,
                    invalid_value,
                    None,
                    self.fields,
                    id=f"{self.field_name}-invalid-{invalid_value!r}",
                )
                for invalid_value in self.invalid
            ),
        ]


_FIELD_TEST_CASES: list[FieldTestCase] = [
    FieldTestCase(
        field_name="int_field",
        valid=[ValidCase(0), ValidCase(-6), ValidCase(3)],
        # Rejects float (even if whole number), bool, string, None, list.
        invalid=[4.0, math.inf, "1", None, [4], True],
    ),
    FieldTestCase(
        field_name="bool_field",
        valid=[ValidCase(True), ValidCase(False)],
        # Rejects int (bool is a subclass of int but the reverse is not accepted), string, None, list.
        invalid=[1, "True", None, [True]],
    ),
    FieldTestCase(
        field_name="str_field",
        valid=[ValidCase(""), ValidCase("text"), ValidCase("1")],
        # Rejects int, bool, None, list, Path, Enum.
        invalid=[1, True, None, ["text"], pathlib.Path("a"), ExampleEnum.a],
    ),
    FieldTestCase(
        field_name="path_field",
        valid=[
            # Stores as pathlib.Path; serializes to string; accepts string input.
            ValidCase(pathlib.Path("."), serialized=".", alternates=(".",)),
            ValidCase(pathlib.Path("text"), serialized="text", alternates=("text",)),
            ValidCase(pathlib.Path("/a/b/c.d"), serialized="/a/b/c.d", alternates=("/a/b/c.d",)),
        ],
        # Rejects int, bool, None, list.
        invalid=[1, True, None, [pathlib.Path("a")]],
    ),
    FieldTestCase(
        field_name="float_field",
        valid=[
            # Accepts int and float; stores and serializes as float; inf and nan are valid.
            ValidCase(4.0),
            ValidCase(math.pi),
            ValidCase(math.inf),
            ValidCase(math.nan),
            ValidCase(3.0, alternates=(3,)),  # int input coerced to float
        ],
        # Rejects None, list, string, bool, numpy scalar.
        invalid=[None, [4.7], "0.0", True, numpy.float64(3)],
    ),
    FieldTestCase(
        field_name="optional_field",
        valid=[ValidCase(None), ValidCase(""), ValidCase("text")],
        # Rejects bool, int, list.
        invalid=[True, 6, [None]],
    ),
    FieldTestCase(
        field_name="union_field",
        valid=[ValidCase(""), ValidCase(0), ValidCase("text"), ValidCase(7)],
        # Rejects float, list, bool.
        invalid=[6.0, [""], True],
    ),
    FieldTestCase(
        field_name="implicit_field",
        valid=[
            # _validate() sets "implicit" when not provided; explicit value overrides.
            ValidCase("implicit"),
            ValidCase(""),
            ValidCase("text"),
        ],
        invalid=[],  # Any string is valid; invalids are covered by str_field tests.
    ),
    FieldTestCase(
        field_name="list_field",
        valid=[
            # Stores as list; accepts tuple input; duplicates preserved.
            ValidCase([]),
            ValidCase([1], alternates=((1,),)),
            ValidCase([3, 4, 6], alternates=((3, 4, 6),)),
            ValidCase([4, 5, 4], alternates=((4, 5, 4),)),
        ],
        # Rejects float, dict, bool, string.
        invalid=[6.0, {}, True, "text"],
    ),
    FieldTestCase(
        field_name="tuple_field",
        valid=[
            # Stores as tuple; serializes as list; accepts list or tuple input.
            ValidCase([], serialized=[], alternates=((),)),
            ValidCase([1], serialized=[1], alternates=((1,),)),
            ValidCase([3, 4, 6], serialized=[3, 4, 6], alternates=((3, 4, 6),)),
            ValidCase([4, 5, 4], serialized=[4, 5, 4], alternates=((4, 5, 4),)),
        ],
        # Rejects float, dict, bool, string.
        invalid=[6.0, {}, True, "text"],
    ),
    FieldTestCase(
        field_name="set_field",
        valid=[
            # Deduplicates; serializes as list; accepts list/tuple/set input.
            # Note: CPython iterates small-int sets in insertion/hash order, matching sorted order here.
            ValidCase([], serialized=[], alternates=(set(), ())),
            ValidCase([1], serialized=[1], alternates=({1}, (1,))),
            ValidCase([3, 4, 6], serialized=[3, 4, 6], alternates=({3, 4, 6}, (3, 4, 6))),
            ValidCase([4, 5], serialized=[4, 5], alternates=({4, 5}, [4, 5, 4], (4, 5, 4))),  # deduplication
        ],
        # Rejects float, dict, bool, string.
        invalid=[6.0, {}, True, "text"],
    ),
    FieldTestCase(
        field_name="dict_field",
        valid=[ValidCase({}), ValidCase({1: 2, 3: 4})],
        # Rejects bool keys, wrong value types, nested dict values, None, int, set, list, string.
        invalid=[{True: 2}, {4: "3"}, {4: {1: 4}}, None, 4, {1}, [5, 7], "text"],
    ),
    FieldTestCase(
        field_name="type_field",
        valid=[
            # Accepts type objects that are subclasses of int; serializes as repr string.
            ValidCase(int, serialized=str(int)),
            ValidCase(bool, serialized=str(bool)),
            ValidCase(IntSubclass, serialized=str(IntSubclass)),
        ],
        # Rejects non-type values.
        invalid=[5, None, [], "text"],
    ),
    FieldTestCase(
        field_name="enum_field",
        valid=[
            # Accepts enum values and their string equivalents; serializes as string.
            ValidCase(ExampleEnum.a, serialized="a", alternates=("a",)),
            ValidCase(ExampleEnum.b, serialized="b", alternates=("b",)),
            ValidCase(ExampleEnum.c, serialized="c", alternates=("c",)),
        ],
        # Rejects non-string, None, list, and strings not in the enum.
        invalid=[5, None, [], "d"],
    ),
    FieldTestCase(
        field_name="complex_field",
        valid=[
            ValidCase({}),
            ValidCase({"3": None, "text": [], "0": [["", 3], ["a", -7]]}),
            ValidCase({"0": [[".", 8]]}),
        ],
        # Rejects non-string dict keys.
        invalid=[{False: [["", 3]]}],
    ),
    FieldTestCase(
        field_name="tuple_fixed_length_field",
        valid=[
            # Fixed-length (int, str) tuple; stores and serializes as list; accepts list or tuple input.
            ValidCase([0, ""], alternates=((0, ""),)),
            ValidCase([5, "text"], alternates=((5, "text"),)),
            ValidCase([7, "True"], alternates=((7, "True"),)),
        ],
        # Rejects wrong length (too short/long) and wrong element types.
        invalid=[(), (5,), ("", 0), ("0", "True"), (0, "", "text")],
        cls=ExampleVerboseConfig,
        fields=["tuple_fixed_length_field"],
    ),
    FieldTestCase(
        field_name="nested_field",
        valid=[
            # Non-empty sub-configs only: empty nested config serializes back to {} (no nested_field key).
            ValidCase({"int_field": 3}),
            ValidCase({"int_field": 3, "str_field": "text"}),
            ValidCase({"list_field": [1, 2], "dict_field": {1: 2}}),
        ],
        # Rejects None, non-dict, and dicts with invalid sub-field values.
        invalid=[None, 5, {"int_field": "not_an_int"}, {"int_field": True}],
        cls=ExampleNestedConfig,
    ),
    FieldTestCase(
        field_name="positive_field",
        valid=[ValidCase(0), ValidCase(5)],
        # Rejects values failing check_field(>=0); type invalids already covered by int_field tests.
        invalid=[-1],
        cls=ExampleCheckFieldConfig,
    ),
    FieldTestCase(
        field_name="optional_positive_field",
        valid=[ValidCase(None), ValidCase(0), ValidCase(5)],
        # Rejects negative values; None bypasses the validator (skip_valid_if_none).
        invalid=[-1],
        cls=ExampleSkipIfNoneConfig,
    ),
]


@pytest.mark.parametrize(
    ("field_name", "cls", "value", "expected", "fields"),
    [case for field_test_case in _FIELD_TEST_CASES for case in field_test_case.params],
)
def test_field(field_name: str, cls: type, value: Any, expected: ValidCase | None, fields: list[str] | None):
    if expected is None:
        check_invalid_config({field_name: value}, cls=cls)
    else:
        check_config(
            {field_name: value},
            *({field_name: alternate} for alternate in expected.alternates),
            serialized_config={field_name: expected.serialized},
            cls=cls,
            fields=fields,
        )


def test_implicit_field_value():
    # When implicit_field is not provided, _validate() fills it in as "implicit".
    config = ExampleConfig.from_dict({})
    Assert.eq(config.implicit_field, "implicit")
    # Implicitly-set fields are not included in the serialized dict; all other fields are default,
    # so the empty config serializes back to {}.
    Assert.eq(config.to_dict(), {})


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


def test_nested_field_empty():
    # An empty sub-config is accepted; sub-fields take their defaults.
    config = ExampleNestedConfig.from_dict({"nested_field": {}})
    Assert.eq(config.nested_field.int_field, 0)
    Assert.eq(config.nested_field.str_field, "")


def test_process_field_transforms_value():
    # process_field transforms the value during validation; input 5 is stored as 10.
    Assert.eq(ExampleProcessFieldConfig.from_dict({"doubled_field": 5}).doubled_field, 10)


def test_field_update_default():
    Assert.eq(ExampleUpdatedDefaultConfig.from_dict({}).int_field, 42)
    Assert.eq(ExampleConfig.from_dict({}).int_field, 0)  # parent default unchanged


def test_field_update_hint():
    assert "str_field" in ExampleUpdatedHintConfig.from_dict({}).to_dict(verbose=FieldVerboseLevel.core)
    assert "str_field" not in ExampleConfig.from_dict({}).to_dict(verbose=FieldVerboseLevel.core)  # parent unchanged
