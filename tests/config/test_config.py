import typing

import pytest
import yaml

from fast_llm.config import (
    Config,
    Field,
    FieldHint,
    FieldVerboseLevel,
    NoAutoValidate,
    UpdateType,
    ValidationError,
    config_class,
)
from fast_llm.utils import Assert, check_equal_nested, header
from tests.config.common import ExampleConfig, ExampleNestedConfig

# --- Dynamic dispatch fixtures ---


@config_class(registry=True)
class AnimalConfig(Config):
    name: str = Field(default="", hint=FieldHint.optional)


@config_class(dynamic_type={AnimalConfig: "dog"})
class DogConfig(AnimalConfig):
    breed: str = Field(default="mutt", hint=FieldHint.optional)


@config_class(dynamic_type={AnimalConfig: "cat"})
class CatConfig(AnimalConfig):
    indoor: bool = Field(default=True, hint=FieldHint.optional)


# --- Verbose level fixtures ---


@config_class()
class ExampleHintConfig(Config):
    """One field at each FieldHint importance level for testing verbose output filtering."""

    core_field: int = Field(default=1, hint=FieldHint.core)
    architecture_field: int = Field(default=2, hint=FieldHint.architecture)
    optional_field: int = Field(default=3, hint=FieldHint.optional)
    performance_field: int = Field(default=4, hint=FieldHint.performance)
    expert_field: int = Field(default=5, hint=FieldHint.expert)


# --- Lifecycle ---


def test_auto_validate():
    assert (config := ExampleConfig())._validated
    with pytest.raises(RuntimeError):
        config.bool_field = True
    config.bool_field = False

    assert ExampleConfig.from_dict({})._validated

    with NoAutoValidate():
        assert not (config := ExampleConfig())._validated

    config.bool_field = True

    config.validate()

    assert config._validated
    with pytest.raises(RuntimeError):
        config.bool_field = False
    config.bool_field = True

    with NoAutoValidate():
        assert not (config := ExampleConfig.from_dict({}))._validated
    config.validate()
    assert config._validated


def test_multiple_validation_errors_all_reported():
    with pytest.raises(ValidationError) as exc_info:
        ExampleConfig.from_dict({"int_field": "not_an_int", "float_field": "not_a_float"})
    error_message = str(exc_info.value)
    assert "int_field" in error_message
    assert "float_field" in error_message


# --- compare() ---


def test_compare_equal_returns_none():
    config_a = ExampleConfig.from_dict({"int_field": 5})
    config_b = ExampleConfig.from_dict({"int_field": 5})
    assert config_a.compare(config_b) is None


def test_compare_different():
    config_a = ExampleConfig.from_dict({"int_field": 5})
    config_b = ExampleConfig.from_dict({"int_field": 7})
    with pytest.raises(ValueError):
        config_a.compare(config_b)
    # Custom log_fn receives the difference instead of raising.
    messages = []
    config_a.compare(config_b, log_fn=messages.append)
    assert len(messages) == 1


# --- strict mode ---


@pytest.mark.parametrize(
    ("config_dict", "cls"),
    [
        ({"int_field": 3, "unknown_field": 5}, ExampleConfig),
        ({"nested_field": {"int_field": 3, "unknown_sub_field": 5}}, ExampleNestedConfig),
    ],
    ids=["top_level", "nested"],
)
def test_strict_unknown_field_raises(config_dict, cls):
    with pytest.raises(ValidationError):
        cls.from_dict(config_dict)


def test_strict_false_unknown_field_ignored():
    config = ExampleConfig.from_dict({"int_field": 3, "unknown_field": 5}, strict=False)
    assert config.int_field == 3
    assert not hasattr(config, "unknown_field")


def test_strict_false_unknown_nested_field_ignored():
    config = ExampleNestedConfig.from_dict({"nested_field": {"int_field": 3, "unknown_sub_field": 5}}, strict=False)
    assert config.nested_field.int_field == 3


# --- Dynamic dispatch ---


@pytest.mark.parametrize(
    ("input_dict", "expected_cls", "expected_field", "expected_value"),
    [
        ({"type": "dog", "breed": "labrador"}, DogConfig, "breed", "labrador"),
        ({"type": "cat", "indoor": False}, CatConfig, "indoor", False),
    ],
    ids=["dog", "cat"],
)
def test_dynamic_dispatch_selects_subclass(input_dict, expected_cls, expected_field, expected_value):
    config = AnimalConfig.from_dict(input_dict)
    assert isinstance(config, expected_cls)
    Assert.eq(getattr(config, expected_field), expected_value)


def test_dynamic_dispatch_type_serialized():
    config = DogConfig.from_dict({"breed": "poodle"})
    result = config.to_dict()
    Assert.eq(result["type"], "dog")
    Assert.eq(result["breed"], "poodle")


def test_dynamic_dispatch_unknown_type_raises():
    with pytest.raises(ValidationError):
        AnimalConfig.from_dict({"type": "fish"})


def test_dynamic_dispatch_roundtrip():
    original = DogConfig.from_dict({"breed": "husky"})
    roundtrip = AnimalConfig.from_dict(original.to_dict())
    assert isinstance(roundtrip, DogConfig)
    Assert.eq(roundtrip.breed, "husky")


# --- Renamed fields ---


def test_renamed_field():
    with pytest.warns(DeprecationWarning, match="old_int_field"):
        config = ExampleConfig.from_dict({"old_int_field": 5})
    Assert.eq(config.int_field, 5)
    # New name works without a deprecation warning.
    Assert.eq(ExampleConfig.from_dict({"int_field": 7}).int_field, 7)


def test_renamed_field_with_transform():
    with pytest.warns(DeprecationWarning, match="original_float_field"):
        config = ExampleConfig.from_dict({"original_float_field": 4.0})
    Assert.eq(config.float_field, 8.0)


# --- Verbose levels ---

# At verbose >= optional (10), the base Config.type field (hint=feature, importance=10) also appears.
_VERBOSE_LEVEL_CASES = [
    (FieldVerboseLevel.explicit, {}),
    (FieldVerboseLevel.core, {"core_field": 1, "architecture_field": 2}),
    (FieldVerboseLevel.optional, {"core_field": 1, "architecture_field": 2, "optional_field": 3, "type": None}),
    (
        FieldVerboseLevel.performance,
        {"core_field": 1, "architecture_field": 2, "optional_field": 3, "performance_field": 4, "type": None},
    ),
    (
        FieldVerboseLevel.debug,
        {
            "core_field": 1,
            "architecture_field": 2,
            "optional_field": 3,
            "performance_field": 4,
            "expert_field": 5,
            "type": None,
        },
    ),
]


@pytest.mark.parametrize(("verbose", "expected"), _VERBOSE_LEVEL_CASES)
def test_verbose_level(verbose, expected):
    check_equal_nested(ExampleHintConfig.from_dict({}).to_dict(verbose=verbose), expected)


# --- Field definition error fixtures ---


with pytest.raises(ValueError, match="default_factory"):
    # Defining this at module level triggers Field.__init__ validation immediately.
    @config_class()
    class _BothDefaultAndFactoryConfig(Config):
        x: list = Field(default=[], default_factory=list, hint=FieldHint.optional)


with pytest.raises(ValueError, match="default_factory"):

    @config_class()
    class _ConfigAsDefaultFactoryConfig(Config):
        nested: ExampleConfig = Field(default_factory=ExampleConfig, hint=FieldHint.optional)


with pytest.raises(TypeError, match="__post_init__"):

    @config_class()
    class _PostInitConfig(Config):
        def __post_init__(self):
            pass


@config_class()
class _AbstractConfig(Config):
    _abstract: typing.ClassVar[bool] = True


# --- Abstract config ---


def test_abstract_config_raises():
    with pytest.raises(ValidationError, match="abstract"):
        _AbstractConfig()


# --- Delete on validated config ---


def test_delattr_after_validation_raises():
    config = ExampleConfig.from_dict({})
    with pytest.raises(RuntimeError, match="delete"):
        del config.int_field


# --- to_logs / __repr__ ---


@pytest.mark.parametrize(
    ("cls", "config_dict", "expected_core_dict"),
    [
        (ExampleConfig, {}, {"core_field": 4}),
        (ExampleConfig, {"int_field": 3}, {"int_field": 3, "core_field": 4}),
        (
            ExampleConfig,
            {"int_field": 3, "str_field": "hello"},
            {"int_field": 3, "str_field": "hello", "core_field": 4},
        ),
        (
            ExampleNestedConfig,
            {"nested_field": {"int_field": 5}},
            {"core_field": 4, "nested_field": {"int_field": 5, "core_field": 4}},
        ),
    ],
)
def test_repr_and_to_logs(cls, config_dict, expected_core_dict):
    config = cls.from_dict(config_dict)
    expected = (
        f"\n{header(config._get_class_name(), 80, '-')}"
        f"\n{yaml.safe_dump(expected_core_dict, sort_keys=False)}"
        f"{header('end', 80, '-')}"
    )
    Assert.eq(repr(config), expected)
    messages = []
    config.to_logs(log_fn=messages.append)
    Assert.eq(len(messages), 1)
    Assert.eq(messages[0], expected)


# --- Validated config as update ---


def test_validated_config_as_update_raises():
    validated = ExampleConfig.from_dict({"int_field": 1})
    with pytest.raises(ValueError, match="Validated"):
        ExampleConfig.from_dict({}, validated, update_type=UpdateType.update)
