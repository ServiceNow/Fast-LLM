import enum
import pathlib

import pytest

from fast_llm.config import Config, Field, FieldHint, ValidationError, config_class
from fast_llm.utils import check_equal_nested


class ExampleEnum(enum.StrEnum):
    a = "a"
    b = "b"
    c = "c"


@config_class()
class ExampleConfig(Config):
    int_field: int = Field(default=0, hint=FieldHint.optional)
    bool_field: bool = Field(default=False, hint=FieldHint.optional)
    str_field: str = Field(default="", hint=FieldHint.optional)
    path_field: pathlib.Path = Field(default="", hint=FieldHint.optional)
    float_field: float = Field(default=4.0, hint=FieldHint.optional)
    optional_field: str | None = Field(default=None, hint=FieldHint.optional)
    union_field: str | int = Field(default=7, hint=FieldHint.optional)
    implicit_field: str = Field(default=None, hint=FieldHint.optional)
    list_field: list[int] = Field(default_factory=list, hint=FieldHint.optional)
    tuple_field: tuple[int, ...] = Field(default=(), hint=FieldHint.optional)
    # tuple_fixed_length_field: tuple[int, str] = Field(default=(5, "text"), hint=FieldHint.optional)
    set_field: set[int] = Field(default_factory=set, hint=FieldHint.optional)
    dict_field: dict[int, int] = Field(default_factory=dict, hint=FieldHint.optional)
    type_field: type[int] = Field(default=int, hint=FieldHint.optional)
    enum_field: ExampleEnum = Field(default=ExampleEnum.a, hint=FieldHint.optional)
    core_field: int = Field(default=4, hint=FieldHint.core)
    complex_field: dict[str | int, list[tuple[str, int]] | None] = Field(default_factory=dict, hint=FieldHint.optional)

    def _validate(self) -> None:
        with self._set_implicit_default():
            if self.implicit_field is None:
                self.implicit_field = "implicit"
        super()._validate()


@config_class()
class ExampleVerboseConfig(Config):
    # These fields will have non-empty default serialized values.
    list_default_field: list[int] = Field(default_factory=lambda: [0], hint=FieldHint.optional)
    tuple_default_field: tuple[int, ...] = Field(default=(0, 1), hint=FieldHint.optional)
    tuple_fixed_length_field: tuple[int, str] = Field(default=(5, "text"), hint=FieldHint.optional)
    set_default_field: set[int] = Field(default_factory=lambda: {0, 1, 2}, hint=FieldHint.optional)
    dict_default_field: dict[str, int] = Field(default_factory=lambda: {"0": 0, "1": 1}, hint=FieldHint.optional)
    explicit_field: str = Field(default=None, hint=FieldHint.optional)

    def _validate(self) -> None:
        if self.explicit_field is None:
            self.explicit_field = "explicit"
        super()._validate()


@config_class()
class ExampleNestedConfig(ExampleConfig):
    nested_field: ExampleConfig = Field(hint=FieldHint.core)


def check_config(
    internal_config,
    *alternate,
    serialized_config=None,
    cls: type[Config] = ExampleConfig,
    fields: list[str] | None = None,
):
    serialized_config = serialized_config if serialized_config else alternate[0] if alternate else internal_config
    for init_config in (internal_config, *alternate):
        config = cls.from_dict(init_config)
        serialized_config_ = config.to_dict()
        internal_config_ = config.to_dict(serialized=False)
        if fields is None:
            check_equal_nested(serialized_config_, serialized_config)
            check_equal_nested(internal_config_, internal_config)
        else:
            for field in fields:
                check_equal_nested(serialized_config_[field], serialized_config[field])
                check_equal_nested(internal_config_[field], internal_config[field])


def check_invalid_config(config, cls: type[Config] = ExampleConfig):
    with pytest.raises(ValidationError):
        cls.from_dict(config)
