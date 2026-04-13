import pytest

from fast_llm.config import NoAutoValidate, UpdateType
from fast_llm.utils import check_equal_nested
from tests.config.common import ExampleNestedConfig

TEST_CONFIGS = (
    (
        # Empty config: updating nothing changes nothing.
        {},
        {},
        {},
        None,
    ),
    (
        # Flat fields: update adds new fields and overwrites shared fields; unrelated base fields survive.
        {"int_field": 4, "str_field": "text"},
        {"float_field": 3.0, "str_field": ""},
        {"int_field": 4, "float_field": 3.0, "str_field": ""},
        None,
    ),
    (
        # Nested field: update merges sub-fields; override replaces the whole nested config.
        {"nested_field": {"int_field": 4, "str_field": "text"}},
        {"nested_field": {"float_field": 3.0, "str_field": ""}},
        {"nested_field": {"int_field": 4, "float_field": 3.0, "str_field": ""}},
        {"nested_field": {"float_field": 3.0, "str_field": ""}},
    ),
    (
        # Top-level and nested fields together: top-level fields and nested sub-fields both update correctly.
        {"int_field": 1, "nested_field": {"int_field": 4, "str_field": "text"}},
        {"str_field": "new", "nested_field": {"float_field": 3.0}},
        {
            "int_field": 1,
            "str_field": "new",
            "nested_field": {"int_field": 4, "float_field": 3.0, "str_field": "text"},
        },
        {"int_field": 1, "str_field": "new", "nested_field": {"float_field": 3.0}},
    ),
    (
        # Update from empty: base has no fields set; all update fields appear in result.
        {},
        {"int_field": 7, "str_field": "hello"},
        {"int_field": 7, "str_field": "hello"},
        None,
    ),
    (
        # Update to empty: update has no fields set; base is preserved unchanged.
        {"int_field": 7, "str_field": "hello"},
        {},
        {"int_field": 7, "str_field": "hello"},
        None,
    ),
    (
        # Collection fields: list and dict fields in update replace their counterparts in base.
        {"int_field": 1, "list_field": [1, 2, 3]},
        {"list_field": [4, 5]},
        {"int_field": 1, "list_field": [4, 5]},
        None,
    ),
)


@pytest.mark.parametrize(("base", "update", "updated", "overridden"), TEST_CONFIGS)
def test_update(base, update, updated, overridden) -> None:
    if overridden is None:
        overridden = updated
    with NoAutoValidate():
        update_config = ExampleNestedConfig.from_dict(update)
    check_equal_nested(ExampleNestedConfig.from_dict(base, update, update_type=UpdateType.update).to_dict(), updated)
    check_equal_nested(
        ExampleNestedConfig.from_dict(base).to_copy(update_config, update_type=UpdateType.update).to_dict(),
        updated,
    )
    check_equal_nested(
        ExampleNestedConfig.from_dict(base, update, update_type=UpdateType.override).to_dict(), overridden
    )
    check_equal_nested(
        ExampleNestedConfig.from_dict(base).to_copy(update_config, update_type=UpdateType.override).to_dict(),
        overridden,
    )
