import pytest

from fast_llm.config import NoAutoValidate, UpdateType
from fast_llm.utils import check_equal_nested
from tests.config.common import ExampleNestedConfig

TEST_CONFIGS = (
    (
        # Empty config
        {},
        {},
        {},
        None,
    ),
    (
        # Update unset field; don't update set field; update
        {"int_field": 4, "str_field": "text"},
        {"float_field": 3.0, "str_field": ""},
        {"int_field": 4, "float_field": 3.0, "str_field": ""},
        None,
    ),
    (
        # Update/override nested field.
        {"nested_field": {"int_field": 4, "str_field": "text"}},
        {"nested_field": {"float_field": 3.0, "str_field": ""}},
        {"nested_field": {"int_field": 4, "float_field": 3.0, "str_field": ""}},
        {"nested_field": {"float_field": 3.0, "str_field": ""}},
    ),
    # TODO: Add more complex cases
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
