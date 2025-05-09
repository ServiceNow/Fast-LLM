import pytest

from fast_llm.config import NoAutoValidate
from tests.config.common import ExampleConfig


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
