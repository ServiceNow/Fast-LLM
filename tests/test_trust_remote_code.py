import pytest

from fast_llm.engine.config_utils import runnable
from fast_llm.engine.config_utils.runnable import RunnableConfig, get_trust_remote_code


@pytest.fixture(autouse=True)
def _reset_trust_flags():
    # The flags are process-wide module globals, so save and restore around each test.
    saved = (runnable._trust_remote_code, runnable._trust_all_remote_code)
    yield
    runnable._trust_remote_code, runnable._trust_all_remote_code = saved


@pytest.mark.parametrize(
    ("master", "trust_all", "config_flag", "expected"),
    [
        # Default: every call is denied, no matter what the config says.
        (False, False, False, False),
        (False, False, True, False),
        # Master switch on: per-call config field decides.
        (True, False, False, False),
        (True, False, True, True),
        # `--trust-all-remote-code` overrides the per-call field and implies the master switch.
        (True, True, False, True),
        (True, True, True, True),
        (False, True, False, True),
        (False, True, True, True),
    ],
)
def test_get_trust_remote_code(master: bool, trust_all: bool, config_flag: bool, expected: bool):
    runnable._trust_remote_code = master or trust_all
    runnable._trust_all_remote_code = trust_all
    assert get_trust_remote_code(config_flag) is expected


def test_cli_flags_are_off_by_default():
    parser = RunnableConfig._get_parser()
    parsed = parser.parse_args([])
    assert parsed.trust_remote_code is False
    assert parsed.trust_all_remote_code is False


def test_cli_flags_set_module_globals():
    # Round-trip the flags through the parser into the module globals to lock down the wiring.
    parser = RunnableConfig._get_parser()
    parsed = parser.parse_args(["--trust-remote-code"])
    assert parsed.trust_remote_code is True
    assert parsed.trust_all_remote_code is False

    parsed = parser.parse_args(["--trust-all-remote-code"])
    assert parsed.trust_remote_code is False
    assert parsed.trust_all_remote_code is True
