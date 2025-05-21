import pytest


def pytest_addoption(parser):
    parser.addoption("--skip-slow", action="store_true")
    parser.addoption(
        "--run-extra-slow",
        action="store_true",
        default=False,
        help="Run tests marked as extra_slow",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: Test is slow.")
    config.addinivalue_line(
        "markers", "extra_slow: Mark test as extra slow and skip unless --run-extra-slow is given."
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--skip-slow"):
        skip_slow = pytest.mark.skip(reason="Skipping slow tests")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    if not config.getoption("--run-extra-slow"):
        skip_extra_slow = pytest.mark.skip(reason="need --run-extra-slow option to run")
        for item in items:
            if "extra_slow" in item.keywords:
                item.add_marker(skip_extra_slow)
