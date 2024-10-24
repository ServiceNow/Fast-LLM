import pytest


def pytest_addoption(parser):
    parser.addoption("--skip-slow", action="store_true")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: Test is slow.")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--skip-slow"):
        skip_slow = pytest.mark.skip(reason="Skipping slow tests")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
