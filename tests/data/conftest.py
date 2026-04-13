import pathlib

import pytest


@pytest.fixture(scope="session")
def data_result_path(result_path: pathlib.Path) -> pathlib.Path:
    return result_path / "data"
