import pathlib
import shutil
import typing

import pytest
import yaml

from fast_llm.data.preparation.dataset_discovery.config import DatasetDiscoveryConfig
from fast_llm.utils import check_equal_nested
from tests.utils.dataset import get_alt_test_dataset, get_common_test_dataset


@pytest.mark.parametrize(
    ("name", "paths", "ignore_paths", "expected_config"),
    (
        ("single_dataset", (".",), (), {"type": "memmap", "path": "dataset_0.fast_llm_dataset"}),
        (
            "same_directory",
            (".", "."),
            (),
            {
                "type": "blended",
                "name": "same_directory",
                "datasets": [
                    {"type": "memmap", "path": "dataset_0.fast_llm_dataset"},
                    {"type": "memmap", "path": "dataset_1.fast_llm_dataset"},
                ],
                "weights": [47178, 46208],
            },
        ),
        (
            "different_directory",
            ("dataset0", "dataset1"),
            (),
            {
                "type": "blended",
                "name": "different_directory",
                "datasets": [
                    {"type": "memmap", "path": "dataset0/dataset_0.fast_llm_dataset"},
                    {"type": "memmap", "path": "dataset1/dataset_1.fast_llm_dataset"},
                ],
                "weights": [47178, 46208],
            },
        ),
        (
            "ignore",
            ("dataset0", "dataset1"),
            ("dataset1",),
            {"type": "memmap", "path": "dataset0/dataset_0.fast_llm_dataset"},
        ),
        (
            "local_and_nested",
            (".", "dataset"),
            (),
            {
                "type": "blended",
                "name": "local_and_nested",
                "datasets": [
                    {"type": "memmap", "path": "dataset/dataset_1.fast_llm_dataset"},
                    {"type": "memmap", "path": "dataset_0.fast_llm_dataset"},
                ],
                "weights": [46208, 47178],
            },
        ),
        (
            "local_blended_and_nested",
            (".", ".", "dataset"),
            (),
            {
                "type": "blended",
                "name": "local_blended_and_nested",
                "datasets": [
                    {"type": "memmap", "path": "dataset/dataset_2.fast_llm_dataset"},
                    {
                        "type": "blended",
                        "name": "local_blended_and_nested_local",
                        "datasets": [
                            {"type": "memmap", "path": "dataset_0.fast_llm_dataset"},
                            {"type": "memmap", "path": "dataset_1.fast_llm_dataset"},
                        ],
                        "weights": [47178, 46208],
                    },
                ],
                "weights": [47178, 93386],
            },
        ),
        (
            "local_and_nested_blended",
            (".", "dataset", "dataset"),
            (),
            {
                "type": "blended",
                "name": "local_and_nested_blended",
                "datasets": [
                    {
                        "type": "blended",
                        "name": "dataset",
                        "datasets": [
                            {"type": "memmap", "path": "dataset/dataset_1.fast_llm_dataset"},
                            {"type": "memmap", "path": "dataset/dataset_2.fast_llm_dataset"},
                        ],
                        "weights": [46208, 47178],
                    },
                    {"type": "memmap", "path": "dataset_0.fast_llm_dataset"},
                ],
                "weights": [93386, 47178],
            },
        ),
        (
            "complex",
            (
                ".",
                "dataset1",
                "dataset1/dataset3",
                "dataset2",
                "dataset3",
                "dataset1/dataset4",
                "dataset1/dataset4/dataset5",
            ),
            # Should ignore "dataset3" but not "dataset1/dataset3"
            ("dataset3", "dataset1/dataset4"),
            {
                "type": "blended",
                "name": "complex",
                "datasets": [
                    {
                        "type": "blended",
                        "name": "dataset1",
                        "datasets": [
                            {"type": "memmap", "path": "dataset1/dataset3/dataset_2.fast_llm_dataset"},
                            {"type": "memmap", "path": "dataset1/dataset_1.fast_llm_dataset"},
                        ],
                        "weights": [47178, 46208],
                    },
                    {"type": "memmap", "path": "dataset2/dataset_3.fast_llm_dataset"},
                    {"type": "memmap", "path": "dataset_0.fast_llm_dataset"},
                ],
                "weights": [93386, 46208, 47178],
            },
        ),
    ),
)
def test_dataset_discovery(
    data_result_path: pathlib.Path, name: str, paths: tuple[pathlib.Path], ignore_paths, expected_config: dict
):
    """Test end-to-end discovery with multiple datasets in various structure."""
    test_dataset_path = [get_common_test_dataset()[0], get_alt_test_dataset()[0]]
    (dataset_path := data_result_path / f"dataset_discovery/{name}").mkdir(parents=True)
    for index, path in enumerate(paths):
        (path_ := dataset_path / path).mkdir(parents=True, exist_ok=True)
        shutil.copy(
            test_dataset_path[index % 2] / "shard_0_0.fast_llm_dataset", path_ / f"dataset_{index}.fast_llm_dataset"
        )
        # Add some files to ignore.
        path_.joinpath("junk.txt").touch()

    # Run dataset discovery
    config = DatasetDiscoveryConfig(
        directory=dataset_path,
        output=data_result_path / f"dataset_discovery/configs/{name}.yaml",
        ignore_paths=ignore_paths,
    )
    config.run()

    generated_config = yaml.safe_load(config.output.open())
    print(generated_config)
    check_equal_nested(generated_config, _set_paths_in_config(expected_config, dataset_path.resolve()))


def _set_paths_in_config(config: dict[str, typing.Any], base_path: pathlib.Path):
    config = config.copy()
    if "path" in config:
        config["path"] = str(base_path / config["path"])
    if "datasets" in config:
        config["datasets"] = [_set_paths_in_config(dataset, base_path) for dataset in config["datasets"]]
    return config
