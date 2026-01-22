"""
Tests for the dataset discovery preparator.
"""

import pathlib

import pytest

from fast_llm.data.preparator.dataset_discovery.config import DatasetDiscoveryConfig
from fast_llm.data.preparator.dataset_discovery.prepare import DatasetDiscoveryPreparator


class TestDatasetDiscovery:
    """Test dataset discovery that scans .fast_llm_dataset files."""

    def test_find_dataset_files(self, tmp_path: pathlib.Path):
        """Test finding .fast_llm_dataset files in directory tree."""
        # Create test directory structure
        (tmp_path / "subdir1").mkdir()
        (tmp_path / "subdir2").mkdir()
        (tmp_path / "subdir1" / "nested").mkdir()

        # Create some .fast_llm_dataset files
        (tmp_path / "dataset1.fast_llm_dataset").touch()
        (tmp_path / "subdir1" / "dataset2.fast_llm_dataset").touch()
        (tmp_path / "subdir1" / "nested" / "dataset3.fast_llm_dataset").touch()
        (tmp_path / "subdir2" / "dataset4.fast_llm_dataset").touch()

        # Create some other files that should be ignored
        (tmp_path / "readme.txt").touch()
        (tmp_path / "subdir1" / "config.yaml").touch()

        # Create config
        config = DatasetDiscoveryConfig(
            directory=tmp_path,
            output=tmp_path / "output.yaml",
        )

        # Create preparator
        preparator = DatasetDiscoveryPreparator(config)

        # Find dataset files
        dataset_files = preparator._find_dataset_files(tmp_path)

        # Should find all 4 .fast_llm_dataset files
        assert len(dataset_files) == 4
        assert all(f.suffix == ".fast_llm_dataset" for f in dataset_files)

    def test_find_dataset_files_with_ignore(self, tmp_path: pathlib.Path):
        """Test finding .fast_llm_dataset files with ignore paths."""
        # Create test directory structure
        (tmp_path / "keep").mkdir()
        (tmp_path / "ignore").mkdir()

        # Create dataset files
        (tmp_path / "keep" / "dataset1.fast_llm_dataset").touch()
        (tmp_path / "ignore" / "dataset2.fast_llm_dataset").touch()

        # Create config with ignore path
        config = DatasetDiscoveryConfig(
            directory=tmp_path,
            output=tmp_path / "output.yaml",
            ignore_paths=[pathlib.Path("ignore")],
        )

        # Create preparator
        preparator = DatasetDiscoveryPreparator(config)

        # Find dataset files
        dataset_files = preparator._find_dataset_files(tmp_path, ignore_paths=config.ignore_paths)

        # Should find only 1 file (dataset2 should be ignored)
        assert len(dataset_files) == 1
        assert dataset_files[0].name == "dataset1.fast_llm_dataset"

    def test_group_files_by_directory(self, tmp_path: pathlib.Path):
        """Test grouping dataset files by directory."""
        # Create files
        files = [
            tmp_path / "dataset1.fast_llm_dataset",
            tmp_path / "dataset2.fast_llm_dataset",
            tmp_path / "subdir" / "dataset3.fast_llm_dataset",
        ]

        # Group by directory
        groups = DatasetDiscoveryPreparator._group_files_by_directory(files)

        # Should have 2 groups
        assert len(groups) == 2
        assert len(groups[tmp_path]) == 2
        assert len(groups[tmp_path / "subdir"]) == 1

    def test_build_directory_tree(self, tmp_path: pathlib.Path):
        """Test building directory tree."""
        # Create nested directories
        (tmp_path / "a" / "b" / "c").mkdir(parents=True)

        # Create groups
        groups = {
            tmp_path: [],
            tmp_path / "a": [],
            tmp_path / "a" / "b": [],
            tmp_path / "a" / "b" / "c": [],
        }

        # Build tree
        tree = DatasetDiscoveryPreparator._build_directory_tree(groups, tmp_path)

        # Verify tree structure
        assert tmp_path / "a" in tree[tmp_path]
        assert tmp_path / "a" / "b" in tree[tmp_path / "a"]
        assert tmp_path / "a" / "b" / "c" in tree[tmp_path / "a" / "b"]

    def test_create_memmap_config(self, tmp_path: pathlib.Path):
        """Test creating memmap config for dataset file."""
        dataset_file = tmp_path / "dataset.fast_llm_dataset"
        dataset_file.touch()

        config = DatasetDiscoveryConfig(
            directory=tmp_path,
            output=tmp_path / "output.yaml",
        )
        preparator = DatasetDiscoveryPreparator(config)

        # Create config
        memmap_config = preparator._create_memmap_config_for_dataset(dataset_file)

        # Verify config structure
        assert memmap_config["type"] == "memmap"
        assert memmap_config["path"] == str(dataset_file)

    def test_get_directory_name(self, tmp_path: pathlib.Path):
        """Test directory naming."""
        root = tmp_path
        subdir = tmp_path / "data" / "train"

        # Test root directory
        name = DatasetDiscoveryPreparator._get_directory_name(root, root)
        assert name == root.name

        # Test subdirectory
        name = DatasetDiscoveryPreparator._get_directory_name(subdir, root)
        assert name == "data_train"

        # Test with suffix
        name = DatasetDiscoveryPreparator._get_directory_name(subdir, root, "_local")
        assert name == "data_train_local"

    @pytest.mark.slow
    def test_dataset_discovery_e2e_single_dataset(self, tmp_path: pathlib.Path):
        """Test end-to-end discovery with a single dataset."""
        import shutil

        import yaml

        from tests.utils.dataset import get_common_test_dataset

        # Get a prepared test dataset
        dataset_path, _, _, _ = get_common_test_dataset()

        # Copy the .fast_llm_dataset file to temp directory
        dataset_files = list(dataset_path.glob("*.fast_llm_dataset"))
        assert len(dataset_files) > 0, "No dataset files found in test dataset"

        test_dataset = dataset_files[0]
        (tmp_path / "datasets").mkdir()
        shutil.copy(test_dataset, tmp_path / "datasets" / "dataset.fast_llm_dataset")

        # Run dataset discovery
        output_path = tmp_path / "discovered_config.yaml"
        config = DatasetDiscoveryConfig(
            directory=tmp_path / "datasets",
            output=output_path,
        )
        config.run()

        # Verify output file was created
        assert output_path.exists()

        # Load and verify the generated config
        with open(output_path) as f:
            content = f.read()
            # Check header comments
            assert "# This file was generated with fast_llm.data.preparator.dataset_discovery" in content
            assert "weights are token-counts in billions" in content
            assert f"#   directory: {tmp_path / 'datasets'}" in content

            # Parse YAML
            f.seek(0)
            generated_config = yaml.safe_load(f)

        # Single dataset should be returned directly (not blended)
        assert generated_config["type"] == "memmap"
        assert "dataset.fast_llm_dataset" in generated_config["path"]

    @pytest.mark.slow
    def test_dataset_discovery_e2e_multiple_datasets(self, tmp_path: pathlib.Path):
        """Test end-to-end discovery with multiple datasets in flat structure."""
        import shutil

        import yaml

        from tests.utils.dataset import get_alt_test_dataset, get_common_test_dataset

        # Get two different test datasets
        dataset1_path, _, _, _ = get_common_test_dataset()
        dataset2_path, _, _, _ = get_alt_test_dataset()

        # Copy dataset files to temp directory
        (tmp_path / "datasets").mkdir()
        dataset1_file = list(dataset1_path.glob("*.fast_llm_dataset"))[0]
        dataset2_file = list(dataset2_path.glob("*.fast_llm_dataset"))[0]

        shutil.copy(dataset1_file, tmp_path / "datasets" / "dataset1.fast_llm_dataset")
        shutil.copy(dataset2_file, tmp_path / "datasets" / "dataset2.fast_llm_dataset")

        # Run dataset discovery
        output_path = tmp_path / "discovered_config.yaml"
        config = DatasetDiscoveryConfig(
            directory=tmp_path / "datasets",
            output=output_path,
        )
        config.run()

        # Verify output file was created
        assert output_path.exists()

        # Load and verify the generated config
        with open(output_path) as f:
            generated_config = yaml.safe_load(f)

        # Multiple datasets should create a blended config
        assert generated_config["type"] == "blended"
        assert len(generated_config["datasets"]) == 2
        assert len(generated_config["weights"]) == 2

        # Verify all weights are positive (in billions)
        assert all(w > 0 for w in generated_config["weights"])

        # Verify datasets are memmap configs
        for dataset_config in generated_config["datasets"]:
            assert dataset_config["type"] == "memmap"
            assert "dataset" in dataset_config["path"]

    @pytest.mark.slow
    def test_dataset_discovery_e2e_hierarchical_structure(self, tmp_path: pathlib.Path):
        """Test end-to-end discovery with hierarchical directory structure."""
        import shutil

        import yaml

        from tests.utils.dataset import get_alt_test_dataset, get_common_test_dataset

        # Get test datasets
        dataset1_path, _, _, _ = get_common_test_dataset()
        dataset2_path, _, _, _ = get_alt_test_dataset()

        # Create hierarchical structure
        (tmp_path / "root").mkdir()
        (tmp_path / "root" / "group1").mkdir()
        (tmp_path / "root" / "group2").mkdir()

        dataset1_file = list(dataset1_path.glob("*.fast_llm_dataset"))[0]
        dataset2_file = list(dataset2_path.glob("*.fast_llm_dataset"))[0]

        # Place datasets in hierarchy
        shutil.copy(dataset1_file, tmp_path / "root" / "dataset_a.fast_llm_dataset")
        shutil.copy(dataset2_file, tmp_path / "root" / "dataset_b.fast_llm_dataset")
        shutil.copy(dataset1_file, tmp_path / "root" / "group1" / "dataset_c.fast_llm_dataset")
        shutil.copy(dataset2_file, tmp_path / "root" / "group2" / "dataset_d.fast_llm_dataset")

        # Run dataset discovery
        output_path = tmp_path / "discovered_config.yaml"
        config = DatasetDiscoveryConfig(
            directory=tmp_path / "root",
            output=output_path,
        )
        config.run()

        # Load and verify the generated config
        with open(output_path) as f:
            generated_config = yaml.safe_load(f)

        # Should create hierarchical blended config
        assert generated_config["type"] == "blended"

        # Root should have 3 items: local group + 2 subdirs
        assert len(generated_config["datasets"]) == 3

        # First item should be local datasets grouped with "_local" suffix
        local_group = generated_config["datasets"][0]
        assert local_group["type"] == "blended"
        assert "_local" in local_group["name"]
        assert len(local_group["datasets"]) == 2

        # Next two should be subdirectory datasets (single dataset each, so memmap type)
        # Check that one is from group1 and one from group2
        subdir_paths = [generated_config["datasets"][1]["path"], generated_config["datasets"][2]["path"]]
        assert any("group1" in path for path in subdir_paths)
        assert any("group2" in path for path in subdir_paths)

    @pytest.mark.slow
    def test_dataset_discovery_e2e_with_ignore_paths(self, tmp_path: pathlib.Path):
        """Test end-to-end discovery with ignore_paths."""
        import shutil

        import yaml

        from tests.utils.dataset import get_common_test_dataset

        # Get test dataset
        dataset_path, _, _, _ = get_common_test_dataset()
        dataset_file = list(dataset_path.glob("*.fast_llm_dataset"))[0]

        # Create directory structure
        (tmp_path / "datasets" / "keep").mkdir(parents=True)
        (tmp_path / "datasets" / "ignore").mkdir(parents=True)

        # Place datasets
        shutil.copy(dataset_file, tmp_path / "datasets" / "keep" / "dataset1.fast_llm_dataset")
        shutil.copy(dataset_file, tmp_path / "datasets" / "ignore" / "dataset2.fast_llm_dataset")

        # Run dataset discovery with ignore_paths
        output_path = tmp_path / "discovered_config.yaml"
        config = DatasetDiscoveryConfig(
            directory=tmp_path / "datasets",
            output=output_path,
            ignore_paths=[pathlib.Path("ignore")],
        )
        config.run()

        # Load and verify the generated config
        with open(output_path) as f:
            content = f.read()
            # Check ignore_paths in header
            assert "ignore_paths:" in content
            assert "ignore" in content

            # Parse YAML
            f.seek(0)
            generated_config = yaml.safe_load(f)

        # Should only include the dataset from "keep" directory
        # Single dataset, so should be memmap (not blended)
        assert generated_config["type"] == "memmap"
        assert "keep" in generated_config["path"]
        assert "ignore" not in generated_config["path"]

    @pytest.mark.slow
    def test_dataset_discovery_e2e_empty_directory(self, tmp_path: pathlib.Path):
        """Test that discovery fails gracefully on empty directory."""
        # Create empty directory
        (tmp_path / "empty").mkdir()

        # Run dataset discovery - should raise ValueError
        output_path = tmp_path / "output.yaml"
        config = DatasetDiscoveryConfig(
            directory=tmp_path / "empty",
            output=output_path,
        )

        with pytest.raises(ValueError, match="No .fast_llm_dataset files found"):
            config.run()
