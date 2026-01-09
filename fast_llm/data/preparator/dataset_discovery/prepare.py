"""
Dataset discovery preparator.

This module discovers datasets by directly scanning for .fast_llm_dataset files
and reading token counts from their binary headers.
"""

import logging
import pathlib
from collections import defaultdict

import yaml

from fast_llm.data.dataset.memmap import MemmapDataset
from fast_llm.data.preparator.config import DatasetPreparator
from fast_llm.data.preparator.dataset_discovery.config import DatasetDiscoveryConfig

logger = logging.getLogger(__name__)


class DatasetDiscoveryPreparator[ConfigType: DatasetDiscoveryConfig](DatasetPreparator[ConfigType]):
    """
    Preparator for discovering datasets by scanning .fast_llm_dataset files.

    Scans a directory tree for .fast_llm_dataset files and reads token counts
    from their binary headers to generate a hierarchical blended config.
    """

    _config: DatasetDiscoveryConfig

    def run(self) -> None:
        """
        Run the dataset discovery preparator.
        """
        # Generate the hierarchical config by finding .fast_llm_dataset files
        config = self._create_hierarchical_config(
            self._config.directory.resolve(),
            ignore_paths=self._config.ignore_paths,
        )

        # Write the config to the output file with header comment
        self._config.output.parent.mkdir(parents=True, exist_ok=True)
        with open(self._config.output, "w") as f:
            # Write header comment
            f.write(
                "# This file was generated with fast_llm.data.preparator.dataset_discovery; "
                "weights are token-counts in billions.\n"
            )
            f.write(f"# Configuration:\n")
            f.write(f"#   directory: {self._config.directory}\n")
            if self._config.ignore_paths:
                f.write(f"#   ignore_paths:\n")
                for ignore_path in self._config.ignore_paths:
                    f.write(f"#     - {ignore_path}\n")
            f.write("\n")
            # Write the YAML config
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Generated dataset config saved to {self._config.output}")

        # Print a preview of the config
        logger.info("\nGenerated config preview:")
        preview = yaml.safe_dump(config, default_flow_style=False, sort_keys=False)
        for line in preview.split("\n")[:50]:
            logger.info(line)

        if len(preview.split("\n")) > 50:
            logger.info("... (truncated)")

    @staticmethod
    def _is_subpath(path: pathlib.Path, parent: pathlib.Path) -> bool:
        """Check if path is under parent directory."""
        try:
            path.relative_to(parent)
            return True
        except ValueError:
            return False

    def _find_dataset_files(
        self, root_dir: pathlib.Path, ignore_paths: list[pathlib.Path] | None = None
    ) -> list[pathlib.Path]:
        """
        Recursively find all .fast_llm_dataset files in the directory tree.

        Args:
            root_dir: Root directory to search
            ignore_paths: List of paths to ignore (can be absolute or relative to root_dir)

        Returns:
            List of paths to .fast_llm_dataset files
        """
        # Normalize ignore paths to absolute paths
        ignore_paths_absolute = set()
        if ignore_paths:
            for ignore_path in ignore_paths:
                if ignore_path.is_absolute():
                    ignore_paths_absolute.add(ignore_path.resolve())
                else:
                    ignore_paths_absolute.add((root_dir / ignore_path).resolve())

        # Find all .fast_llm_dataset files and filter out ignored ones
        dataset_files = []
        for dataset_file in root_dir.rglob("*.fast_llm_dataset"):
            dataset_file_resolved = dataset_file.resolve()

            # Check if this file is under any ignored path
            is_ignored = any(
                self._is_subpath(dataset_file_resolved, ignore_path) for ignore_path in ignore_paths_absolute
            )

            if not is_ignored:
                dataset_files.append(dataset_file)

        # Sort by path for consistent ordering
        return sorted(dataset_files)

    @staticmethod
    def _read_memmap_num_tokens(memmap_path: pathlib.Path) -> int:
        """Read number of tokens from a .fast_llm_dataset memmap file."""

        if not memmap_path.exists():
            logger.warning(f"Memmap file not found: {memmap_path}")
            return 0

        try:
            reader_config = MemmapDataset.read_reader_config(memmap_path)
            return reader_config.num_tokens
        except Exception as e:
            logger.warning(f"Failed to read memmap file {memmap_path}: {e}")
            return 0

    def _get_token_count(self, dataset_file: pathlib.Path) -> float | None:
        """
        Get token count in billions for a .fast_llm_dataset file.

        Returns:
            Token count in billions, or None if the file couldn't be read
        """
        num_tokens = self._read_memmap_num_tokens(dataset_file)
        if num_tokens == 0:
            logger.warning(f"  - {dataset_file.name}: skipping (0 tokens or read error)")
            return None
        logger.debug(f"  - {dataset_file.name}: {num_tokens:,} tokens")
        return num_tokens / 1e9

    def _create_memmap_config_for_dataset(self, dataset_file: pathlib.Path) -> dict:
        """
        Create a memmap config dictionary for a .fast_llm_dataset file.

        Args:
            dataset_file: Path to the .fast_llm_dataset file

        Returns:
            Dictionary representing a memmap dataset config
        """
        return {"type": "memmap", "path": str(dataset_file)}

    @staticmethod
    def _get_directory_name(directory: pathlib.Path, root_dir: pathlib.Path, suffix: str = "") -> str:
        """
        Generate a name for a directory relative to root.

        Args:
            directory: The directory to name
            root_dir: The root directory
            suffix: Optional suffix to append to the name

        Returns:
            A string name for the directory
        """
        rel_path = directory.relative_to(root_dir) if directory != root_dir else pathlib.Path(".")
        base_name = str(rel_path).replace("/", "_").replace(".", root_dir.name)
        return f"{base_name}{suffix}" if suffix else base_name

    @staticmethod
    def _group_files_by_directory(dataset_files: list[pathlib.Path]) -> dict[pathlib.Path, list[pathlib.Path]]:
        """
        Group dataset files by their parent directory.

        Args:
            dataset_files: List of dataset file paths

        Returns:
            Dictionary mapping directory paths to lists of dataset files in that directory
        """
        groups: dict[pathlib.Path, list[pathlib.Path]] = defaultdict(list)
        for dataset_file in dataset_files:
            groups[dataset_file.parent].append(dataset_file)

        return dict(groups)

    @staticmethod
    def _build_directory_tree(
        groups: dict[pathlib.Path, list[pathlib.Path]], root_dir: pathlib.Path
    ) -> dict[pathlib.Path, set[pathlib.Path]]:
        """
        Build a tree structure of directories showing parent-child relationships.

        Args:
            groups: Dictionary mapping directories to their dataset files
            root_dir: Root directory

        Returns:
            Dictionary mapping each directory to its immediate child directories
        """
        tree: dict[pathlib.Path, set[pathlib.Path]] = {root_dir: set()}

        for directory in groups.keys():
            # Add all ancestors to the tree
            current = directory
            while current != root_dir and current.parent != current:
                parent = current.parent
                if parent not in tree:
                    tree[parent] = set()
                if current not in tree:
                    tree[current] = set()
                tree[parent].add(current)
                current = parent

        return tree

    def _create_directory_config(
        self,
        directory: pathlib.Path,
        groups: dict[pathlib.Path, list[pathlib.Path]],
        tree: dict[pathlib.Path, set[pathlib.Path]],
        root_dir: pathlib.Path,
    ) -> tuple[dict, float] | None:
        """
        Recursively create a blended config for a directory and its subdirectories.

        Args:
            directory: Current directory to process
            groups: Dictionary mapping directories to their dataset files
            tree: Directory tree structure
            root_dir: Root directory

        Returns:
            Tuple of (config dictionary, total token count in billions), or None if directory has no datasets
        """
        local_datasets = []
        local_tokens = []

        # Collect dataset files directly in this directory (not in subdirectories)
        if directory in groups:
            for dataset_file in sorted(groups[directory]):
                token_count = self._get_token_count(dataset_file)
                if token_count is not None:  # Skip files that couldn't be read
                    local_datasets.append(self._create_memmap_config_for_dataset(dataset_file))
                    local_tokens.append(token_count)

        # Recursively process subdirectories
        subdir_datasets = []
        subdir_tokens = []
        if directory in tree:
            for subdir in sorted(tree[directory]):
                subdir_result = self._create_directory_config(subdir, groups, tree, root_dir)
                if subdir_result is not None:
                    subdir_config, subdir_token_count = subdir_result
                    subdir_datasets.append(subdir_config)
                    subdir_tokens.append(subdir_token_count)

        # Combine local and subdirectory datasets
        if local_datasets and subdir_datasets:
            # If multiple local datasets, group them together
            if len(local_datasets) > 1:
                local_total_tokens = sum(local_tokens)
                local_group = {
                    "type": "blended",
                    "name": self._get_directory_name(directory, root_dir, "_local"),
                    "datasets": local_datasets,
                    "weights": local_tokens,
                }
                all_datasets = [local_group] + subdir_datasets
                all_tokens = [local_total_tokens] + subdir_tokens
            else:
                all_datasets = local_datasets + subdir_datasets
                all_tokens = local_tokens + subdir_tokens
        elif local_datasets:
            all_datasets = local_datasets
            all_tokens = local_tokens
        elif subdir_datasets:
            all_datasets = subdir_datasets
            all_tokens = subdir_tokens
        else:
            return None

        total_tokens = sum(all_tokens)

        # Don't wrap a single dataset
        if len(all_datasets) == 1:
            return all_datasets[0], total_tokens

        # Multiple datasets - create blended config
        return {
            "type": "blended",
            "name": self._get_directory_name(directory, root_dir),
            "datasets": all_datasets,
            "weights": all_tokens,
        }, total_tokens

    def _create_hierarchical_config(
        self,
        root_dir: pathlib.Path,
        ignore_paths: list[pathlib.Path] | None = None,
    ) -> dict:
        """
        Create a hierarchical blended dataset config from all .fast_llm_dataset files in a directory.

        Datasets in the same directory are grouped together with weights proportional to token counts,
        and these groups are nested following the directory structure.

        Args:
            root_dir: Root directory to search for datasets
            ignore_paths: List of paths to ignore (can be absolute or relative to root_dir)

        Returns:
            Dictionary representing the hierarchical blended dataset config
        """
        logger.info(f"Discovering .fast_llm_dataset files in {root_dir}...")

        if ignore_paths:
            logger.info(f"Ignoring {len(ignore_paths)} path(s):")
            for ignore_path in ignore_paths:
                logger.info(f"  - {ignore_path}")

        dataset_files = self._find_dataset_files(root_dir, ignore_paths=ignore_paths)

        if not dataset_files:
            raise ValueError(f"No .fast_llm_dataset files found in {root_dir}")

        logger.debug(f"Found {len(dataset_files)} dataset file(s):")
        for dataset_file in dataset_files:
            logger.debug(f"  - {dataset_file.relative_to(root_dir)}")

        # Group dataset files by directory
        groups = self._group_files_by_directory(dataset_files)

        # Build directory tree
        tree = self._build_directory_tree(groups, root_dir)

        # Create hierarchical config
        result = self._create_directory_config(root_dir, groups, tree, root_dir)

        if result is None:
            raise ValueError("Failed to create config")

        config, total_tokens = result

        logger.info(f"Total tokens across all datasets: {total_tokens:.2f}B")

        return config
