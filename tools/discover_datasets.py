"""
Tool to recursively discover datasets in a directory and generate a blended dataset config.

This tool walks through a directory tree, identifies datasets by their fast_llm_config*.yaml files,
and generates a config file that blends all discovered datasets with weights proportional to token counts.
"""

import argparse
import logging
import pathlib
from collections import defaultdict

import yaml

from fast_llm.config import Field, config_class
from fast_llm.engine.config_utils.runnable import RunnableConfig

logger = logging.getLogger(__name__)


def _is_subpath(path: pathlib.Path, parent: pathlib.Path) -> bool:
    """Check if path is under parent directory."""
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def find_dataset_configs(root_dir: pathlib.Path, ignore_paths: list[pathlib.Path] | None = None) -> list[pathlib.Path]:
    """
    Recursively find all fast_llm_config*.yaml files in the directory tree.

    Args:
        root_dir: Root directory to search
        ignore_paths: List of paths to ignore (can be absolute or relative to root_dir)

    Returns:
        List of paths to fast_llm_config*.yaml files
    """
    # Normalize ignore paths to absolute paths
    ignore_paths_absolute = set()
    if ignore_paths:
        for ignore_path in ignore_paths:
            if ignore_path.is_absolute():
                ignore_paths_absolute.add(ignore_path.resolve())
            else:
                ignore_paths_absolute.add((root_dir / ignore_path).resolve())

    # Find all fast_llm_config*.yaml files and filter out ignored ones
    config_files = []
    for config_file in root_dir.rglob("fast_llm_config*.yaml"):
        config_file_resolved = config_file.resolve()

        # Check if this file is under any ignored path
        is_ignored = any(_is_subpath(config_file_resolved, ignore_path) for ignore_path in ignore_paths_absolute)

        if not is_ignored:
            config_files.append(config_file)

    # Sort by path for consistent ordering
    return sorted(config_files)


def load_dataset_config(config_path: pathlib.Path) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def _read_memmap_num_tokens(memmap_path: pathlib.Path) -> int:
    """Read number of tokens from a memmap file."""
    import json

    from fast_llm.data.dataset.memmap import FILE_HEADER
    from fast_llm.data.sample.abstract import MemmapIndexDatasetReaderConfig

    if not memmap_path.exists():
        logger.warning(f"Memmap file not found: {memmap_path}")
        return 0

    try:
        with memmap_path.open("rb") as stream:
            header = stream.read(len(FILE_HEADER))
            if header != FILE_HEADER:
                logger.warning(f"Invalid memmap file format: {memmap_path}")
                return 0
            stream.seek(int.from_bytes(stream.read(8), signed=False))
            config_bytes = stream.read(int.from_bytes(stream.read(4), signed=False))
            reader_config = MemmapIndexDatasetReaderConfig.from_dict(json.loads(config_bytes.decode("utf-8")))
            return reader_config.num_tokens
    except Exception as e:
        logger.warning(f"Failed to read memmap file {memmap_path}: {e}")
        return 0


def _resolve_path(path: str | pathlib.Path, relative_to: pathlib.Path) -> pathlib.Path:
    """Resolve a path relative to a base directory if not absolute."""
    path = pathlib.Path(path)
    return path if path.is_absolute() else relative_to / path


def _get_config_num_tokens(config_dict: dict, base_dir: pathlib.Path) -> int:
    """Get number of tokens from a config dict (handles inline configs recursively)."""
    dataset_type = config_dict.get("type")

    if dataset_type == "file":
        file_path = _resolve_path(config_dict["path"], base_dir)
        return get_dataset_num_tokens(file_path)

    if dataset_type == "memmap":
        memmap_path = _resolve_path(config_dict.get("path", ""), base_dir)
        return _read_memmap_num_tokens(memmap_path)

    if dataset_type in ["blended", "sampled", "concatenated"]:
        return sum(_get_config_num_tokens(sub, base_dir) for sub in config_dict.get("datasets", []))

    if dataset_type == "slice":
        base_config = config_dict.get("dataset", {})
        begin = config_dict.get("begin", 0)
        end = config_dict.get("end", 1)
        base_tokens = _get_config_num_tokens(base_config, base_dir)
        return int(base_tokens * (end - begin))

    logger.warning(f"Unsupported inline config type '{dataset_type}'")
    return 0


def get_dataset_num_tokens(config_path: pathlib.Path) -> int:
    """
    Load a dataset config and get its number of tokens.

    Args:
        config_path: Path to the dataset config file

    Returns:
        Number of tokens in the dataset
    """
    # Import preprocessing and sample configs to register them
    import fast_llm.data.preprocessing.image_patch  # noqa
    import fast_llm.data.preprocessing.language_model  # noqa
    import fast_llm.data.sample.language_model  # noqa
    import fast_llm.data.sample.patch  # noqa
    import fast_llm.data.sample.range  # noqa
    import fast_llm.data.sample.token  # noqa

    config_dict = load_dataset_config(config_path)
    return _get_config_num_tokens(config_dict, config_path.parent)


def _get_token_count(config_file: pathlib.Path) -> float:
    """
    Get token count in billions for a dataset config file.
    """
    num_tokens = get_dataset_num_tokens(config_file)
    logger.info(f"  - {config_file.name}: {num_tokens:,} tokens")
    return num_tokens / 1e9


def _create_dataset_reference(config_file: pathlib.Path, use_file_refs: bool) -> dict:
    """
    Create a dataset reference or inline config.

    Args:
        config_file: Path to the dataset config file
        use_file_refs: If True, create a file reference; if False, inline the config

    Returns:
        Dictionary representing the dataset
    """
    if use_file_refs:
        return {"type": "file", "path": str(config_file)}
    else:
        return load_dataset_config(config_file)


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


def create_blended_config(
    config_files: list[pathlib.Path],
    name: str = "blended",
    use_file_refs: bool = True,
) -> dict:
    """
    Create a blended dataset config from a list of config files.

    Args:
        config_files: List of paths to dataset config files
        name: Name for the blended dataset
        use_file_refs: If True, use file references (type: file, path: ...).
                      If False, inline the full configs.

    Returns:
        Dictionary representing a blended dataset config
    """
    if len(config_files) == 0:
        raise ValueError("No config files provided")

    if len(config_files) == 1:
        # If only one dataset, just reference it directly
        if use_file_refs:
            return {
                "type": "file",
                "path": str(config_files[0]),
            }
        else:
            return load_dataset_config(config_files[0])

    # Build datasets and weights in a single pass
    logger.info("Calculating token counts for blended dataset weights...")
    datasets = []
    weights = []

    for config_file in config_files:
        # Add dataset reference or inline config
        if use_file_refs:
            datasets.append({"type": "file", "path": str(config_file)})
        else:
            datasets.append(load_dataset_config(config_file))

        # Get token count for weight
        weights.append(_get_token_count(config_file))

    return {
        "type": "blended",
        "name": name,
        "datasets": datasets,
        "weights": weights,
    }


def group_configs_by_directory(config_files: list[pathlib.Path]) -> dict[pathlib.Path, list[pathlib.Path]]:
    """
    Group config files by their parent directory.

    Args:
        config_files: List of config file paths

    Returns:
        Dictionary mapping directory paths to lists of config files in that directory
    """
    groups: dict[pathlib.Path, list[pathlib.Path]] = defaultdict(list)
    for config_file in config_files:
        groups[config_file.parent].append(config_file)

    return dict(groups)


def build_directory_tree(
    groups: dict[pathlib.Path, list[pathlib.Path]], root_dir: pathlib.Path
) -> dict[pathlib.Path, set[pathlib.Path]]:
    """
    Build a tree structure of directories showing parent-child relationships.

    Args:
        groups: Dictionary mapping directories to their config files
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


def create_directory_config(
    directory: pathlib.Path,
    groups: dict[pathlib.Path, list[pathlib.Path]],
    tree: dict[pathlib.Path, set[pathlib.Path]],
    root_dir: pathlib.Path,
    use_file_refs: bool,
) -> tuple[dict, float] | None:
    """
    Recursively create a blended config for a directory and its subdirectories.

    Args:
        directory: Current directory to process
        groups: Dictionary mapping directories to their config files
        tree: Directory tree structure
        root_dir: Root directory
        use_file_refs: Whether to use file references

    Returns:
        Tuple of (config dictionary, total token count in billions), or None if directory has no datasets
    """
    local_datasets = []
    local_tokens = []

    # Collect configs directly in this directory (not in subdirectories)
    if directory in groups:
        for config_file in sorted(groups[directory]):
            local_datasets.append(_create_dataset_reference(config_file, use_file_refs))
            local_tokens.append(_get_token_count(config_file))

    # Recursively process subdirectories
    subdir_datasets = []
    subdir_tokens = []
    if directory in tree:
        for subdir in sorted(tree[directory]):
            subdir_result = create_directory_config(subdir, groups, tree, root_dir, use_file_refs)
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
                "name": _get_directory_name(directory, root_dir, "_local"),
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
        "name": _get_directory_name(directory, root_dir),
        "datasets": all_datasets,
        "weights": all_tokens,
    }, total_tokens


def create_hierarchical_config(
    root_dir: pathlib.Path,
    use_file_refs: bool = True,
    ignore_paths: list[pathlib.Path] | None = None,
) -> dict:
    """
    Create a hierarchical blended dataset config from all datasets in a directory.

    Datasets in the same directory are grouped together with weights proportional to token counts,
    and these groups are nested following the directory structure.

    Args:
        root_dir: Root directory to search for datasets
        use_file_refs: If True, use file references (type: file).
                      If False, inline the full configs.
        ignore_paths: List of paths to ignore (can be absolute or relative to root_dir)

    Returns:
        Dictionary representing the hierarchical blended dataset config
    """
    logger.info(f"Discovering datasets in {root_dir}...")

    if ignore_paths:
        logger.info(f"Ignoring {len(ignore_paths)} path(s):")
        for ignore_path in ignore_paths:
            logger.info(f"  - {ignore_path}")

    config_files = find_dataset_configs(root_dir, ignore_paths=ignore_paths)

    if not config_files:
        raise ValueError(f"No fast_llm_config*.yaml files found in {root_dir}")

    logger.info(f"Found {len(config_files)} dataset config(s):")
    for config_file in config_files:
        logger.info(f"  - {config_file.relative_to(root_dir)}")

    # Group configs by directory
    groups = group_configs_by_directory(config_files)

    # Build directory tree
    tree = build_directory_tree(groups, root_dir)

    # Create hierarchical config
    result = create_directory_config(root_dir, groups, tree, root_dir, use_file_refs)

    if result is None:
        raise ValueError("Failed to create config")

    config, total_tokens = result

    logger.info(f"Total tokens across all datasets: {total_tokens:,}")

    return config


@config_class()
class DiscoverDatasetsConfig(RunnableConfig):
    """
    Configuration for the dataset discovery tool.
    """

    directory: pathlib.Path = Field(desc="Directory to search for datasets recursively")
    output: pathlib.Path = Field(desc="Output path for the generated config YAML file")
    use_file_refs: bool = Field(default=True, desc="Use file references (type: file) instead of inlining configs")
    ignore_paths: list[pathlib.Path] = Field(
        default_factory=list,
        desc="List of paths to ignore during dataset discovery (can be absolute or relative to directory)",
    )

    def run(self):
        """
        Run the dataset discovery tool.
        """
        # Validate directory exists
        if not self.directory.exists():
            raise ValueError(f"Directory does not exist: {self.directory}")

        if not self.directory.is_dir():
            raise ValueError(f"Path is not a directory: {self.directory}")

        # Generate the hierarchical config
        config = create_hierarchical_config(
            self.directory.resolve(),
            use_file_refs=self.use_file_refs,
            ignore_paths=self.ignore_paths,
        )

        # Write the config to the output file with header comment
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output, "w") as f:
            # Write header comment
            f.write(
                "# This file was generated with tools/discover_datasets.py; weights are token-counts in billions.\n"
            )
            f.write(f"# Configuration:\n")
            f.write(f"#   directory: {self.directory}\n")
            f.write(f"#   use_file_refs: {self.use_file_refs}\n")
            if self.ignore_paths:
                f.write(f"#   ignore_paths:\n")
                for ignore_path in self.ignore_paths:
                    f.write(f"#     - {ignore_path}\n")
            f.write("\n")
            # Write the YAML config
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Generated dataset config saved to {self.output}")

        # Print a preview of the config
        logger.info("\nGenerated config preview:")
        preview = yaml.safe_dump(config, default_flow_style=False, sort_keys=False)
        for line in preview.split("\n")[:50]:  # Show first 50 lines
            logger.info(line)

        if len(preview.split("\n")) > 50:
            logger.info("... (truncated)")


def main():
    """
    Command-line entry point.
    """
    parser = argparse.ArgumentParser(description="Discover datasets and generate hierarchical blended config")
    parser.add_argument("directory", type=pathlib.Path, help="Directory to search for datasets recursively")
    parser.add_argument(
        "-o", "--output", type=pathlib.Path, required=True, help="Output path for the generated config YAML file"
    )
    parser.add_argument("--no-file-refs", action="store_true", help="Inline configs instead of using file references")
    parser.add_argument(
        "--ignore",
        type=pathlib.Path,
        action="append",
        dest="ignore_paths",
        help="Path to ignore during dataset discovery (can be specified multiple times)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Create and run the config
    config = DiscoverDatasetsConfig(
        directory=args.directory,
        output=args.output,
        use_file_refs=not args.no_file_refs,
        ignore_paths=args.ignore_paths or [],
    )
    config.run()


if __name__ == "__main__":
    # Support both CLI usage and Fast-LLM's config system
    import sys

    # Check if using argparse-style CLI (positional arg without --config)
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-") and sys.argv[1] != "--config":
        main()
    else:
        DiscoverDatasetsConfig.parse_and_run()
