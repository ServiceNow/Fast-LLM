"""
Tool to recursively discover datasets in a directory and generate a blended dataset config.

This tool is a command-line wrapper around the DatasetDiscoveryPreparator.
For programmatic usage, use fast_llm.data.preparator.dataset_discovery directly.
"""

import argparse
import logging
import pathlib

from fast_llm.data.preparator.dataset_discovery import DatasetDiscoveryConfig


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
    config = DatasetDiscoveryConfig(
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
        DatasetDiscoveryConfig.parse_and_run()
