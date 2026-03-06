import logging
import pathlib

import yaml

from fast_llm.data.dataset.memmap.memmap import MemmapDataset
from fast_llm.data.preparation.config import DatasetPreparator
from fast_llm.data.preparation.dataset_discovery.config import DatasetDiscoveryConfig

logger = logging.getLogger(__name__)


class DatasetDiscoveryPreparator[ConfigType: DatasetDiscoveryConfig](DatasetPreparator[ConfigType]):
    """
    Preparator for discovering datasets by scanning .fast_llm_dataset files.

    Scans a directory tree for .fast_llm_dataset files and reads token counts
    from their binary headers to generate a hierarchical blended config.
    """

    _config: DatasetDiscoveryConfig
    _directory: pathlib.Path
    _ignore_paths: set[pathlib.Path]

    def run(self) -> None:
        """
        Run the dataset discovery preparator.
        """
        # Generate the hierarchical config by finding .fast_llm_dataset files
        self._directory = self._config.directory.resolve()
        if not self._directory.is_dir():
            raise ValueError(f"Path is not a directory: {self._directory}")

        logger.info(f"Discovering .fast_llm_dataset files in {self._directory}...")

        self._ignore_paths = {(self._directory / ignore_path).resolve() for ignore_path in self._config.ignore_paths}

        if self._ignore_paths:
            logger.info(f"Ignoring {len(self._ignore_paths)} path(s):")
            for ignore_path in self._ignore_paths:
                logger.info(f"  - {ignore_path}")

        # Create hierarchical config
        config, total_tokens = self._create_directory_config(self._directory)

        if config is None:
            raise ValueError("No valid dataset file found.")

        logger.info(f"Total tokens across all datasets: {total_tokens:,}")

        # Write the config to the output file with header comment
        self._config.output.parent.mkdir(parents=True, exist_ok=True)
        with open(self._config.output, "w") as f:
            # Write header comment
            f.write(
                "# This file was generated with fast_llm.data.preparator.dataset_discovery; "
                "weights are token-counts in billions.\n"
            )
            f.write(f"# Configuration:\n")
            f.write(f"#   directory: {self._directory}\n")
            if self._ignore_paths:
                f.write(f"#   ignore_paths:\n")
                for ignore_path in self._ignore_paths:
                    f.write(f"#     - {ignore_path.relative_to(self._directory)}\n")
            f.write("\n")
            # Write the YAML config
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Generated dataset config saved to {self._config.output}")

        logger.info(f"\nGenerated config: \n{yaml.safe_dump(config, default_flow_style=False, sort_keys=False)}")

    def _create_directory_config(
        self,
        directory: pathlib.Path,
    ) -> tuple[dict | None, float]:
        """
        Recursively create a blended config for a directory and its subdirectories.
        """
        local_datasets = []
        local_tokens = []
        all_datasets = []
        all_tokens = []

        # Collect dataset files directly in this directory (not in subdirectories)
        for subpath in directory.iterdir():
            if any(subpath.is_relative_to(ignore_path) for ignore_path in self._ignore_paths):
                continue
            if subpath.is_dir():
                subdir_config, subdir_token_count = self._create_directory_config(subpath)
                if subdir_config is not None:
                    all_datasets.append(subdir_config)
                    all_tokens.append(subdir_token_count)
            elif subpath.is_file():
                if subpath.suffix != ".fast_llm_dataset":
                    continue
                try:
                    num_tokens = MemmapDataset("", subpath).num_tokens
                    if num_tokens == 0:
                        raise ValueError(f"Dataset is empty")
                except Exception as e:
                    logger.warning(f"Failed to read memmap file {subpath}: {e}")
                else:
                    logger.info(f"{subpath.relative_to(self._directory)}: {num_tokens:,} tokens")
                    local_datasets.append({"type": "memmap", "path": str(subpath)})
                    local_tokens.append(num_tokens)
            else:
                logger.warning(f"Failed to read path {subpath}")

        # Generate a name for a directory relative to root.
        directory_name = (
            str(directory.relative_to(self._directory)).replace("/", "_").replace(".", self._directory.name)
        )

        if local_datasets:
            all_tokens.append(sum(local_tokens))
            all_datasets.append(
                {
                    "type": "blended",
                    "name": directory_name + "_local" if all_datasets else directory_name,
                    "datasets": local_datasets,
                    "weights": local_tokens,
                }
                if len(local_datasets) > 1
                else local_datasets[0]
            )

        if len(all_datasets) > 1:
            return {
                "type": "blended",
                "name": directory_name,
                "datasets": all_datasets,
                "weights": all_tokens,
            }, sum(all_tokens)
        elif len(all_datasets) == 1:
            return all_datasets[0], all_tokens[0]
        else:
            return None, 0
