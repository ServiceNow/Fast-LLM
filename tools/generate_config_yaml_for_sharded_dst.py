import argparse
import pathlib

import yaml

from fast_llm.data.dataset.gpt.config import GPTMemmapDatasetConfig
from fast_llm.data.dataset.gpt.memmap import GPTMemmapDataset
from fast_llm.data.preparator.gpt_memmap.prepare import GPTMemmapDatasetPreparator, GPTMemmapDatasetPreparatorConfig

"""
This script is intended to be used only for creation of fast_llm_config.yaml files for sharded datasets encoded with older version of the prepare command.
"""


def read_dataset_shard_config(shard_path):
    """
    Read a dataset shard from the given path.

    Args:
        shard_path: Path to the shard prefix (without .idx or .bin extension)

    Returns:
        A GPTMemmapDataset instance
    """
    # Convert to pathlib.Path if it's a string
    path = pathlib.Path(shard_path) if isinstance(shard_path, str) else shard_path

    # Create a GPTMemmapDataset instance
    # The name parameter is just for identification
    dataset = GPTMemmapDataset(name=path.name, prefix=path)

    # Print basic information about the dataset
    print(f"Dataset: {dataset.name}")
    print(f"Number of documents: {dataset._num_documents}")
    print(f"Number of tokens: {dataset.num_tokens}")

    return GPTMemmapDatasetConfig.from_dict(
        {
            "type": "memmap",
            "path": path.name.replace(".bin", ""),
            "num_documents": dataset._num_documents,
            "num_tokens": dataset.num_tokens,
        }
    )


def get_preparator(prepare_config: GPTMemmapDatasetPreparatorConfig) -> GPTMemmapDatasetPreparator:
    config = GPTMemmapDatasetPreparatorConfig.from_dict(
        {
            "output_path": prepare_config.output_path,
            "dataset": {"path": prepare_config.dataset.path},
            "tokenizer": {"path": prepare_config.tokenizer.path},
        },
        {},
    )
    return config.get_dataset_preparator_class()(config=config)


def main(config_dict):
    prepare_config = GPTMemmapDatasetPreparatorConfig.from_dict(config_dict)
    destination = pathlib.Path(prepare_config.output_path)

    shards = list(destination.glob("shard_*.bin"))
    dataset_configs = [read_dataset_shard_config(shard) for shard in shards]

    preparator = get_preparator(prepare_config)
    preparator.generate_config_yaml_for_sharded_dst(dataset_configs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate config YAML for sharded datasets")
    parser.add_argument(
        "--prepare_config",
        type=str,
        required=False,
        default=None,  # "/home/toolkit/dev/Fast-LLM/.vscode/prepare_dst.yaml",
        help="Path to the prepare config YAML file",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=False,
        default="/mnt/datasets/tokenized/Mistral-Nemo-Base-2407/FineWeb2/deu_Latn/",
        help="Path to the dataset path",
    )
    args = parser.parse_args()

    if args.prepare_config:
        with open(args.prepare_config) as f:
            config_dict = yaml.safe_load(f)
    else:
        assert args.dataset_path is not None, "Please provide a prepare config YAML file or dataset path"
        config_dict = {
            "output_path": args.dataset_path,
            "dataset": {"path": "unknown"},
            "tokenizer": {"path": "no_tokenizer"},
        }
    main(config_dict)
