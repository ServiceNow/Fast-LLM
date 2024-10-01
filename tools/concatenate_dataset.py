import json
import logging
import pathlib

from fast_llm.config import Config, Field, config_class
from fast_llm.data.mmap import MMapIndexedDataset
from fast_llm.engine.config_utils.logging import configure_logging

logger = logging.getLogger(__name__)


@config_class()
class ConcatenateDatasetConfig(Config):
    directory: pathlib.Path = Field()
    output_name: str = Field(default="fast_llm_dataset.json")
    # A lower bound on the number of tokens in a dataset.
    # Normally we would like each dataset split to contain at least a few samples,
    # i.e. we want num_tokens >= sequence_length * min_split * min_samples_per_split.
    # For example with a (999, 1, 0) split , 8K sequence length, we need at least 8M tokens
    # for a single validation sample, possibly more if the split is imperfect.
    min_tokens: int | None = Field(default=None)


def concatenate_dataset(config: ConcatenateDatasetConfig):
    config.to_logs()
    assert config.directory.is_dir()
    output_file = config.directory / config.output_name
    assert not output_file.exists(), str(output_file)
    datasets = []

    logger.info(f"Loading datasets from {config.directory}")
    for path in config.directory.glob("**/*.idx"):
        prefix = path.with_suffix("")
        logger.info(str(prefix))
        dataset = MMapIndexedDataset(prefix)
        dataset_dict = {
            "prefix": str(prefix.relative_to(config.directory)),
            "num_documents": dataset.num_documents,
            "num_tokens": dataset.num_tokens,
        }
        if config.min_tokens is not None and dataset_dict["num_tokens"] < config.min_tokens:
            logger.info(
                f"Ignoring dataset {dataset_dict['prefix']} with {dataset_dict['num_tokens']:,} tokens"
                f" (requiring at least {config.min_tokens:,} tokens)"
            )
        else:
            datasets.append(dataset_dict)
    total_documents = sum(dataset["num_documents"] for dataset in datasets)
    total_tokens = sum(dataset["num_tokens"] for dataset in datasets)
    logger.info(f"Found {total_documents:,} documents, {total_tokens:,} tokens in {len(datasets)} dataset files")
    for dataset in datasets:
        dataset["weight"] = dataset["num_tokens"] / total_tokens
        logger.info(
            f'{dataset["prefix"]}: documents = {dataset["num_documents"]:,}, tokens = {dataset["num_tokens"]:,}, weight = {dataset["weight"]:.6f}'
        )
    logger.info(f"Saving merged dataset to {output_file}")
    json.dump({"datasets": datasets}, output_file.open("w"))


def main(args=None):
    configure_logging()
    config: ConcatenateDatasetConfig = ConcatenateDatasetConfig.from_flat_args(args)
    concatenate_dataset(config)


if __name__ == "__main__":
    main()
