import json
import logging
import pathlib

from fast_llm.config import Field, config_class
from fast_llm.data.dataset.gpt.memmap import GPTMemmapDataset
from fast_llm.engine.config_utils.runnable import RunnableConfig

logger = logging.getLogger(__name__)


@config_class()
class ConcatenateDatasetConfig(RunnableConfig):
    directory: pathlib.Path = Field()
    output_name: str = Field(default="fast_llm_dataset.json")
    # A lower bound on the number of tokens in a dataset.
    # Normally we would like each dataset split to contain at least a few samples,
    # i.e. we want num_tokens >= sequence_length * min_split * min_samples_per_split.
    # For example with a (999, 1, 0) split , 8K sequence length, we need at least 8M tokens
    # for a single validation sample, possibly more if the split is imperfect.
    min_tokens: int | None = Field(default=None)

    def run(self):
        self.to_logs()
        assert self.directory.is_dir()
        output_file = self.directory / self.output_name
        assert not output_file.exists(), str(output_file)
        datasets = []

        logger.info(f"Loading datasets from {self.directory}")
        for path in self.directory.glob("**/*.idx"):
            prefix = path.with_suffix("")
            logger.info(str(prefix))
            dataset = GPTMemmapDataset("dataset", prefix)
            dataset_dict = {
                "prefix": str(prefix.relative_to(self.directory)),
                "num_documents": dataset.num_documents,
                "num_tokens": dataset.num_tokens,
            }
            if self.min_tokens is not None and dataset_dict["num_tokens"] < self.min_tokens:
                logger.info(
                    f"Ignoring dataset {dataset_dict['prefix']} with {dataset_dict['num_tokens']:,} tokens"
                    f" (requiring at least {self.min_tokens:,} tokens)"
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


if __name__ == "__main__":
    ConcatenateDatasetConfig.parse_and_run()
