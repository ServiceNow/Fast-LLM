import abc
import argparse
import json
import os
import pathlib
import typing
from multiprocessing import Pool

import numpy as np
import torch.distributed

from fast_llm.config import Config, Field, FieldHint, check_field, config_class
from fast_llm.data.config import TokenizerConfig
from fast_llm.data.gpt.memmap import GPTMemmapDataset
from fast_llm.data.tokenizer import Tokenizer
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.engine.config_utils.runnable import RunnableConfig
from fast_llm.utils import Assert, Registry


@config_class
class DistributedConfig(Config):
    default_world_size: typing.ClassVar[int] = int(os.environ.get("WORLD_SIZE", 1))
    default_rank: typing.ClassVar[int] = int(os.environ.get("RANK", 0))
    world_size: int = Field(
        default=None,
        desc="Size of the world group. Typically provided by torchrun or equivalent through the `WORLD_SIZE` environment variable.",
        hint=FieldHint.expert,
        valid=check_field(Assert.gt, 0),
    )
    rank: int = Field(
        default=None,
        desc="Rank of the local process. Typically provided by torchrun or equivalent through the `RANK` environment variable.",
        hint=FieldHint.expert,
        valid=check_field(Assert.geq, 0),
    )
    backend: str = Field(
        default="gloo",
        desc="Distributed backend to use.",
        hint=FieldHint.optional,
        valid=check_field(Assert.incl, torch.distributed.Backend.backend_list),
    )

    def _validate(self):
        if self.world_size is None:
            self.world_size = self.default_world_size
        if self.rank is None:
            self.rank = self.default_rank
        super()._validate()
        Assert.in_range(self.rank, 0, self.world_size)


@config_class()
class DatasetPreparatorConfig(RunnableConfig):
    _abstract = True
    model_name: typing.ClassVar[str]

    output_path: pathlib.Path = Field(
        desc="Output directory for the processed dataset.",
        hint=FieldHint.core,
    )
    distributed: DistributedConfig = Field(
        default_factory=DistributedConfig,
        desc="Configuration for distributed processing.",
        hint=FieldHint.feature,
    )

    @classmethod
    def get_dataset_preparator_class(cls) -> typing.Type["DatasetPreparator"]:
        raise NotImplementedError

    def _get_runnable(self, parsed: argparse.Namespace) -> typing.Callable[[], None]:
        dataset_preparator = self.get_dataset_preparator_class()(config=self)
        return dataset_preparator.run


class DatasetPreparator(abc.ABC):
    _abstract = True
    _config: DatasetPreparatorConfig
    config_class: typing.ClassVar[type[DatasetPreparatorConfig]] = DatasetPreparatorConfig

    def __init__(self, config: DatasetPreparatorConfig) -> None:
        Assert.custom(isinstance, config, self.config_class)
        config.validate()
        self._config = config

    def run(self) -> None:
        raise NotImplementedError


@config_class
class GPTDatasetConfig(Config):
    name_or_path: str = Field(
        desc="Name or path of the dataset.",
        hint=FieldHint.core,
    )
    config_name: None | str = Field(
        default=None,
        desc="Specific configuration name for the dataset.",
        hint=FieldHint.optional,
    )
    split: str = Field(
        default="train",
        desc="Split of the dataset to use.",
        hint=FieldHint.optional,
    )
    field: str = Field(
        default="text",
        desc="Field of the dataset to use.",
        hint=FieldHint.optional,
    )
    data_type: DataType = Field(
        default=None,
        desc="Data type of the dataset field.",
        hint=FieldHint.derived,
    )
    trust_remote_code: bool = Field(
        default=False,
        desc="Trust remote code when downloading the dataset.",
        hint=FieldHint.optional,
    )
    disable_disk_space_check: bool = Field(
        default=False,
        desc="Disable disk space check. Useful for environments where disk space is not accurately reported.",
        hint=FieldHint.optional,
    )


@config_class()
class GPTDatasetPreparatorConfig(DatasetPreparatorConfig):
    _abstract = False
    model_name: typing.ClassVar[str] = "gpt"

    tokens_per_shard: int = Field(
        default=1_000_000_000,
        desc="Approximate number of tokens per shard.",
        hint=FieldHint.feature,
        valid=check_field(Assert.geq, 100_000),
    )
    loading_workers: int = Field(
        default=1,
        desc="Number of workers in load_dataset() call.",
        hint=FieldHint.optional,
        valid=check_field(Assert.geq, 1),
    )
    tokenize_workers: int = Field(
        default=1,
        desc="Number of workers for tokenization.",
        hint=FieldHint.optional,
        valid=check_field(Assert.geq, 1),
    )
    saving_workers: int = Field(
        default=1,
        desc="Number of processes for saving the data.",
        hint=FieldHint.optional,
        valid=check_field(Assert.geq, 1),
    )
    clean_output: bool = Field(
        default=False,
        desc="Remove downloaded dataset after processing.",
        hint=FieldHint.optional,
    )
    dataset: GPTDatasetConfig = Field(
        default_factory=GPTDatasetConfig,
        desc="Configuration for the dataset.",
        hint=FieldHint.feature,
    )
    tokenizer: TokenizerConfig = Field(
        default_factory=TokenizerConfig,
        desc="Configuration for the tokenizer.",
        hint=FieldHint.feature,
    )
    _tokenizer: Tokenizer = Field(
        init=False,
        desc="The tokenizer instance.",
        hint=FieldHint.derived,
    )

    def _validate(self):
        Assert.not_none(self.tokenizer.path)
        self._tokenizer = Tokenizer(config=self.tokenizer)
        if self.dataset.data_type is None:
            # Decide the datatype based on the tokenizer vocabulary size
            vocab_size = self._tokenizer.vocab_size
            if vocab_size <= np.iinfo(np.int16).max:
                self.dataset.data_type = DataType.int16
            # elif vocab_size <= np.iinfo(np.uint16).max:
            #     self.dataset.data_type = DataType.uint16  # Not supported by Fast-LLM's DataType
            elif vocab_size <= np.iinfo(np.int32).max:
                self.dataset.data_type = DataType.int32
            else:
                raise ValueError(f"Tokenizer vocabulary size {vocab_size} is too large. This is likely an error.")
        super()._validate()

    @classmethod
    def get_dataset_preparator_class(cls):
        return GPTDatasetPreparator


class GPTDatasetPreparator(DatasetPreparator):
    _abstract = False
    _config: GPTDatasetPreparatorConfig
    config_class = GPTDatasetPreparatorConfig

    def _tokenize_batch(self, batch):
        input_ids = [
            np.array(self._config._tokenizer.tokenize(text), dtype=self._config.dataset.data_type.numpy)
            for text in batch[self._config.dataset.field]
        ]
        num_tokens = [len(x) for x in input_ids]
        return {
            "input_ids": input_ids,
            "num_tokens": num_tokens,
        }

    def _save_shard(self, args) -> dict:
        from tqdm import tqdm

        shard_idx, shard_dataset = args
        prefix = f"shard_{self._config.distributed.rank}_{shard_idx}"
        shard_output_path = self._config.output_path / prefix
        documents = [
            np.array(item["input_ids"], dtype=self._config.dataset.data_type.numpy)
            for item in tqdm(shard_dataset, desc=f"Saving shard {shard_idx}", unit="docs")
        ]
        GPTMemmapDataset.write_dataset(prefix=shard_output_path, documents=documents)
        dataset_dict = {
            "prefix": prefix,
            "num_documents": len(documents),
            "num_tokens": sum(len(doc) for doc in documents),
        }
        return dataset_dict

    def run(self):
        import datasets
        import transformers
        from tqdm import tqdm

        # Set transformers logging verbosity
        transformers.logging.set_verbosity_error()

        if self._config.dataset.disable_disk_space_check:
            datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory=".": True

        # Initialize distributed processing
        if self._config.distributed.world_size > 1:
            torch.distributed.init_process_group(
                backend=self._config.distributed.backend,
                rank=self._config.distributed.rank,
                world_size=self._config.distributed.world_size,
            )

        # Prepare output directory
        self._config.output_path.mkdir(parents=True, exist_ok=True)

        # Download dataset if necessary on rank 0
        download_path = self._config.output_path / "downloaded_dataset"
        download_path_ok = download_path / "ok"
        if self._config.distributed.rank == 0 and not download_path_ok.exists():
            datasets.load_dataset(
                path=self._config.dataset.name_or_path,
                name=self._config.dataset.config_name,
                split=self._config.dataset.split,
                num_proc=self._config.loading_workers,
                trust_remote_code=self._config.dataset.trust_remote_code,
            ).save_to_disk(download_path, num_proc=self._config.saving_workers)
            download_path_ok.touch()

        # Synchronize processes to wait for the download to finish
        if self._config.distributed.world_size > 1:
            torch.distributed.barrier()

        # Load and shard the dataset on each rank
        dataset = datasets.load_from_disk(download_path).shard(
            num_shards=self._config.distributed.world_size,
            index=self._config.distributed.rank,
        )
        if self._config.dataset.field not in dataset.column_names:
            raise ValueError(f"Dataset does not have field '{self._config.dataset.field}'.")

        # Tokenize the dataset in parallel
        tokenized_dataset = dataset.map(
            self._tokenize_batch,
            batched=True,
            num_proc=self._config.tokenize_workers,
            desc="Tokenizing batches",
        )

        # Calculate total number of tokens
        total_tokens = sum(tqdm(tokenized_dataset["num_tokens"], desc="Counting tokens", unit="tokens"))

        # Split dataset into shards based on number of tokens
        num_shards = int(np.ceil(total_tokens / self._config.tokens_per_shard))
        shards = [
            (i, tokenized_dataset.shard(num_shards=num_shards, index=i))
            for i in tqdm(range(num_shards), desc="Creating shards")
        ]

        # Use multiprocessing to save each shard in parallel on all ranks
        with Pool(processes=self._config.saving_workers) as pool:
            dataset_dicts = pool.map(self._save_shard, shards)

        # Gather dataset_dicts from all ranks to rank 0
        if self._config.distributed.world_size > 1:
            if self._config.distributed.rank == 0:
                all_dataset_dicts = [None] * self._config.distributed.world_size
                torch.distributed.gather_object(dataset_dicts, all_dataset_dicts, dst=0)
                dataset_dicts = [item for sublist in all_dataset_dicts for item in sublist]
            else:
                torch.distributed.gather_object(dataset_dicts, [], dst=0)

        # Create a metadata file on rank 0
        if self._config.distributed.rank == 0:
            total_tokens = sum(dataset_dict["num_tokens"] for dataset_dict in dataset_dicts)
            for dataset_dict in dataset_dicts:
                dataset_dict["weight"] = float(dataset_dict["num_tokens"]) / float(total_tokens)
            output_file = self._config.output_path / "fast_llm_dataset.json"
            json.dump({"datasets": dataset_dicts}, output_file.open("w"))

        # Finalize distributed processing
        if self._config.distributed.world_size > 1:
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
        
        # Clean up downloaded dataset
        if self._config.clean_output and self._config.distributed.rank == 0:
            download_path.unlink(missing_ok=True)


dataset_preparator_registry = Registry(
    "DatasetPreparator",
    {
        dataset_preparator.model_name: dataset_preparator
        for dataset_preparator in [
            GPTDatasetPreparatorConfig,
        ]
    },
)


class PrepareDatasetConfig(RunnableConfig):
    @classmethod
    def _get_parser(cls):
        parser = super()._get_parser()
        parser.add_argument(
            "model_type",
            choices=dataset_preparator_registry.keys(),
            help="The Fast-LLM model type to use. Must be defined in the model registry in `fast_llm.models.auto`.",
        )
        return parser

    @classmethod
    def _from_parsed_args(cls, parsed: argparse.Namespace, unparsed: list[str]):
        return dataset_preparator_registry[parsed.model_type]._from_parsed_args(parsed, unparsed)


if __name__ == "__main__":
    PrepareDatasetConfig.parse_and_run()
