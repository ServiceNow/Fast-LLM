import json
import logging
import multiprocessing
import pathlib
import shutil
import typing

import datasets
import huggingface_hub
import numpy as np
import requests
import torch.distributed
import tqdm
import transformers
import yaml

from fast_llm.data.dataset.gpt.config import (
    GPTBlendedDatasetConfig,
    GPTDatasetSliceConfig,
    GPTIndexedDatasetConfig,
    GPTMemmapDatasetConfig,
)
from fast_llm.data.dataset.gpt.memmap import GPTMemmapDataset
from fast_llm.data.dataset.gpt.sampled import GPTSample
from fast_llm.data.preparator.config import DatasetPreparator
from fast_llm.data.preparator.gpt_memmap.config import GPTMemmapDatasetPreparatorConfig
from fast_llm.data.tokenizer import Tokenizer
from fast_llm.engine.config_utils.data_type import DataType, get_unsigned_integer_type
from fast_llm.utils import Assert, normalize_probabilities, padded_cumsum

logger = logging.getLogger(__name__)


class GPTMemmapDatasetPreparator[ConfigType: GPTMemmapDatasetPreparatorConfig](DatasetPreparator[ConfigType]):
    config_class: typing.ClassVar[type[GPTMemmapDatasetPreparatorConfig]] = GPTMemmapDatasetPreparatorConfig

    _tokenizer: Tokenizer
    _data_type: DataType

    def _tokenize_batch(self, batch: dict[str, list[typing.Any]]) -> dict[str, list[typing.Any]]:
        input_ids = [
            np.array(self._tokenizer.tokenize(text), dtype=self._data_type.numpy)
            for text in batch[self._config.dataset.field]
        ]
        num_tokens = [len(x) for x in input_ids]
        return {
            "input_ids": input_ids,
            "num_tokens": num_tokens,
        }

    def _tokenize_batch_with_spans(self, batch: dict[str, list[typing.Any]]) -> dict[str, list[typing.Any]]:
        input_ids, token_spans = map(
            list,
            zip(
                *[
                    (
                        np.array(input_ids, dtype=self._data_type.numpy),
                        np.array(token_spans, dtype=np.int32).reshape(-1, 2),
                    )
                    for input_ids, token_spans in [
                        self._tokenizer.tokenize_with_spans(text, char_spans)
                        for text, char_spans in zip(
                            batch[self._config.dataset.field], batch[self._config.dataset.loss_masking_spans]
                        )
                    ]
                ]
            ),
        )
        num_tokens = [len(x) for x in input_ids]
        return {
            "input_ids": input_ids,
            "token_spans": token_spans,
            "num_tokens": num_tokens,
        }

    def _save_shard(self, args: tuple[int, datasets.Dataset]) -> GPTMemmapDatasetConfig:
        shard_idx, shard_dataset = args
        prefix = f"shard_{self._config.distributed.rank}_{shard_idx}"
        shard_output_path = self._config.output_path / prefix

        def _document_generator():
            if "token_spans" in shard_dataset.column_names and self._config.dataset.loss_masking_spans is not None:
                for item in tqdm.tqdm(shard_dataset, desc=f"Saving shard {shard_idx}", unit="docs"):
                    yield GPTSample(
                        np.array(item["input_ids"], dtype=self._data_type.numpy),
                        np.array(item["token_spans"], dtype=np.int32).reshape(-1, 2),
                    )
            else:
                for item in tqdm.tqdm(shard_dataset, desc=f"Saving shard {shard_idx}", unit="docs"):
                    yield GPTSample(np.array(item["input_ids"], dtype=self._data_type.numpy))

        GPTMemmapDataset.write_dataset(prefix=shard_output_path, documents=_document_generator())

        return GPTMemmapDatasetConfig.from_dict(
            {
                "type": "memmap",
                "path": prefix,
                "num_documents": len(shard_dataset),  # Use the length of the shard dataset directly
                "num_tokens": sum(len(doc["input_ids"]) for doc in shard_dataset),
            }
        )

    def _load_dataset(self) -> datasets.Dataset:
        dataset = datasets.load_dataset(
            path=self._config.dataset.path,
            name=self._config.dataset.config_name,
            data_dir=self._config.dataset.data_directory,
            data_files=self._config.dataset.data_files,
            split=self._config.dataset.split,
            num_proc=self._config.loading_workers,
            trust_remote_code=self._config.dataset.trust_remote_code,
        )
        assert isinstance(dataset, datasets.Dataset)
        return dataset

    def _get_croissant_metadata(self):
        token = huggingface_hub.HfFolder.get_token()
        try:
            # Retrieve the dataset metadata in croissant format
            url = f"https://huggingface.co/api/datasets/{self._config.dataset.path}/croissant"
            if token is None:
                response = requests.get(url)
            else:
                response = requests.get(url, headers={"Authorization": f"Bearer {token}"})

            if response.status_code != 200:
                logger.warning(
                    f"Failed to get croissant metadata, status_code: {response.status_code}, body: {response.text}"
                )
                return None

            data = response.json()
        except Exception as e:
            logger.warning(f"Failed to get croissant metadata, {e}")
            return None
        if "error" in data:
            logger.warning(f"Failed to get croissant metadata, error: {data['error']}")
            return None

        return data

    def _save_croissant_metadata(self):
        dataset_path = pathlib.Path(self._config.dataset.path)
        croissant_path = pathlib.Path(self._config.output_path) / "croissant.json"

        if dataset_path.is_dir():
            # If the dataset is local, check if it has the metadata file and copy it
            croissant_file = dataset_path / "croissant.json"
            if croissant_file.is_file():
                shutil.copy(croissant_file, croissant_path)
            else:
                logger.warning(f"Source local dataset {self._config.dataset.path} does not have croissant file")
                return
        else:
            # If the dataset is on HF hub, retrieve the metadata if provided and save it
            data = self._get_croissant_metadata()
            if data is not None:
                json.dump(data, croissant_path.open("w"))

    def run(self) -> None:
        # Set transformers logging verbosity
        transformers.logging.set_verbosity_error()

        # Disable disk space check if requested
        if self._config.dataset.disable_disk_space_check:
            datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory=".": True

        # Load tokenizer
        self._tokenizer = Tokenizer(config=self._config.tokenizer)

        # Decide the datatype based on the tokenizer vocabulary size
        self._data_type = (
            get_unsigned_integer_type(self._tokenizer.vocab_size)
            if self._config.dataset.data_type is None
            else self._config.dataset.data_type
        )

        # Initialize distributed processing
        if self._config.distributed.world_size > 1:
            torch.distributed.init_process_group(
                backend=self._config.distributed.backend,
                rank=self._config.distributed.rank,
                world_size=self._config.distributed.world_size,
            )

        # Prepare output directory
        self._config.output_path.mkdir(parents=True, exist_ok=True)

        if pathlib.Path(self._config.dataset.path).is_dir():
            # Dataset is already downloaded, load from disk
            dataset = self._load_dataset()
        else:
            # Dataset is not downloaded, download on rank 0
            if self._config.distributed.rank == 0:
                dataset = self._load_dataset()

            # Synchronize processes to wait for the download to finish on rank 0
            if self._config.distributed.world_size > 1:
                torch.distributed.barrier()

            # Load the downloaded dataset on remaining ranks
            if self._config.distributed.rank != 0:
                dataset = self._load_dataset()

            # Synchronize processes to wait for the dataset to load on remaining ranks
            if self._config.distributed.world_size > 1:
                torch.distributed.barrier()

        assert isinstance(dataset, datasets.Dataset)
        dataset = dataset.shard(
            num_shards=self._config.distributed.world_size,
            index=self._config.distributed.rank,
        )
        if self._config.dataset.field not in dataset.column_names:
            raise ValueError(f"Dataset does not have field '{self._config.dataset.field}'.")
        if self._config.dataset.loss_masking_spans is not None:
            if self._config.dataset.loss_masking_spans not in dataset.column_names:
                raise ValueError(f"Dataset does not have spans field '{self._config.dataset.loss_masking_spans}'.")
            tokenize_fn = self._tokenize_batch_with_spans
        else:
            tokenize_fn = self._tokenize_batch

        # Tokenize the dataset in parallel
        tokenized_dataset = dataset.map(
            tokenize_fn,
            batched=True,
            num_proc=self._config.tokenize_workers,
            desc="Tokenizing batches",
        )

        # Calculate total number of tokens
        total_tokens = sum(tqdm.tqdm(tokenized_dataset["num_tokens"], desc="Counting tokens", unit="tokens"))

        # Split dataset into shards based on number of tokens
        num_shards = int(np.ceil(total_tokens / self._config.tokens_per_shard))
        shards = [
            (i, tokenized_dataset.shard(num_shards=num_shards, index=i))
            for i in tqdm.tqdm(range(num_shards), desc="Creating shards")
        ]

        # Use multiprocessing to save each shard in parallel on all ranks
        with multiprocessing.Pool(processes=self._config.saving_workers) as pool:
            dataset_configs = pool.map(self._save_shard, shards)

        # Gather dataset_dicts from all ranks to rank 0
        if self._config.distributed.world_size > 1:
            if self._config.distributed.rank == 0:
                all_dataset_configs = [None] * self._config.distributed.world_size
                torch.distributed.gather_object(dataset_configs, all_dataset_configs, dst=0)
                dataset_configs = [item for sublist in all_dataset_configs for item in sublist]
            else:
                torch.distributed.gather_object(dataset_configs, [], dst=0)

        if self._config.distributed.rank == 0:
            # Create the config file(s) on rank 0
            if self._config.splits:
                for split_name, split_config in self._split_and_blend_dataset_configs(
                    dataset_configs, self._config.splits
                ).items():
                    self._save_dataset_config(
                        split_config, self._config.output_path / f"fast_llm_config_{split_name}.yaml"
                    )
            else:
                self._save_dataset_config(
                    self._blend_dataset_configs(dataset_configs), self._config.output_path / f"fast_llm_config.yaml"
                )

            # Save metadata on rank 0
            self._save_croissant_metadata()

        # Finalize distributed processing
        if self._config.distributed.world_size > 1:
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()

    @classmethod
    def _save_dataset_config(cls, dataset_config: GPTIndexedDatasetConfig, output_path: pathlib.Path) -> None:
        logger.info(f"Saving config to {output_path}")
        yaml.safe_dump(
            dataset_config.to_serialized(),
            output_path.open("w"),
        )

    @classmethod
    def _blend_dataset_configs(cls, dataset_configs: list[GPTMemmapDatasetConfig]) -> GPTIndexedDatasetConfig:
        if len(dataset_configs) == 1:
            return dataset_configs[0]
        return GPTIndexedDatasetConfig.from_dict(
            {
                "type": "blended",
                "datasets": dataset_configs,
                "weights": [dataset_config.num_tokens for dataset_config in dataset_configs],
            }
        )

    @classmethod
    def _split_and_blend_dataset_configs(
        cls, dataset_configs: list[GPTMemmapDatasetConfig], splits: dict[str, int | float]
    ) -> dict[str, GPTIndexedDatasetConfig]:
        split_cumsum = padded_cumsum(normalize_probabilities(list(splits.values()), return_array=True)).tolist()
        dataset_sizes = [dataset_config.num_tokens for dataset_config in dataset_configs]
        dataset_probabilities = normalize_probabilities(dataset_sizes)
        dataset_cumsums = padded_cumsum(dataset_probabilities).tolist()
        dataset_splits = {}

        for split_index, split_name in enumerate(splits):
            datasets_in_split = []
            dataset_tokens_in_split = []
            for dataset_index, dataset_config in enumerate(dataset_configs):
                split_begin_in_dataset = max(
                    (split_cumsum[split_index] - dataset_cumsums[dataset_index])
                    / dataset_probabilities[dataset_index],
                    0,
                )
                split_end_in_dataset = min(
                    (split_cumsum[split_index + 1] - dataset_cumsums[dataset_index])
                    / dataset_probabilities[dataset_index],
                    1,
                )
                if split_begin_in_dataset == 0 and split_end_in_dataset == 1:
                    # All the dataset belongs to the split.
                    datasets_in_split.append(dataset_configs[dataset_index])
                    dataset_tokens_in_split.append(dataset_sizes[dataset_index])
                elif split_end_in_dataset > split_begin_in_dataset:
                    # Part of the dataset belongs to the split.
                    sizes_cumsum = dataset_config.build().get_document_sizes().cumsum()
                    Assert.eq(sizes_cumsum[-1], dataset_config.num_tokens)
                    begin_index = _get_nearest_split(sizes_cumsum, split_begin_in_dataset * dataset_config.num_tokens)
                    end_index = _get_nearest_split(sizes_cumsum, split_end_in_dataset * dataset_config.num_tokens)
                    if end_index > begin_index:
                        datasets_in_split.append(
                            GPTDatasetSliceConfig.from_dict(
                                {
                                    "type": "slice",
                                    "dataset": dataset_configs[dataset_index],
                                    "begin": begin_index / dataset_config.num_documents,
                                    "end": end_index / dataset_config.num_documents,
                                }
                            )
                        )
                        dataset_tokens_in_split.append(
                            sizes_cumsum[end_index - 1].item()
                            - (sizes_cumsum[begin_index - 1].item() if begin_index > 0 else 0)
                        )

                # [else] None of the dataset belongs to the split.

            if len(datasets_in_split) == 0:
                # This is a big problem, but we don't want to crash the whole run.
                logger.error(f"Datasets split {split_name} is empty!")
            elif len(datasets_in_split) == 1:
                dataset_splits[split_name] = datasets_in_split[0]
            else:
                dataset_splits[split_name] = GPTBlendedDatasetConfig.from_dict(
                    {
                        "type": "blended",
                        "datasets": datasets_in_split,
                        "weights": dataset_tokens_in_split,
                    }
                )

        return dataset_splits


def _get_nearest_split(cumsum: np.ndarray, value: float) -> int:
    left = cumsum.searchsorted(value, side="right")
    if left == len(cumsum):
        return left.item()
    return left + 1 if (value - cumsum[left]) / (cumsum[left + 1] - cumsum[left]) > 0.5 else left
