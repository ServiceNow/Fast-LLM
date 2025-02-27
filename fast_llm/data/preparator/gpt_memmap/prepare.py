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

from fast_llm.data.dataset.gpt.memmap import GPTMemmapDataset
from fast_llm.data.dataset.gpt.sampled import GPTSample
from fast_llm.data.preparator.config import DatasetPreparator
from fast_llm.data.preparator.gpt_memmap.config import GPTMemmapDatasetPreparatorConfig
from fast_llm.data.tokenizer import Tokenizer
from fast_llm.engine.config_utils.data_type import DataType

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

    def _save_shard(self, args: tuple[int, datasets.Dataset]) -> dict[str, typing.Any]:
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

        dataset_dict = {
            "prefix": prefix,
            "num_documents": len(shard_dataset),  # Use the length of the shard dataset directly
            "num_tokens": sum(len(doc["input_ids"]) for doc in shard_dataset),
        }
        return dataset_dict

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

        # Set data type if not provided
        if self._config.dataset.data_type is None:
            # Decide the datatype based on the tokenizer vocabulary size
            vocab_size = self._tokenizer.vocab_size
            if vocab_size <= np.iinfo(np.int16).max:
                self._data_type = DataType.int16
            # elif vocab_size <= np.iinfo(np.uint16).max:
            #     self._data_type = DataType.uint16  # Not supported by Fast-LLM's DataType
            elif vocab_size <= np.iinfo(np.int32).max:
                self._data_type = DataType.int32
            else:
                raise ValueError(f"Tokenizer vocabulary size {vocab_size} is too large. This is likely an error.")
        else:
            self._data_type = self._config.dataset.data_type

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

            self._save_croissant_metadata()

            # Create an index file on rank 0
            index_file = self._config.output_path / "index.txt"
            index_file.open("w").writelines([dataset_dict["prefix"] + "\n" for dataset_dict in dataset_dicts])

        # Finalize distributed processing
        if self._config.distributed.world_size > 1:
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
