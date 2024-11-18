import json
import multiprocessing
import shutil

import datasets
import numpy as np
import torch.distributed
import tqdm
import transformers

from fast_llm.data.gpt.memmap import GPTMemmapDataset
from fast_llm.data.preparator.config import DatasetPreparator
from fast_llm.data.preparator.gpt_memmap.config import GPTMemmapDatasetPreparatorConfig
from fast_llm.data.tokenizer import Tokenizer
from fast_llm.engine.config_utils.data_type import DataType


class GPTMemmapDatasetPreparator(DatasetPreparator):
    _config: GPTMemmapDatasetPreparatorConfig
    config_class = GPTMemmapDatasetPreparatorConfig

    _tokenizer: Tokenizer
    _data_type: DataType

    def _tokenize_batch(self, batch):
        input_ids = [
            np.array(self._tokenizer.tokenize(text), dtype=self._data_type.numpy)
            for text in batch[self._config.dataset.field]
        ]
        num_tokens = [len(x) for x in input_ids]
        return {
            "input_ids": input_ids,
            "num_tokens": num_tokens,
        }

    def _save_shard(self, args) -> dict:
        shard_idx, shard_dataset = args
        prefix = f"shard_{self._config.distributed.rank}_{shard_idx}"
        shard_output_path = self._config.output_path / prefix
        documents = [
            np.array(item["input_ids"], dtype=self._data_type.numpy)
            for item in tqdm.tqdm(shard_dataset, desc=f"Saving shard {shard_idx}", unit="docs")
        ]
        GPTMemmapDataset.write_dataset(prefix=shard_output_path, documents=documents)
        dataset_dict = {
            "prefix": prefix,
            "num_documents": len(documents),
            "num_tokens": sum(len(doc) for doc in documents),
        }
        return dataset_dict

    def run(self):
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

        # Download dataset if necessary on rank 0
        download_path = self._config.output_path / "downloaded_dataset"
        download_path_ok = download_path / "ok"
        if self._config.distributed.rank == 0 and not download_path_ok.exists():
            datasets.load_dataset(
                path=self._config.dataset.path,
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

        # Finalize distributed processing
        if self._config.distributed.world_size > 1:
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()

        # Clean up downloaded dataset
        if self._config.remove_downloads and self._config.distributed.rank == 0:
            shutil.rmtree(download_path, ignore_errors=True)
