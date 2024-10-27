import json
import os
from functools import cached_property
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, logging

from fast_llm.config import Field, FieldHint, check_field, config_class
from fast_llm.data.mmap import MMapIndexedDataset
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.engine.config_utils.runnable import RunnableConfig
from fast_llm.utils import Assert

logging.set_verbosity_error()


@config_class
class PrepareDatasetConfig(RunnableConfig):
    dataset_name_or_path: str = Field(
        desc="Name or path of the dataset.",
        hint=FieldHint.core,
    )
    tokenizer_path_or_name: str = Field(
        desc="Path or name of the tokenizer.",
        hint=FieldHint.core,
    )
    output_dir: str = Field(
        desc="Output directory for the processed dataset.",
        hint=FieldHint.core,
    )
    num_processes_load: int = Field(
        default=1,
        desc="Number of workers in load_dataset() call.",
        hint=FieldHint.optional,
        valid=check_field(Assert.geq, 0),
    )
    num_processes_map: int = Field(
        default=1,
        desc="Number of workers in .map() call.",
        hint=FieldHint.optional,
        valid=check_field(Assert.geq, 0),
    )
    num_processes_save: int = Field(
        default=1,
        desc="Number of processes for saving the mmap'ed datasets.",
        hint=FieldHint.optional,
        valid=check_field(Assert.geq, 0),
    )
    num_tokens_per_shard: int = Field(
        default=1000000000,
        desc="Approximate number of tokens per shard.",
        hint=FieldHint.optional,
        valid=check_field(Assert.geq, 1),
    )
    dataset_config_name: None | str = Field(
        default=None,
        desc="Specific configuration name for the dataset.",
        hint=FieldHint.optional,
    )
    dataset_split: str = Field(
        default="train",
        desc="Split of the dataset to use.",
        hint=FieldHint.optional,
    )
    dataset_field: str = Field(
        default="text",
        desc="Field of the dataset to use.",
        hint=FieldHint.optional,
    )
    dataset_dtype: DataType = Field(
        default=None,
        desc="Data type of the dataset field.",
        hint=FieldHint.derived,
    )

    @cached_property
    def tokenizer(self):
        return AutoTokenizer.from_pretrained(self.tokenizer_path_or_name)

    def _validate(self):
        if self.dataset_dtype is None:
            # Decide the dtype based on the tokenizer vocabulary size
            vocab_size = len(self.tokenizer)

            if vocab_size <= np.iinfo(np.int8).max:
                self.dataset_dtype = DataType.int8
            elif vocab_size <= np.iinfo(np.int16).max:
                self.dataset_dtype = DataType.int16
            elif vocab_size <= np.iinfo(np.int32).max:
                self.dataset_dtype = DataType.int32
            elif vocab_size <= np.iinfo(np.int64).max:
                self.dataset_dtype = DataType.int64
            else:
                raise ValueError(
                    f"Tokenizer vocabulary size {vocab_size} is too large for supported dtypes in MMapIndexedDataset."
                )
        super()._validate()

    def _tokenize_text(self, text):
        tokens = self.tokenizer(
            text,
            truncation=False,
            padding=False,
            add_special_tokens=True,
        )["input_ids"]
        return np.array(tokens, dtype=self.dataset_dtype.numpy)

    def _tokenize_batch(self, batch):
        input_ids = [self._tokenize_text(text) for text in batch[self.dataset_field]]
        num_tokens = [len(x) for x in input_ids]
        return {
            "input_ids": input_ids,
            "num_tokens": num_tokens,
        }

    def _save_shard(self, args) -> dict:
        shard_idx, shard_dataset = args
        prefix = f"shard_{shard_idx}"
        shard_output_path = Path(self.output_dir) / prefix
        documents = [
            np.array(item["input_ids"], dtype=self.dataset_dtype.numpy)
            for item in tqdm(shard_dataset, desc=f"Saving shard {shard_idx}", unit="docs")
        ]
        MMapIndexedDataset.write_dataset(prefix=shard_output_path, documents=documents)
        dataset_dict = {
            "prefix": prefix,
            "num_documents": len(documents),
            "num_tokens": sum(len(doc) for doc in documents),
        }
        return dataset_dict

    def run(self):
        # Load dataset
        dataset = load_dataset(
            path=self.dataset_name_or_path,
            name=self.dataset_config_name,
            split=self.dataset_split,
            num_proc=self.num_processes_load,
        )
        if self.dataset_field not in dataset.column_names:
            raise ValueError(f"Dataset does not have field '{self.dataset_field}'.")

        # Tokenize the dataset
        tokenized_dataset = dataset.map(
            self._tokenize_batch,
            batched=True,
            num_proc=self.num_processes_map,
            desc="Tokenizing batches",
        )

        # Calculate total number of tokens
        total_tokens = sum(tqdm(tokenized_dataset["num_tokens"], desc="Counting tokens", unit="tokens"))

        # Split dataset into shards
        num_shards = int(np.ceil(total_tokens / self.num_tokens_per_shard))
        shards = [
            (i, tokenized_dataset.shard(num_shards=num_shards, index=i))
            for i in tqdm(range(num_shards), desc="Creating shards")
        ]

        # Prepare output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Use multiprocessing to save each shard in parallel
        with Pool(processes=self.num_processes_save) as pool:
            dataset_dicts = pool.map(self._save_shard, shards)

        # Create a metadata file
        total_tokens = sum(dataset_dict["num_tokens"] for dataset_dict in dataset_dicts)
        for dataset_dict in dataset_dicts:
            dataset_dict["weight"] = float(dataset_dict["num_tokens"]) / float(total_tokens)
        output_file = Path(self.output_dir) / "fast_llm_dataset.json"
        json.dump({"datasets": dataset_dicts}, output_file.open("w"))


if __name__ == "__main__":
    PrepareDatasetConfig.parse_and_run()
