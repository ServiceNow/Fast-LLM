import json
import logging
import math
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

from fast_llm.data.dataset.config import (
    BlendedDatasetConfig,
    DatasetSliceConfig,
    IndexedDatasetConfig,
    MemmapDatasetConfig,
    SampledDatasetConfig,
)
from fast_llm.data.dataset.memmap import MemmapDataset
from fast_llm.data.preparator.config import DatasetPreparator
from fast_llm.data.preparator.gpt_memmap.config import GPTMemmapDatasetPreparatorConfig, LanguageModelSourceConfig
from fast_llm.data.sample.language_model import LanguageModelSample, LanguageModelWriter
from fast_llm.data.sample.range import RangeSample
from fast_llm.data.sample.token import TokenSample
from fast_llm.data.tokenizer import Tokenizer
from fast_llm.engine.config_utils.data_type import DataType, get_unsigned_integer_type
from fast_llm.utils import Assert, normalize_probabilities, padded_cumsum

logger = logging.getLogger(__name__)


class GPTMemmapDatasetPreparator[ConfigType: GPTMemmapDatasetPreparatorConfig](DatasetPreparator[ConfigType]):
    _tokenizer: Tokenizer
    _data_type: DataType
    _sample_type: typing.ClassVar[type[LanguageModelSample]] = LanguageModelSample
    _config: GPTMemmapDatasetPreparatorConfig

    def __init__(self, config: ConfigType):
        super().__init__(config)
        self._source_shema: LanguageModelSourceConfig = self._config.dataset.source_shema

    def _save_shard(self, args: tuple[int, datasets.Dataset]) -> MemmapDatasetConfig:
        shard_index, shard_dataset = args
        file_name = f"shard_{self._config.distributed.rank}_{shard_index}.fast_llm_dataset"

        MemmapDataset.write_dataset(
            self._config.output_path / file_name,
            tqdm.tqdm((sample["sample"] for sample in shard_dataset), desc=f"Saving shard {shard_index}", unit="docs"),
            LanguageModelWriter,
        )

        return MemmapDatasetConfig.from_dict({"type": "memmap", "path": file_name})

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
        self._tokenizer = self._config.tokenizer.get_tokenizer()

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

        downloaded = pathlib.Path(self._config.dataset.path).is_dir()
        if self._config.distributed.world_size > 1:
            torch.distributed.barrier()

        if downloaded:
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

        for column_name in self._source_shema.columns:
            if column_name not in dataset.column_names:
                raise ValueError(f"Dataset does not have field '{column_name}'.")

        # Tokenize the dataset in parallel
        prepared_dataset = dataset.map(
            self._prepare_batch,
            batched=True,
            num_proc=self._config.tokenize_workers,
            desc="Tokenizing batches",
        )

        # Split dataset into shards based on number of tokens
        num_shards = math.ceil(
            sum(len(sample) for sample in prepared_dataset["samples"]) / self._config.tokens_per_shard
        )
        shards = [
            (i, prepared_dataset.shard(num_shards=num_shards, index=i))
            for i in tqdm.tqdm(range(num_shards), desc="Creating shards")
        ]

        # Use multiprocessing to save each shard in parallel on all ranks
        with multiprocessing.Pool(processes=self._config.saving_workers) as pool:
            dataset_configs = pool.map(self._save_shard, shards)

        self.generate_config_yaml_for_sharded_dst(dataset_configs)

    def _prepare_batch(self, batch: dict[str, list[typing.Any]]) -> dict[str, list[LanguageModelSample]]:
        # Gather values by sample using zip*
        sample_column_values = zip(*(batch[column_name] for column_name in self._source_shema.columns))
        # Convert to dicts using column names.
        sample_dicts = (
            {column_name: column_value for column_name, column_value in zip(self._source_shema.columns, sample_data)}
            for sample_data in sample_column_values
        )
        # Prepare each sample, wrap in dict for the `Dataset` interface
        return {"samples": [self._prepare_sample(sample_dict) for sample_dict in sample_dicts]}

    def _prepare_sample(self, sample: dict[str, typing.Any]) -> LanguageModelSample:
        text = sample[self._source_shema.text_column]
        all_spans = []
        if self._source_shema.has_loss_masking_span:
            # TODO: ====== What is the input format? ======
            # Spans are typically stored in the (begin, last) format. We convert to (begin, end) range format.
            loss_masking_spans = _sort_spans(
                (begin, last + 1)
                for begin, last in np.array(sample[self._source_shema.loss_masking_spans_column], dtype=np.int32)
                .reshape(-1, 2)
                .tolist()
            )
            all_spans.extend(loss_masking_spans)

        if self._source_shema.has_preference_spans:
            # TODO: ===== Was `self._config.dataset.field` (bug?) ======
            full_chosen_text = (
                text + sample[self._source_shema.chosen_spans_column] + self._tokenizer.tokenizer.eos_token
            )
            full_rejected_text = (
                self._tokenizer.tokenizer.bos_token + text + sample[self._source_shema.rejected_spans_column]
            )
            # compute chosen span
            chosen_spans = [[len(text), len(full_chosen_text)]]

            # compute rejected span
            rejected_span = [
                [
                    len(full_chosen_text) + len(self._tokenizer.tokenizer.bos_token) + len(text),
                    len(full_chosen_text) + len(full_rejected_text),
                ]
            ]
            # pack texts
            text = full_chosen_text + full_rejected_text
            all_spans.extend(chosen_spans + rejected_span)

        tokens = torch.tensor(
            self._tokenizer.tokenize_with_spans(text, True, True, spans=_sort_spans(all_spans)),
            dtype=self._data_type.torch,
        )
        sample_size = len(tokens)

        return LanguageModelSample(
            TokenSample(tokens, [sample_size]),
            RangeSample(loss_masking_spans, sample_size) if self._source_shema.has_loss_masking_span else None,
            RangeSample(chosen_spans, sample_size) if self._source_shema.has_preference_spans else None,
            RangeSample(rejected_span, sample_size) if self._source_shema.has_preference_spans else None,
        )

    def generate_config_yaml_for_sharded_dst(self, dataset_configs: list[MemmapDatasetConfig]) -> None:
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
                    dataset_configs, self._config.splits, self._config.output_path
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
    def _save_dataset_config(
        cls, dataset_config: IndexedDatasetConfig[_sample_type], output_path: pathlib.Path
    ) -> None:
        logger.info(f"Saving config to {output_path}")
        yaml.safe_dump(
            dataset_config.to_dict(),
            output_path.open("w"),
        )

    @classmethod
    def _blend_dataset_configs(
        cls, dataset_configs: list[MemmapDatasetConfig[_sample_type]]
    ) -> IndexedDatasetConfig[_sample_type]:
        if len(dataset_configs) == 1:
            return dataset_configs[0]
        return SampledDatasetConfig[cls._sample_type].from_dict(
            {
                "type": "blended",
                "datasets": dataset_configs,
                "weights": [dataset_config.num_tokens for dataset_config in dataset_configs],
            }
        )

    @classmethod
    def _split_and_blend_dataset_configs(
        cls,
        dataset_configs: list[MemmapDatasetConfig[_sample_type]],
        splits: dict[str, int | float],
        output_path: pathlib.Path,
    ) -> dict[str, SampledDatasetConfig[_sample_type]]:
        # TODO: ====== Missing `num_tokens`, `num_documents`. ======
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
                    # TODO: Somehow getting a segfault when merging two lines below (numpy bug?).
                    dataset = dataset_config.to_copy({"path": output_path / dataset_config.path}).build()
                    sizes_cumsum = dataset.get_document_sizes().numpy().cumsum()
                    Assert.eq(sizes_cumsum[-1], dataset_config.num_tokens)
                    begin_index = _get_nearest_split(sizes_cumsum, split_begin_in_dataset * dataset_config.num_tokens)
                    end_index = _get_nearest_split(sizes_cumsum, split_end_in_dataset * dataset_config.num_tokens)
                    if end_index > begin_index:
                        datasets_in_split.append(
                            DatasetSliceConfig[cls._sample_type].from_dict(
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
                dataset_splits[split_name] = BlendedDatasetConfig[cls._sample_type].from_dict(
                    {
                        "type": "blended",
                        "datasets": datasets_in_split,
                        "weights": dataset_tokens_in_split,
                    }
                )

        return dataset_splits


def _sort_spans(spans: typing.Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted(spans, key=lambda span: span[0])


def _get_nearest_split(cumsum: np.ndarray, value: float) -> int:
    left = cumsum.searchsorted(value, side="right")
    if left == len(cumsum):
        return left.item()
    return left.item() + 1 if (value - cumsum[left]) / (cumsum[left + 1] - cumsum[left]) > 0.5 else left.item()
