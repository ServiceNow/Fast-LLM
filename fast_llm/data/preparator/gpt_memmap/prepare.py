import collections
import datetime
import enum
import functools
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
from fast_llm.data.preparator.gpt_memmap.config import (
    ConversationSourceConfig,
    DocumentSourceConfig,
    GPTMemmapDatasetPreparatorConfig,
    LanguageModelSourceConfig,
)
from fast_llm.data.preprocessing.abstract import NullPreprocessingConfig
from fast_llm.data.preprocessing.language_model import LanguageModelPreprocessingConfig
from fast_llm.data.preprocessing.tokenizer import Tokenizer
from fast_llm.data.sample.abstract import MemmapIndexDatasetReaderConfig
from fast_llm.data.sample.language_model import LanguageModelSample, LanguageModelWriter
from fast_llm.data.sample.patch import PatchSample
from fast_llm.data.sample.range import RangeSample
from fast_llm.data.sample.token import TokenSample
from fast_llm.engine.config_utils.data_type import DataType, get_integer_type
from fast_llm.engine.config_utils.run import log_main_rank
from fast_llm.utils import normalize_probabilities, padded_cumsum

logger = logging.getLogger(__name__)


class SpanType(enum.StrEnum):
    loss_masking = "loss_masking"
    chosen = "chosen"
    rejected = "rejected"
    image = "image"


class GPTMemmapDatasetPreparator[ConfigType: GPTMemmapDatasetPreparatorConfig](DatasetPreparator[ConfigType]):
    _tokenizer: Tokenizer
    _data_type: DataType
    _sample_type: typing.ClassVar[type[LanguageModelSample]] = LanguageModelSample
    _config: GPTMemmapDatasetPreparatorConfig

    def __init__(self, config: ConfigType):
        super().__init__(config)
        self._source_schema: LanguageModelSourceConfig = self._config.dataset.source_schema

    def _load_dataset(self) -> datasets.Dataset:
        if self._config.dataset.load_from_disk:
            dataset = datasets.load_from_disk(self._config.dataset.path)[self._config.dataset.split]
        else:
            dataset = datasets.load_dataset(
                path=self._config.dataset.path,
                name=self._config.dataset.config_name,
                data_dir=self._config.dataset.data_directory,
                data_files=self._config.dataset.data_files,
                split=self._config.dataset.split,
                num_proc=self._config.num_workers,
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

        # Validate chat template for conversation format
        if isinstance(self._source_schema, ConversationSourceConfig):
            self._tokenizer.validate_chat_template()

        # Decide the datatype based on the tokenizer vocabulary size
        self._data_type = (
            get_integer_type(self._tokenizer.vocab_size)
            if self._config.dataset.data_type is None
            else self._config.dataset.data_type
        )

        # Initialize distributed processing
        if self._config.distributed.world_size > 1:
            log_main_rank(f"> Initializing distributed process groups ...")
            torch.distributed.init_process_group(
                backend=self._config.distributed.backend,
                rank=self._config.distributed.rank,
                world_size=self._config.distributed.world_size,
                timeout=datetime.timedelta(seconds=self._config.distributed.timeout),
            )

        # Prepare output directory
        self._config.output_path.mkdir(parents=True, exist_ok=True)

        log_main_rank(f"> Loading dataset `{self._config.dataset.path}` ...")
        if self._config.distributed.world_size == 1:
            dataset = self._load_dataset()
        elif self._config.distributed.rank == 0:
            # Load first on rank 0 to prevent parallel downloads.
            dataset = self._load_dataset()
            torch.distributed.barrier()
        else:
            torch.distributed.barrier()
            # Load the downloaded dataset on remaining ranks
            dataset = self._load_dataset()

        dataset = dataset.shard(
            num_shards=self._config.distributed.world_size,
            index=self._config.distributed.rank,
        )

        for column_name in self._source_schema.columns:
            if column_name not in dataset.column_names:
                raise ValueError(f"Dataset does not have field '{column_name}'.")

        # Split dataset into shards based on number of tokens
        num_shards = math.ceil(len(dataset) / self._config.documents_per_shard)
        shards = [(i, dataset.shard(num_shards=num_shards, index=i)) for i in range(num_shards)]

        log_main_rank(f"> Preparing samples on {self._config.num_workers} workers ...")

        # Use multiprocessing to save each shard in parallel on all ranks
        with multiprocessing.Pool(processes=self._config.num_workers) as pool:
            dataset_and_reader_configs = pool.map(self._prepare_shard, shards)

        log_main_rank(f"> Generating dataset config ...")
        self.generate_config_yaml_for_sharded_dst(dataset_and_reader_configs)

    def _prepare_shard(
        self, args: tuple[int, datasets.Dataset]
    ) -> tuple[MemmapDatasetConfig, MemmapIndexDatasetReaderConfig]:
        shard_index, shard_dataset = args
        file_name = f"shard_{self._config.distributed.rank}_{shard_index}.fast_llm_dataset"

        reader_config = MemmapDataset.write_dataset(
            self._config.output_path / file_name,
            (
                self._prepare_sample(sample)
                for sample in tqdm.tqdm(shard_dataset, desc=f"Saving shard {shard_index}", unit="docs")
            ),
            LanguageModelWriter,
            self._preprocessing_config,
        )
        return MemmapDatasetConfig.from_dict({"type": "memmap", "path": file_name}), reader_config

    @functools.cached_property
    def _preprocessing_config(self) -> LanguageModelPreprocessingConfig:
        return LanguageModelPreprocessingConfig(
            tokenizer=self._config.tokenizer,
            vocab_size=self._tokenizer.vocab_size,
            image_patches=(
                self._config.image_patches if self._source_schema.has_images else NullPreprocessingConfig()
            ),
            use_loss_masking_spans=self._source_schema.has_loss_masking_span,
            use_preference_spans=self._source_schema.has_preference_spans,
        )

    def _prepare_sample(self, sample: dict[str, typing.Any]) -> LanguageModelSample:
        token_spans_by_type = collections.defaultdict(list)
        image_patches = image_token_maps = image_position_ids = patch_counts = None

        if isinstance(self._source_schema, ConversationSourceConfig):
            # Conversation format: tokenize messages and get loss masking spans from chat template
            tokens, loss_masking_spans = self._tokenizer.tokenize_chat(
                sample[self._source_schema.messages],
                True,
                True,
                data_type=self._data_type,
            )
            token_spans_by_type[SpanType.loss_masking] = loss_masking_spans
        elif isinstance(self._source_schema, DocumentSourceConfig):
            # Document format: use the text-spans pipeline
            text = sample[self._source_schema.text]
            all_spans = []

            if self._source_schema.has_loss_masking_span:
                # Spans are typically stored in the (begin, last) format. We convert to (begin, end) range format.
                loss_masking_spans = _sort_spans(
                    (SpanType.loss_masking, (begin, last + 1))
                    for begin, last in np.array(sample[self._source_schema.loss_masking_spans], dtype=np.int32)
                    .reshape(-1, 2)
                    .tolist()
                )
                all_spans.extend(loss_masking_spans)

            if self._source_schema.has_preference_spans:
                full_chosen_text = text + sample[self._source_schema.chosen_span] + self._tokenizer.tokenizer.eos_token
                full_rejected_text = (
                    self._tokenizer.tokenizer.bos_token + text + sample[self._source_schema.rejected_span]
                )
                # compute chosen span
                chosen_spans = [(SpanType.chosen, (len(text), len(full_chosen_text)))]

                # compute rejected span
                rejected_span = [
                    (
                        SpanType.rejected,
                        (
                            len(full_chosen_text) + len(self._tokenizer.tokenizer.bos_token) + len(text),
                            len(full_chosen_text) + len(full_rejected_text),
                        ),
                    )
                ]
                # pack texts
                text = full_chosen_text + full_rejected_text
                all_spans.extend(chosen_spans + rejected_span)

            if self._source_schema.has_images:
                # Get the images and positions, sorted by position.
                images, image_positions = (
                    zip(
                        *sorted(
                            zip(
                                sample[self._source_schema.images],
                                sample[self._source_schema.image_positions],
                                strict=True,
                            ),
                            key=lambda x: x[1],
                        )
                    )
                    if len(sample[self._source_schema.images]) > 0
                    else ([], [])
                )
                # Get the image patches and associated data.
                image_patches, image_position_ids, image_token_maps, image_token_ids, patch_counts = (
                    self._config.image_patches.get_patches_from_images(images, self._data_type)
                )
                patch_count_cumsum = padded_cumsum(patch_counts).tolist()
                # Add an empty "span" at each image position so we know where to insert them in the tokenized sequence.
                all_spans.extend([(SpanType.image, (position, position)) for position in image_positions])

            # Sort the spans by location (begin), keeping track of their type.
            # Note: overlapping spans are not supported (explicit assertion in the tokenizer).
            span_types, spans = zip(*_sort_spans(all_spans)) if all_spans else ([], [])
            # Tokenize the text, and determine the span locations in the tokenized text.
            tokens, token_spans = self._tokenizer.tokenize_with_spans(
                text, True, True, text_spans=spans, data_type=self._data_type
            )

            # Gather token spans by type.
            if self._source_schema.has_images:
                # Insert the image token ids in the token sequence and shift the spans accordingly.
                tokens_shift = 0
                image_index = 0
                for span_type, (begin, end) in zip(span_types, token_spans, strict=True):
                    # Account for the tokens already inserted.
                    begin = begin + tokens_shift
                    end = end + tokens_shift
                    if span_type == SpanType.image:
                        # Shift the token map to the image location.
                        image_token_maps[
                            patch_count_cumsum[image_index] : patch_count_cumsum[image_index + 1]
                        ] += begin
                        # Insert the placeholder and image break tokens.
                        tokens = torch.cat([tokens[:begin], image_token_ids[image_index], tokens[begin:]])
                        tokens_shift += len(image_token_ids[image_index])
                        image_index += 1
                    else:
                        token_spans_by_type[span_type].append((begin, end))
            else:
                for span_type, token_span in zip(span_types, token_spans, strict=True):
                    token_spans_by_type[span_type].append(token_span)
        else:
            raise NotImplementedError(f"Unsupported source schema type: {type(self._source_schema)}")

        sample_size = len(tokens)

        return LanguageModelSample(
            TokenSample(tokens, [sample_size]),
            (
                RangeSample(token_spans_by_type[SpanType.loss_masking], sample_size)
                if self._source_schema.has_loss_masking_span
                else None
            ),
            (
                RangeSample(token_spans_by_type[SpanType.chosen], sample_size)
                if self._source_schema.has_preference_spans
                else None
            ),
            (
                # `tokenize_with_spans` excludes the final eod token from the rejected span, but we want to include it.
                RangeSample([(begin, end + 1) for begin, end in token_spans_by_type[SpanType.rejected]], sample_size)
                if self._source_schema.has_preference_spans
                else None
            ),
            (
                PatchSample(image_patches, image_token_maps, image_position_ids, sample_size, patch_counts)
                if self._source_schema.has_images
                else None
            ),
        )

    def generate_config_yaml_for_sharded_dst(
        self, dataset_and_reader_configs: list[tuple[MemmapDatasetConfig, MemmapIndexDatasetReaderConfig]]
    ) -> None:
        # Gather dataset_dicts from all ranks to rank 0
        if self._config.distributed.world_size > 1:
            if self._config.distributed.rank == 0:
                all_dataset_and_reader_configs = [None] * self._config.distributed.world_size
                torch.distributed.gather_object(dataset_and_reader_configs, all_dataset_and_reader_configs, dst=0)
                dataset_and_reader_configs = [item for sublist in all_dataset_and_reader_configs for item in sublist]
            else:
                torch.distributed.gather_object(dataset_and_reader_configs, [], dst=0)

        if self._config.distributed.rank == 0:
            # Create the config file(s) on rank 0
            dataset_configs, reader_configs = zip(*dataset_and_reader_configs)
            if self._config.splits:
                for split_name, (split_config, metadata) in self._split_and_blend_dataset_configs(
                    dataset_configs, reader_configs, self._config.splits, self._config.output_path
                ).items():
                    self._save_dataset_config(
                        split_config,
                        metadata,
                        output_path=self._config.output_path / f"fast_llm_config_{split_name}.yaml",
                    )
            else:
                self._save_dataset_config(
                    *self._blend_dataset_configs(dataset_configs, reader_configs),
                    output_path=self._config.output_path / f"fast_llm_config.yaml",
                )

            # Save metadata on rank 0
            self._save_croissant_metadata()

        # Finalize distributed processing
        if self._config.distributed.world_size > 1:
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()

    @classmethod
    def _save_dataset_config(
        cls,
        dataset_config: IndexedDatasetConfig[_sample_type],
        metadata: dict[str, typing.Any],
        output_path: pathlib.Path,
    ) -> None:
        logger.info(f"Saving config to {output_path}")
        yaml.safe_dump({"config": dataset_config.to_dict(), "metadata": metadata}, output_path.open("w"))

    @classmethod
    def _blend_dataset_configs(
        cls,
        dataset_configs: list[MemmapDatasetConfig[_sample_type]],
        reader_configs: list[MemmapIndexDatasetReaderConfig],
    ) -> tuple[IndexedDatasetConfig[_sample_type], dict[str, typing.Any]]:
        datasets_metadata = [reader_config.get_metadata() for reader_config in reader_configs]
        if len(dataset_configs) == 1:
            return dataset_configs[0], datasets_metadata[0]
        return SampledDatasetConfig[cls._sample_type].from_dict(
            {
                "type": "blended",
                "datasets": dataset_configs,
                "weights": [reader_config.num_tokens for reader_config in reader_configs],
            }
        ), reader_configs[0].blend_metadata(datasets_metadata)

    def _split_and_blend_dataset_configs(
        self,
        dataset_configs: list[MemmapDatasetConfig[_sample_type]],
        reader_configs: list[MemmapIndexDatasetReaderConfig],
        splits: dict[str, int | float],
        output_path: pathlib.Path,
    ) -> dict[str, tuple[SampledDatasetConfig[_sample_type], dict[str, typing.Any]]]:
        split_cumsum = padded_cumsum(normalize_probabilities(list(splits.values()), return_array=True)).tolist()
        dataset_sizes = [reader_config.num_tokens for reader_config in reader_configs]
        dataset_probabilities = normalize_probabilities(dataset_sizes)
        dataset_cumsums = padded_cumsum(dataset_probabilities).tolist()
        dataset_splits = {}

        for split_index, split_name in enumerate(splits):
            datasets_in_split = []
            datasets_metadata = []
            for dataset_index, (dataset_config, reader_config) in enumerate(
                zip(dataset_configs, reader_configs, strict=True)
            ):
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
                    datasets_metadata.append(reader_config.get_metadata())

                elif split_end_in_dataset > split_begin_in_dataset:
                    # Part of the dataset belongs to the split.
                    # TODO: Somehow getting a segfault when merging two lines below (numpy bug?).
                    dataset = dataset_config.to_copy({"path": output_path / dataset_config.path}).build(
                        self._preprocessing_config
                    )
                    begin_index, end_index, metadata = dataset.reader.get_split(
                        split_begin_in_dataset, split_end_in_dataset
                    )
                    if end_index > begin_index:
                        datasets_in_split.append(
                            DatasetSliceConfig[self._sample_type].from_dict(
                                {
                                    "type": "slice",
                                    "dataset": dataset_configs[dataset_index],
                                    "begin": begin_index / len(reader_config),
                                    "end": end_index / len(reader_config),
                                }
                            )
                        )
                        datasets_metadata.append(metadata)

                # [else] None of the dataset belongs to the split.

            if len(datasets_in_split) == 0:
                # This is a big problem, but we don't want to crash the whole run.
                logger.error(f"Datasets split {split_name} is empty!")
            elif len(datasets_in_split) == 1:
                dataset_splits[split_name] = (datasets_in_split[0], datasets_metadata[0])
            else:
                dataset_splits[split_name] = (
                    BlendedDatasetConfig[self._sample_type].from_dict(
                        {
                            "type": "blended",
                            "datasets": datasets_in_split,
                            "weights": [dataset_metadata["num_tokens"] for dataset_metadata in datasets_metadata],
                        }
                    ),
                    reader_configs[0].blend_metadata(datasets_metadata),
                )

        return dataset_splits


def _sort_spans(spans: typing.Iterable[tuple[SpanType, tuple[int, int]]]) -> list[tuple[SpanType, tuple[int, int]]]:
    return sorted(spans, key=lambda span: span[1][0])


def _get_nearest_split(cumsum: np.ndarray, value: float) -> int:
    left = cumsum.searchsorted(value, side="right")
    if left == len(cumsum):
        return left.item()
    return left.item() + 1 if (value - cumsum[left]) / (cumsum[left + 1] - cumsum[left]) > 0.5 else left.item()
