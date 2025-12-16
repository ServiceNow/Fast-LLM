import dataclasses
import enum
import functools
import itertools
import logging
import math
import pathlib
import typing

from fast_llm.config import Config, Field, FieldHint, FieldUpdate, UpdateType, check_field, config_class
from fast_llm.data.dataset.abstract import SamplableDataset, SampledDataset
from fast_llm.data.sample.abstract import Sample
from fast_llm.redis.config import RedisConfig
from fast_llm.utils import Assert, normalize_probabilities

if typing.TYPE_CHECKING:
    from fast_llm.data.dataset.abstract_iterable import SamplableIterableDataset, SampledIterableDataset
    from fast_llm.data.dataset.indexed import ConcatenatedDataset, DatasetSlice, IndexedDataset
    from fast_llm.data.sample.language_model import LanguageModelSample
    from fast_llm.engine.distributed.distributed import Distributed

logger = logging.getLogger(__name__)


class ShufflingType(str, enum.Enum):
    # Shuffle all epochs together. Not extendable.
    full = "full"
    # Shuffle all epochs separately. Default mode, recommended if the dataset doesn't come pre-shuffled.
    epoch = "epoch"
    # Shuffle all epochs except the first one. Recommended for pre-shuffled datasets, especially big ones.
    skip_first_epoch = "skip_first_epoch"
    # Disable shuffling entirely.
    disabled = "disabled"


@config_class()
class SamplingConfig(Config):
    """
    A dataset-dependent configuration for sampling.
    """

    seed: int = Field(
        default=784569,
        desc="Seed for random sampling.",
        hint=FieldHint.feature,
    )
    gpu: bool = Field(
        default=True,
        desc="Enable fast sampling on GPU."
        " Note that random sampling works differently on GPU,"
        " so the sample won't match the CPU equivalent.",
        hint=FieldHint.feature,
    )
    shuffle: ShufflingType = Field(
        default=ShufflingType.epoch,
        desc="Shuffling strategy.",
        hint=FieldHint.feature,
    )


@dataclasses.dataclass(kw_only=True)
class SamplingParameters:
    """
    Sampling parameters set externally to the dataset and data, ex. determined by the trainer or model.
    """

    sequence_length: int
    num_samples: int
    truncate_documents: bool = True
    # How many extra tokens to add to the sequence length.
    # This is used to provide labels even for the last tokens in the sequence.
    extra_tokens: int = 1


@dataclasses.dataclass(kw_only=True)
class SamplingData:
    """
    Holds all the necessary information for sampling, including dataset-dependent ones (`SamplingConfig`),
    usage-dependent ones (`SamplingParameters`), and others set by the `Data`.
    """

    # TODO: Have a separate configuration (subset?) for `build`?
    config: SamplingConfig
    parameters: SamplingParameters
    cache_directory: pathlib.Path | None
    # TODO: This prevents the sampling config from being pickled in multiprocessing.
    distributed: "Distributed"
    dataset_name: str
    # Using a mutable rather than an int so it's shared with all copies made with `update`.
    _rank_counter: typing.Iterator[int] = itertools.count

    def update_config(self, update: SamplingConfig):
        return dataclasses.replace(
            self, config=self.config.from_dict(self.config, update.to_dict(), update_type=UpdateType.update)
        )

    def get_next_rank(self) -> int:
        # Counter that loops over ranks to try to distribute workloads evenly between ranks.
        return next(self._rank_counter()) % self.distributed.config.world_size


@config_class()
class DatasetConfig[SampleType: Sample](Config):
    _abstract: typing.ClassVar[bool] = True


@config_class(registry=True)
class SampledDatasetConfig[SampleType: Sample](DatasetConfig[SampleType]):
    """
    A sampled dataset containing a prepared list or iterable of samples to be indexed sequentially (as-is) during training.
    """

    def build_and_sample(
        self, sampling: SamplingData
    ) -> "SampledDataset[SampleType] | SampledIterableDataset[SampleType]":
        raise NotImplementedError()


@config_class()
class SamplableDatasetConfig[SampleType: Sample](SampledDatasetConfig[SampleType]):
    def build(self) -> "SamplableDataset[SampleType] | SamplableIterableDataset[SampleType]":
        raise NotImplementedError()

    def build_and_sample(
        self, sampling: SamplingData
    ) -> "SampledDataset[SampleType] | SampledIterableDataset[SampleType]":
        return self.build().sample(sampling)


@config_class()
class IndexedDatasetConfig[SampleType: Sample](SamplableDatasetConfig[SampleType]):
    def build(self) -> "IndexedDataset[SampleType]":
        raise NotImplementedError()


@config_class(dynamic_type={SampledDatasetConfig: "concatenated"})
class ConcatenatedDatasetConfig[SampleType: Sample](SamplableDatasetConfig[SampleType]):
    """
    Concatenate multiple indexed datasets as if they were one.
    TODO: Make a post-sampling version? (staged training)
    """

    _abstract = False
    name: str = Field(
        default="concatenated",
        desc="The name of the dataset.",
        hint=FieldHint.core,
    )
    datasets: list[IndexedDatasetConfig[SampleType]] = Field(
        default_factory=list,
        desc="The datasets to concatenate.",
        hint=FieldHint.core,
        valid=check_field(functools.partial(Assert.custom, lambda x: len(x) > 0)),
    )

    def build(self) -> "ConcatenatedDataset":
        from fast_llm.data.dataset.indexed import ConcatenatedDataset

        return ConcatenatedDataset(self.name, [dataset.build() for dataset in self.datasets])


@config_class(dynamic_type={SampledDatasetConfig: "slice"})
class DatasetSliceConfig[SampleType: Sample](SamplableDatasetConfig[SampleType]):
    """
    Use a fraction of an indexed dataset, specified by the range (begin, end).
    Typically used to subsample a dataset, or to reserve part of the dataset for validation and/or testing.
    Ex. use (0.0, 0.9) for train, (0.9, 1.0) for validation for a 90%-10% split.
    TODO: This is suboptimal (duplication between train/test, unnecessary sub-datasets in the case of concatenation,
        leads to higher resource usage than necessary; more open files?)
    """

    _abstract = False
    dataset: IndexedDatasetConfig[SampleType] = Field(
        default=None,
        desc="The dataset to split.",
        hint=FieldHint.core,
    )
    begin: float = Field(
        default=0,
        desc="The beginning of the dataset split, as a fraction of the total samples.",
        hint=FieldHint.core,
    )
    end: float = Field(
        default=1,
        desc="The end of the dataset split, as a fraction of the total samples.",
        hint=FieldHint.core,
    )

    def build(self) -> "DatasetSlice":
        from fast_llm.data.dataset.indexed import DatasetSlice

        dataset = self.dataset.build()
        size = len(dataset)
        return DatasetSlice[SampleType](
            f"{dataset.name}_{self.begin}_{self.end}",
            dataset,
            round(self.begin * size),
            round(self.end * size),
        )


@config_class(dynamic_type={SampledDatasetConfig: "sampled"})
class SampledDatasetUpdateConfig[SampleType: Sample](SampledDatasetConfig[SampleType]):
    """
    Wrap a dataset to explicitly sample from it and optionally update its configuration parameters.
    Only explicitly set parameters (not None) will be updated, other will still be taken from `build_and_sample`'s argument.
    """

    _abstract = True
    sampling: SamplingConfig = Field(
        desc="Optional override to sampling configuration parameters.",
        hint=FieldHint.core,
    )
    dataset: SampledDatasetConfig[SampleType] = Field(
        desc="The dataset to sample from.",
        hint=FieldHint.core,
    )

    def build_and_sample(self, data: SamplingData) -> SampledDataset[SampleType]:
        return self.dataset.build_and_sample(data.update_config(self.sampling))


@config_class(dynamic_type={SampledDatasetConfig: "blended"})
class BlendedDatasetConfig[SampleType: Sample](SampledDatasetConfig[SampleType]):
    _abstract = False
    name: str = Field(
        default="blended",
        desc="The name of the dataset.",
        hint=FieldHint.core,
    )
    datasets: list[SampledDatasetConfig[SampleType]] = Field(
        default_factory=list,
        desc="The datasets to blend.",
        hint=FieldHint.core,
    )
    weights: list[float] = Field(
        default_factory=list,
        desc="The blending weight of each dataset.",
        hint=FieldHint.core,
    )

    def _validate(self) -> None:
        self.weights = normalize_probabilities(self.weights)
        super()._validate()
        Assert.geq(len(self.datasets), 2)
        Assert.eq(len(self.datasets), len(self.weights))

    def build_and_sample(
        self,
        sampling: SamplingData,
    ) -> SampledDataset[SampleType]:
        from fast_llm.data.dataset.blended import BlendedDataset

        # Build and sample the datasets.

        sampled_datasets = [
            dataset.build_and_sample(
                # Blending is deterministic and the error will never be higher than 1.
                dataclasses.replace(
                    sampling,
                    parameters=dataclasses.replace(
                        sampling.parameters,
                        num_samples=math.ceil(weight * sampling.parameters.num_samples) + 1,
                    ),
                    # TODO: Seed may not be unique for nested blended datasets.
                    config=sampling.config.to_copy({"seed": sampling.config.seed + i * 697}),
                ),
            )
            for i, (dataset, weight) in enumerate(zip(self.datasets, self.weights, strict=True))
        ]
        # Blend the datasets.
        return BlendedDataset[SampleType](
            self.name,
            sampled_datasets,
            self.weights,
            sampling,
        )


@config_class(dynamic_type={SampledDatasetConfig: "memmap"})
class MemmapDatasetConfig[SampleType: LanguageModelSample](IndexedDatasetConfig[SampleType]):
    _abstract: typing.ClassVar[bool] = False
    path: pathlib.Path = Field(
        default=None,
        desc="The path to the dataset, excluding the `.bin` or `.idx` suffix.",
        hint=FieldHint.core,
    )

    def build(self) -> "IndexedDataset[SampleType]":
        name = str(self.path).replace("/", "__")
        if self.path.is_file():
            from fast_llm.data.dataset.memmap import MemmapDataset

            return MemmapDataset[SampleType](name, self.path)
        elif self.path.with_suffix(".bin").is_file() and self.path.with_suffix(".idx").is_file():
            logger.warning(
                "Using the legacy memmap dataset format."
                " This format is deprecated and will be removed in a future release."
                " Please recreate the dataset in the new memmap format."
            )
            from fast_llm.data.dataset.gpt.legacy_memmap import LegacyMemmapDataset

            return LegacyMemmapDataset[SampleType](name, self.path)
        else:
            raise FileNotFoundError(self.path)


@config_class()
class StreamingDatasetRedisConfig(RedisConfig):
    stream_key: str = FieldUpdate(default="fast_llm_streaming")

    payload_key: str = FieldUpdate(
        default="data",
    )


class IngestionType(str, enum.Enum):
    CONSUMER_GROUP = "consumer_group"
    ONE_STREAM = "one_stream"
    N_STREAMS = "n_streams"


class HashType(str, enum.Enum):
    MESSAGE_INDEX = "message_index"
    """Use the index of the received message for hashing. Provides precise distribution but may not be well shuffled."""

    MESSAGE_ID = "message_id"
    """Hash messages based on their unique message ID. Good for probabilistic distribution.
       Redis message IDs are regenerated each time, so this is not reproducible.
    """

    MESSAGE_BODY = "message_body"
    """Hash messages based on their payload content (bytes). Distributes messages roughly evenly.
       Deterministic based on message content, but not perfectly balanced across ranks.
    """

    PRODUCER_PROVIDED = "producer_provided"
    """Use the hash or index provided by the producer. Allows deterministic splitting and perfect balance."""


@config_class(dynamic_type={SampledDatasetConfig: "streaming"})
class StreamingDatasetConfig[SampleType: LanguageModelSample](SamplableDatasetConfig[SampleType]):
    """
    Configuration for a streaming dataset that reads training data from a Redis stream.
    """

    _abstract = False

    redis: StreamingDatasetRedisConfig = Field(
        desc="Redis connection and stream settings used to fetch incoming training data.",
        hint=FieldHint.core,
    )

    group_name: str = Field(
        default="fast_llm_dp_group",
        desc="Name of the Redis consumer group used for data-parallel reading.",
        hint=FieldHint.core,
    )

    consumer_name_prefix: str = Field(
        default="fast_llm_dp_group_consumer",
        desc="Prefix used to generate unique consumer names for each rank in Redis consumer group.",
        hint=FieldHint.core,
    )

    ingestion_type: IngestionType = Field(
        default=IngestionType.CONSUMER_GROUP,
        desc="Strategy used to ingest data from Redis streams (consumer group, single stream, or multiple streams).",
        hint=FieldHint.core,
    )

    hash_type: HashType = Field(
        default=HashType.MESSAGE_ID,
        desc="How to compute hash for assigning messages to ranks.",
        hint=FieldHint.core,
    )

    hash_key: str = Field(
        default="hash",
        desc="Key in the message dict containing the hash or index provided by the producer.",
        hint=FieldHint.core,
    )

    ack_period_per_consumer: int = Field(
        default=10,
        desc="Number of messages after which the consumer acknowledges received IDs back to the Redis hash.",
        hint=FieldHint.core,
    )

    def build_and_sample(self, sampling: SamplingData) -> "SampledIterableDataset[SampleType]":
        from fast_llm.data.dataset.streaming import StreamingDataset

        return StreamingDataset[SampleType](self, sampling.distributed).sample(sampling)
