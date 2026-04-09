import enum
import functools
import itertools
import logging
import math
import pathlib
import typing

from fast_llm.config import Config, Field, FieldHint, check_field, config_class
from fast_llm.data.dataset.abstract import SamplableDataset, SampledDataset
from fast_llm.data.document.abstract import Document
from fast_llm.utils import Assert, normalize_probabilities

if typing.TYPE_CHECKING:
    from fast_llm.data.dataset.indexed import ConcatenatedDataset, DatasetSlice, IndexedDataset
    from fast_llm.data.document.language_model import LanguageModelDocument

logger = logging.getLogger(__name__)


class ShufflingType(enum.StrEnum):
    """Strategy for shuffling dataset samples across training epochs."""

    # Shuffle all epochs together. Not extendable.
    full = "full"
    # Shuffle all epochs separately. Default mode, recommended if the dataset doesn't come pre-shuffled.
    epoch = "epoch"
    # Shuffle all epochs except the first one. Recommended for pre-shuffled datasets, especially big ones.
    skip_first_epoch = "skip_first_epoch"
    # Disable shuffling entirely.
    disabled = "disabled"


@config_class()
class SamplingConfigBase(Config):
    """
    A dataset-dependent configuration for sampling.
    """

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
    micro_batch_size: int = Field(
        default=2048,
        desc="Size of individual micro-batches.",
        hint=FieldHint.core,
        valid=check_field(Assert.gt, 0),
    )
    maximum_document_length: int | None = Field(
        default=None,
        desc="Maximum number of tokens in a document."
        " Document exceeding this size will be truncated or dropped depending on `truncate_documents`.",
        hint=FieldHint.core,
    )
    truncate_documents: bool | None = Field(
        default=True,
        desc=(
            "If enabled, documents may be truncated while being packed to fit the sequence length."
            "Otherwise, sequences will be padded such that every document lies entirely within a sample"
            " (and documents exceeding the sequence length will be skipped altogether)."
        ),
        hint=FieldHint.feature,
    )


@config_class()
class SamplingConfig(SamplingConfigBase):
    """
    Holds all the necessary information for sampling.
    """

    # How many extra tokens to add to the sequence length.
    # This is used to provide labels even for the last tokens in the sequence.
    predicted_tokens: int = Field(default=1)
    token_cumsum_rate: int = Field(
        default=10,
        desc="Sampling interval for the token cumulative sum index."
        " A smaller value reduces per-sample seek time at the cost of a larger index.",
        hint=FieldHint.performance,
        valid=check_field(Assert.gt, 0),
    )
    cache_directory: pathlib.Path | None = Field(default=None)
    dataset_name: str = Field(default="dataset")
    world_size: int = Field(default=1)
    rank: int = Field(default=0)
    _rank_counter: typing.Iterator[int] = Field(init=False)

    def _validate(self):
        # Using itertools.count to make the field mutable.
        self._rank_counter = itertools.count()
        super()._validate()

    def is_running_next(self) -> bool:
        # Counter that loops over ranks to try to distribute workloads evenly between ranks.
        return next(self._rank_counter) % self.world_size == self.rank

    @functools.cached_property
    def sample_size(self) -> int:
        return self.micro_batch_size + self.predicted_tokens

    @functools.cached_property
    def sampling_maximum_document_length(self) -> int:
        if self.maximum_document_length is None:
            return self.sample_size
        else:
            return min(self.maximum_document_length, self.sample_size)


@config_class()
class DatasetConfig[DocumentType: Document](Config):
    """Abstract base configuration for all dataset types."""

    _abstract: typing.ClassVar[bool] = True


@config_class(registry=True)
class SampledDatasetConfig[DocumentType: Document](DatasetConfig[DocumentType]):
    """
    A sampled dataset containing a prepared list of samples to be indexed sequentially (as-is) during training.
    """

    def build_and_sample(self, config: SamplingConfig, num_samples: int, seed: int) -> SampledDataset[DocumentType]:
        raise NotImplementedError()


@config_class()
class SamplableDatasetConfig[DocumentType: Document](SampledDatasetConfig[DocumentType]):
    """Abstract configuration for datasets that can be built and then sampled."""

    def build(self) -> SamplableDataset[DocumentType]:
        raise NotImplementedError()

    def build_and_sample(self, config: SamplingConfig, num_samples: int, seed: int) -> SampledDataset[DocumentType]:
        return self.build().sample(config, num_samples, seed)


@config_class()
class IndexedDatasetConfig[DocumentType: Document](SamplableDatasetConfig[DocumentType]):
    """Abstract configuration for indexed datasets that support random access by index."""

    def build(self) -> "IndexedDataset[DocumentType]":
        raise NotImplementedError()


@config_class(dynamic_type={SampledDatasetConfig: "concatenated"})
class ConcatenatedDatasetConfig[DocumentType: Document](SamplableDatasetConfig[DocumentType]):
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
    datasets: list[IndexedDatasetConfig[DocumentType]] = Field(
        default_factory=list,
        desc="The datasets to concatenate.",
        hint=FieldHint.core,
        valid=check_field(functools.partial(Assert.custom, lambda x: len(x) > 0)),
    )

    def build(self) -> "ConcatenatedDataset":
        from fast_llm.data.dataset.indexed import ConcatenatedDataset

        return ConcatenatedDataset(self.name, [dataset.build() for dataset in self.datasets])


@config_class(dynamic_type={SampledDatasetConfig: "slice"})
class DatasetSliceConfig[DocumentType: Document](SamplableDatasetConfig[DocumentType]):
    """
    Use a fraction of an indexed dataset, specified by the range (begin, end).
    Typically used to subsample a dataset, or to reserve part of the dataset for validation and/or testing.
    Ex. use (0.0, 0.9) for train, (0.9, 1.0) for validation for a 90%-10% split.
    TODO: This is suboptimal (duplication between train/test, unnecessary sub-datasets in the case of concatenation,
        leads to higher resource usage than necessary; more open files?)
    """

    _abstract = False
    dataset: IndexedDatasetConfig[DocumentType] = Field(
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
        return DatasetSlice[DocumentType](
            f"{dataset.name}_{self.begin}_{self.end}",
            dataset,
            round(self.begin * size),
            round(self.end * size),
        )


@config_class(dynamic_type={SampledDatasetConfig: "blended"})
class BlendedDatasetConfig[DocumentType: Document](SampledDatasetConfig[DocumentType]):
    """Mixes multiple datasets together, sampling from each according to specified weights."""

    _abstract = False
    name: str = Field(
        default="blended",
        desc="The name of the dataset.",
        hint=FieldHint.core,
    )
    datasets: list[SampledDatasetConfig[DocumentType]] = Field(
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

    def build_and_sample(self, config: SamplingConfig, num_samples: int, seed: int) -> SampledDataset[DocumentType]:
        from fast_llm.data.dataset.blended import BlendedDataset

        # Build and sample the datasets.

        sampled_datasets = [
            dataset.build_and_sample(
                # Blending is deterministic and the error will never be higher than 1.
                config,
                num_samples=math.ceil(weight * num_samples) + 1,
                # TODO: Seed may not be unique for nested blended datasets.
                seed=seed + i * 697,
            )
            for i, (dataset, weight) in enumerate(zip(self.datasets, self.weights, strict=True))
        ]
        # Blend the datasets.
        return BlendedDataset[DocumentType](
            self.name,
            sampled_datasets,
            self.weights,
            config,
            num_samples,
        )


REDIS_DATA_STREAM = "fast_llm_streaming"
REDIS_GROUP_NAME = "fast_llm_group"


@config_class()
class RedisConfig(Config):
    """Configuration for connecting to a Redis server (host, port, timeout)."""

    REDIS_FIELD: typing.ClassVar[str] = "data"
    REDIS_FIELD_B: typing.ClassVar[bytes] = REDIS_FIELD.encode()
    REDIS_GROUP_NAME: typing.ClassVar[str] = "fast_llm_group"
    REDIS_GROUP_NAME_B: typing.ClassVar[bytes] = REDIS_GROUP_NAME.encode()

    # TODO: Move elsewhere? (Also used in trainer) Get it from the trainer in sampling config?
    host: str = Field(
        default="localhost",
        desc="Hostname or IP address of the Redis server.",
        hint=FieldHint.core,
    )

    port: int = Field(
        default=6379,
        desc="Port number on which the Redis server is running.",
        hint=FieldHint.core,
    )
    timeout: float = Field(default=600.0, desc="Timeout (seconds) for sending and receiving data.")

    def get_client(self):
        import redis

        return redis.Redis(self.host, self.port)


@config_class(dynamic_type={SampledDatasetConfig: "streaming"})
class StreamingDatasetConfig[DocumentType: LanguageModelDocument](RedisConfig, SamplableDatasetConfig[DocumentType]):
    """
    Configuration for a streaming dataset that reads training data from a Redis stream.
    """

    _abstract = False

    def build_and_sample(self, config: SamplingConfig, num_samples: int, seed: int) -> SampledDataset[DocumentType]:
        from fast_llm.data.dataset.streaming import RedisStreamingDataset

        return RedisStreamingDataset[StreamingDatasetConfig, DocumentType](self).sample(config, num_samples, seed)
