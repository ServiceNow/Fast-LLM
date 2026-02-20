import dataclasses
import enum
import functools
import itertools
import logging
import math
import pathlib
import typing

from fast_llm.config import Config, Field, FieldHint, UpdateType, check_field, config_class
from fast_llm.data.dataset.abstract import SamplableDataset, SampledDataset
from fast_llm.data.document.abstract import Document
from fast_llm.data.preprocessing.abstract import PreprocessingConfig
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.utils import Assert, normalize_probabilities

if typing.TYPE_CHECKING:
    from fast_llm.data.dataset.indexed import ConcatenatedDataset, DatasetSlice, IndexedDataset

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

    @functools.cached_property
    def total_length(self) -> int:
        return self.sequence_length + self.extra_tokens


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
    distributed_config: DistributedConfig
    dataset_name: str
    preprocessing: PreprocessingConfig
    # Using a mutable rather than an int so it's shared with all copies made with `update`.
    _rank_counter: typing.Iterator[int] = itertools.count

    def update_config(self, update: SamplingConfig):
        return dataclasses.replace(
            self, config=self.config.from_dict(self.config, update.to_dict(), update_type=UpdateType.update)
        )

    def get_next_rank(self) -> int:
        # Counter that loops over ranks to try to distribute workloads evenly between ranks.
        return next(self._rank_counter()) % self.distributed_config.world_size


@config_class()
class DatasetConfig[DocumentType: Document](Config):
    _abstract: typing.ClassVar[bool] = True


@config_class(registry=True)
class SampledDatasetConfig[DocumentType: Document](DatasetConfig[DocumentType]):
    """
    A sampled dataset containing a prepared list of samples to be indexed sequentially (as-is) during training.
    """

    def build_and_sample(self, sampling: SamplingData) -> SampledDataset[DocumentType]:
        raise NotImplementedError()


@config_class()
class SamplableDatasetConfig[DocumentType: Document](SampledDatasetConfig[DocumentType]):
    def build(self, preprocessing: PreprocessingConfig) -> SamplableDataset[DocumentType]:
        raise NotImplementedError()

    def build_and_sample(self, sampling: SamplingData) -> SampledDataset[DocumentType]:
        return self.build(sampling.preprocessing).sample(sampling)


@config_class()
class IndexedDatasetConfig[DocumentType: Document](SamplableDatasetConfig[DocumentType]):
    def build(self, preprocessing: PreprocessingConfig) -> "IndexedDataset[DocumentType]":
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

    def build(self, preprocessing: PreprocessingConfig) -> "ConcatenatedDataset":
        from fast_llm.data.dataset.indexed import ConcatenatedDataset

        return ConcatenatedDataset(self.name, [dataset.build(preprocessing) for dataset in self.datasets])


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

    def build(self, preprocessing: PreprocessingConfig) -> "DatasetSlice":
        from fast_llm.data.dataset.indexed import DatasetSlice

        dataset = self.dataset.build(preprocessing)
        size = len(dataset)
        return DatasetSlice[DocumentType](
            f"{dataset.name}_{self.begin}_{self.end}",
            dataset,
            round(self.begin * size),
            round(self.end * size),
        )


@config_class(dynamic_type={SampledDatasetConfig: "sampled"})
class SampledDatasetUpdateConfig[DocumentType: Document](SampledDatasetConfig[DocumentType]):
    """
    Wrap a dataset to explicitly sample from it and optionally update its configuration parameters.
    Only explicitly set parameters (not None) will be updated, other will still be taken from `build_and_sample`'s argument.
    """

    _abstract = True
    sampling: SamplingConfig = Field(
        desc="Optional override to sampling configuration parameters.",
        hint=FieldHint.core,
    )
    dataset: SampledDatasetConfig[DocumentType] = Field(
        desc="The dataset to sample from.",
        hint=FieldHint.core,
    )

    def build_and_sample(self, data: SamplingData) -> SampledDataset[DocumentType]:
        return self.dataset.build_and_sample(data.update_config(self.sampling))


@config_class(dynamic_type={SampledDatasetConfig: "blended"})
class BlendedDatasetConfig[DocumentType: Document](SampledDatasetConfig[DocumentType]):
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

    def build_and_sample(
        self,
        sampling: SamplingData,
    ) -> SampledDataset[DocumentType]:
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
        return BlendedDataset[DocumentType](
            self.name,
            sampled_datasets,
            self.weights,
            sampling,
        )
