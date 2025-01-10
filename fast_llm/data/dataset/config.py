import dataclasses
import functools
import math
import pathlib
import typing

from fast_llm.config import Config, Field, FieldHint, check_field, config_class
from fast_llm.data.dataset.abstract import (
    PhaseSplits,
    SamplableDataset,
    SamplableSplitDataset,
    SampledDataset,
    SampledSplitDataset,
)
from fast_llm.engine.distributed.config import PhaseType
from fast_llm.utils import Assert


@config_class()
class DatasetConfig(Config):
    _abstract = True


@dataclasses.dataclass
class SamplingConfig:
    num_samples: int = 1
    seed: int = 0
    cache_directory: pathlib.Path | None = None
    verbose: bool = True
    distributed: typing.Any = None


@config_class()
class SampledSplitDatasetConfig(DatasetConfig):

    def build_split_sample(
        self,
        config: PhaseSplits[SamplingConfig],
        default_phase: PhaseType = PhaseType.training,
    ) -> SampledSplitDataset:
        raise NotImplementedError()

    @property
    def sampled(self):
        # Generally hard-coded, but some classes allow for more flexible values.
        return True

    @property
    def split(self):
        # Generally hard-coded, but some classes allow for more flexible values.
        return True


@config_class()
class SampledDatasetConfig(SampledSplitDatasetConfig):
    """
    A sampled dataset containing a prepared list of samples to be indexed sequentially (as-is) during training.
    (See `fast_llm.data.sampler.Sampler`.)
    """

    def build_sample(self, config: SamplingConfig) -> SampledDataset:
        raise NotImplementedError()

    def build_split_sample(
        self,
        config: PhaseSplits[SamplingConfig],
        default_phase: PhaseType = PhaseType.training,
    ) -> SampledSplitDataset:
        dataset = self.build_sample(config[default_phase])
        return SampledSplitDataset(dataset.name, {default_phase: dataset})

    @property
    def sampled(self):
        return True

    @property
    def split(self):
        return False


@config_class()
class SamplableSplitDatasetConfig(SampledSplitDatasetConfig):

    def build_split(
        self,
        default_phase: PhaseType = PhaseType.training,
    ) -> SamplableSplitDataset:
        raise NotImplementedError()

    def build_split_sample(
        self,
        config: PhaseSplits[SamplingConfig],
        default_phase: PhaseType = PhaseType.training,
    ) -> SampledSplitDataset:
        split_dataset = self.build_split(default_phase)
        # TODO: Name
        return SampledSplitDataset(
            "dataset",
            {phase: split_dataset[phase].sample(phase_config) for phase, phase_config in config.items()},
        )

    @property
    def sampled(self):
        return False

    @property
    def split(self):
        return True


@config_class()
class SamplableDatasetConfig(SampledDatasetConfig, SamplableSplitDatasetConfig):
    def build(self) -> SamplableDataset:
        raise NotImplementedError()

    def build_sample(self, config: SamplingConfig) -> SampledDataset:
        return self.build().sample(config)

    def build_split(
        self,
        default_phase: PhaseType = PhaseType.training,
    ) -> SamplableSplitDataset:
        dataset = self.build()
        return SamplableSplitDataset(dataset.name, {default_phase: dataset})

    @property
    def sampled(self):
        return False

    @property
    def split(self):
        return False


@config_class()
class BlendedDatasetConfig(SampledDatasetConfig):
    # [(?sampled, ?split)] -> (sampled, ?split)
    name: str = Field(
        default="blended",
        desc="The name of the dataset.",
        hint=FieldHint.core,
    )
    datasets: list[SampledDatasetConfig] = Field(
        default_factory=list,
        desc="The datasets to blend.",
        hint=FieldHint.core,
        valid=check_field(functools.partial(Assert.custom, lambda x: len(x) > 0)),
    )
    weights: list[float] = Field(
        default_factory=list,
        desc="The blending weight of each dataset.",
        hint=FieldHint.core,
    )

    def __post_init__(self):
        Assert.eq(len(self.datasets), len(self.weights))

    @property
    def split(self):
        return any(dataset.split for dataset in self.datasets)

    def build_sample(
        self,
        config: SamplingConfig,
    ) -> SampledDataset:
        from fast_llm.data.dataset.blended import BlendedDataset

        assert not self.split

        # Build and sample the datasets.
        sampled_datasets = [
            dataset.build_sample(
                # Blending is deterministic and the error will never be higher than 1.
                config.to_copy({"num_samples": math.ceil(weight * config.num_samples) + 1}),
            )
            for dataset, weight in zip(self.datasets, self.weights, strict=True)
        ]
        # Blend the datasets.
        return BlendedDataset(
            self.name,
            sampled_datasets,
            self.weights,
            config,
        )

    def build_split_sample(
        self,
        config: PhaseSplits[SamplingConfig],
        default_phase: PhaseType = PhaseType.training,
    ) -> SampledSplitDataset:
        from fast_llm.data.dataset.blended import BlendedDataset

        if not self.split:
            # Take the base class shortcut using build_sample if it's available.
            return super().build_split_sample(config, default_phase)

        # Build, sample and split the datasets.
        sampled_datasets = [
            dataset.build_split_sample(
                # Blending is deterministic and the error will never be higher than 1.
                PhaseSplits[SamplingConfig](
                    {
                        phase: phase_config.to_copy({"num_samples": math.ceil(weight * phase_config.num_samples) + 1})
                        for phase, phase_config in config.items()
                    }
                ),
                default_phase,
            )
            for dataset, weight in zip(self.datasets, self.weights, strict=True)
        ]

        # Blend the datasets for each phase.
        return SampledSplitDataset[BlendedDataset](
            self.name,
            {
                phase: BlendedDataset(
                    f"{self.name}_{phase.value}",
                    [dataset[phase] for dataset in sampled_datasets],
                    self.weights,
                    phase_config,
                )
                for phase, phase_config in config.items()
            },
        )
