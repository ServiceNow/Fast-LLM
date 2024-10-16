import json
import logging
import math
import pathlib
import typing
import warnings

import numpy as np
import torch
import torch.utils.data

from fast_llm.data.config import AbstractData, DataConfig, DatasetSource
from fast_llm.data.dataset import BlendedDataset, SampledDataset, Sampler
from fast_llm.data.gpt import DummyGPTDataset, GPTDataset, GPTSampledDataset
from fast_llm.data.mmap import MMapIndexedDataset
from fast_llm.data.tokenizer import Tokenizer
from fast_llm.engine.config_utils.run import get_run, log_main_rank
from fast_llm.engine.distributed.config import DistributedConfig, PhaseType
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.schedule.config import BatchConfig
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)


def normalize_probs(p: list[float]) -> list[float]:
    p = np.array(p)
    Assert.custom(lambda x: np.all(x >= 0), p)
    p_sum = p.sum()
    Assert.gt(p_sum, 0)
    return (p / p_sum).tolist()


class Data(AbstractData):
    """
    A global class for all dataset needs, including loading, splitting, sampling and iteration.
    Currently hard-coded to a GPT dataset.
    TODO: Separate generic and GPT classes.
    """

    _sampled_datasets: dict[PhaseType, dict[str, SampledDataset]]
    _blended_datasets: dict[PhaseType, SampledDataset]
    _tokenizer: Tokenizer | None
    _distributed: Distributed
    _cache_dir: pathlib.Path | None
    _samples_per_phase: dict[PhaseType, int]
    _phases: typing.ClassVar[tuple[PhaseType, ...]] = (PhaseType.training, PhaseType.validation, PhaseType.test)

    def __init__(
        self,
        config: DataConfig,
        distributed_config: DistributedConfig,
        vocab_size: int,
        max_sequence_length: int,
    ):
        """
        Create the data and gather some basic information on the dataset(s).
        Should be `setup` before use.
        """
        self._config = config.validate()
        self._distributed_config = distributed_config.validate()
        self._vocab_size = vocab_size
        self._max_sequence_length = max_sequence_length
        Assert.eq(len(self._config.split), len(self._phases))
        self._phase_split = {
            phase: ratio for phase, ratio in zip(self._phases, normalize_probs(self._config.split)) if ratio > 0
        }

        data_base_path = None
        if self._config.format == DatasetSource.file:
            Assert.eq(len(self._config.path), 1)
            data_path = pathlib.Path(self._config.path[0])
            dataset_defs = json.load(data_path.open("r"))
            data_base_path = data_path.parent
            dataset_prefixes = [dataset_def["prefix"] for dataset_def in dataset_defs["datasets"]]
            dataset_weights = normalize_probs([dataset_def["weight"] for dataset_def in dataset_defs["datasets"]])
            self._build_and_sample_dataset = self._build_and_sample_gpt_dataset
        elif self._config.format == DatasetSource.list:
            Assert.geq(len(self._config.path), 1)
            if len(self._config.path) == 1:
                dataset_prefixes, dataset_weights = [self._config.path[0].strip()], [1.0]
            else:
                Assert.custom(lambda x: x % 2 == 0, len(self._config.path))
                dataset_prefixes = [x.strip() for x in self._config.path[1::2]]
                assert len(dataset_prefixes) == len(set(dataset_prefixes))
                dataset_weights = normalize_probs([float(x) for x in self._config.path[::2]])
            self._build_and_sample_dataset = self._build_and_sample_gpt_dataset
        elif self._config.format == DatasetSource.sample:
            Assert.eq(len(self._config.path), 1)
            dataset_prefixes, dataset_weights = [self._config.path[0].strip()], [1.0]
            self._build_and_sample_dataset = self._build_and_sample_dummy_dataset
        elif self._config.format == DatasetSource.random:
            Assert.eq(len(self._config.path), 0)
            dataset_prefixes, dataset_weights = [None], [1.0]
            self._build_and_sample_dataset = self._build_and_sample_dummy_dataset
        else:
            raise NotImplementedError(self._config.format)

        dataset_names = [
            f"dataset_{i}_{'dummy' if prefix is None else prefix.replace('/','__')}"
            for i, prefix in enumerate(dataset_prefixes)
        ]
        self._num_datasets = len(dataset_names)
        self._dataset_prefixes = {
            name: (
                None
                if prefix is None
                else (
                    pathlib.Path(prefix).resolve()
                    if data_base_path is None
                    else (pathlib.Path(data_base_path) / prefix).resolve()
                )
            )
            for name, prefix in zip(dataset_names, dataset_prefixes)
        }
        self._dataset_weights = {name: weight for name, weight in zip(dataset_names, dataset_weights)}

    def setup(self, distributed: Distributed, samples_per_phase: dict[PhaseType, int]):
        """
        Load the datasets, and prepare or load the samplings.
        This may take a while and a significant amount of cpu memory.
        """
        run = get_run()
        Assert.leq(set(samples_per_phase), set(self._phase_split))
        log_main_rank(f"Preparing {self._num_datasets} datasets. This may take several minutes.")
        self._tokenizer = Tokenizer(self._config.tokenizer) if self._config.fim.rate > 0 else None
        self._distributed = distributed
        self._cache_dir = run.dataset_cache_dir
        self._samples_per_phase = samples_per_phase
        if self._cache_dir is None:
            warnings.warn(f"Using the dataset directory for the index cache.")

        # Build and split datasets.
        self._sampled_datasets = {phase: {} for phase in self._samples_per_phase}
        for i, (name, weight) in enumerate(self._dataset_weights.items()):
            if i % 100 == 0 and i > 0:
                log_main_rank(f"Prepared {i} of {self._num_datasets} datasets.")
            dataset_samples_per_phase = {}
            for phase, samples_per_phase in self._samples_per_phase.items():
                expected_samples = self._dataset_weights[name] * samples_per_phase
                # Add 5 times the standard deviation (of a binomial distribution)
                # so the probability of sampling more than this amount during blending is negligible.
                dataset_samples_per_phase[phase] = math.ceil(
                    expected_samples
                    + 5 * math.sqrt(expected_samples * self._dataset_weights[name] * (1 - self._dataset_weights[name]))
                )
            sampled_datasets = self._build_and_sample_dataset(name, dataset_samples_per_phase)
            for phase, dataset in sampled_datasets.items():
                self._sampled_datasets[phase][name] = dataset

        self._blended_datasets = {
            phase: (
                list(datasets.values())[0]
                if len(datasets) == 1
                else BlendedDataset(
                    list(datasets.values()),
                    weights=[self._dataset_weights[name] for name in datasets],
                    name=phase.value,
                    num_samples=self._samples_per_phase[phase],
                    cache_dir=self._cache_dir,
                    group=self._distributed.world_group,
                    verbose=run.is_main_rank,
                    data_sample_warn_time_ms=self._config.data_sample_warn_time_ms,
                )
            )
            for phase, datasets in self._sampled_datasets.items()
        }

    def get_iterator(
        self,
        batch_config: BatchConfig,
        phase: PhaseType,
        *,
        consumed_samples: int,
        num_workers: int,
        prefetch_factor: int | None = None,
    ):
        Assert.incl(phase, self._blended_datasets)
        Assert.in_range_incl(batch_config.sequence_length, 1, self._max_sequence_length)
        log_main_rank(f"Initializing {phase} data iterator from sample {consumed_samples}...")
        return iter(
            torch.utils.data.DataLoader(
                self._blended_datasets[phase],  # noqa
                batch_sampler=Sampler(
                    total_samples=len(self._blended_datasets[phase]),
                    begin_index=consumed_samples,
                    micro_batch_size=batch_config.micro_batch_size,
                    data_rank=self._distributed.config.batch_data_rank,
                    data_parallel=self._distributed.config.batch_data_parallel,
                ),
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                pin_memory=True,
                multiprocessing_context=self._config.multiprocessing_context.value if num_workers > 0 else None,
            )
        )

    def _build_and_sample_gpt_dataset(self, name: str, dataset_samples_per_phase: dict[PhaseType, int]):
        dataset_split = GPTDataset.from_splits(
            name, MMapIndexedDataset(self._dataset_prefixes[name]), self._phase_split
        )

        sampled_datasets = {}
        for phase, num_samples in dataset_samples_per_phase.items():
            if num_samples == 0:
                continue
            sampled_datasets[phase] = GPTSampledDataset(
                dataset_split[phase],
                num_samples=num_samples,
                sequence_length=self._max_sequence_length,
                seed=self._distributed.config.seed,
                group=self._distributed.world_group,
                config=self._config,
                tokenizer=self._tokenizer,
                cache_dir=self._dataset_prefixes[name].parent if self._cache_dir is None else self._cache_dir,
                verbose=self._num_datasets <= 5,
            )
        return sampled_datasets

    def _build_and_sample_dummy_dataset(self, name: str, dataset_samples_per_phase: dict[PhaseType, int]):
        return {
            phase: DummyGPTDataset(
                self._dataset_prefixes[name],
                dataset_samples_per_phase[phase],
                self._max_sequence_length,
                self._vocab_size,
                name,
            )
            for phase in dataset_samples_per_phase
        }
