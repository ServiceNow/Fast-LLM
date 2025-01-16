import json
import logging
import math
import pathlib
import typing
import warnings

import torch
import torch.utils.data

from fast_llm.data.data.abstract import Data
from fast_llm.data.data.gpt.config import GPTDataConfig
from fast_llm.data.dataset.abstract import CopySplitDataset, PhaseSplits, SampledSplitDataset
from fast_llm.data.dataset.blended import BlendedDataset
from fast_llm.data.dataset.gpt.config import DatasetSource, GPTSamplingConfig
from fast_llm.data.dataset.gpt.dummy import GPTDummyDataset
from fast_llm.data.dataset.gpt.fim import FimDataset
from fast_llm.data.dataset.gpt.indexed import GPTDatasetSlice
from fast_llm.data.dataset.gpt.memmap import GPTMemmapDataset
from fast_llm.data.dataset.monitor import DatasetMonitor
from fast_llm.data.iterator import SampledDatasetIterator
from fast_llm.data.tokenizer import Tokenizer
from fast_llm.engine.config_utils.run import log_main_rank
from fast_llm.engine.distributed.config import DistributedConfig, PhaseType
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.schedule.config import BatchConfig
from fast_llm.utils import Assert, normalize_probabilities

logger = logging.getLogger(__name__)


class GPTData[ConfigType: GPTDataConfig](Data[ConfigType]):
    """
    A global class for all dataset needs, including loading, splitting, sampling and iteration.
    Currently hard-coded to a GPT dataset.
    TODO: Separate generic and GPT classes.
    """

    _datasets: SampledSplitDataset
    _tokenizer: Tokenizer | None
    _phases: typing.ClassVar[tuple[PhaseType, ...]] = (PhaseType.training, PhaseType.validation, PhaseType.test)
    _is_setup: bool = False

    def __init__(
        self,
        config: GPTDataConfig,
        distributed_config: DistributedConfig,
        vocab_size: int,
        max_sequence_length: int,
    ):
        """
        Create the data and gather some basic information on the dataset(s).
        Should be `setup` before use.
        """
        super().__init__(config, distributed_config)
        self._vocab_size = vocab_size
        self._max_sequence_length = max_sequence_length
        Assert.eq(len(self._config.split), len(self._phases))
        self._phase_split = {
            phase: ratio
            for phase, ratio in zip(self._phases, normalize_probabilities(self._config.split))
            if ratio > 0
        }

        data_base_path = None
        if self._config.format == DatasetSource.file:
            Assert.eq(len(self._config.path), 1)
            data_path = pathlib.Path(self._config.path[0])
            dataset_defs = json.load(data_path.open("r"))
            data_base_path = data_path.parent
            dataset_prefixes = [dataset_def["prefix"] for dataset_def in dataset_defs["datasets"]]
            dataset_weights = normalize_probabilities(
                [dataset_def["weight"] for dataset_def in dataset_defs["datasets"]]
            )
            self._build_and_sample_dataset = self._build_and_sample_gpt_dataset
        elif self._config.format == DatasetSource.list:
            Assert.geq(len(self._config.path), 1)
            if len(self._config.path) == 1:
                dataset_prefixes, dataset_weights = [self._config.path[0].strip()], [1.0]
            else:
                Assert.custom(lambda x: x % 2 == 0, len(self._config.path))
                dataset_prefixes = [x.strip() for x in self._config.path[1::2]]
                assert len(dataset_prefixes) == len(set(dataset_prefixes))
                dataset_weights = normalize_probabilities([float(x) for x in self._config.path[::2]])
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

    def setup(
        self,
        distributed: "Distributed",
        samples_per_phase: dict[PhaseType, int],
        cache_directory: pathlib.Path,
    ) -> None:
        """
        Load the datasets, and prepare or load the samplings.
        This may take a while and a significant amount of cpu memory.
        """
        super().setup(distributed, samples_per_phase, cache_directory)
        Assert.leq(set(samples_per_phase), set(self._phase_split))
        log_main_rank(f"Preparing {self._num_datasets} datasets. This may take several minutes.")
        self._tokenizer = Tokenizer(self._config.tokenizer) if self._config.fim.rate > 0 else None
        self._distributed = distributed
        self._samples_per_phase = samples_per_phase
        if self._cache_directory is None:
            # TODO: Avoid this
            warnings.warn(f"Using the dataset directory for the index cache.")

        datasets_and_weights = []
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

            sampling_configs = PhaseSplits[GPTSamplingConfig](
                {
                    phase: GPTSamplingConfig(
                        num_samples=dataset_samples_per_phase[phase],
                        seed=self._distributed_config.seed,
                        cache_directory=(
                            self._dataset_prefixes[name].parent
                            if self._cache_directory is None and isinstance(self._dataset_prefixes[name], pathlib.Path)
                            else self._cache_directory
                        ),
                        verbose=self._num_datasets <= 5,
                        distributed=self._distributed,
                        sequence_length=self._max_sequence_length,
                        vocab_size=self._vocab_size,
                        tokenizer=self._tokenizer,
                    )
                    for phase, num_samples in dataset_samples_per_phase.items()
                    if num_samples > 0
                }
            )
            datasets_and_weights.append(
                (self._build_and_sample_dataset(name, sampling_configs), self._dataset_weights[name])
            )

        if len(datasets_and_weights) == 1:
            datasets = datasets_and_weights[0][0]
        else:
            datasets = BlendedDataset.apply(
                "blended",
                datasets_and_weights,
                PhaseSplits[GPTSamplingConfig](
                    {
                        phase: GPTSamplingConfig(
                            num_samples=samples_per_phase,
                            seed=self._distributed_config.seed,
                            cache_directory=None if self._cache_directory is None else self._cache_directory,
                            verbose=self._num_datasets <= 5,
                            distributed=self._distributed,
                            sequence_length=self._max_sequence_length,
                            vocab_size=self._vocab_size,
                            tokenizer=self._tokenizer,
                        )
                        for phase, samples_per_phase in self._samples_per_phase.items()
                    }
                ),
            )
        self._datasets = SampledSplitDataset[GPTDatasetSlice](
            "monitor",
            {
                phase: DatasetMonitor(dataset, self._config.data_sample_warn_time_ms)
                for phase, dataset in datasets.items()
            },
        )
        self._is_setup = True

    @property
    def tokenizer(self) -> Tokenizer:
        assert self._is_setup
        return self._tokenizer

    def get_iterator(
        self,
        batch_config: BatchConfig,
        phase: PhaseType,
        *,
        consumed_samples: int,
        num_workers: int,
        prefetch_factor: int | None = None,
    ) -> typing.Iterator[typing.Any]:
        assert self._is_setup
        Assert.incl(phase, self._datasets)
        Assert.in_range_incl(batch_config.sequence_length, 1, self._max_sequence_length)
        log_main_rank(f"Initializing {phase} data iterator from sample {consumed_samples}...")
        return iter(
            torch.utils.data.DataLoader(
                self._datasets[phase],  # noqa
                batch_sampler=SampledDatasetIterator(
                    total_samples=len(self._datasets[phase]),
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

    def _build_and_sample_gpt_dataset(self, name: str, sampling_configs: PhaseSplits[GPTSamplingConfig]):
        datasets = GPTDatasetSlice.from_splits(
            GPTMemmapDataset(name, self._dataset_prefixes[name]), self._phase_split
        ).sample(sampling_configs)
        if self._config.fim.rate > 0:
            datasets = SampledSplitDataset[GPTDatasetSlice](
                "fim",
                {
                    phase: FimDataset(self.config.fim, dataset, sampling_configs[phase])
                    for phase, dataset in datasets.items()
                },
            )
        return datasets

    def _build_and_sample_dummy_dataset(self, name: str, sampling_configs: PhaseSplits[GPTSamplingConfig]):
        return CopySplitDataset(
            f"{name}_split",
            GPTDummyDataset(name),
            list(sampling_configs),
        ).sample(sampling_configs)
