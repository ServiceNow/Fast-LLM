import logging
import pathlib
import time
import warnings

import torch
import torch.utils.data

from fast_llm.core.distributed import safe_barrier
from fast_llm.data.data.abstract import Data
from fast_llm.data.data.gpt.config import GPTDataConfig
from fast_llm.data.dataset.abstract import PhaseSplits, SampledSplitDataset
from fast_llm.data.dataset.gpt.config import GPTSamplingConfig
from fast_llm.data.iterator import SampledDatasetIterator
from fast_llm.data.tokenizer import Tokenizer
from fast_llm.engine.config_utils.run import get_run, log_main_rank
from fast_llm.engine.distributed.config import DistributedConfig, PhaseType
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.schedule.config import BatchConfig
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)


class GPTData(Data):
    """
    A global class for all dataset needs, including loading, splitting, sampling and iteration.
    Currently hard-coded to a GPT dataset.
    TODO: Separate generic and GPT classes.
    """

    _datasets: SampledSplitDataset
    _config: GPTDataConfig
    _tokenizer: Tokenizer | None
    _distributed: Distributed
    _cache_directory: pathlib.Path | None
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
        self._sampling_rank = -1
        self._sampling_time = None

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def max_sequence_length(self) -> int:
        return self._max_sequence_length

    def setup(self, distributed: Distributed, samples_per_phase: PhaseSplits[int]):
        """
        Load the datasets, and prepare or load the samplings.
        This may take a while and a significant amount of cpu memory.
        """
        super().setup(distributed, samples_per_phase)
        run = get_run()
        log_main_rank(f"Preparing dataset. This may take several minutes.")
        self._tokenizer = Tokenizer(self._config.tokenizer) if self._config.fim.rate > 0 else None

        if run.experiment_directory is None:
            warnings.warn(f"Using the dataset directory for the index cache.")
            self._cache_directory = None
        else:
            self._cache_directory = run.experiment_directory / "dataset_cache"
        sampling_config = PhaseSplits[GPTSamplingConfig](
            {
                phase: GPTSamplingConfig(
                    num_samples=samples_per_phase[phase],
                    sequence_length=self._max_sequence_length,
                    seed=self._distributed_config.seed,
                    cache_directory=self._cache_directory,
                    verbose=True,
                )
                for phase, num_samples in samples_per_phase.items()
                if num_samples > 0
            }
        )
        self._datasets = self._config.dataset.build_split_sample(self, sampling_config)
        safe_barrier(self._distributed.world_group, "build_split_sample")
        self._is_setup = True

    def get_next_sampling_rank_and_verbose(self) -> tuple[int, bool]:
        sampling_time = time.perf_counter()
        verbose = self._sampling_time is None or sampling_time - self._sampling_time > 60
        if verbose:
            self._sampling_time = sampling_time
        if self._config.distributed_data_sampling:
            self._sampling_rank += 1
            return self._sampling_rank % self._distributed_config.world_size, verbose
        else:
            return 0, verbose

    @property
    def tokenizer(self):
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
    ):
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
