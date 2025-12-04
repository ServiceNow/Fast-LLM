import logging
import pathlib
import typing
import warnings

import torch
import torch.utils.data

from fast_llm.core.distributed import safe_barrier
from fast_llm.data.data.abstract import Data
from fast_llm.data.data.data_loader_wrapper import DistributedDataLoaderWrapper
from fast_llm.data.data.gpt.config import GPTDataConfig
from fast_llm.data.dataset.abstract import SampledDataset
from fast_llm.data.dataset.abstract_iterable import SampledIterableDataset
from fast_llm.data.dataset.gpt.config import GPTSamplingData, GPTSamplingParameters
from fast_llm.data.dataset.monitor import DatasetMonitor
from fast_llm.data.iterator import SampledDatasetIterator
from fast_llm.data.sample.language_model import LanguageModelBatch
from fast_llm.data.sample.pipeline_rl import PipelineRLBatch
from fast_llm.engine.config_utils.run import log_main_rank
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.models.gpt.config import GPTBatchConfig
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)


class GPTData[ConfigType: GPTDataConfig](Data[ConfigType]):
    """
    A global class for all dataset needs, including loading, splitting, sampling and iteration.
    """

    _datasets: dict[str, SampledDataset]
    _sampling_parameters: dict[str, GPTSamplingParameters]
    _is_setup: bool = False

    def __init__(
        self,
        config: GPTDataConfig,
        distributed_config: DistributedConfig,
    ):
        """
        Create the data and gather some basic information on the dataset(s).
        Should be `setup` before use.
        """
        super().__init__(config, distributed_config)

    def setup(
        self,
        distributed: "Distributed",
        sampling_parameters: dict[str, GPTSamplingParameters],
        cache_directory: pathlib.Path,
        timeout: float | None = None,
    ) -> None:
        """
        Load the datasets, and prepare or load the samplings.
        This may take a while and a significant amount of cpu memory.
        """
        super().setup(distributed, sampling_parameters, cache_directory)

        # Check and raise an error if a used dataset is not defined.
        for dataset_name in self._sampling_parameters.keys():
            if dataset_name not in self._config.datasets:
                raise ValueError(f"Dataset {dataset_name} not found.")

        # Check and warn if there are defined datasets that are not used.
        unused_datasets = self._config.datasets.keys() - self._sampling_parameters.keys()
        if unused_datasets:
            warnings.warn(
                f"The following datasets are defined but not used: {', '.join(unused_datasets)}. "
                "Ensure this is intentional, or update the configuration accordingly."
            )

        log_main_rank(f"Preparing dataset. This may take several minutes.")

        if self._cache_directory is None:
            # TODO: Avoid this
            warnings.warn(f"Using the dataset directory for the index cache.")

        self._datasets = {}
        for dataset_name, sampling_parameters in self._sampling_parameters.items():
            if sampling_parameters.num_samples > 0:
                sampling = GPTSamplingData(
                    config=self._config.sampling,
                    parameters=sampling_parameters,
                    cache_directory=self._cache_directory,
                    distributed=distributed,
                    dataset_name=dataset_name,
                )
                dataset = self._config.datasets[dataset_name].build_and_sample(sampling)
                if isinstance(dataset, SampledDataset):
                    self._datasets[dataset_name] = DatasetMonitor(dataset, self._config.data_sample_warn_time_ms)
                else:
                    # Do not set monitor for iterable dataset as monitor only works with map style datasets
                    assert isinstance(dataset, SampledIterableDataset)
                    self._datasets[dataset_name] = dataset

        safe_barrier(self._distributed.world_group, "data_preparation", timeout)
        self._is_setup = True

    def get_iterator(
        self,
        batch_config: GPTBatchConfig,
        dataset_name: str,
        *,
        consumed_samples: int,
        num_workers: int,
        prefetch_factor: int | None = None,
        timeout: float = 60,
    ) -> typing.Iterator[LanguageModelBatch]:
        assert self._is_setup

        # Some dataset names may come from phases and are capitalized,
        # so we need to normalize them before use.
        dataset_name = dataset_name.lower()

        Assert.incl(dataset_name, self._datasets)
        sampling_parameters = self._sampling_parameters[dataset_name]
        Assert.in_range_incl(batch_config.sequence_length, 1, sampling_parameters.sequence_length)
        log_main_rank(f"Initializing {dataset_name} dataset iterator from sample {consumed_samples}...")

        dataset = self._datasets[dataset_name]

        if isinstance(dataset, SampledDataset):
            data_loader = torch.utils.data.DataLoader(
                dataset,  # noqa
                batch_sampler=SampledDatasetIterator(
                    total_samples=len(self._datasets[dataset_name]),
                    begin_index=consumed_samples,
                    micro_batch_size=batch_config.micro_batch_size,
                    data_rank=self._distributed.config.batch_data_rank,
                    data_parallel=self._distributed.config.batch_data_parallel,
                ),
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                pin_memory=True,
                collate_fn=LanguageModelBatch.from_samples,
                multiprocessing_context=self._config.multiprocessing_context.value if num_workers > 0 else None,
            )

        elif isinstance(dataset, SampledIterableDataset):
            if (
                self.distributed.model_and_sequence_data_group is None
                or self.distributed.model_and_sequence_data_group.rank() == 0
            ):
                rank = 0
                data_loader = torch.utils.data.DataLoader(
                    dataset,  # noqa
                    batch_size=batch_config.micro_batch_size,
                    num_workers=0 if num_workers == 0 else 1,
                    prefetch_factor=prefetch_factor,
                    pin_memory=True,
                    collate_fn=PipelineRLBatch.from_samples,
                    multiprocessing_context=self._config.multiprocessing_context.value if num_workers > 0 else None,
                )
            else:
                rank = self.distributed.model_and_sequence_data_group.rank()
                data_loader = None
            data_loader = DistributedDataLoaderWrapper(
                data_loader, rank, self.distributed.model_and_sequence_data_group
            )

        return iter(data_loader)
