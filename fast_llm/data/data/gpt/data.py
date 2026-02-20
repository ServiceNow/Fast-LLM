import functools
import logging
import typing
import warnings

import torch
import torch.utils.data

from fast_llm.data.batch.config import LanguageModelBatchPreprocessingConfig
from fast_llm.data.batch.language_model import LanguageModelPreprocessedBatch
from fast_llm.data.data.abstract import Data
from fast_llm.data.data.data_loader import SampledDatasetIterator
from fast_llm.data.data.gpt.config import GPTDataConfig
from fast_llm.data.dataset.abstract import SampledDataset
from fast_llm.data.dataset.config import SamplingParameters
from fast_llm.data.dataset.gpt.config import GPTSamplingData
from fast_llm.data.dataset.monitor import DatasetMonitor
from fast_llm.data.document.language_model import LanguageModelBatch, LanguageModelDocument
from fast_llm.engine.config_utils.run import log_main_rank
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.models.gpt.config import GPTBatchConfig
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)


class GPTData[ConfigType: GPTDataConfig](Data[ConfigType]):
    """
    A global class for all dataset needs, including loading, splitting, sampling and iteration.
    """

    _datasets: dict[str, SampledDataset]
    # _sampling_parameters: dict[str, SamplingParameters]
    _preprocessing: dict[str, LanguageModelBatchPreprocessingConfig]

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
        self._datasets = {}
        self._preprocessing = {}

    def sample_dataset(
        self,
        dataset_name: str,
        config: LanguageModelBatchPreprocessingConfig,
        num_samples: int,
    ) -> None:
        assert self._is_setup
        Assert.gt(num_samples, 0)
        if dataset_name not in self._config.datasets:
            raise ValueError(f"Dataset {dataset_name} not found.")
        if dataset_name in self._datasets:
            raise ValueError(f"Dataset {dataset_name} is already sampled.")

        log_main_rank(f"Sampling dataset {dataset_name}. This may take several minutes.")

        if self._cache_directory is None:
            # TODO: Avoid this
            warnings.warn(f"The index cache will be saved in the dataset directory.")

        sampling_parameters = SamplingParameters(
            sequence_length=config.batch.sequence_length,
            num_samples=num_samples,
            truncate_documents=config.batch.truncate_documents,
            extra_tokens=config.predicted_tokens,
        )

        sampling = GPTSamplingData(
            config=self._config.sampling,
            parameters=sampling_parameters,
            preprocessing=config,
            cache_directory=self._cache_directory,
            distributed_config=self._distributed_config,
            dataset_name=dataset_name,
        )
        self._preprocessing[dataset_name] = config
        dataset = self._config.datasets[dataset_name].build_and_sample(sampling)
        self._datasets[dataset_name] = DatasetMonitor(dataset, self._config.data_sample_warn_time_ms)

    def get_iterator(
        self,
        batch_config: GPTBatchConfig,
        dataset_name: str,
        *,
        consumed_samples: int,
        num_workers: int,
        prefetch_factor: int | None = None,
        timeout: float = 60,
        preprocess: bool = True,
    ) -> typing.Iterator[LanguageModelPreprocessedBatch]:
        assert self._is_setup

        # Some dataset names may come from phases and are capitalized,
        # so we need to normalize them before use.
        dataset_name = dataset_name.lower()

        Assert.incl(dataset_name, self._datasets)
        log_main_rank(f"Initializing {dataset_name} dataset iterator from sample {consumed_samples}...")

        return iter(
            torch.utils.data.DataLoader(
                self._datasets[dataset_name],  # noqa
                batch_sampler=SampledDatasetIterator(
                    total_samples=len(self._datasets[dataset_name]),
                    begin_index=consumed_samples,
                    micro_batch_size=self._preprocessing[dataset_name].batch.micro_batch_size,
                    data_rank=self._distributed_config.batch_data_rank,
                    data_parallel=self._distributed_config.batch_data_parallel,
                ),
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                pin_memory=True,
                collate_fn=functools.partial(self._collate_fn, dataset_name=dataset_name, preprocess=preprocess),
                multiprocessing_context=self._config.multiprocessing_context.value if num_workers > 0 else None,
            )
        )

    def _collate_fn(
        self,
        documents: list[list[LanguageModelDocument]],
        dataset_name: str,
        preprocess: bool = True,
    ) -> LanguageModelPreprocessedBatch | LanguageModelBatch:
        documents = [document for documents_ in documents for document in documents_]
        if preprocess:
            return LanguageModelPreprocessedBatch.from_documents(documents, self._preprocessing[dataset_name])
        else:
            return LanguageModelBatch.from_documents(documents, self._preprocessing[dataset_name].total_length)
