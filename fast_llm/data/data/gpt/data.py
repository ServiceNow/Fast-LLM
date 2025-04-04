import dataclasses
import logging
import pathlib
import typing
import warnings
from functools import partial

import numpy as np
import torch
import torch.utils.data

from fast_llm.core.distributed import safe_barrier
from fast_llm.data.data.abstract import Data
from fast_llm.data.data.gpt.config import GPTDataConfig
from fast_llm.data.dataset.abstract import SampledDataset
from fast_llm.data.dataset.gpt.config import GPTSamplingData
from fast_llm.data.dataset.gpt.sampled import GPTSample
from fast_llm.data.dataset.monitor import DatasetMonitor
from fast_llm.data.iterator import SampledDatasetIterator
from fast_llm.data.tokenizer import Tokenizer
from fast_llm.engine.config_utils.run import log_main_rank
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.schedule.config import BatchConfig
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class GPTBatch:
    token_ids: torch.Tensor
    loss_masking_spans: list[torch.Tensor] | None = None
    sequence_lengths: list[torch.Tensor] | None = None


def gpt_data_collate_fn(
    batch: list[GPTSample], use_loss_masking_spans: bool, cross_document_attention: bool
) -> GPTBatch:
    stacked_ids = np.stack([sample.token_ids for sample in batch])
    stacked_spans = None
    sequence_lengths = None
    if use_loss_masking_spans:
        stacked_spans = [torch.from_numpy(sample.loss_masking_spans) for sample in batch]
    if not cross_document_attention:
        sequence_lengths = [torch.tensor(sample.sequence_lengths) for sample in batch]
    return GPTBatch(
        token_ids=torch.from_numpy(stacked_ids), loss_masking_spans=stacked_spans, sequence_lengths=sequence_lengths
    )


class GPTData[ConfigType: GPTDataConfig](Data[ConfigType]):
    """
    A global class for all dataset needs, including loading, splitting, sampling and iteration.
    Currently hard-coded to a GPT dataset.
    TODO: Separate generic and GPT classes.
    """

    _datasets: dict[str, SampledDataset]
    _tokenizer: Tokenizer | None
    _is_setup: bool = False

    def __init__(
        self,
        config: GPTDataConfig,
        distributed_config: DistributedConfig,
        vocab_size: int,
        max_sequence_length: int,
        cross_document_attention: bool = True,
    ):
        """
        Create the data and gather some basic information on the dataset(s).
        Should be `setup` before use.
        """
        super().__init__(config, distributed_config)
        self._vocab_size = vocab_size
        self._max_sequence_length = max_sequence_length
        self._cross_document_attention = cross_document_attention

    def setup(
        self,
        distributed: "Distributed",
        samples_per_dataset: dict[str, int],
        cache_directory: pathlib.Path,
        timeout: float | None = None,
    ) -> None:
        """
        Load the datasets, and prepare or load the samplings.
        This may take a while and a significant amount of cpu memory.
        """
        # Check and raise an error if a used dataset is not defined.
        for dataset_name in samples_per_dataset.keys():
            if dataset_name not in self._config.datasets:
                raise ValueError(f"Dataset {dataset_name} not found.")

        # Check and warn if there are defined datasets that are not used.
        unused_datasets = self._config.datasets.keys() - samples_per_dataset.keys()
        if unused_datasets:
            warnings.warn(
                f"The following datasets are defined but not used: {', '.join(unused_datasets)}. "
                "Ensure this is intentional, or update the configuration accordingly."
            )

        super().setup(distributed, samples_per_dataset, cache_directory)
        log_main_rank(f"Preparing dataset. This may take several minutes.")
        self._tokenizer = None if self._config.tokenizer.path is None else Tokenizer(self._config.tokenizer)

        if self._cache_directory is None:
            # TODO: Avoid this
            warnings.warn(f"Using the dataset directory for the index cache.")

        self._datasets = {}
        for dataset_name, num_samples in samples_per_dataset.items():
            if num_samples > 0:
                sampling = GPTSamplingData(
                    num_samples=num_samples,
                    config=self._config.sampling,
                    cache_directory=self._cache_directory,
                    distributed=distributed,
                    dataset_name=dataset_name,
                    sequence_length=self._max_sequence_length,
                    vocab_size=self._vocab_size,
                    tokenizer=self._tokenizer,
                    truncate_documents=self._config.truncate_documents,
                    cross_document_attention=self._cross_document_attention,
                )
                dataset = self._config.datasets[dataset_name].build_and_sample(sampling)
                self._datasets[dataset_name] = DatasetMonitor(dataset, self._config.data_sample_warn_time_ms)

        safe_barrier(self._distributed.world_group, "data_preparation", timeout)
        self._is_setup = True

    @property
    def tokenizer(self) -> Tokenizer:
        assert self._is_setup
        return self._tokenizer

    def get_iterator(
        self,
        batch_config: BatchConfig,
        dataset_name: str,
        *,
        consumed_samples: int,
        num_workers: int,
        prefetch_factor: int | None = None,
        timeout: float = 60,
    ) -> typing.Iterator[typing.Any]:
        assert self._is_setup

        # Some dataset names may come from phases and are capitalized,
        # so we need to normalize them before use.
        dataset_name = dataset_name.lower()

        Assert.incl(dataset_name, self._datasets)
        Assert.in_range_incl(batch_config.sequence_length, 1, self._max_sequence_length)
        log_main_rank(f"Initializing {dataset_name} dataset iterator from sample {consumed_samples}...")
        return iter(
            torch.utils.data.DataLoader(
                self._datasets[dataset_name],  # noqa
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
                collate_fn=partial(
                    gpt_data_collate_fn,
                    use_loss_masking_spans=self._config.sampling.use_loss_masking_spans,
                    cross_document_attention=self._cross_document_attention,
                ),
                multiprocessing_context=self._config.multiprocessing_context.value if num_workers > 0 else None,
            )
        )
