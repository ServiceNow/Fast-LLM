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
from fast_llm.data.dataset.gpt.config import GPTSamplingData, GPTSamplingParameters
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
    masked_token_ids: torch.Tensor | None = None
    loss_masking_spans: list[torch.Tensor] | None = None
    sequence_lengths: list[torch.Tensor] | None = None
    mask_indexes: torch.Tensor | None = None
    mask_probabilities: torch.Tensor | None = None
    loss_weights: torch.Tensor | None = None
    in_context_length: torch.Tensor | None = None


def do_mask(x, mask, mask_token_id):
    x = x.clone()
    x[mask] = mask_token_id
    return x


def do_uniform(x, is_uniform, vocab_size):
    # WARNING! "Shuffle" was really meant to mean "uniformly sample among all non-mask tokens"
    x = x.clone()
    uniform = torch.randint(0, vocab_size, x.size())
    x[is_uniform] = uniform[is_uniform]
    return x


def prepare_batch(
        data_ids,
        positions,
        padded,
        mask_token_id,
        vocab_size,
        context_length,
        p_mask,
        *,
        p_uniform=0.0,
        ar_factor=1.0,
        un_factor=1.0,
        last_factor=0.0,
        in_mask=None, 
        in_uniform=None
    ):
    B, L = positions.size()
    in_context_length = context_length
    context_length = context_length.unsqueeze(1).expand(B, L)
    p_mask = p_mask.unsqueeze(1)

    in_context = positions <= context_length

    if in_mask is None:
        in_mask = (~in_context) & (torch.rand(B, L) < p_mask)

    if in_uniform is None:
        in_uniform = (~in_context) & (~in_mask) & (torch.rand(B, L) < p_uniform)
    in_clean = (~in_context) & (~in_mask) & (~in_uniform)

    loss_weights = (~padded) * (
        ar_factor * in_context.float()
        + in_mask.float() / p_mask
        + un_factor * (
            (1 - p_uniform) * in_uniform.float()
            + p_uniform * in_clean.float()
        ) / (1 - p_mask)
    )

    if last_factor >= 0.0:
        last_weights = last_factor * torch.ones(B, 1, device=loss_weights.device)
        loss_weights[:, -1] = last_weights.squeeze(1)

    input_ids = do_uniform(do_mask(data_ids, in_mask, mask_token_id), in_uniform, vocab_size)

    return {
        "in_context": in_context[:, 1:],  # Only for tokens to be predicted
        "in_mask": in_mask[:, 1:],
        "in_uniform": in_uniform[:, 1:],
        "in_clean": in_clean[:, 1:],
        "input_ids": input_ids,
        "target_ids": data_ids,
        "loss_weights": loss_weights[:, 1:],
        "in_context_length": in_context_length,
        
    }


def gpt_data_collate_fn(batch: list[GPTSample], sampling_parameters: GPTSamplingParameters) -> GPTBatch:
    stacked_ids = np.stack([sample.token_ids for sample in batch])
    stacked_spans = None
    sequence_lengths = None
    mask_indexes = None
    mask_probabilities = None
    loss_weights = None
    token_ids = torch.from_numpy(stacked_ids)

    if sampling_parameters.diffusion.enabled:
        diffusion_config = sampling_parameters.diffusion
        batch_size, seq_len = token_ids.shape[0], token_ids.shape[1]
        data_ids = token_ids
        padded = torch.zeros_like(data_ids, dtype=torch.bool)
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len)
        C = torch.randint(0, seq_len, (batch_size,), dtype=torch.long)

        batch_data = prepare_batch(
            data_ids=data_ids,
            positions=positions,
            padded=padded,
            mask_token_id=diffusion_config.mask_token_id,
            vocab_size=sampling_parameters.vocab_size,
            context_length=C,  
            p_mask=torch.full((batch_size,), diffusion_config.max_mask_prob),  
            p_uniform=0.0, 
            ar_factor=1.0,
            un_factor=1.0,
            last_factor=0.0
        )
        
        # token_ids = batch_data["input_ids"]
        masked_token_ids = batch_data["input_ids"]

        mask_indexes = batch_data["in_mask"]
        mask_probabilities = torch.full_like(mask_indexes, diffusion_config.max_mask_prob, dtype=torch.float32)  # Use float32 to match model dtype
        loss_weights = batch_data["loss_weights"]
        in_context_length = batch_data["in_context_length"]

    if sampling_parameters.use_loss_masking_spans:
        stacked_spans = [torch.from_numpy(sample.loss_masking_spans) for sample in batch]

    if not sampling_parameters.cross_document_attention:
        sequence_lengths = [torch.tensor(sample.sequence_lengths) for sample in batch]

    return GPTBatch(
        token_ids=token_ids,
        masked_token_ids=masked_token_ids,
        loss_masking_spans=stacked_spans,
        sequence_lengths=sequence_lengths,
        mask_indexes=mask_indexes,
        mask_probabilities=mask_probabilities,
        loss_weights=loss_weights,
        in_context_length=in_context_length
    )


class GPTData[ConfigType: GPTDataConfig](Data[ConfigType]):
    """
    A global class for all dataset needs, including loading, splitting, sampling and iteration.
    Currently hard-coded to a GPT dataset.
    TODO: Separate generic and GPT classes.
    """

    _datasets: dict[str, SampledDataset]
    _sampling_parameters: dict[str, GPTSamplingParameters]
    _tokenizer: Tokenizer | None
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
        self._tokenizer = None if self._config.tokenizer.path is None else Tokenizer(self._config.tokenizer)

        if self._cache_directory is None:
            # TODO: Avoid this
            warnings.warn(f"Using the dataset directory for the index cache.")

        self._datasets = {}
        for dataset_name, sampling_parameters in self._sampling_parameters.items():
            if self._tokenizer is not None:
                # TODO: Too constraining?
                Assert.eq(self._tokenizer.vocab_size, sampling_parameters.vocab_size)
            if sampling_parameters.num_samples > 0:
                sampling = GPTSamplingData(
                    config=self._config.sampling,
                    parameters=sampling_parameters,
                    cache_directory=self._cache_directory,
                    distributed=distributed,
                    dataset_name=dataset_name,
                    tokenizer=self._tokenizer,
                    truncate_documents=self._config.truncate_documents,
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
        sampling_parameters = self._sampling_parameters[dataset_name]
        Assert.in_range_incl(batch_config.sequence_length, 1, sampling_parameters.sequence_length)
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
                    sampling_parameters=sampling_parameters,
                ),
                multiprocessing_context=self._config.multiprocessing_context.value if num_workers > 0 else None,
            )
        )
