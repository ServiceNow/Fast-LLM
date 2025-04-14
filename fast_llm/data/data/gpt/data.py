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
    token_ids: torch.Tensor  # [batch_size, sequence_length]
    labels: torch.Tensor  # [batch_size, sequence_length] original tokens before masking
    attention_mask: torch.Tensor  # [batch_size, sequence_length]
    loss_masking_spans: list[torch.Tensor] | None = None
    sequence_lengths: list[torch.Tensor] | None = None


def gpt_data_collate_fn(
    batch: list[GPTSample], 
    use_loss_masking_spans: bool, 
    cross_document_attention: bool,
    random_token_masking: bool = False,
    masking_probability: float = 0.15,
    mask_token_id: int | None = None,
    vocab_size: int | None = None,
    mask_replace_prob: float = 0.8,
    random_replace_prob: float = 0.1,
    use_diffusion: bool = False,
    diffusion_noise_schedule: str = "cosine",
    diffusion_timesteps: int = 1000,
    diffusion_loss_type: str = "mlm",
) -> GPTBatch:
    """
    Collate function that supports both standard MLM and diffusion-based noisy MLM.
    """
    stacked_ids = np.stack([sample.token_ids for sample in batch])
    batch_size, seq_length = stacked_ids.shape
    
    # Convert to tensor early to use PyTorch's random functions
    token_ids = torch.from_numpy(stacked_ids)
    labels = token_ids.clone()
    attention_mask = torch.ones_like(token_ids)
    
    if random_token_masking or use_diffusion:
        Assert.not_none(vocab_size, "vocab_size must be provided when using masking or diffusion")
        Assert.not_none(mask_token_id, "mask_token_id must be provided when using masking or diffusion")
        
        if use_diffusion:
            # Initialize diffusion process
            from fast_llm.layers.diffusion.diffusion import get_noise_schedule, get_alphas, apply_forward_diffusion
            
            # Get noise schedule and compute alphas
            betas = get_noise_schedule(diffusion_noise_schedule, diffusion_timesteps)
            alphas, alpha_bars, _ = get_alphas(betas)
            
            # Sample random timesteps for batch
            t = torch.randint(0, diffusion_timesteps, (batch_size,), device=token_ids.device)
            
            # Apply forward diffusion
            token_ids, noise_mask = apply_forward_diffusion(
                token_ids,
                t,
                alpha_bars,
                vocab_size,
                mask_token_id,
            )
            
            # Set labels to -100 for non-corrupted tokens
            labels = torch.where(noise_mask, labels, -100)
            
        else:
            # Standard MLM masking
            mask_prob = torch.full_like(token_ids, masking_probability, dtype=torch.float32)
            mask = torch.bernoulli(mask_prob).bool()
            
            # Don't mask padding tokens
            padding_mask = token_ids.eq(0)
            mask.masked_fill_(padding_mask, False)
            
            # Set labels to -100 for non-masked tokens
            labels = torch.where(mask, labels, -100)
            
            # Apply BERT-style masking (80% MASK, 10% random, 10% unchanged)
            mask_or_random = torch.bernoulli(torch.full_like(token_ids, mask_replace_prob + random_replace_prob, dtype=torch.float32)).bool() & mask
            random_or_keep = torch.bernoulli(torch.full_like(token_ids, random_replace_prob / (1 - mask_replace_prob), dtype=torch.float32)).bool() & mask_or_random
            
            # Replace with mask token
            token_ids = torch.where(mask_or_random & ~random_or_keep, torch.full_like(token_ids, mask_token_id), token_ids)
            
            # Replace with random token
            random_tokens = torch.randint_like(token_ids, 1, vocab_size)  # Start from 1 to avoid padding token
            token_ids = torch.where(random_or_keep, random_tokens, token_ids)
    
    stacked_spans = None
    sequence_lengths = None
    if use_loss_masking_spans:
        stacked_spans = [torch.from_numpy(sample.loss_masking_spans) for sample in batch]
    if not cross_document_attention:
        sequence_lengths = [torch.tensor(sample.sequence_lengths) for sample in batch]
    
    return GPTBatch(
        token_ids=token_ids,
        labels=labels,
        attention_mask=attention_mask,
        loss_masking_spans=stacked_spans,
        sequence_lengths=sequence_lengths,
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
                    random_token_masking=self._config.sampling.random_token_masking,
                    masking_probability=self._config.sampling.masking_probability,
                    mask_token_id=self._config.sampling.mask_token_id,
                    vocab_size=self._vocab_size,
                    mask_replace_prob=self._config.sampling.mask_replace_prob,
                    random_replace_prob=self._config.sampling.random_replace_prob,
                    use_diffusion=self._config.sampling.use_diffusion,
                    diffusion_noise_schedule=self._config.sampling.diffusion_noise_schedule,
                    diffusion_timesteps=self._config.sampling.diffusion_timesteps,
                    diffusion_loss_type=self._config.sampling.diffusion_loss_type,
                ),
                multiprocessing_context=self._config.multiprocessing_context.value if num_workers > 0 else None,
            )
        )
