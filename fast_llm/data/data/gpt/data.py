import dataclasses
import logging
import pathlib
import typing
import warnings
from functools import partial

import numpy as np
import torch
import torch.utils.data

from fast_llm.config import DiffusionStyle
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
    loss_masking_spans: list[torch.Tensor] | None = None
    sequence_lengths: list[torch.Tensor] | None = None
    chosen_spans: list[torch.Tensor] | None = None
    rejected_spans: list[torch.Tensor] | None = None
    mask_indexes: torch.Tensor | None = None
    mask_probabilities: torch.Tensor | None = None
    masked_token_ids: torch.Tensor | None = None
    loss_weights: torch.Tensor | None = None
    in_context_length: torch.Tensor | None = None
    in_context: torch.Tensor | None = None


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
    in_uniform=None,
):

    B, L = positions.size()
    context_length = context_length.unsqueeze(1).expand(B, L)
    p_mask = p_mask.unsqueeze(1)

    # Reminder: a context_length of zero still has one in_context token (à la <BOS>)
    in_context = positions <= context_length
    if in_mask is None:
        in_mask = (~in_context) & (torch.rand(B, L) < p_mask)

    if in_uniform is None:
        in_uniform = (~in_context) & (~in_mask) & (torch.rand(B, L) < p_uniform)
    in_clean = (~in_context) & (~in_mask) & (~in_uniform)

    loss_weights = (~padded)[:, 1:] * torch.cat(
        [
            ar_factor * in_context[:, 1:]
            + in_mask[:, 1:] / p_mask
            + un_factor * ((1 - p_uniform) * in_uniform[:, 1:] + p_uniform * in_clean[:, 1:]) / (1 - p_mask),
            last_factor * torch.ones(B, 1),
        ],
        dim=1,
    )

    input_ids = do_uniform(do_mask(data_ids[:, :-1], in_mask, mask_token_id), in_uniform, vocab_size)

    # print(
    #     f"{'Name':<20} {'Shape/Value':<30}\n"
    #     f"{'-'*50}\n"
    #     f"{'input_ids':<20} {str(input_ids.shape):<30}\n"
    #     f"{'in_context':<20} {str(in_context.shape):<30}\n"
    #     f"{'in_mask':<20} {str(in_mask.shape):<30}\n"
    #     f"{'in_uniform':<20} {str(in_uniform.shape):<30}\n"
    #     f"{'in_clean':<20} {str(in_clean.shape):<30}\n"
    #     f"{'loss_weights':<20} {str(loss_weights.shape):<30}\n"
    #     f"{'in_context_length':<20} {str(context_length):<30}\n"
    # )

    return {
        "in_context": in_context,  # Only for tokens to be predicted
        "in_mask": in_mask,
        "in_uniform": in_uniform,
        "in_clean": in_clean,
        "input_ids": input_ids,
        # "target_ids": data_ids,
        "loss_weights": loss_weights,
        # "in_context_length": context_length,
    }


def gpt_data_collate_fn(batch: list[GPTSample], sampling_parameters: GPTSamplingParameters) -> GPTBatch:

    stacked_ids = np.stack([sample.token_ids for sample in batch])
    stacked_spans = None
    sequence_lengths = None
    mask_indexes = None
    mask_probabilities = None
    masked_token_ids = None

    loss_weights = None
    in_context_length = None
    in_context = None

    token_ids = torch.from_numpy(stacked_ids)

    if sampling_parameters.diffusion.style == DiffusionStyle.masked:
        diffusion_config = sampling_parameters.diffusion

        batch_size, seq_len = token_ids.shape
        diffusion_config.mask_token_id
        positions = torch.arange(seq_len - 1).unsqueeze(0).expand(batch_size, seq_len - 1)
        padded = torch.zeros_like(token_ids, dtype=torch.bool)

        # Generate a random tensor of batch size to seed masking probabilities
        t = torch.rand((batch_size,))

        # Compute the mask probabilities for every sequence in the batch
        p_mask = (1 - (2 * diffusion_config.epsilon)) * t + diffusion_config.epsilon

        # Do we need to clamp at max_mask_prob?
        # p_mask = torch.min(p_mask, torch.tensor(diffusion_config.max_mask_prob))

        # Input has an additional token for shitting, is [0, 1, 2, 3, 4] -> [1, 2, 3, 4]

        # index   [0, 1, 2, 3, 4, 5] ->
        # The labels are already left shifted x = [A, B, C, D, E, F] ->
        #                                 embd =  [A, B, C, D, E]
        #                                label =  [B, C, D, E, F]
        # Last input token is dropped from the processing

        # Generate random values for all tokens in the batch and only mask the positions\
        # where the value is smaller than the mask probability
        # mask_indexes = torch.rand((batch_size, seq_len)) < p_mask[:, None]

        # Need further classification of this padding - 1% data to have partial sequences and padding
        # if diffusion_config.pad_prob > 0:
        #     pad_mask = torch.rand((batch_size,), device=device) < diffusion_config.pad_prob
        #     if pad_mask.any():
        #         mask_indexes[pad_mask] = True

        # Replace masked tokens with the mask token ID to create input for the model.
        # masked_token_ids = torch.where(mask_indexes, mask_token_id, token_ids)
        # masked_token_ids = masked_token_ids[:, :-1]  # Remove the last token, which is not used for prediction.

        # mask_indexes = mask_indexes[:, 1:]  # Shift left so previous token to mask is the index for loss.
        # mask_probabilities = p_mask

        batch_data = prepare_batch(
            data_ids=token_ids,
            positions=positions,
            padded=padded,
            mask_token_id=diffusion_config.mask_token_id,
            vocab_size=sampling_parameters.vocab_size,
            context_length=-torch.ones(batch_size, dtype=torch.int),  # No context length for masked diffusion
            p_mask=p_mask,
            p_uniform=0.0,  # no uniform shuffling of tokens
            ar_factor=0.0,
            un_factor=0.0,
            last_factor=0.0,
        )

        # token_ids = batch_data["input_ids"]
        masked_token_ids = batch_data["input_ids"]

        mask_indexes = batch_data["in_mask"]
        # mask_probabilities = torch.full_like(mask_indexes, diffusion_config.max_mask_prob, dtype=token_ids.dtype)
        loss_weights = batch_data["loss_weights"]
        # in_context_length = C
        in_context = batch_data["in_context"]

    elif sampling_parameters.diffusion.style == DiffusionStyle.ar_masked:
        diffusion_config = sampling_parameters.diffusion
        batch_size, seq_len = token_ids.shape
        data_ids = token_ids
        padded = torch.zeros_like(data_ids, dtype=torch.bool)
        positions = torch.arange(seq_len - 1).unsqueeze(0).expand(batch_size, seq_len - 1)

        # TODO:
        # 90% of the batch: C = random [0, seq_len // 4], 10%: C = random in [0, seq_len-2)
        prob = torch.rand(1)
        C = torch.where(
            prob > diffusion_config.context_sampler,
            torch.randint(0, seq_len // 4, (batch_size,), dtype=torch.long),
            torch.randint(0, seq_len - 2, (batch_size,), dtype=torch.long),
        )
        # C = torch.randint(0, (seq_len - 2), (batch_size,), dtype=torch.long)
        # C = -torch.ones(batch_size, dtype=torch.int)
        # Generate a random tensor of batch size to seed masking probabilities
        t = torch.rand((batch_size,))
        # Compute the mask probabilities for every sequence in the batch leaving extrams 0 & 1
        p_mask = (1 - (2 * diffusion_config.epsilon)) * t + diffusion_config.epsilon

        batch_data = prepare_batch(
            data_ids=data_ids,
            positions=positions,
            padded=padded,
            mask_token_id=diffusion_config.mask_token_id,
            vocab_size=sampling_parameters.vocab_size,
            context_length=C,
            p_mask=p_mask,
            p_uniform=0.0,  # no uniform shuffling of tokens
            ar_factor=diffusion_config.ar_factor,
            un_factor=1.0,
            last_factor=0.0,
        )

        # token_ids = batch_data["input_ids"]
        masked_token_ids = batch_data["input_ids"]

        mask_indexes = batch_data["in_mask"]
        # mask_probabilities = torch.full_like(mask_indexes, diffusion_config.max_mask_prob, dtype=token_ids.dtype)
        loss_weights = batch_data["loss_weights"]
        in_context_length = C
        in_context = batch_data["in_context"]

    if sampling_parameters.use_loss_masking_spans:
        stacked_spans = [torch.from_numpy(sample.loss_masking_spans) for sample in batch]

    stacked_chosen_spans = None
    stacked_rejected_spans = None
    if sampling_parameters.use_loss_masking_spans:
        stacked_spans = [torch.from_numpy(sample.loss_masking_spans) for sample in batch]
    if sampling_parameters.use_preference_loss_spans:
        stacked_chosen_spans = [torch.from_numpy(sample.chosen_span) for sample in batch]
        stacked_rejected_spans = [torch.from_numpy(sample.rejected_span) for sample in batch]
    if not sampling_parameters.cross_document_attention:
        sequence_lengths = [torch.tensor(sample.sequence_lengths) for sample in batch]

    return GPTBatch(
        token_ids=token_ids,
        loss_masking_spans=stacked_spans,
        sequence_lengths=sequence_lengths,
        chosen_spans=stacked_chosen_spans,
        rejected_spans=stacked_rejected_spans,
        mask_indexes=mask_indexes,
        mask_probabilities=mask_probabilities,
        masked_token_ids=masked_token_ids,
        loss_weights=loss_weights,
        in_context_length=in_context_length,
        in_context=in_context,
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
                # NOTE: Some models like Qwen2-1.5B-Instruct
                # have vocab_size bigger in model config than in tokenizer
                # TODO: Still, is it too constraining?
                Assert.geq(sampling_parameters.vocab_size, self._tokenizer.vocab_size)
            if sampling_parameters.num_samples > 0:
                sampling = GPTSamplingData(
                    config=self._config.sampling,
                    parameters=sampling_parameters,
                    cache_directory=self._cache_directory,
                    distributed=distributed,
                    dataset_name=dataset_name,
                    tokenizer=self._tokenizer,
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
