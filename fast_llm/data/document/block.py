import dataclasses
import functools
import typing

import torch

from fast_llm.data.document.abstract import ModelInput
from fast_llm.data.document.config import LengthPreprocessingConfig
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedDimNames
from fast_llm.layers.attention.config import AttentionKwargs
from fast_llm.layers.language_model.config import LanguageModelKwargs
from fast_llm.utils import Assert, padded_cumsum


@dataclasses.dataclass(kw_only=True)
class BlockModelInput(ModelInput):
    token_dim: TensorDim = None
    hidden_token_dim: TensorDim = None
    sequence_k_dim: TensorDim = None
    unpadded_length: int = None  # Number of tokens in the current input excluding padding at the end.
    sequence_length: int = None  # Total number of tokens across all inputs, including padding.
    lengths: list[int] = None
    cumulative_lengths_q: torch.Tensor | None = None
    cumulative_lengths_k: torch.Tensor | None = None
    max_length_q: torch.Tensor | None = None
    max_length_k: torch.Tensor | None = None
    document_index_q: torch.Tensor | None = None
    document_index_k: torch.Tensor | None = None
    position_index: torch.Tensor | None = None

    def to_kwargs(self) -> dict[str, typing.Any]:
        return {
            **super().to_kwargs(),
            LanguageModelKwargs.token_dim: self.token_dim,
            LanguageModelKwargs.hidden_token_dim: self.hidden_token_dim,
            LanguageModelKwargs.sequence_k_dim: self.sequence_k_dim,
            LanguageModelKwargs.num_tokens: self.unpadded_length,
            LanguageModelKwargs.sequence_length: self.sequence_length,
            LanguageModelKwargs.lengths: self.lengths,
            AttentionKwargs.cu_seqlens_q: self.cumulative_lengths_q,
            AttentionKwargs.cu_seqlens_k: self.cumulative_lengths_k,
            AttentionKwargs.max_seqlen_q: self.max_length_q,
            AttentionKwargs.max_seqlen_k: self.max_length_k,
            AttentionKwargs.document_index_q: self.document_index_q,
            AttentionKwargs.document_index_k: self.document_index_k,
            LanguageModelKwargs.position_ids: self.position_index,
        }


@dataclasses.dataclass(kw_only=True)
class LengthModelInputPreprocessor:
    lengths: list[int]
    sequence_k_past: int
    first_document_begin: int
    last_document_end: int
    device: torch.device
    unpadded_length: int
    sequence_length: int

    def preprocess(self, model_input: BlockModelInput, config: LengthPreprocessingConfig):
        model_input.lengths = self.lengths
        model_input.unpadded_length = self.unpadded_length
        model_input.sequence_length = self.sequence_length
        sequence_data_dim = config.distributed.get_distributed_dim(DistributedDimNames.sequence_data)
        model_input.token_dim = TensorDim(
            "token",
            self.length * sequence_data_dim.size,
            sequence_data_dim,
        )
        model_input.hidden_token_dim = (
            TensorDim(
                "token_tp",
                self.length * sequence_data_dim.size,
                config.distributed.get_distributed_dim(DistributedDimNames.tensor_and_sequence_data),
            )
            if config.distributed.sequence_tensor_parallel
            else model_input.token_dim
        )
        model_input.sequence_k_dim = TensorDim("sequence_k", self.sequence_k_past + self.length)

        if not config.causal:
            # TODO: Support non-causal cropping (needs to know about the future too).
            Assert.eq(model_input.sequence_k_dim.global_size, self.last_document_end)

        if config.return_cumulative_sequence_lengths:
            model_input.cumulative_lengths_q, model_input.cumulative_lengths_k = self.cumulative_lengths
        if config.return_max_sequence_lengths or config.return_document_index:
            model_input.max_length_q, model_input.max_length_k = self.max_lengths
        if config.return_document_index:
            model_input.document_index_q, model_input.document_index_k = self.document_index
        if config.return_position_index:
            model_input.position_index = self.position_index

    @functools.cached_property
    def length(self) -> int:
        return sum(self.lengths)

    @functools.cached_property
    def cumulative_lengths(self) -> tuple[torch.Tensor, torch.Tensor]:
        cumulative_lengths_q = torch.from_numpy(padded_cumsum(self.lengths)).to(dtype=torch.int32, device=self.device)
        cumulative_lengths_k = cumulative_lengths_q + self.sequence_k_past
        cumulative_lengths_k[0] = self.first_document_begin
        return cumulative_lengths_q, cumulative_lengths_k

    @functools.cached_property
    def max_lengths(self) -> tuple[torch.Tensor, torch.Tensor]:
        max_length_q = max(self.lengths)
        max_length_k = max(max_length_q, self.sequence_k_past + self.lengths[0] - self.first_document_begin)
        return (
            torch.full((1,), max_length_q, dtype=torch.int32, device=self.device),
            torch.full((1,), max_length_k, dtype=torch.int32, device=self.device),
        )

    @functools.cached_property
    def document_index(self) -> tuple[torch.Tensor, torch.Tensor]:
        cumulative_lengths_q, cumulative_lengths_k = self.cumulative_lengths
        # Note: index starts at 1. Index 0 is for sequence k before `self.current_document_begin`.
        return (
            torch.searchsorted(
                cumulative_lengths_q, torch.arange(self.length, device=self.device), side="right", out_int32=True
            ),
            torch.searchsorted(
                cumulative_lengths_k,
                torch.arange(self.sequence_k_past + self.length, device=self.device),
                side="right",
                out_int32=True,
            ),
        )

    @functools.cached_property
    def position_index(self) -> torch.Tensor:
        _, document_index_k = self.document_index
        _, cumulative_lengths_k = self.cumulative_lengths
        document_begins = cumulative_lengths_k[
            document_index_k[self.sequence_k_past : self.sequence_k_past + self.length] - 1
        ]
        return (
            torch.arange(
                self.sequence_k_past, self.sequence_k_past + self.length, dtype=torch.int32, device=self.device
            )
            - document_begins
        )
