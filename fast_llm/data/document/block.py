import dataclasses
import functools
import typing

import torch

from fast_llm.data.document.abstract import ModelInput
from fast_llm.data.document.config import LengthPreprocessingConfig
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedDimNames
from fast_llm.layers.attention.config import AttentionKwargs
from fast_llm.layers.block.config import BlockKwargs
from fast_llm.layers.language_model.config import LanguageModelKwargs
from fast_llm.utils import Assert, padded_cumsum


@dataclasses.dataclass(kw_only=True)
class BlockModelInput(ModelInput):
    token_dim: TensorDim = None
    hidden_token_dim: TensorDim = None
    sequence_k_dim: TensorDim = None
    key_value_token_dim: TensorDim = None
    unpadded_length: int = None  # Number of tokens in the current input excluding padding at the end.
    sequence_length: int = None  # Total number of tokens across all inputs, including padding.
    lengths: list[int] = None
    cumulative_lengths_q: torch.Tensor | None = None
    cumulative_lengths_k: torch.Tensor | None = None
    max_length_q: int | None = None
    max_length_k: int | None = None
    min_length_q: int | None = None
    min_length_k: int | None = None
    document_index_q: torch.Tensor | None = None
    document_index_k: torch.Tensor | None = None
    global_document_index_q: torch.Tensor | None = None
    num_documents_in_sequence: int | None = None
    position_index: torch.Tensor | None = None
    first_document_begin: int = 0

    def to_kwargs(self) -> dict[str, typing.Any]:
        return {
            **super().to_kwargs(),
            LanguageModelKwargs.token_dim: self.token_dim,
            LanguageModelKwargs.hidden_token_dim: self.hidden_token_dim,
            LanguageModelKwargs.sequence_k_dim: self.sequence_k_dim,
            LanguageModelKwargs.key_value_token_dim: self.key_value_token_dim,
            LanguageModelKwargs.num_tokens: self.unpadded_length,
            LanguageModelKwargs.sequence_length: self.sequence_length,
            LanguageModelKwargs.lengths: self.lengths,
            AttentionKwargs.cu_seqlens_q: self.cumulative_lengths_q,
            AttentionKwargs.cu_seqlens_k: self.cumulative_lengths_k,
            AttentionKwargs.max_seqlen_q: self.max_length_q,
            AttentionKwargs.max_seqlen_k: self.max_length_k,
            AttentionKwargs.min_seqlen_q: self.min_length_q,
            AttentionKwargs.min_seqlen_k: self.min_length_k,
            AttentionKwargs.document_index_q: self.document_index_q,
            AttentionKwargs.document_index_k: self.document_index_k,
            BlockKwargs.global_document_index_q: self.global_document_index_q,
            BlockKwargs.num_documents_in_sequence: self.num_documents_in_sequence,
            LanguageModelKwargs.position_ids: self.position_index,
            AttentionKwargs.first_document_begin: self.first_document_begin,
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
        data_dim = config.distributed.get_distributed_dim(DistributedDimNames.data)
        model_input.token_dim = TensorDim(
            "token",
            self.length * data_dim.size,
            data_dim,
        )
        model_input.hidden_token_dim = (
            TensorDim(
                "token_tp",
                self.length * data_dim.size,
                config.distributed.get_distributed_dim(DistributedDimNames.tensor_and_data),
            )
            if config.distributed.sequence_tensor_parallel
            else model_input.token_dim
        )

        # Key-value token dim after sequence-data-parallel gather.
        model_input.key_value_token_dim = (
            TensorDim(
                "key_value_token",
                self.length * data_dim.size,
                config.distributed.get_distributed_dim(DistributedDimNames.batch_data),
            )
            if config.distributed.sequence_data_parallel > 1
            else model_input.token_dim
        )
        # Key-value token dim as seen by the attention layer, after concatenating the past and cropping the future.
        model_input.sequence_k_dim = TensorDim("sequence_k", self.sequence_k_past + self.length)

        if not config.causal:
            # TODO: Support non-causal cropping (needs to know about the future too).
            Assert.eq(model_input.sequence_k_dim.global_size, self.last_document_end)

        model_input.first_document_begin = self.first_document_begin
        if config.return_cumulative_sequence_lengths:
            model_input.cumulative_lengths_q, model_input.cumulative_lengths_k = self.cumulative_lengths
        if config.return_max_sequence_lengths or config.return_document_index:
            model_input.max_length_q, model_input.max_length_k = self.max_lengths
        if config.return_min_sequence_lengths:
            model_input.min_length_q, model_input.min_length_k = self.min_lengths
        if config.return_document_index:
            model_input.document_index_q, model_input.document_index_k = self.document_index
        if config.return_position_index:
            model_input.position_index = self.position_index

    @functools.cached_property
    def length(self) -> int:
        return sum(self.lengths)

    @functools.cached_property
    def cumulative_lengths(self) -> tuple[torch.Tensor, torch.Tensor]:
        # `cu_seqlens_k` follows the canonical prefix-sum layout starting at 0, describing the K
        # extent narrowed by `first_document_begin` (the inactive leading prefix from earlier
        # documents brought in by the sequence-data-parallel gather is dropped). Downstream
        # consumers narrow `key_value` by `first_document_begin` to match.
        cumulative_lengths_q = torch.from_numpy(padded_cumsum(self.lengths)).to(dtype=torch.int32, device=self.device)
        cumulative_lengths_k = cumulative_lengths_q + (self.sequence_k_past - self.first_document_begin)
        cumulative_lengths_k[0] = 0
        return cumulative_lengths_q, cumulative_lengths_k

    @functools.cached_property
    def _first_length_k(self) -> int:
        # First doc's K-side length includes the past KV prefix; remaining docs match q-side.
        return self.sequence_k_past + self.lengths[0] - self.first_document_begin

    @functools.cached_property
    def max_lengths(self) -> tuple[int, int]:
        max_length_q = max(self.lengths)
        return max_length_q, max(max_length_q, self._first_length_k)

    @functools.cached_property
    def min_lengths(self) -> tuple[int, int]:
        min_length_q = min(self.lengths)
        min_length_k = min(self._first_length_k, *self.lengths[1:]) if len(self.lengths) > 1 else self._first_length_k
        return min_length_q, min_length_k

    @functools.cached_property
    def document_index(self) -> tuple[torch.Tensor, torch.Tensor]:
        # `document_index_k` is computed against the narrowed K extent (length `_narrow_total_k`),
        # consistent with the canonical `cumulative_lengths_k`. Values start at 1 (no leading
        # "before first active document" entries).
        cumulative_lengths_q, cumulative_lengths_k = self.cumulative_lengths
        return (
            torch.searchsorted(
                cumulative_lengths_q, torch.arange(self.length, device=self.device), side="right", out_int32=True
            ),
            torch.searchsorted(
                cumulative_lengths_k,
                torch.arange(self._narrow_total_k, device=self.device),
                side="right",
                out_int32=True,
            ),
        )

    @functools.cached_property
    def _narrow_total_k(self) -> int:
        return self.sequence_k_past + self.length - self.first_document_begin

    @functools.cached_property
    def position_index(self) -> torch.Tensor:
        # Computed in the narrowed K coordinate space; the position-within-document is invariant
        # under the narrowing shift, so this matches the un-narrowed result.
        _, document_index_k = self.document_index
        _, cumulative_lengths_k = self.cumulative_lengths
        narrow_total = self._narrow_total_k
        document_begins = cumulative_lengths_k[document_index_k[narrow_total - self.length : narrow_total] - 1]
        return (
            torch.arange(narrow_total - self.length, narrow_total, dtype=torch.int32, device=self.device)
            - document_begins
        )
