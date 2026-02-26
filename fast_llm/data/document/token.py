import dataclasses
import functools
import typing

import torch

from fast_llm.data.document.abstract import Batch, Document
from fast_llm.utils import Assert, padded_cumsum


def crop_lengths(lengths: list[int], begin: int, end: int) -> list[int]:
    if len(lengths) == 1:
        # Shortcut for the frequent case of a single document.
        return [end - begin]
    begin_ = 0
    lengths_ = []
    for length in lengths:
        end_ = begin_ + length
        cropped_length = min(end_, end) - max(begin_, begin)
        if cropped_length > 0:
            lengths_.append(cropped_length)
        if end_ > end:
            break
        begin_ = end_
    return lengths_


@dataclasses.dataclass(kw_only=True)
class TokenDocument(Document):
    tokens: torch.Tensor

    def __len__(self) -> int:
        return len(self.tokens)


@dataclasses.dataclass(kw_only=True)
class TokenBatch(TokenDocument, Batch):
    lengths: list[int]
    sequence_k_past: int = 0
    current_document_begin: int = 0

    def __post_init__(self):
        Assert.eq(sum(self.lengths), len(self.tokens))

    @classmethod
    def from_documents(cls, documents: typing.Iterable[TokenDocument]) -> typing.Self:
        return cls(
            tokens=torch.cat([document.tokens for document in documents]),
            lengths=[len(document) for document in documents],
        )

    def crop(self, begin: int, end: int) -> typing.Self:
        Assert.eq(self.sequence_k_past, self.current_document_begin, 0)

        document_begin = 0
        lengths_ = []
        current_document_begin = None
        for length in self.lengths:
            document_end = document_begin + length
            cropped_length = min(document_end, end) - max(document_begin, begin)
            if cropped_length > 0:
                lengths_.append(cropped_length)
                if current_document_begin is None:
                    current_document_begin = document_begin
            if document_end > end:
                break
            document_begin = document_end

        return self.__class__(
            tokens=self.tokens[begin:end],
            lengths=lengths_,
            sequence_k_past=begin,
            current_document_begin=current_document_begin,
        )

    def to_device(self, device: "torch.device | str") -> typing.Self:
        return self.__class__(
            tokens=self.tokens.to(device, non_blocking=True),
            lengths=self.lengths,
            sequence_k_past=self.sequence_k_past,
            current_document_begin=self.current_document_begin,
        )

    @functools.cached_property
    def device(self) -> torch.device:
        return self.tokens.device

    @functools.cached_property
    def cumulative_lengths(self) -> tuple[torch.Tensor, torch.Tensor]:
        cumulative_lengths_q = torch.from_numpy(padded_cumsum(self.lengths)).to(dtype=torch.int32, device=self.device)
        cumulative_lengths_k = cumulative_lengths_q + self.sequence_k_past
        cumulative_lengths_k[0] = self.current_document_begin
        return cumulative_lengths_q, cumulative_lengths_k

    @functools.cached_property
    def max_lengths(self) -> tuple[torch.Tensor, torch.Tensor]:
        max_length_q = max(self.lengths)
        max_length_k = max(max_length_q, self.sequence_k_past + self.lengths[0] - self.current_document_begin)
        return (
            torch.full((1,), max_length_q, dtype=torch.int32, device=self.device),
            torch.full((1,), max_length_k, dtype=torch.int32, device=self.device),
        )

    @functools.cached_property
    def document_index(self) -> tuple[torch.Tensor, torch.Tensor]:
        cumulative_lengths_q, cumulative_lengths_k = self.cumulative_lengths
        # Note: index starts at 1. Index 0 is for sequence k before `self.current_document_begin`.
        return (
            torch.searchsorted(cumulative_lengths_q, torch.arange(len(self.tokens)), side="right"),
            torch.searchsorted(
                cumulative_lengths_k, torch.arange(self.sequence_k_past + len(self.tokens)), side="right"
            ),
        )

    @functools.cached_property
    def position_index(self) -> torch.Tensor:
        _, document_index_k = self.document_index
        _, cumulative_lengths_k = self.cumulative_lengths
        document_begins = cumulative_lengths_k[
            document_index_k[self.sequence_k_past : self.sequence_k_past + len(self.tokens)] - 1
        ]
        return (
            torch.arange(
                self.sequence_k_past, self.sequence_k_past + len(self.tokens), dtype=torch.int32, device=self.device
            )
            - document_begins
        )
