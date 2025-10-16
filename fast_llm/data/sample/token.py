import typing

import torch

from fast_llm.data.sample.abstract import Batch, Sample
from fast_llm.utils import Assert


class TokenSample(Sample):
    def __init__(self, tokens: torch.Tensor, lengths: list[int] | None = None):
        self.tokens = tokens
        # Length of each document in the sample. TODO: Use cumsums instead?
        if lengths is None:
            lengths = [len(tokens)]
        else:
            Assert.eq(sum(lengths), len(tokens))
        self.lengths = lengths

    @classmethod
    def from_documents(cls, documents: typing.Iterable[typing.Self]) -> typing.Self:
        return cls(
            torch.cat([document.tokens for document in documents]),
            sum((document.lengths for document in documents), []),
        )

    def crop(self, begin: int, end: int) -> typing.Self:
        sample_size = end - begin
        if self.lengths == [len(self.tokens)]:
            # Shortcut for the frequent case of a single document.
            lengths = [sample_size]
        else:
            begin_ = 0
            lengths = []
            for length in self.lengths:
                end_ = begin_ + length
                cropped_length = min(end_, end) - max(begin_, begin)
                if cropped_length > 0:
                    lengths.append(cropped_length)
                if end_ > end:
                    break
                begin_ = end_
        return self.__class__(self.tokens[begin:end], lengths)

    def __len__(self) -> int:
        return len(self.tokens)


class TokenBatch(Batch):
    def __init__(self, tokens: torch.Tensor, lengths: list[list[int]] | None) -> None:
        self.tokens = tokens
        if lengths is None:
            lengths = [[tokens.size(1)]] * tokens.size(0)
        self.lengths = lengths

    @classmethod
    def from_samples(cls, samples: typing.Iterable[TokenSample]) -> typing.Self:
        return cls(
            torch.stack([sample.tokens for sample in samples]),
            [sample.lengths for sample in samples],
        )

    def to_samples(self) -> list[TokenSample]:
        return [TokenSample(tokens, lengths) for tokens, lengths in zip(self.tokens, self.lengths, strict=True)]

    def crop(self, begin: int, end: int) -> typing.Self:
        return self.__class__(
            self.tokens[:, begin:end], [sample.crop(begin, end).lengths for sample in self.to_samples()]
        )

    def to_device_(self, device: "torch.device | str"):
        # Also standardize the dtype while we're here.
        self.tokens = self.tokens.to(device, dtype=torch.int64, non_blocking=True)
