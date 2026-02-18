import dataclasses
import logging
import typing

import torch

from fast_llm.data.document.abstract import Batch, Document
from fast_llm.data.document.patch import PatchBatch, PatchDocument
from fast_llm.data.document.range import RangeBatch, RangeDocument
from fast_llm.data.document.token import TokenBatch, TokenDocument
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)


@dataclasses.dataclass(kw_only=True)
class LanguageModelDocument(Document):
    tokens: TokenDocument
    loss_masking_spans: RangeDocument | None = None
    chosen_spans: RangeDocument | None = None
    rejected_spans: RangeDocument | None = None
    image_patches: PatchDocument | None = None

    def __len__(self) -> int:
        return len(self.tokens)


@dataclasses.dataclass(kw_only=True)
class LanguageModelBatch(LanguageModelDocument, Batch):
    tokens: TokenBatch
    loss_masking_spans: RangeBatch | None = None
    chosen_spans: RangeBatch | None = None
    rejected_spans: RangeBatch | None = None
    image_patches: PatchBatch | None = None
    num_tokens: int  # Number of tokens in the micro-batch excluding padding at the end.

    @classmethod
    def from_documents(
        cls, documents: typing.Iterable[LanguageModelDocument], pad_to_size: int | None = None
    ) -> typing.Self:
        num_tokens = sum(len(document) for document in documents)
        if pad_to_size is not None:
            Assert.geq(pad_to_size, num_tokens)
            padding = pad_to_size - num_tokens
            if padding > 0:
                documents = documents + [
                    LanguageModelDocument(
                        tokens=TokenDocument(tokens=documents[0].tokens.tokens.new_full([padding], -100))
                    )
                ]
        sizes = [len(document) for document in documents]
        return cls(
            tokens=TokenBatch.from_documents([document.tokens for document in documents]),
            loss_masking_spans=RangeBatch.from_documents(
                [document.loss_masking_spans for document in documents], sizes
            ),
            chosen_spans=RangeBatch.from_documents([document.chosen_spans for document in documents], sizes),
            rejected_spans=RangeBatch.from_documents([document.rejected_spans for document in documents], sizes),
            image_patches=PatchBatch.from_documents([document.image_patches for document in documents], sizes),
            num_tokens=num_tokens,
        )

    def crop(self, begin: int, end: int) -> typing.Self:
        return self.__class__(
            tokens=self.tokens.crop(begin, end),
            loss_masking_spans=_crop_optional(self.loss_masking_spans, begin, end),
            chosen_spans=_crop_optional(self.chosen_spans, begin, end),
            rejected_spans=_crop_optional(self.rejected_spans, begin, end),
            image_patches=_crop_optional(self.image_patches, begin, end),
            num_tokens=min(end, self.num_tokens) - begin,
        )

    def to_device_(self, device: "torch.device | str"):
        self.tokens.to_device_(device)
        if self.loss_masking_spans is not None:
            self.loss_masking_spans.to_device_(device)
        if self.chosen_spans is not None:
            self.chosen_spans.to_device_(device)
        if self.rejected_spans is not None:
            self.rejected_spans.to_device_(device)
        if self.image_patches is not None:
            self.image_patches.to_device_(device)


def _merge_optional[T](fn: typing.Callable, args: typing.Iterable) -> T | None:
    return None if any(arg is None for arg in args) else fn(args)


def _crop_optional[T: Document](sample: T, begin: int, end: int) -> T | None:
    return None if sample is None else sample.crop(begin, end)
