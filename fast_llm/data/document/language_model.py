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
    num_tokens: int = None  # Number of tokens in the micro-batch excluding padding at the end.

    def __post_init__(self):
        if self.num_tokens is None:
            self.num_tokens = len(self.tokens)

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
            tokens=_crop_optional(self.tokens, begin, end),
            loss_masking_spans=_crop_optional(self.loss_masking_spans, begin, end),
            chosen_spans=_crop_optional(self.chosen_spans, begin, end),
            rejected_spans=_crop_optional(self.rejected_spans, begin, end),
            image_patches=_crop_optional(self.image_patches, begin, end),
            num_tokens=min(end, self.num_tokens) - begin,
        )

    def to_device(self, device: "torch.device | str"):
        return self.__class__(
            tokens=_to_device_optional(self.tokens, device),
            loss_masking_spans=_to_device_optional(self.loss_masking_spans, device),
            chosen_spans=_to_device_optional(self.chosen_spans, device),
            rejected_spans=_to_device_optional(self.rejected_spans, device),
            image_patches=_to_device_optional(self.image_patches, device),
            num_tokens=self.num_tokens,
        )


def _merge_optional[T](fn: typing.Callable, args: typing.Iterable) -> T | None:
    return None if any(arg is None for arg in args) else fn(args)


def _crop_optional[T: Batch](batch: T, begin: int, end: int) -> T | None:
    return None if batch is None else batch.crop(begin, end)


def _to_device_optional[T: Batch](batch: T, device: "torch.device | str") -> T | None:
    return None if batch is None else batch.to_device(device)
