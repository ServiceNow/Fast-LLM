import dataclasses
import typing

import torch

from fast_llm.data.document.abstract import Batch, Document
from fast_llm.utils import Assert, padded_cumsum


def filter_lengths(lengths: list[int], filter: torch.Tensor) -> list[int]:
    length_cumsum = padded_cumsum(lengths)
    filtered_lengths = (filter[begin:end].sum().item() for begin, end in zip(length_cumsum[:-1], length_cumsum[1:]))
    return [length for length in filtered_lengths if length > 0]


@dataclasses.dataclass(kw_only=True)
class PatchDocument(Document):
    """
    A reusable component holding a set of fixed-shape patches (ex. images, audio, video),
    each of which providing a single token embedding in a multimodal model.
    """

    patches: torch.Tensor
    token_map: torch.Tensor
    positions: torch.Tensor  # Position identifier for each patch in the patch grid.
    lengths: list[int]  # Length of each patch group (ex. image) in the document. TODO: Use cumsums instead?

    def __post_init__(self):
        Assert.eq(self.positions.shape, (self.patches.size(0), self.patches.ndim - 2))
        Assert.eq(sum(self.lengths), len(self.patches))


@dataclasses.dataclass(kw_only=True)
class PatchBatch(PatchDocument, Batch):
    @classmethod
    def from_documents(cls, documents: typing.Iterable[PatchDocument], sizes: typing.Iterable[int]) -> typing.Self:
        document_begin = 0
        embedding_maps = []
        for document, size in zip(documents, sizes, strict=True):
            if document is not None:
                embedding_maps.append(document.token_map + document_begin)
            document_begin += size
        return (
            cls(
                patches=torch.cat([document.patches for document in documents if document is not None]),
                token_map=torch.cat(embedding_maps),
                positions=torch.cat([document.positions for document in documents if document is not None]),
                lengths=sum((document.lengths for document in documents if document is not None), []),
            )
            if embedding_maps
            else None
        )

    def crop(self, begin: int, end: int) -> typing.Self:
        patch_filter = (self.token_map >= begin) & (self.token_map < end)
        return self.__class__(
            patches=self.patches[patch_filter],
            token_map=self.token_map[patch_filter] - begin,
            positions=self.positions[patch_filter],
            lengths=filter_lengths(self.lengths, patch_filter),
        )

    def to_device(self, device: "torch.device | str") -> typing.Self:
        return self.__class__(
            patches=self.patches.to(device, non_blocking=True),
            token_map=self.token_map.to(device, non_blocking=True),
            positions=self.positions.to(device, non_blocking=True),
            lengths=self.lengths,
        )
