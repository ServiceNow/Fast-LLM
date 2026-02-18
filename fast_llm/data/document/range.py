import dataclasses
import typing

from fast_llm.data.document.abstract import Batch, Document


@dataclasses.dataclass(kw_only=True)
class RangeDocument(Document):
    """
    A reusable component holding a set of ranges in a sample.
    """

    ranges: list[tuple[int, int]]


@dataclasses.dataclass(kw_only=True)
class RangeBatch(RangeDocument, Batch):
    @classmethod
    def from_documents(
        cls, documents: typing.Iterable[RangeDocument | None], sizes: typing.Iterable[int]
    ) -> typing.Self:
        """
        Used to merge ranges from multiple documents, i.e. when multiple documents are packed together.
        """
        document: RangeDocument
        ranges = []
        document_begin = 0
        for document, size in zip(documents, sizes, strict=True):
            if document is not None:
                for begin, end in document.ranges:
                    ranges.append((begin + document_begin, end + document_begin))
            document_begin += size
        return cls(ranges=ranges) if ranges else None

    def crop(self, begin: int, end: int) -> typing.Self:
        cropped_ranges = ((max(begin_ - begin, 0), min(end_ - begin, end - begin)) for begin_, end_ in self.ranges)
        return self.__class__(ranges=[(begin_, end_) for begin_, end_ in cropped_ranges if end_ > begin_])
