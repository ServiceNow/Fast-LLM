import dataclasses
import typing

import torch

from fast_llm.data.document.abstract import Batch, Document


@dataclasses.dataclass(kw_only=True)
class TokenDataDocument(Document):
    """
    A reusable component holding tensor-valued data of fixed dtype and shape for each token.
    """

    data: torch.Tensor


@dataclasses.dataclass(kw_only=True)
class TokenDataBatch(Batch, TokenDataDocument):
    @classmethod
    def from_documents(
        cls, documents: typing.Sequence[TokenDataDocument | None], sizes: typing.Iterable[int]
    ) -> typing.Self | None:
        """
        Used to merge ranges from multiple documents, i.e. when multiple documents are packed together.
        """
        data = [document.data for document in documents if document is not None]
        return TokenDataBatch(data=torch.cat(data)) if len(data) == len(documents) else None

    def get_cropped_data(self, begin: int, end: int) -> torch.Tensor:
        return self.data[begin:end]
