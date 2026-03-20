import dataclasses
import typing

import torch

from fast_llm.data.document.abstract import Batch, Document
from fast_llm.utils import Assert


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
        cls,
        documents: typing.Sequence[TokenDataDocument | None],
        sizes: typing.Iterable[int],
        pad_to_size: int | None = None,
    ) -> typing.Self | None:
        """
        Used to merge ranges from multiple documents, i.e. when multiple documents are packed together.
        """
        data = [document.data for document in documents if document is not None]
        if len(data) == len(documents):
            lengths = [len(data_) for data_ in data]
            Assert.eq(lengths, sizes)
            if pad_to_size is not None:
                unpadded_length = sum(lengths)
                Assert.geq(pad_to_size, unpadded_length)
                padding = pad_to_size - unpadded_length
                if padding > 0:
                    data.append(data[0].new_zeros(padding, *data[0].shape[1:]))
            return TokenDataBatch(data=torch.cat(data))
        else:
            return None

    def get_cropped_data(self, begin: int, end: int) -> torch.Tensor:
        return self.data[begin:end]
