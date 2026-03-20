import dataclasses
import functools
import typing

import torch

from fast_llm.data.document.abstract import Batch, Document
from fast_llm.data.document.block import BlockModelInput, LengthModelInputPreprocessor
from fast_llm.data.document.config import LengthPreprocessingConfig
from fast_llm.tensor import TensorMeta
from fast_llm.utils import Assert


@dataclasses.dataclass(kw_only=True)
class TokenDocument(Document):
    tokens: torch.Tensor

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def device(self) -> torch.device:
        return self.tokens.device


@dataclasses.dataclass(kw_only=True)
class TokenModelInput(BlockModelInput, TokenDocument):
    @functools.cached_property
    def is_meta(self) -> bool:
        return isinstance(self.tokens, TensorMeta)


@dataclasses.dataclass(kw_only=True)
class TokenBatch(Batch, TokenDocument):
    _model_input_class: typing.ClassVar[type[TokenModelInput]] = TokenModelInput
    lengths: list[int]
    unpadded_length: int = None

    def __post_init__(self):
        Assert.eq(sum(self.lengths), len(self.tokens))
        if self.unpadded_length is None:
            self.unpadded_length = len(self.tokens)

    @classmethod
    def from_documents(cls, documents: typing.Sequence[TokenDocument], pad_to_size: int | None = None) -> typing.Self:
        tokens = [document.tokens for document in documents]
        lengths = [len(document) for document in documents]
        unpadded_length = sum(lengths)
        if pad_to_size is not None:
            Assert.geq(pad_to_size, unpadded_length)
            padding = pad_to_size - unpadded_length
            if padding > 0:
                tokens.append(tokens[0].new_full([padding], -100))
                lengths.append(padding)
        return cls(
            tokens=torch.cat(tokens),
            lengths=lengths,
            unpadded_length=unpadded_length,
        )

    def _get_cropped_lengths(self, begin: int, end: int) -> tuple[list[int], int, int]:
        document_begin = 0
        lengths = []
        for length in self.lengths:
            document_end = document_begin + length
            cropped_length = min(document_end, end) - max(document_begin, begin)
            if cropped_length > 0:
                if not lengths:
                    first_document_begin = document_begin
                lengths.append(cropped_length)
            if document_end > end:
                break
            document_begin = document_end

        return lengths, first_document_begin, document_end

    def _get_model_input(self, begin: int, end: int, config: LengthPreprocessingConfig):
        model_input = self._model_input_class(tokens=self.tokens[begin:end])
        lengths, first_document_begin, last_document_end = self._get_cropped_lengths(begin, end)

        LengthModelInputPreprocessor(
            lengths=lengths,
            sequence_k_past=begin,
            first_document_begin=first_document_begin,
            last_document_end=last_document_end,
            device=self.device,
            unpadded_length=min(end, self.unpadded_length) - begin,
            sequence_length=len(self.tokens),
        ).preprocess(model_input, config)

        Assert.eq(model_input.token_dim.size, end - begin)
        if self.tokens.device.type == "meta":
            model_input.tokens = TensorMeta.from_dims(
                (model_input.token_dim,), tensor_name=f"tokens_{begin}_to_{end}", dtype=torch.int64
            )
        return model_input
