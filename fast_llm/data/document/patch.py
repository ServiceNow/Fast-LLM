import dataclasses
import typing

import torch

from fast_llm.data.document.abstract import Batch, Document
from fast_llm.data.document.block import BlockModelInput, LengthModelInputPreprocessor
from fast_llm.data.document.config import PatchPreprocessingConfig
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.layers.language_model.config import LanguageModelKwargs
from fast_llm.layers.vision.config import VisionKwargs
from fast_llm.tensor import TensorMeta
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
class PatchModelInput(BlockModelInput):
    patches: torch.Tensor
    token_map: torch.Tensor
    positions: torch.Tensor
    namespace: str

    def to_kwargs(self) -> dict[str, typing.Any]:
        return {
            self.namespace: {
                **super().to_kwargs(),
                VisionKwargs.patches: self.patches,
                VisionKwargs.patch_positions: self.positions,
                VisionKwargs.device: self.patches.device,
            },
            LanguageModelKwargs.embedding_map: self.token_map,
        }


@dataclasses.dataclass(kw_only=True)
class PatchBatch(Batch, PatchDocument):
    @classmethod
    def from_documents(
        cls, documents: typing.Iterable[PatchDocument], sizes: typing.Iterable[int]
    ) -> typing.Self | None:
        # Note: `sizes` refers to the number of tokens in each document, not the number of patches.
        # But `pad_to_sizes` refers to patches. TODO: Make less confusing?
        document_begin = 0
        documents_ = []
        for document, size in zip(documents, sizes, strict=True):
            if document is not None:
                documents_.append(
                    PatchDocument(
                        patches=document.patches,
                        token_map=document.token_map + document_begin,
                        positions=document.positions,
                        lengths=document.lengths,
                    )
                )
            document_begin += size

        if not documents_:
            return None
        return cls(
            patches=torch.cat([document.patches for document in documents_]),
            token_map=torch.cat([document.token_map for document in documents_]),
            positions=torch.cat([document.positions for document in documents_]),
            lengths=sum((document.lengths for document in documents_), []),
        )

    def get_model_input(self, begin: int, end: int, config: PatchPreprocessingConfig) -> PatchModelInput:
        Assert.eq(self.patches.shape[1:], config.shape)
        if is_meta := (self.patches.device.type == "meta"):
            model_input = PatchModelInput(
                patches=self.patches[begin:end],
                token_map=self.token_map[begin:end],
                positions=self.positions[begin:end],
                namespace=config.namespace,
            )
            pad_size = 0
            unpadded_length = end - begin

        else:
            # Here `begin` and `end` refer to token rather than patch positions,
            # so we build a filter from the token map to get the corresponding patch positions.
            # TODO: ====== Should it actually refer to patch positions so model inputs have balanced sizes?? ======
            patch_filter = (self.token_map >= begin) & (self.token_map < end)
            patches = self.patches[patch_filter]
            if config.normalization is not None:
                patches = config.normalization.normalize(patches)
            patches = patches.to(config.distributed.compute_dtype.torch)

            # TODO: ====== Avoid excessive padding ======
            unpadded_length = len(patches)
            pad_size = end - begin - unpadded_length
            model_input = PatchModelInput(
                patches=torch.cat([patches, patches.new_zeros(pad_size, *patches.shape[1:])]),
                token_map=self.token_map[patch_filter] - begin,
                positions=torch.cat(
                    [self.positions[patch_filter], self.positions.new_zeros(pad_size, *self.positions.shape[1:])]
                ),
                namespace=config.namespace,
            )

        patch_begin = 0
        lengths = []
        for length in self.lengths:
            patch_end = patch_begin + length
            filtered_length = end - begin if is_meta else patch_filter[patch_begin:patch_end].sum().item()
            if filtered_length > 0:
                if not lengths:
                    sequence_k_past = patch_end - filtered_length
                    first_document_begin = patch_begin
                lengths.append(filtered_length)
            if patch_end >= end:
                break
            elif len(lengths) > 1:
                # We assume the token map is ordered, so only the first and last patch may be cropped.
                Assert.eq(filtered_length, length)
            patch_begin = patch_end

        if pad_size > 0:
            lengths.append(pad_size)

        LengthModelInputPreprocessor(
            lengths=lengths,
            sequence_k_past=sequence_k_past,
            first_document_begin=first_document_begin,
            last_document_end=patch_end + pad_size,
            device=self.patches.device,
            unpadded_length=unpadded_length,
            sequence_length=len(self.patches),
        ).preprocess(model_input, config)

        if is_meta:
            model_input.patches = TensorMeta.from_dims(
                (model_input.token_dim, *(TensorDim(f"patch_dim_{i}", size) for i, size in enumerate(config.shape))),
                tensor_name=f"patches_{begin}_to_{end}",
                dtype=torch.float32,
            )
        return model_input
