import dataclasses
import logging
import typing

import torch

from fast_llm.data.document.abstract import ModelInput
from fast_llm.data.document.config import LanguageModelBatchPreprocessingConfig
from fast_llm.data.document.patch import PatchBatch, PatchDocument, PatchModelInput
from fast_llm.data.document.range import RangeBatch, RangeDocument
from fast_llm.data.document.token import TokenBatch, TokenDocument, TokenModelInput
from fast_llm.layers.language_model.config import LanguageModelKwargs
from fast_llm.utils import div

logger = logging.getLogger(__name__)


@dataclasses.dataclass(kw_only=True)
class LanguageModelDocument(TokenDocument):
    loss_masking_spans: RangeDocument | None = None
    chosen_spans: RangeDocument | None = None
    rejected_spans: RangeDocument | None = None
    image_patches: PatchDocument | None = None


@dataclasses.dataclass(kw_only=True)
class LanguageModelTargetInput(ModelInput):
    tokens: torch.Tensor | None = None
    mask: torch.Tensor | None = None


@dataclasses.dataclass(kw_only=True)
class LanguageModelInput(TokenModelInput):
    targets: list[LanguageModelTargetInput] = dataclasses.field(default_factory=list)
    image_patches: PatchModelInput | None = None

    def set_children_attributes(self) -> None:
        if self.image_patches is not None:
            self.image_patches.set_parent_attributes(self)

    def to_kwargs(self) -> dict[str, typing.Any]:
        # TODO: Avoid conversion, use `LanguageModelMicroBatch` directly instead.
        out = {
            **super().to_kwargs(),
            LanguageModelKwargs.token_ids: self.tokens,
            LanguageModelKwargs.phase: self.phase,
            LanguageModelKwargs.device: self.tokens.device,
            LanguageModelKwargs.labels: [target.tokens for target in self.targets],
            LanguageModelKwargs.loss_mask: [target.mask for target in self.targets],
            LanguageModelKwargs.output_hidden_states: self.output_hidden_states,
            LanguageModelKwargs.hidden_states: self.hidden_states,
        }
        if self.image_patches is not None:
            out.update(self.image_patches.to_kwargs())
            out[LanguageModelKwargs.token_ids] = self.tokens
        return out


@dataclasses.dataclass(kw_only=True)
class LanguageModelBatch(TokenBatch):
    _model_input_class: typing.ClassVar[type[LanguageModelInput]] = LanguageModelInput
    loss_masking_spans: RangeBatch | None = None
    image_patches: PatchBatch | None = None

    @classmethod
    def from_documents(
        cls, documents: typing.Iterable[LanguageModelDocument], pad_to_size: int | None = None
    ) -> typing.Self:
        batch = super().from_documents(documents, pad_to_size)
        batch.loss_masking_spans = RangeBatch.from_documents(
            [document.loss_masking_spans for document in documents], batch.lengths
        )
        batch.image_patches = PatchBatch.from_documents(
            [document.image_patches for document in documents], batch.lengths
        )
        return batch

    def get_model_inputs(self, config: LanguageModelBatchPreprocessingConfig) -> list[LanguageModelInput]:
        total_input_length = len(self.tokens) - config.num_labels
        input_length = div(total_input_length, config.micro_batch_splits)

        model_inputs = []
        presents = None
        local_input_length = div(input_length, config.distributed.sequence_data_parallel)
        for micro_sequence_index, sequence_k_past in enumerate(
            range(
                local_input_length * config.distributed.sequence_data_rank,
                total_input_length,
                input_length,
            )
        ):
            model_input = self._get_model_input(sequence_k_past, sequence_k_past + local_input_length, config)

            model_input.pasts = presents
            presents = None if micro_sequence_index == config.micro_batch_splits - 1 else []
            model_input.presents = presents
            model_input.set_children_attributes()

            model_inputs.append(model_input)

        return model_inputs

    def _get_model_input(
        self, begin: int, end: int, config: LanguageModelBatchPreprocessingConfig
    ) -> LanguageModelInput:
        model_input = super()._get_model_input(begin, end, config)
        model_input.phase = config.phase

        if config.use_image_patches:
            model_input.image_patches = self.image_patches.get_model_input(begin, end, config.vision_encoder)

        for prediction_distance in range(1, config.num_labels + 1):
            label_begin = begin + prediction_distance
            label_end = end + prediction_distance
            labels = self.tokens[label_begin:label_end].clone()

            # Apply loss masking spans.
            if config.use_loss_masking_spans and self.loss_masking_spans is not None:
                for span_begin, span_end in self.loss_masking_spans.get_cropped_ranges(label_begin, label_end):
                    labels[span_begin:span_end] = -100

            # Mask cross-document predictions.
            cropped_lengths, _, _ = self._get_cropped_lengths(label_begin, label_end)
            document_begin = cropped_lengths[0]
            for length in cropped_lengths[1:]:
                labels[document_begin : document_begin + prediction_distance] = -100
                document_begin += length

            # Labels contain all four sources of masking: padding, user-defined spans, image placeholders, cross-document predictions.
            model_input.targets.append(
                LanguageModelTargetInput(
                    tokens=labels,
                    mask=labels > 0 if config.return_prediction_mask else None,
                )
            )

        return model_input
