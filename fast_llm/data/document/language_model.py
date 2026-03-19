import dataclasses
import logging
import typing

import torch

from fast_llm.data.document.abstract import ModelInput
from fast_llm.data.document.config import LanguageModelBatchPreprocessingConfig
from fast_llm.data.document.patch import PatchBatch, PatchDocument, PatchModelInput
from fast_llm.data.document.range import RangeBatch, RangeDocument
from fast_llm.data.document.token import TokenBatch, TokenDocument, TokenModelInput
from fast_llm.data.document.token_data import TokenDataBatch, TokenDataDocument
from fast_llm.layers.language_model.config import LanguageModelKwargs
from fast_llm.utils import div

logger = logging.getLogger(__name__)


@dataclasses.dataclass(kw_only=True)
class LanguageModelDocument(TokenDocument):
    loss_masking_spans: RangeDocument | None = None
    chosen_spans: RangeDocument | None = None
    rejected_spans: RangeDocument | None = None
    image_patches: PatchDocument | None = None
    advantages: TokenDataDocument | None = None
    old_log_probabilities: TokenDataDocument | None = None


@dataclasses.dataclass(kw_only=True)
class LanguageModelTargetInput(ModelInput):
    tokens: torch.Tensor | None = None
    mask: torch.Tensor | None = None
    advantages: torch.Tensor | None = None
    old_log_probabilities: torch.Tensor | None = None
    num_labels_in_seq: torch.Tensor | None = None


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
            LanguageModelKwargs.advantages: [target.advantages for target in self.targets],
            LanguageModelKwargs.old_log_probabilities: [target.old_log_probabilities for target in self.targets],
            LanguageModelKwargs.num_labels_in_seq: [target.num_labels_in_seq for target in self.targets],
        }
        if self.image_patches is not None:
            out.update(self.image_patches.to_kwargs())
            out[LanguageModelKwargs.token_ids] = self.tokens
        return out

    def to_device_(self, device: "torch.device") -> typing.Self:
        super().to_device_(device)
        for target in self.targets:
            target.to_device_(device)
        return self


@dataclasses.dataclass(kw_only=True)
class LanguageModelBatch(TokenBatch):
    _model_input_class: typing.ClassVar[type[LanguageModelInput]] = LanguageModelInput
    loss_masking_spans: RangeBatch | None = None
    image_patches: PatchBatch | None = None
    advantages: TokenDataBatch | None = None
    old_log_probabilities: TokenDataBatch | None = None

    @classmethod
    def from_documents(
        cls, documents: typing.Sequence[LanguageModelDocument], pad_to_size: int | None = None
    ) -> typing.Self:
        batch = super().from_documents(documents, pad_to_size)
        # We don't want to use `batch.lengths` because it may include a padding length.
        lengths = [len(document) for document in documents]
        batch.loss_masking_spans = RangeBatch.from_documents(
            [document.loss_masking_spans for document in documents], lengths
        )
        batch.image_patches = PatchBatch.from_documents([document.image_patches for document in documents], lengths)
        batch.advantages = TokenDataBatch.from_documents([document.advantages for document in documents], lengths)
        batch.old_log_probabilities = TokenDataBatch.from_documents(
            [document.old_log_probabilities for document in documents], lengths
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
            cropped_lengths, _, _ = self._get_cropped_lengths(begin, label_end)
            document_begin = cropped_lengths[0]
            for length in cropped_lengths[1:]:
                labels[max(document_begin - prediction_distance, 0) : document_begin] = -100
                document_begin += length

            # Labels contain all four sources of masking: padding, user-defined spans, image placeholders, cross-document predictions.
            target_input = LanguageModelTargetInput(
                tokens=labels,
                mask=labels > 0 if config.return_prediction_mask else None,
            )

            if config.use_grpo_data and not model_input.is_meta:
                target_input.advantages = self.advantages.get_cropped_data(label_begin, label_end)

                target_input.old_log_probabilities = self.old_log_probabilities.get_cropped_data(
                    label_begin, label_end
                )

                # Compute num_labels_in_seq per document: for each document segment, broadcast
                # the count of response tokens (labels >= 0) to all token positions in that segment.
                # cropped_lengths already computed above for cross-document masking.
                parts, pos = [], 0
                for length in cropped_lengths:
                    n = max(float((labels[pos : pos + length] >= 0).sum()), 1.0)
                    parts.append(torch.full([length], n, dtype=torch.float32))
                    pos += length
                target_input.num_labels_in_seq = torch.cat(parts)

            model_input.targets.append(target_input)

        return model_input
