import dataclasses
import logging
import typing

import torch

from fast_llm.core.distributed import allreduce_scalar
from fast_llm.data.document.abstract import ModelInput
from fast_llm.data.document.config import LanguageModelBatchPreprocessingConfig
from fast_llm.data.document.patch import PatchBatch, PatchDocument, PatchModelInput
from fast_llm.data.document.range import RangeBatch, RangeDocument
from fast_llm.data.document.token import TokenBatch, TokenDocument, TokenModelInput
from fast_llm.data.document.token_data import TokenDataBatch, TokenDataDocument
from fast_llm.engine.distributed.distributed import Distributed
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
    label_counts: torch.Tensor | None = None
    num_labels: int | None = None
    num_labels_in_batch: int | None = None

    @classmethod
    def share_batch_data(cls, model_inputs: "list[LanguageModelTargetInput]", distributed: "Distributed"):
        if model_inputs[0].num_labels is not None and model_inputs[0].num_labels_in_batch is None:
            # We sum over sequences but not within a sequence.
            num_labels_in_batch = allreduce_scalar(
                sum(model_input.num_labels for model_input in model_inputs),
                dtype=torch.int32,
                group=distributed.batch_data_group,
            )
            for model_input in model_inputs:
                model_input.num_labels_in_batch = num_labels_in_batch


@dataclasses.dataclass(kw_only=True)
class LanguageModelInput(TokenModelInput):
    targets: list[LanguageModelTargetInput] = dataclasses.field(default_factory=list)
    image_patches: PatchModelInput | None = None

    @classmethod
    def share_batch_data(cls, model_inputs: "list[LanguageModelInput]", distributed: "Distributed"):
        super().share_batch_data(model_inputs, distributed)
        for targets in zip(*(model_input.targets for model_input in model_inputs), strict=True):
            targets[0].share_batch_data(targets, distributed)
        model_inputs[0].image_patches.share_batch_data(
            [model_input.image_patches for model_input in model_inputs], distributed
        )

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
            LanguageModelKwargs.label_counts: [target.label_counts for target in self.targets],
            LanguageModelKwargs.num_labels_in_batch: [target.num_labels_in_batch for target in self.targets],
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
        batch.advantages = TokenDataBatch.from_documents(
            [document.advantages for document in documents], lengths, pad_to_size
        )
        batch.old_log_probabilities = TokenDataBatch.from_documents(
            [document.old_log_probabilities for document in documents], lengths, pad_to_size
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
            model_input.phase = config.phase

            if config.use_image_patches:
                model_input.image_patches = self.image_patches.get_model_input(
                    sequence_k_past, sequence_k_past + local_input_length, config.vision_encoder
                )

            model_input.pasts = presents
            presents = None if micro_sequence_index == config.micro_batch_splits - 1 else []
            model_input.presents = presents
            model_input.set_children_attributes()

            model_inputs.append(model_input)

        self._set_target_inputs(model_inputs, config)

        return model_inputs

    def _set_target_inputs(
        self, model_inputs: list[LanguageModelInput], config: LanguageModelBatchPreprocessingConfig
    ):
        labels = self.tokens.clone()

        # Apply loss masking spans.
        if config.use_loss_masking_spans and self.loss_masking_spans is not None:
            for span_begin, span_end in self.loss_masking_spans.ranges:
                labels[span_begin:span_end] = -100

        for prediction_distance in range(1, config.num_labels + 1):
            # Mask cross-document predictions.
            document_begin = 0
            for length in self.lengths:
                labels[document_begin + prediction_distance - 1] = -100
                document_begin += length

            prediction_labels = labels[
                prediction_distance : len(self.tokens) - config.num_labels + prediction_distance
            ].clone()
            mask = prediction_labels >= 0
            label_counts = self._get_label_counts(mask) if config.return_label_counts else None

            for input_index, model_input in enumerate(model_inputs):
                begin = model_input.sequence_k_dim.size
                end = begin + model_input.token_dim.size

                # Labels contain all four sources of masking: padding, user-defined spans, image placeholders, cross-document predictions.
                target_input = LanguageModelTargetInput(
                    tokens=labels[begin:end],
                    mask=mask[begin:end] if config.return_prediction_mask else None,
                    label_counts=label_counts[begin:end] if config.return_label_counts else None,
                    # Set value for the first input only so `share_batch_data` generated the correct sum.
                    # TODO: ====== Make optional?
                    num_labels=mask.sum(dtype=torch.int32).item() if input_index == 0 else 0,
                )
                if config.use_grpo_data and not model_input.is_meta:
                    target_input.advantages = self.advantages.get_cropped_data(
                        begin + prediction_distance, end + prediction_distance
                    )
                    target_input.old_log_probabilities = self.old_log_probabilities.get_cropped_data(
                        begin + prediction_distance, end + prediction_distance
                    )

                model_input.targets.append(target_input)

    def _get_label_counts(self, mask: torch.Tensor):
        # Count the number of non-masked labels in each document through cumulative sums.
        mask_cumsum = torch.cat([mask.new_zeros(1), mask.cumsum(0)])
        length_cumsum = torch.tensor([0] + self.lengths, device=self.device).cumsum(0)
        label_count_cumsum = mask_cumsum[length_cumsum]
        labels_per_document = label_count_cumsum[1:] - label_count_cumsum[:-1]
        # Expand to one entry per token: find each token's document index via the sorted
        # length cumsum, then look up that document's label count.
        # TODO: Document index already computed in `LengthModelInputPreprocessor`.
        document_index = torch.searchsorted(
            length_cumsum[1:], torch.arange(len(mask), device=self.device), side="right"
        )
        return labels_per_document[document_index]
