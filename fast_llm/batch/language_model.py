import dataclasses
import typing

import torch

from fast_llm.batch.config import LanguageModelBatchPreprocessingConfig
from fast_llm.data.sample.language_model import LanguageModelSample
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig, DistributedDimNames


@dataclasses.dataclass
class LanguageModelBatchNew:
    tokens: torch.Tensor
    token_dim: TensorDim
    hidden_token_dim: TensorDim
    sequence_k_dim: TensorDim
    # TODO: Adjust names
    num_tokens: int  # Number of tokens in the micro-batch excluding padding at the end.
    sequence_length: int  # Total number of tokens across all micro-batches, including padding.
    document_lengths: list[int]
    labels: list[torch.Tensor] = dataclasses.field(default_factory=list)
    prediction_masks: list[torch.Tensor] = dataclasses.field(default_factory=list)
    cumulative_lengths_q: torch.Tensor | None = None
    cumulative_lengths_k: torch.Tensor | None = None
    max_length_q: torch.Tensor | None = None
    max_length_k: torch.Tensor | None = None
    document_index: torch.Tensor | None = None
    position_index: torch.Tensor | None = None
    chosen_spans: list[tuple[int, int]] | None = None
    rejected_spans: list[tuple[int, int]] | None = None

    def to_device_(self, device: torch.device):
        self.tokens = self.tokens.to(device, non_blocking=True)
        if self.cumulative_lengths_q is not None:
            self.cumulative_lengths_q = self.cumulative_lengths_q.to(device, non_blocking=True)
        if self.cumulative_lengths_k is not None:
            self.cumulative_lengths_k = self.cumulative_lengths_k.to(device, non_blocking=True)
        if self.max_length_q is not None:
            self.max_length_q = self.max_length_q.to(device, non_blocking=True)
        if self.max_length_k is not None:
            self.max_length_k = self.max_length_k.to(device, non_blocking=True)
        if self.document_index is not None:
            self.document_index = self.document_index.to(device, non_blocking=True)
        if self.position_index is not None:
            self.position_index = self.position_index.to(device, non_blocking=True)

    @classmethod
    def from_documents(
        cls,
        config: LanguageModelBatchPreprocessingConfig,
        distributed_config: DistributedConfig,
        documents: list[LanguageModelSample],
        device: torch.device | None = None,
    ) -> list[typing.Self]:
        num_tokens = sum(len(document) for document in documents)
        padding = config.batch.sequence_length + config.predicted_tokens - num_tokens
        sample = LanguageModelSample.from_documents(documents + [documents[0].get_padding(padding)])
        # sample.tokens.lengths
        # lengths = [len(document) for document in documents]
        # num_tokens = sum(lengths)

        if device is None:
            device = sample.tokens.tokens.device
        sample.to_device_(device)

        token_dim = TensorDim(
            "token",
            config.batch.micro_sequence_length,
            distributed_config.get_distributed_dim(DistributedDimNames.sequence_data),
        )
        hidden_token_dim = (
            (
                "token_tp",
                token_dim.global_size,
                distributed_config.get_distributed_dim(DistributedDimNames.tensor_and_data),
            )
            if distributed_config.sequence_tensor_parallel
            else token_dim
        )
        micro_batches = []
        for micro_sequence_index, sequence_k_past in enumerate(
            range(
                token_dim.size * distributed_config.sequence_data_rank,
                config.batch.sequence_length,
                token_dim.global_size,
            )
        ):
            sequence_k = sequence_k_past + token_dim.size
            sequence_k_dim = TensorDim("sequence_k", sequence_k)
            cropped_sample = sample.crop(sequence_k_past, sequence_k)

            # document_lengths, cumulative_lengths_q, cumulative_lengths_k, first_document_index, remaining_tokens = crop_lengths(
            #    sample.tokens.lengths, sequence_k_past, sequence_k_past + token_dim.size)

            micro_batch = LanguageModelBatchNew(
                tokens=sample.tokens.tokens[sequence_k_past:sequence_k],
                token_dim=token_dim,
                hidden_token_dim=hidden_token_dim,
                sequence_k_dim=sequence_k_dim,
                num_tokens=min(sequence_k, num_tokens) - sequence_k_past,
                sequence_length=config.batch.sequence_length,
                document_lengths=sample.tokens.lengths,
            )
            if config.return_cumulative_sequence_lengths:
                micro_batch.cumulative_lengths_q, micro_batch.cumulative_lengths_k = (
                    cropped_sample.tokens.get_cumulative_lengths(device)
                )
            if config.return_max_sequence_lengths:
                micro_batch.max_length_q, micro_batch.max_length_k = cropped_sample.tokens.get_max_lengths(device)
            if config.return_document_index:
                micro_batch.document_index = cropped_sample.tokens.get_document_index()
            if config.return_position_index:
                micro_batch.position_index = cropped_sample.tokens.get_position_index()
            if config.use_preference_spans:
                micro_batch.chosen_spans = cropped_sample.chosen_spans.ranges
                micro_batch.rejected_spans = cropped_sample.rejected_spans.ranges

            for prediction_distance in range(1, config.predicted_tokens + 1):
                label_begin = sequence_k_past + prediction_distance
                label_end = sequence_k + prediction_distance
                label_tokens = sample.tokens.crop(label_begin, label_end)
                labels = label_tokens.tokens.clone()

                # Apply loss masking spans.
                if config.use_loss_masking_spans:
                    for span_begin, span_end in sample.loss_masking_spans.crop(label_begin, label_end).ranges:
                        labels[span_begin:span_end] = -100

                # Mask cross-document predictions.
                document_end = 0
                for length in label_tokens.lengths:
                    document_end += length
                    labels[max(document_end - prediction_distance, 0) : document_end] = -100

                # Labels contain all four sources of masking: padding, user-defined spans, image placeholders, cross-document predictions.
                micro_batch.labels.append(labels)
                if config.return_prediction_mask:
                    # TODO: Does the prediction mask really need all sources of masking?
                    #   (i.e. lack of labels doesn't mean we can't do predictions and compute other losses.)
                    micro_batch.prediction_masks.append(labels > 0)

            micro_batches.append(micro_batch)
        return micro_batches
