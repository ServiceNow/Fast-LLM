import dataclasses
import typing

import torch

from fast_llm.data.batch.config import LanguageModelBatchPreprocessingConfig, MicroBatch, PreprocessedBatch
from fast_llm.data.document.language_model import LanguageModelBatch, LanguageModelDocument
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedDimNames
from fast_llm.tensor import TensorMeta


@dataclasses.dataclass
class LanguageModelMicroBatch(MicroBatch):
    tokens: torch.Tensor
    token_dim: TensorDim
    hidden_token_dim: TensorDim
    sequence_k_dim: TensorDim
    # TODO: Adjust names
    num_tokens: int  # Number of tokens in the micro-batch excluding padding at the end.
    sequence_length: int  # Total number of tokens across all micro-batches, including padding.
    document_lengths: list[int]
    is_meta: bool
    labels: list[torch.Tensor] = dataclasses.field(default_factory=list)
    prediction_masks: list[torch.Tensor] = dataclasses.field(default_factory=list)
    cumulative_lengths_q: torch.Tensor | None = None
    cumulative_lengths_k: torch.Tensor | None = None
    max_length_q: torch.Tensor | None = None
    max_length_k: torch.Tensor | None = None
    document_index: torch.Tensor | None = None
    position_index: torch.Tensor | None = None
    # TODO: ====== Preference spans? ======

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


@dataclasses.dataclass
class LanguageModelPreprocessedBatch[
    ConfigType: LanguageModelBatchPreprocessingConfig, MicroBatchType: LanguageModelMicroBatch
](PreprocessedBatch[ConfigType, MicroBatchType]):
    def __init__(self, config: LanguageModelBatchPreprocessingConfig, micro_batches: list[MicroBatchType]):
        super().__init__(config, micro_batches)

    @classmethod
    def from_documents(
        cls,
        documents: list[LanguageModelDocument],
        config: ConfigType,
        device: torch.device | None = None,
    ) -> typing.Self:
        batch = LanguageModelBatch.from_documents(
            documents, pad_to_size=config.batch.micro_batch_size * config.total_length
        )
        return cls.from_batch(batch, config=config, device=device)

    @classmethod
    def from_batch(
        cls,
        batch: LanguageModelBatch,
        config: ConfigType,
        device: torch.device | None = None,
    ) -> typing.Self:
        if device is None:
            device = batch.tokens.tokens.device
        batch.to_device_(device)
        is_meta = device.type == "meta"

        token_dim = TensorDim(
            "token",
            config.batch.micro_sequence_length,
            config.distributed.get_distributed_dim(DistributedDimNames.sequence_data),
        )
        hidden_token_dim = (
            (
                "token_tp",
                token_dim.global_size,
                config.distributed.get_distributed_dim(DistributedDimNames.tensor_and_data),
            )
            if config.distributed.sequence_tensor_parallel
            else token_dim
        )
        micro_batches = []
        for micro_sequence_index, sequence_k_past in enumerate(
            range(
                token_dim.size * config.distributed.sequence_data_rank,
                config.batch.sequence_length,
                token_dim.global_size,
            )
        ):
            sequence_k = sequence_k_past + token_dim.size
            sequence_k_dim = TensorDim("sequence_k", sequence_k)
            cropped_sample = batch.crop(sequence_k_past, sequence_k)
            if is_meta:
                tokens = TensorMeta.from_dims(
                    (token_dim,), tensor_name=f"tokens_{sequence_k_past}_to_{sequence_k-1}", dtype=torch.int64
                )
            else:
                tokens = batch.tokens.tokens[sequence_k_past:sequence_k]
            micro_batch = LanguageModelMicroBatch(
                tokens=tokens,
                token_dim=token_dim,
                hidden_token_dim=hidden_token_dim,
                sequence_k_dim=sequence_k_dim,
                num_tokens=min(sequence_k, batch.num_tokens) - sequence_k_past,
                sequence_length=config.batch.sequence_length,
                document_lengths=batch.tokens.lengths,
                is_meta=is_meta,
            )
            if not is_meta:
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

                for prediction_distance in range(1, config.predicted_tokens + 1):
                    label_begin = sequence_k_past + prediction_distance
                    label_end = sequence_k + prediction_distance
                    label_tokens = batch.tokens.crop(label_begin, label_end)
                    labels = label_tokens.tokens.clone()

                    # Apply loss masking spans.
                    if config.use_loss_masking_spans and batch.loss_masking_spans is not None:
                        for span_begin, span_end in batch.loss_masking_spans.crop(label_begin, label_end).ranges:
                            labels[span_begin:span_end] = -100

                    # Mask cross-document predictions.
                    document_begin = label_tokens.lengths[0]
                    for length in label_tokens.lengths[1:]:
                        labels[document_begin : document_begin + prediction_distance] = -100
                        document_begin += length

                    # Labels contain all four sources of masking: padding, user-defined spans, image placeholders, cross-document predictions.
                    micro_batch.labels.append(labels)
                    if config.return_prediction_mask:
                        # TODO: Does the prediction mask really need all sources of masking?
                        #   (i.e. lack of labels doesn't mean we can't do predictions and compute other losses.)
                        micro_batch.prediction_masks.append(labels > 0)

            micro_batches.append(micro_batch)
        return cls(micro_batches=micro_batches, config=config)
