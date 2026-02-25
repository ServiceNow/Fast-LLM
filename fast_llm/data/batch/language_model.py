import dataclasses
import typing

import torch

from fast_llm.data.batch.config import LanguageModelBatchPreprocessingConfig, ModelInput, PreprocessedBatch
from fast_llm.data.document.language_model import LanguageModelBatch, LanguageModelDocument
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedDimNames
from fast_llm.layers.attention.config import AttentionKwargs
from fast_llm.layers.language_model.config import LanguageModelKwargs
from fast_llm.tensor import TensorMeta
from fast_llm.utils import div


@dataclasses.dataclass
class LanguageModelInput(ModelInput):
    config: LanguageModelBatchPreprocessingConfig
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
    document_index_q: torch.Tensor | None = None
    document_index_k: torch.Tensor | None = None
    position_index: torch.Tensor | None = None
    # A set of intermediate the model should store in `hidden_states` for downstream usage,
    # referred by name or regex pattern.
    # Tensor names are generally of the form `{module_name}.{tensor_name}`.
    # This field is typically populated downstream, depending on the task.
    output_hidden_states: set[str] = dataclasses.field(default_factory=list)
    # The model will populate this with the hidden states specified by `output_hidden_states`,
    # together with the metadata necessary to reconstruct the global tensor.
    hidden_states: dict[str, tuple[TensorMeta, torch.Tensor]] = dataclasses.field(default_factory=dict)
    # Cached intermediate states (ex. key and value tensors) from earlier in the sequence.
    pasts: list[typing.Any] | None = None
    # If defined, the model will store intermediate states for downstream computation. Used together with `pasts`.
    presents: list[typing.Any] | None = None
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
        if self.document_index_q is not None:
            self.document_index_q = self.document_index_q.to(device, non_blocking=True)
        if self.document_index_k is not None:
            self.document_index_k = self.document_index_k.to(device, non_blocking=True)
        if self.position_index is not None:
            self.position_index = self.position_index.to(device, non_blocking=True)

    def to_kwargs(self) -> dict[str, typing.Any]:
        # TODO: Avoid conversion, use `LanguageModelMicroBatch` directly instead.
        return {
            LanguageModelKwargs.phase: self.config.phase,
            LanguageModelKwargs.device: self.tokens.device,
            LanguageModelKwargs.token_dim: self.token_dim,
            LanguageModelKwargs.hidden_token_dim: self.hidden_token_dim,
            LanguageModelKwargs.sequence_k_dim: self.sequence_k_dim,
            LanguageModelKwargs.num_tokens: self.num_tokens,
            LanguageModelKwargs.sequence_length: self.sequence_length,
            LanguageModelKwargs.sequence_lengths: self.document_lengths,
            LanguageModelKwargs.labels: self.labels,
            LanguageModelKwargs.loss_mask: self.prediction_masks,
            AttentionKwargs.cu_seqlens_q: self.cumulative_lengths_q,
            AttentionKwargs.cu_seqlens_k: self.cumulative_lengths_k,
            AttentionKwargs.max_seqlen_q: self.max_length_q,
            AttentionKwargs.max_seqlen_k: self.max_length_k,
            AttentionKwargs.document_index_q: self.document_index_q,
            AttentionKwargs.document_index_k: self.document_index_k,
            LanguageModelKwargs.position_ids: self.position_index,
            LanguageModelKwargs.output_hidden_states: self.output_hidden_states,
            LanguageModelKwargs.hidden_states: self.hidden_states,
            AttentionKwargs.past_key_values: self.pasts,
            AttentionKwargs.presents: self.presents,
        }


@dataclasses.dataclass
class LanguageModelPreprocessedBatch[
    ConfigType: LanguageModelBatchPreprocessingConfig, ModelInputType: LanguageModelInput
](PreprocessedBatch[ConfigType, ModelInputType]):
    def __init__(self, config: LanguageModelBatchPreprocessingConfig, micro_batches: list[ModelInputType]):
        super().__init__(config, micro_batches)

    @classmethod
    def from_documents(
        cls,
        documents: list[LanguageModelDocument],
        config: ConfigType,
        pad_to_size: int | None = None,
        device: torch.device | None = None,
    ) -> typing.Self:
        batch = LanguageModelBatch.from_documents(documents, pad_to_size)
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
        batch = batch.to_device(device)
        is_meta = device.type == "meta"
        total_input_length = len(batch) - config.predicted_tokens
        input_length = div(total_input_length, config.micro_batch_splits)

        token_dim = TensorDim(
            "token",
            input_length,
            config.distributed.get_distributed_dim(DistributedDimNames.sequence_data),
        )
        hidden_token_dim = (
            (
                "token_tp",
                input_length,
                config.distributed.get_distributed_dim(DistributedDimNames.tensor_and_data),
            )
            if config.distributed.sequence_tensor_parallel
            else token_dim
        )
        micro_batches = []
        presents = None
        for micro_sequence_index, sequence_k_past in enumerate(
            range(
                token_dim.size * config.distributed.sequence_data_rank,
                total_input_length,
                token_dim.global_size,
            )
        ):
            pasts = presents
            presents = None if micro_sequence_index == config.micro_batch_splits - 1 else []
            sequence_k = sequence_k_past + token_dim.size
            sequence_k_dim = TensorDim("sequence_k", sequence_k)
            cropped_sample = batch.crop(sequence_k_past, sequence_k)
            if is_meta:
                tokens = TensorMeta.from_dims(
                    (token_dim,), tensor_name=f"tokens_{sequence_k_past}_to_{sequence_k-1}", dtype=torch.int64
                )
            else:
                tokens = batch.tokens.tokens[sequence_k_past:sequence_k]
            micro_batch = LanguageModelInput(
                config=config,
                tokens=tokens,
                token_dim=token_dim,
                hidden_token_dim=hidden_token_dim,
                sequence_k_dim=sequence_k_dim,
                num_tokens=min(sequence_k, batch.num_tokens) - sequence_k_past,
                sequence_length=total_input_length,
                document_lengths=batch.tokens.lengths,
                is_meta=is_meta,
                pasts=pasts,
                presents=presents,
            )
            if not is_meta:
                if config.return_cumulative_sequence_lengths:
                    micro_batch.cumulative_lengths_q, micro_batch.cumulative_lengths_k = (
                        cropped_sample.tokens.cumulative_lengths
                    )
                if config.return_max_sequence_lengths or config.return_document_index:
                    micro_batch.max_length_q, micro_batch.max_length_k = cropped_sample.tokens.max_lengths
                if config.return_document_index:
                    micro_batch.document_index_q, micro_batch.document_index_k = cropped_sample.tokens.document_index
                if config.return_position_index:
                    micro_batch.position_index = cropped_sample.tokens.position_index

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
