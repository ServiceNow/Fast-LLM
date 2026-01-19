import logging
import re
import typing

import torch

from fast_llm.data.sample.language_model import LanguageModelBatch
from fast_llm.engine.base_model.base_model import BaseModel
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig, DistributedDimNames, PhaseType
from fast_llm.engine.inference.runner import InferenceRunner
from fast_llm.engine.multi_stage.fast_llm_model import FastLLMModel
from fast_llm.layers.attention.config import AttentionKwargs
from fast_llm.layers.block.config import BlockDimNames, BlockKwargs
from fast_llm.layers.language_model.kwargs import LanguageModelKwargs
from fast_llm.layers.language_model.language_model import LanguageModel
from fast_llm.models.gpt.config import GPTBaseModelConfig, GPTBatchConfig, GPTModelConfig
from fast_llm.models.gpt.megatron import get_init_megatron
from fast_llm.tensor import ParameterMeta, TensorMeta
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)


class GPTBaseModel[ConfigType: GPTBaseModelConfig](LanguageModel[ConfigType], BaseModel[ConfigType]):
    """
    A transformer-based language model generalizing the GPT model architecture.
    """

    _config: ConfigType

    def __init__(
        self,
        config: ConfigType,
        distributed_config: DistributedConfig,
    ):
        super().__init__(config, distributed_config, lr_scale=config.lr_scale, peft=config.peft)
        if self._config.use_megatron_initialization:
            for param in self.parameters():
                Assert.custom(isinstance, param, ParameterMeta)
                param.init_parameter = get_init_megatron(param, self._config.decoder.block, config.hidden_size)  # Noqa

    def preprocess_meta(
        self, batch_meta: GPTBatchConfig | LanguageModelBatch, phase: PhaseType
    ) -> list[tuple[TensorMeta, dict]]:
        # TODO Remove (Move batch splitting elsewhere)
        # TODO: Use parallel/sequential dims, distinguish micro and full batch/sequence

        if isinstance(batch_meta, GPTBatchConfig):
            micro_batch_size = batch_meta.micro_batch_size
            sequence_length = batch_meta.sequence_length
            micro_sequence_length = batch_meta.micro_sequence_length
            truncate_documents = batch_meta.truncate_documents
        else:
            micro_batch_size, sequence_length = batch_meta.tokens.tokens.shape
            if phase != PhaseType.inference:
                sequence_length -= self._config.head.prediction_heads
            micro_sequence_length = sequence_length
            truncate_documents = True

        batch_data = self._distributed_config.get_distributed_dim(DistributedDimNames.batch_data)
        batch_dim = TensorDim(BlockDimNames.batch, micro_batch_size * batch_data.size, batch_data)

        if micro_sequence_length is None:
            micro_sequence_length = sequence_length
        else:
            Assert.multiple(sequence_length, micro_sequence_length)

        # TODO: Calculate hidden dims elsewhere?
        sequence_q_dim = TensorDim(
            BlockDimNames.sequence_q,
            micro_sequence_length,
            self._distributed_config.get_distributed_dim(DistributedDimNames.sequence_data),
        )
        hidden_sequence_q_dim = (
            TensorDim(
                BlockDimNames.sequence_q_tp,
                micro_sequence_length,
                self._distributed_config.get_distributed_dim(DistributedDimNames.tensor_and_sequence_data),
            )
            if self._distributed_config.sequence_tensor_parallel
            else sequence_q_dim
        )

        need_sequence_first = hidden_sequence_q_dim.size != sequence_length
        if self._config.sequence_first is None:
            sequence_first = need_sequence_first
        else:
            sequence_first = self._config.sequence_first
            assert not (need_sequence_first and not sequence_first)

        hidden_dims = (
            (hidden_sequence_q_dim, batch_dim, self._hidden_dim)
            if sequence_first
            else (batch_dim, hidden_sequence_q_dim, self._hidden_dim)
        )

        common_kwargs = {
            LanguageModelKwargs.phase: phase,
            AttentionKwargs.sequence_first: sequence_first,
            AttentionKwargs.hidden_dims: hidden_dims,
            AttentionKwargs.sequence_length: sequence_length,
            AttentionKwargs.sequence_q_dim: sequence_q_dim,
            LanguageModelKwargs.mask_inputs: not truncate_documents,
        }

        sequence_k_pasts = range(
            sequence_q_dim.size * self._distributed_config.sequence_data_rank,
            sequence_length,
            micro_sequence_length,
        )
        reference_preprocessed_metas = {}
        for name, reference_model in self._reference_models.items():
            reference_preprocessed_metas[name] = reference_model.fast_llm_model.base_model.preprocess_meta(
                batch_meta, PhaseType.inference
            )
            Assert.eq(len(reference_preprocessed_metas[name]), len(sequence_k_pasts))

        preprocessed_meta = []
        for i, sequence_k_past in enumerate(sequence_k_pasts):
            sequence_k = sequence_k_past + sequence_q_dim.size
            sequence_k_dim = TensorDim(BlockDimNames.sequence_k, sequence_k)

            tokens = TensorMeta.from_dims(
                hidden_dims[:2], tensor_name=f"tokens_{sequence_k_past}_to_{sequence_k-1}", dtype=torch.int64
            )

            kwargs = {
                **common_kwargs,
                AttentionKwargs.sequence_k_dim: sequence_k_dim,
            }
            if phase != PhaseType.inference:
                kwargs[LanguageModelKwargs.labels] = TensorMeta.from_dims(
                    hidden_dims[:2], tensor_name="labels", dtype=torch.int64
                )
            reference_kwargs = {}
            for name, reference_preprocessed_meta in reference_preprocessed_metas.items():
                reference_tokens, reference_kwargs_ = reference_preprocessed_meta[i]
                for key in (
                    AttentionKwargs.sequence_first,
                    AttentionKwargs.sequence_length,
                    AttentionKwargs.sequence_q_dim,
                    AttentionKwargs.sequence_k_dim,
                ):
                    Assert.eq(reference_kwargs_[key], kwargs[key])
                reference_kwargs[name] = reference_kwargs_
            kwargs["reference_models"] = reference_kwargs

            preprocessed_meta.append((tokens, kwargs))

        return preprocessed_meta

    def preprocess_batch(
        self,
        batch: LanguageModelBatch,
        preprocessed_meta: list[tuple[TensorMeta, dict]] | None = None,
        *,
        phase: PhaseType,
        iteration: int,
        metrics: dict | None = None,
    ) -> list[tuple[torch.Tensor, dict]]:
        # TODO Move batch splitting elsewhere, align interface with LayerBase
        assert self._is_setup

        batch.to_device_(self._distributed.device)

        if preprocessed_meta is None:
            preprocessed_meta = self.preprocess_meta(batch, phase)

        distillation_models = self._config.decoder.get_distillation_models()
        # TODO: Support multiple distillation models?
        assert len(distillation_models) <= 1
        reference_logits = [{} for _ in preprocessed_meta]
        if phase != PhaseType.inference:
            for name, reference_model in self._reference_models.items():
                reference_preprocessed_meta = [
                    (tokens_meta, kwargs_meta["reference_models"][name])
                    for tokens_meta, kwargs_meta in preprocessed_meta
                ]

                # Set output_hidden_states in reference metadata before preprocessing if needed for distillation
                if name in distillation_models:
                    reference_output_hidden_states = [r"decoder\.\d+\.mixer_output$"]
                    for _, ref_kwargs_meta in reference_preprocessed_meta:
                        ref_kwargs_meta[BlockKwargs.output_hidden_states] = [
                            re.compile(pattern) for pattern in reference_output_hidden_states
                        ]

                reference_batch = reference_model.fast_llm_model.base_model.preprocess_batch(
                    batch,
                    reference_preprocessed_meta,
                    phase=PhaseType.inference,
                    iteration=iteration,
                )

                # TODO: Do things work with >1?
                Assert.eq(len(reference_batch), len(preprocessed_meta), 1)
                for i, (reference_tokens, reference_kwargs) in enumerate(reference_batch):
                    reference_model.forward(reference_tokens, reference_kwargs, iteration=iteration)
                    reference_logits[i][f"{name}_logits"] = reference_kwargs["logits"]
                    if BlockKwargs.hidden_states in reference_kwargs and reference_kwargs[BlockKwargs.hidden_states]:
                        # Extract activations from hidden_states dict (stored by _debug method)
                        # Format: {layer_name: (meta, tensor), ...}
                        activations = {
                            layer_name: tensor
                            for layer_name, (meta, tensor) in reference_kwargs[BlockKwargs.hidden_states].items()
                        }
                        reference_logits[i][f"{name}_activations"] = activations

        preprocessed = []
        presents = None
        for i, (_, kwargs_meta) in enumerate(preprocessed_meta):
            tokens_end = kwargs_meta[AttentionKwargs.sequence_k_dim].size
            tokens_begin = tokens_end - kwargs_meta[AttentionKwargs.sequence_q_dim].size
            cropped_tokens = batch.tokens.crop(tokens_begin, tokens_end)

            # TODO: Add pasts/presents to meta input?
            # Use lists as pointers so `past_key_values` is populated during the previous micro_sequence.
            pasts = presents
            presents = None if i == len(preprocessed_meta) - 1 else []

            # Create activation mask for activation distillation
            # This mask should:
            # - Be 0 on padding tokens (added at the end when documents aren't truncated)
            # - Be 1 on image placeholder tokens (token value -100 but not padding)
            # - Be 1 on all other valid tokens (ignores loss-masking-spans)
            #
            # Note: Padding is added as a separate document with all tokens = -100
            # We detect padding by checking if all tokens in a document segment are -100
            activation_mask = torch.ones_like(cropped_tokens.tokens, dtype=torch.bool)

            for sample_index, sample_lengths in enumerate(cropped_tokens.lengths):
                # Iterate through documents in this sample
                pos = 0
                for doc_length in sample_lengths:
                    # Check if this document is padding (all tokens are -100)
                    doc_tokens = cropped_tokens.tokens[sample_index, pos : pos + doc_length]
                    is_padding_doc = torch.all(doc_tokens == -100).item()

                    if is_padding_doc:
                        # This is a padding document, mask it out
                        activation_mask[sample_index, pos : pos + doc_length] = False

                    pos += doc_length

            kwargs: dict[str, typing.Any] = {
                **kwargs_meta,
                AttentionKwargs.past_key_values: pasts,
                AttentionKwargs.presents: presents,
                BlockKwargs.iteration: iteration,
                AttentionKwargs.sequence_lengths: cropped_tokens.lengths,
                BlockKwargs.activation_mask: activation_mask,
                AttentionKwargs.device: self._distributed.device,
                BlockKwargs.hidden_states: {},
                **reference_logits[i],
            }

            # Add activation-distillation targets
            assert len(distillation_models) <= 1
            for distillation_model in distillation_models:
                teacher_key = f"{distillation_model}_activations"
                if teacher_key in reference_logits[i]:
                    kwargs[BlockKwargs.activation_distillation_targets] = reference_logits[i].pop(teacher_key)

            if phase != PhaseType.inference:
                labels_begin = tokens_begin + 1
                labels_end = tokens_end + self._config.head.max_prediction_distance

                labels = batch.tokens.crop(labels_begin, labels_end).tokens
                loss_mask = labels >= 0
                if batch.loss_masking_spans is not None:
                    loss_masking_spans = batch.loss_masking_spans.crop(labels_begin, labels_end)
                    # loss_mask = torch.ones_like(labels, dtype=torch.bool)
                    for sample_index, loss_masking_spans in enumerate(loss_masking_spans.ranges):
                        for begin, end in loss_masking_spans:
                            loss_mask[sample_index, begin:end] = False
                    labels = torch.where(loss_mask, labels, -100)

                if self._config.head.distillation_model is not None or len(distillation_models) > 0:
                    kwargs[LanguageModelKwargs.loss_mask] = loss_mask

                kwargs[LanguageModelKwargs.labels] = (
                    labels.transpose(0, 1) if kwargs[AttentionKwargs.sequence_first] else labels
                ).contiguous()
                if LanguageModelKwargs.loss_mask in kwargs and kwargs[AttentionKwargs.sequence_first]:
                    kwargs[LanguageModelKwargs.loss_mask] = (
                        kwargs[LanguageModelKwargs.loss_mask].transpose(0, 1).contiguous()
                    )

                if batch.chosen_spans is not None:
                    kwargs[LanguageModelKwargs.chosen_spans] = batch.chosen_spans.crop(labels_begin, labels_end).ranges

                if batch.rejected_spans is not None:
                    kwargs[LanguageModelKwargs.rejected_spans] = batch.rejected_spans.crop(
                        labels_begin, labels_end
                    ).ranges

            tokens = (
                cropped_tokens.tokens.transpose(0, 1)
                if kwargs[AttentionKwargs.sequence_first]
                else cropped_tokens.tokens
            ).contiguous()
            self.preprocess(kwargs)
            preprocessed.append((tokens, kwargs))

        return preprocessed

    def get_tied_parameters(self) -> dict[str, tuple[ParameterMeta, tuple[int, ...]]]:
        # TODO: Integrate to the `LayerBase` interface, move to `LanguageModel`, `MultiTokenPrediction`?
        output_weights = self.head.get_output_weights()
        if self._config.tied_embedding_weight:
            output_weights.insert(0, self.embeddings.word_embeddings_weight)
        return {output_weights[0].tensor_name: output_weights} if len(output_weights) > 1 else {}


class GPTModel[ConfigType: GPTModelConfig](FastLLMModel[ConfigType]):
    # TODO: Can we drop class?
    pass


class GPTInferenceRunner(InferenceRunner):
    model_class: typing.ClassVar[type[GPTModel]] = GPTModel
    batch_config_class: typing.ClassVar[type[GPTBatchConfig]] = GPTBatchConfig
