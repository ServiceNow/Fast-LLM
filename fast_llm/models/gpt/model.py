import functools
import logging
import re
import typing

import torch

from fast_llm.config import NoAutoValidate
from fast_llm.data.sample.language_model import LanguageModelBatch
from fast_llm.engine.base_model.base_model import BaseModel
from fast_llm.engine.distributed.config import DistributedConfig, PhaseType
from fast_llm.engine.inference.runner import InferenceRunner
from fast_llm.engine.multi_stage.fast_llm_model import FastLLMModel
from fast_llm.layers.attention.config import AttentionKwargs
from fast_llm.layers.block.config import BlockKwargs
from fast_llm.layers.language_model.config import LanguageModelKwargs
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
        self, batch_config: GPTBatchConfig | LanguageModelBatch, phase: PhaseType
    ) -> tuple[TensorMeta, GPTBatchConfig]:
        # TODO Remove (Move batch splitting elsewhere)
        # TODO: Use parallel/sequential dims, distinguish micro and full batch/sequence

        if isinstance(batch_config, LanguageModelBatch):
            micro_batch_size, sequence_length = batch_config.tokens.tokens.shape
            if phase != PhaseType.inference:
                sequence_length -= self._config.head.prediction_heads
            with NoAutoValidate():
                batch_config = GPTBatchConfig(micro_batch_size=micro_batch_size, sequence_length=sequence_length)
            batch_config.validate()

        tokens = TensorMeta.from_dims(
            (batch_config.token_dim, self._hidden_dim), tensor_name="tokens", dtype=torch.int64
        )

        return tokens, batch_config

    def preprocess_batch(
        self,
        batch: LanguageModelBatch,
        batch_config: GPTBatchConfig | None = None,
        *,
        phase: PhaseType,
        iteration: int,
        metrics: dict | None = None,
    ) -> list[tuple[torch.Tensor, dict]]:
        # TODO Move batch splitting elsewhere, align interface with LayerBase
        assert self._is_setup

        batch.to_device_(self._distributed.device)

        if batch_config is None:
            _, batch_config = self.preprocess_meta(batch, phase)

        reference_preprocessed_batches = {
            name: reference_model.fast_llm_model.base_model.preprocess_batch(
                batch,
                batch_config,
                phase=PhaseType.inference,
                iteration=iteration,
            )
            for name, reference_model in self._reference_models.items()
        }

        preprocessed = []
        presents = None
        for micro_sequence_index, sequence_k_dim in enumerate(batch_config.sequence_k_dims):
            tokens_end = sequence_k_dim.size
            tokens_begin = tokens_end - batch_config.sequence_q_dim.size
            cropped_tokens = batch.tokens.crop(tokens_begin, tokens_end)

            # Use lists as pointers so `past_key_values` is populated during the previous micro_sequence.
            pasts = presents
            presents = None if micro_sequence_index == batch_config.micro_sequences - 1 else []

            # TODO: ======= activation mask ========

            kwargs: dict[str, typing.Any] = {
                BlockKwargs.batch_config: batch_config,
                BlockKwargs.sequence_k_dim: batch_config.sequence_k_dims[micro_sequence_index],
                AttentionKwargs.past_key_values: pasts,
                AttentionKwargs.presents: presents,
                BlockKwargs.iteration: iteration,
                AttentionKwargs.sequence_lengths: cropped_tokens.lengths,
                # BlockKwargs.activation_mask: activation_mask,
                AttentionKwargs.device: self._distributed.device,
                BlockKwargs.hidden_states: {},
            }

            for name, reference_model in self._reference_models.items():
                reference_tokens, reference_kwargs = reference_preprocessed_batches[name][micro_sequence_index]
                if name in self._decoder_reference_models:
                    if BlockKwargs.output_hidden_states not in reference_kwargs[BlockKwargs.output_hidden_states]:
                        reference_kwargs[BlockKwargs.output_hidden_states] = []
                    # TODO: Get the actual names
                    reference_kwargs[BlockKwargs.output_hidden_states].append(
                        re.compile(r"decoder\.\d+\.mixer_output$")
                    )

                reference_model.forward(reference_tokens, reference_kwargs, iteration=iteration)

                if name in self._head_reference_models:
                    kwargs[f"reference_{name}_logits"] = reference_kwargs["logits"]

                if reference_hidden_states := reference_kwargs[BlockKwargs.hidden_states]:
                    kwargs[f"reference_{name}_hidden_states"] = {
                        layer_name: tensor for layer_name, (meta, tensor) in reference_hidden_states.items()
                    }

            if phase != PhaseType.inference:
                labels_begin = tokens_begin + 1
                labels_end = tokens_end + self._config.head.max_prediction_distance
                labels = batch.tokens.crop(labels_begin, labels_end).tokens

                if batch.loss_masking_spans is not None:
                    loss_masking_spans = batch.loss_masking_spans.crop(labels_begin, labels_end)
                    loss_mask = torch.ones_like(labels, dtype=torch.bool)
                    for sample_index, loss_masking_spans in enumerate(loss_masking_spans.ranges):
                        for begin, end in loss_masking_spans:
                            loss_mask[sample_index, begin:end] = False
                    labels = torch.where(loss_mask, labels, -100)

                labels = labels.flatten(0, 1)
                kwargs[LanguageModelKwargs.labels] = labels

                if self._config.head.get_reference_models():  # loss masks only used for distillation currently
                    # loss masks contain all three sources of masking: padding, user-defined spans, image placeholders
                    kwargs[LanguageModelKwargs.loss_mask] = labels >= 0

                if batch.chosen_spans is not None:
                    kwargs[LanguageModelKwargs.chosen_spans] = batch.chosen_spans.crop(labels_begin, labels_end).ranges

                if batch.rejected_spans is not None:
                    kwargs[LanguageModelKwargs.rejected_spans] = batch.rejected_spans.crop(
                        labels_begin, labels_end
                    ).ranges

            tokens = cropped_tokens.tokens.flatten(0, 1)
            self.preprocess(kwargs)
            preprocessed.append((tokens, kwargs))

        return preprocessed

    def get_tied_parameters(self) -> dict[str, tuple[ParameterMeta, tuple[int, ...]]]:
        # TODO: Integrate to the `LayerBase` interface, move to `LanguageModel`, `MultiTokenPrediction`?
        output_weights = self.head.get_output_weights()
        if self._config.tied_embedding_weight:
            output_weights.insert(0, self.embeddings.word_embeddings_weight)
        return {output_weights[0].tensor_name: output_weights} if len(output_weights) > 1 else {}

    @functools.cached_property
    def _decoder_reference_models(self) -> set[str]:
        out = self._config.decoder.get_reference_models()
        Assert.leq(out, self._reference_models.keys())
        Assert.leq(len(out), 1)
        return out

    @functools.cached_property
    def _head_reference_models(self) -> set[str]:
        out = self._config.head.get_reference_models()
        Assert.leq(out, self._reference_models.keys())
        return out


class GPTModel[ConfigType: GPTModelConfig](FastLLMModel[ConfigType]):
    # TODO: Can we drop class?
    pass


class GPTInferenceRunner(InferenceRunner):
    model_class: typing.ClassVar[type[GPTModel]] = GPTModel
    batch_config_class: typing.ClassVar[type[GPTBatchConfig]] = GPTBatchConfig
