import logging
import typing

import torch

from fast_llm.data.data.gpt.data import GPTBatch
from fast_llm.engine.base_model.base_model import BaseModel, Layer, LossDef
from fast_llm.engine.base_model.config import Preprocessor
from fast_llm.engine.config_utils.tensor_space import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig, DistributedDimNames, PhaseType
from fast_llm.engine.inference.runner import InferenceRunner
from fast_llm.engine.multi_stage.fast_llm_model import FastLLMModel
from fast_llm.layers.language_model.config import LanguageModelKwargs, LanguageModelLossNames
from fast_llm.layers.language_model.embedding import WORD_EMBEDDINGS_WEIGHT, LanguageModelEmbedding
from fast_llm.layers.language_model.head import OUTPUT_WEIGHTS, LanguageModelHead
from fast_llm.layers.language_model.preprocessing import PositionEmbeddingPreprocessor
from fast_llm.layers.transformer.config import (
    RoutingType,
    TransformerDimNames,
    TransformerKwargs,
    TransformerLossNames,
)
from fast_llm.layers.transformer.preprocessing import (
    BackupAttentionPreprocessor,
    FlashAttnVarlenPreprocessor,
    RotaryEmbeddingPreprocessor,
)
from fast_llm.layers.transformer.transformer import TransformerLayer
from fast_llm.models.gpt.config import GPTBaseModelConfig, GPTBatchConfig, GPTModelConfig
from fast_llm.models.gpt.megatron import get_init_megatron
from fast_llm.tensor import ParameterMeta, TensorMeta
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)


class GPTBaseModel[ConfigType: GPTBaseModelConfig](BaseModel[ConfigType]):
    """
    A transformer-based language model generalizing the GPT model architecture.
    """

    config_class: typing.ClassVar[type[GPTBaseModelConfig]] = GPTBaseModelConfig
    _rotary_embedding_frequencies: torch.Tensor
    _position_ids: torch.Tensor
    _mask: torch.Tensor
    _mask_value: torch.Tensor
    _tensor_cache_max_sequence_length: int = -1

    def __init__(
        self,
        config: GPTBaseModelConfig,
        distributed_config: DistributedConfig,
    ):
        super().__init__(config, distributed_config)
        self._use_flash_attention = self._config.transformer.do_use_flash_attention(distributed_config)
        if self._config.use_megatron_initialization:
            for param in self.parameters():
                Assert.custom(isinstance, param, ParameterMeta)
                param.init_parameter = get_init_megatron(param, self._config.transformer)  # Noqa
        # `self._reference_models` is not populated at this point, so we pass a mutable dict.
        self._preprocessors: list[Preprocessor] = []
        if self._config.use_absolute_position_embeddings:
            self._preprocessors.append(PositionEmbeddingPreprocessor(self._config, self._tensor_space))
        if self._config.transformer.rotary.enabled:
            self._preprocessors.append(
                RotaryEmbeddingPreprocessor(self._config.transformer.rotary, self._tensor_space)
            )
        if self._use_flash_attention:
            self._preprocessors.append(FlashAttnVarlenPreprocessor(self._config.transformer, self._tensor_space))
        else:
            self._preprocessors.append(BackupAttentionPreprocessor(self._config.transformer, self._tensor_space))

    def get_output_layers(self) -> list[Layer]:
        layers = []
        for i in range(self._config.prediction_heads):
            if i > 0:
                layers.append(
                    TransformerLayer(
                        self._config.transformer,
                        self._tensor_space,
                        # TODO MTP: which index?
                        layer_index=max(self._config.transformer.num_layers, 1),
                        # The last layer only returns the transformer output.
                        # The previous layers return a stack of shared_hidden and transformer_output.
                        return_input=i < self._config.prediction_heads - 1,
                    )
                )
            layers.append(
                LanguageModelHead(
                    self._config,
                    self._tensor_space,
                    prediction_distance=i,
                )
            )
        return layers

    def get_layers(self) -> list[Layer]:
        return [
            LanguageModelEmbedding(self._config, self._tensor_space),
            *[
                TransformerLayer(
                    self._config.transformer,
                    self._tensor_space,
                    layer_index=i + 1,
                    # The last layer only returns the transformer output.
                    # The previous layers return a stack of shared_hidden and transformer_output.
                    return_input=self._config.prediction_heads > 1 and i == self._config.transformer.num_layers - 1,
                )
                for i in range(self._config.transformer.num_layers)
            ],
            *self.get_output_layers(),
        ]

    def preprocess_meta(
        self, batch_meta: GPTBatchConfig | torch.Tensor, phase: PhaseType
    ) -> list[tuple[TensorMeta, dict]]:
        # TODO: How much of this is generalizable?
        # TODO: Use parallel/sequential dims, distinguish micro and full batch/sequence

        if isinstance(batch_meta, GPTBatchConfig):
            micro_batch_size = batch_meta.micro_batch_size
            sequence_length = batch_meta.sequence_length
            micro_sequence_length = batch_meta.micro_sequence_length
        else:
            micro_batch_size, sequence_length = batch_meta.shape
            if phase != PhaseType.inference:
                sequence_length -= self._config.prediction_heads
            micro_sequence_length = sequence_length

        batch_data = self._tensor_space.distributed_config.get_distributed_dim(DistributedDimNames.batch_data)
        batch_dim = TensorDim(TransformerDimNames.batch, micro_batch_size * batch_data.size, batch_data)

        if isinstance(batch_meta, GPTBatchConfig):
            micro_sequence_length = batch_meta.micro_sequence_length

        if micro_sequence_length is None:
            micro_sequence_length = sequence_length
        else:
            Assert.multiple(sequence_length, micro_sequence_length)

        # TODO: Calculate hidden dims elsewhere?
        sequence_q_dim = TensorDim(
            TransformerDimNames.sequence_q,
            micro_sequence_length,
            self._tensor_space.distributed_config.get_distributed_dim(DistributedDimNames.sequence_data),
        )
        hidden_sequence_q_dim = (
            TensorDim(
                TransformerDimNames.sequence_q_tp,
                micro_sequence_length,
                self._tensor_space.distributed_config.get_distributed_dim(
                    DistributedDimNames.tensor_and_sequence_data
                ),
            )
            if self._tensor_space.distributed_config.sequence_tensor_parallel
            else sequence_q_dim
        )

        need_sequence_first = hidden_sequence_q_dim.size != sequence_length
        if self._config.sequence_first is None:
            sequence_first = need_sequence_first
        else:
            sequence_first = self._config.sequence_first
            assert not (need_sequence_first and not sequence_first)

        hidden_dim = self._tensor_space.get_tensor_dim(TransformerDimNames.hidden)
        hidden_dims = (
            (hidden_sequence_q_dim, batch_dim, hidden_dim)
            if sequence_first
            else (batch_dim, hidden_sequence_q_dim, hidden_dim)
        )

        common_kwargs = {
            LanguageModelKwargs.phase: phase,
            TransformerKwargs.sequence_first: sequence_first,
            TransformerKwargs.hidden_dims: hidden_dims,
            TransformerKwargs.sequence_length: sequence_length,
            TransformerKwargs.sequence_q_dim: sequence_q_dim,
        }

        sequence_k_pasts = range(
            sequence_q_dim.size * self._tensor_space.distributed_config.sequence_data_rank,
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
            sequence_k_dim = TensorDim(TransformerDimNames.sequence_k, sequence_k)

            tokens = TensorMeta.from_dims(
                hidden_dims[:2], tensor_name=f"tokens_{sequence_k_past}_to_{sequence_k-1}", dtype=torch.int64
            )

            kwargs = {
                **common_kwargs,
                TransformerKwargs.sequence_k_dim: sequence_k_dim,
            }
            if phase != PhaseType.inference:
                kwargs[LanguageModelKwargs.labels] = TensorMeta.from_dims(
                    hidden_dims[:2], tensor_name="labels", dtype=torch.int64
                )
            for preprocessor in self._preprocessors:
                preprocessor.preprocess_meta(kwargs)
            reference_kwargs = {}
            for name, reference_preprocessed_meta in reference_preprocessed_metas.items():
                reference_tokens, reference_kwargs_ = reference_preprocessed_meta[i]
                for key in (
                    TransformerKwargs.sequence_first,
                    TransformerKwargs.sequence_length,
                    TransformerKwargs.sequence_q_dim,
                    TransformerKwargs.sequence_k_dim,
                ):
                    Assert.eq(reference_kwargs_[key], kwargs[key])
                reference_kwargs[name] = reference_kwargs_
            kwargs["reference_models"] = reference_kwargs

            preprocessed_meta.append((tokens, kwargs))

        return preprocessed_meta

    def preprocess(
        self,
        batch: GPTBatch,
        preprocessed_meta: list[tuple[TensorMeta, dict]] | None = None,
        *,
        phase: PhaseType,
        iteration: int,
        metrics: dict | None = None,
    ) -> list[tuple[torch.Tensor, dict]]:
        # TODO: How much of this is generalizable?
        assert self._is_setup

        if preprocessed_meta is None:
            preprocessed_meta = self.preprocess_meta(batch.token_ids, phase)

        _, common_kwargs = preprocessed_meta[0]
        sequence_q = common_kwargs[TransformerKwargs.sequence_q_dim].size
        sequence_first = common_kwargs[TransformerKwargs.sequence_first]
        prediction_heads: int = self._config.prediction_heads

        batch.token_ids = batch.token_ids.to(
            device=self._tensor_space.distributed.device,
            dtype=torch.int64,
            non_blocking=True,
        )

        reference_logits = [{} for _ in preprocessed_meta]
        for name, reference_model in self._reference_models.items():
            reference_preprocessed_meta = [
                (tokens_meta, kwargs_meta["reference_models"][name]) for tokens_meta, kwargs_meta in preprocessed_meta
            ]

            reference_batch = reference_model.fast_llm_model.base_model.preprocess(
                batch, reference_preprocessed_meta, phase=PhaseType.inference, iteration=iteration
            )

            # TODO: Do things work with >1?
            Assert.eq(len(reference_batch), len(preprocessed_meta), 1)
            for i, (reference_tokens, reference_kwargs) in enumerate(reference_batch):
                reference_model.forward(reference_tokens, reference_kwargs, iteration=iteration)
                reference_logits[i][f"{name}_logits"] = reference_kwargs["logits"]

        if sequence_first:
            # Move the sequence dimension first to make sequence parallel ops more efficient.
            batch.token_ids = batch.token_ids.transpose(0, 1).contiguous()

        preprocessed = []
        presents = None
        for i, (_, kwargs_meta) in enumerate(preprocessed_meta):
            sequence_k = kwargs_meta[TransformerKwargs.sequence_k_dim].size
            if sequence_first:
                tokens = batch.token_ids[sequence_k - sequence_q : sequence_k]
            else:
                # TODO: Avoid multiple contiguous calls?
                tokens = batch.token_ids[:, sequence_k - sequence_q : sequence_k].contiguous()
            if batch.sequence_lengths is not None:
                kwargs_meta[TransformerKwargs.sequence_lengths] = batch.sequence_lengths

            # TODO: Add pasts/presents to meta input?
            # Use lists as pointers so `past_key_values` is populated during the previous micro_sequence.
            pasts = presents
            presents = None if i == len(preprocessed_meta) - 1 else []
            kwargs = {
                **kwargs_meta,
                TransformerKwargs.past_key_values: pasts,
                TransformerKwargs.presents: presents,
            }
            if phase != PhaseType.inference:
                sequence_offset = sequence_k - sequence_q + 1
                if sequence_first:
                    labels = batch.token_ids[sequence_offset : sequence_k + prediction_heads]
                else:
                    # TODO: Avoid multiple contiguous calls?
                    labels = batch.token_ids[:, sequence_offset : sequence_k + prediction_heads].contiguous()
                    # We set label indices to -100 for masked spans, inline with ignore_index in torch.nn.CrossEntropyLoss
                    # TODO: take ignore_index from config
                if batch.loss_masking_spans is not None:
                    # avoid changing input tokens
                    labels = labels.clone()
                    for i, spans in enumerate(batch.loss_masking_spans):
                        if not spans.numel():
                            continue
                        valid_spans = spans[
                            (spans[:, 0] <= sequence_k + prediction_heads - 1) & (spans[:, 1] >= sequence_offset)
                        ]
                        if valid_spans.numel():
                            valid_spans[:, 0].clamp_(min=sequence_offset)
                            valid_spans[:, 1].clamp_(max=sequence_k + prediction_heads - 1)
                            valid_spans -= sequence_offset
                            loss_mask = torch.ones_like(labels, dtype=torch.bool)
                            for start, end in valid_spans:
                                if sequence_first:
                                    loss_mask[start : end + 1, i] = False
                                else:
                                    loss_mask[i, start : end + 1] = False
                            if self._config.distillation_model is not None:
                                kwargs[LanguageModelKwargs.loss_mask] = loss_mask
                            labels = torch.where(loss_mask, labels, -100)
                kwargs[LanguageModelKwargs.labels] = labels
            kwargs.update(reference_logits[i])

            for preprocessor in self._preprocessors:
                preprocessor.preprocess(tokens, kwargs)
            preprocessed.append((tokens, kwargs))

        return preprocessed

    @property
    def embedding(self) -> LanguageModelEmbedding:
        return self.layers[0]

    @property
    def transformer_layers(self) -> list[TransformerLayer]:
        return self.layers[1:-1]

    @property
    def model_head(self) -> LanguageModelHead:
        return self.layers[self.model_head_indices[0]]

    @property
    def model_head_indices(self) -> list[int]:
        return sorted([len(self) - 1 - 2 * i for i in range(self._config.prediction_heads)])

    def get_tied_weights(self) -> dict[str, tuple[ParameterMeta, tuple[int, ...]]]:
        if self._config.tie_word_embeddings:
            return {
                WORD_EMBEDDINGS_WEIGHT: (
                    self.embedding.word_embeddings_weight,
                    (0, *self.model_head_indices),
                )
            }
        elif self._config.prediction_heads > 1:
            return {
                OUTPUT_WEIGHTS: (
                    self.model_head.output_weights,
                    tuple(self.model_head_indices),
                )
            }
        else:
            return {}

    @property
    def loss_defs(self) -> list[LossDef]:
        loss_defs = []
        if (
            self._config.transformer.num_experts > 1
            and self._config.transformer.expert_routing_type == RoutingType.topk
        ):
            loss_defs.append(
                LossDef(
                    name=TransformerLossNames.load_balancing_loss,
                    formatted_name="load balancing loss",
                    count=self._config.transformer.num_layers,
                )
            )
            if self._config.transformer.expert_z_loss_coefficient:
                loss_defs.append(
                    LossDef(
                        name=TransformerLossNames.router_z_loss,
                        formatted_name="router z loss",
                        count=self._config.transformer.num_layers,
                    )
                )
        if self._config.logit_z_loss:
            LossDef(name=LanguageModelLossNames.z_loss, formatted_name="logit z loss", count=1)

        for i in range(self._config.prediction_heads):
            loss_defs.append(
                LossDef(
                    name=LanguageModelLossNames.multi_token_prediction_loss(i),
                    formatted_name=f"language model loss {i}",
                    count=1,
                )
            )
        return loss_defs


class GPTModel[ConfigType: GPTModelConfig](FastLLMModel[ConfigType]):
    config_class: typing.ClassVar[type[GPTModelConfig]] = GPTModelConfig
    base_model_class: typing.ClassVar[type[GPTBaseModel]] = GPTBaseModel


class GPTInferenceRunner(InferenceRunner):
    model_class: typing.ClassVar[type[GPTModel]] = GPTModel
    batch_config_class: typing.ClassVar[type[GPTBatchConfig]] = GPTBatchConfig
