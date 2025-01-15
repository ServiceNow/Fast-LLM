import logging
import typing

import torch

from fast_llm.engine.base_model.base_model import BaseModel, Layer, LossDef
from fast_llm.engine.config_utils.tensor_space import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig, DistributedDimNames, PhaseType
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.multi_stage.fast_llm_model import FastLLMModel
from fast_llm.engine.schedule.config import BatchConfig
from fast_llm.layers.language_model.config import LanguageModelKwargs, LanguageModelLossNames
from fast_llm.layers.language_model.embedding import WORD_EMBEDDINGS_WEIGHT, LanguageModelEmbedding
from fast_llm.layers.language_model.head import LanguageModelHead
from fast_llm.layers.language_model.preprocessing import PositionEmbeddingPreprocessor
from fast_llm.layers.transformer.config import (
    RoutingType,
    TransformerDimNames,
    TransformerKwargs,
    TransformerLossNames,
)
from fast_llm.layers.transformer.preprocessing import BackupAttentionPreprocessor, RotaryEmbeddingPreprocessor
from fast_llm.layers.transformer.transformer import TransformerLayer
from fast_llm.models.gpt.config import GPTBaseModelConfig, GPTModelConfig
from fast_llm.models.gpt.megatron import get_init_megatron
from fast_llm.tensor import ParameterMeta, TensorMeta
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)


class GPTBaseModel[ConfigType: GPTBaseModelConfig](BaseModel[ConfigType]):
    """
    A transformer-based language model generalizing the GPT model architecture.
    """

    config_class: typing.ClassVar[type[GPTBaseModelConfig]] = GPTBaseModelConfig
    _is_setup: bool = False
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
        if self._config.use_absolute_position_embeddings:
            self._position_embedding_preprocessor = PositionEmbeddingPreprocessor(self._config, self._tensor_space)
        if self._config.transformer.rotary.enabled:
            self._rotary_embedding_preprocessor = RotaryEmbeddingPreprocessor(
                self._config.transformer.rotary, self._tensor_space
            )
        if not self._use_flash_attention:
            self._backup_attention_preprocessor = BackupAttentionPreprocessor(
                self._config.transformer, self._tensor_space
            )

    def get_layers(self) -> list[Layer]:
        return [
            LanguageModelEmbedding(self._config, self._tensor_space),
            *[
                TransformerLayer(
                    self._config.transformer,
                    self._tensor_space,
                    layer_index=i + 1,
                )
                for i in range(self._config.transformer.num_layers)
            ],
            LanguageModelHead(self._config, self._tensor_space),
        ]

    def setup(self, distributed: Distributed) -> None:
        assert not self._is_setup
        assert distributed.config is self._tensor_space.distributed_config
        self._tensor_space.setup(distributed)
        self._is_setup = True

    def preprocess_meta(
        self, batch_meta: BatchConfig | torch.Tensor, phase: PhaseType
    ) -> list[tuple[TensorMeta, dict]]:
        # TODO: How much of this is generalizable?
        # TODO: Use parallel/sequential dims, distinguish micro and full batch/sequence

        if isinstance(batch_meta, BatchConfig):
            micro_batch_size = batch_meta.micro_batch_size
            sequence_length = batch_meta.sequence_length
            micro_sequence_length = batch_meta.micro_sequence_length
        else:
            micro_batch_size, sequence_length = batch_meta.shape
            if phase != PhaseType.inference:
                sequence_length -= 1
            micro_sequence_length = sequence_length

        batch_data = self._tensor_space.distributed_config.get_distributed_dim(DistributedDimNames.batch_data)
        batch_dim = TensorDim(TransformerDimNames.batch, micro_batch_size * batch_data.size, batch_data)

        if isinstance(batch_meta, BatchConfig):
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

        preprocessed_meta = []
        for sequence_k_past in range(
            sequence_q_dim.size * self._tensor_space.distributed_config.sequence_data_rank,
            sequence_length,
            micro_sequence_length,
        ):
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
            if self._config.use_absolute_position_embeddings:
                self._position_embedding_preprocessor.preprocess_meta(kwargs)
            if self._config.transformer.rotary.enabled:
                self._rotary_embedding_preprocessor.preprocess_meta(kwargs)
            if not self._use_flash_attention:
                self._backup_attention_preprocessor.preprocess_meta(kwargs)
            preprocessed_meta.append((tokens, kwargs))

        return preprocessed_meta

    def preprocess(
        self,
        batch: torch.Tensor,
        preprocessed_meta: list[tuple[TensorMeta, dict]] | None = None,
        *,
        phase: PhaseType,
        iteration: int,
        metrics: dict | None = None,
    ) -> list[tuple[torch.Tensor, dict]]:
        # TODO: How much of this is generalizable?
        assert self._is_setup

        if preprocessed_meta is None:
            preprocessed_meta = self.preprocess_meta(batch, phase)

        _, common_kwargs = preprocessed_meta[0]
        sequence_q = common_kwargs[TransformerKwargs.sequence_q_dim].size
        sequence_first = common_kwargs[TransformerKwargs.sequence_first]
        sequence_length = common_kwargs[TransformerKwargs.sequence_length]

        batch = batch.to(
            device=self._tensor_space.distributed.device,
            dtype=torch.int64,
            non_blocking=True,
        )
        if sequence_first:
            # Move the sequence dimension first to make sequence parallel ops more efficient.
            batch = batch.transpose(0, 1).contiguous()

        if self._config.use_absolute_position_embeddings:
            self._position_embedding_preprocessor.create_tensors(sequence_length)
        if self._config.transformer.rotary.enabled:
            self._rotary_embedding_preprocessor.create_tensors(sequence_length)
        if not self._use_flash_attention:
            self._backup_attention_preprocessor.create_tensors(sequence_length)

        preprocessed = []
        presents = None
        for i, (tokens_meta, kwargs_meta) in enumerate(preprocessed_meta):
            sequence_k = kwargs_meta[TransformerKwargs.sequence_k_dim].size
            if sequence_first:
                tokens = batch[sequence_k - sequence_q : sequence_k]
            else:
                # TODO: Avoid multiple contiguous calls?
                tokens = batch[:, sequence_k - sequence_q : sequence_k].contiguous()

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
                if sequence_first:
                    labels = batch[sequence_k - sequence_q + 1 : sequence_k + 1]
                else:
                    # TODO: Avoid multiple contiguous calls?
                    labels = batch[:, sequence_k - sequence_q + 1 : sequence_k + 1].contiguous()
                kwargs[LanguageModelKwargs.labels] = labels
            if self._config.use_absolute_position_embeddings:
                self._position_embedding_preprocessor.preprocess(kwargs)
            if self._config.transformer.rotary.enabled:
                self._rotary_embedding_preprocessor.preprocess(kwargs)
            if not self._use_flash_attention:
                self._backup_attention_preprocessor.preprocess(kwargs)
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
        return self.layers[-1]

    def get_tied_weights(self) -> dict[str, tuple[ParameterMeta, tuple[int, ...]]]:
        return (
            {WORD_EMBEDDINGS_WEIGHT: (self.embedding.word_embeddings_weight, (0, len(self) - 1))}
            if self._config.tie_word_embeddings
            else {}
        )

    @property
    def loss_defs(self) -> list[LossDef]:
        loss_defs = [
            LossDef(name=LanguageModelLossNames.language_model_loss, formatted_name="language model loss", count=1)
        ]
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
        return loss_defs


class GPTModel[ConfigType: GPTModelConfig](FastLLMModel[ConfigType]):
    config_class: typing.ClassVar[type[GPTModelConfig]] = GPTModelConfig
    base_model_class: typing.ClassVar[type[GPTBaseModel]] = GPTBaseModel
