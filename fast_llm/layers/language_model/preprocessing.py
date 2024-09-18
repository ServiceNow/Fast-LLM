import logging

import torch

from fast_llm.layers.language_model.config import LanguageModelBaseConfig, LanguageModelKwargs
from fast_llm.layers.transformer.config import TransformerKwargs
from fast_llm.tensor import DefaultDimNames, TensorDim, TensorMeta, TensorSpace
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)


class PositionEmbeddingPreprocessor:
    _scalar_dim: TensorDim
    _rotary_embedding_frequencies: torch.Tensor
    _position_ids: torch.Tensor
    _tensor_cache_max_sequence_length: int = -1

    def __init__(
        self,
        config: LanguageModelBaseConfig,
        tensor_space: TensorSpace,
    ):
        self._config = config
        assert config.use_absolute_position_embeddings
        self._tensor_space = tensor_space
        self._distributed_config = self._tensor_space.distributed_config
        self._scalar_dim = self._tensor_space.get_tensor_dim(DefaultDimNames.scalar)

    def create_tensors(self, sequence_length: int):
        if sequence_length <= self._tensor_cache_max_sequence_length:
            return
        self._tensor_cache_max_sequence_length = sequence_length

        Assert.leq(sequence_length, self._config.num_absolute_position_embeddings)
        self._position_ids = torch.arange(
            0, sequence_length, device=self._tensor_space.distributed.device, dtype=torch.int64
        )

    def preprocess(self, kwargs: dict):
        sequence_k = kwargs[TransformerKwargs.sequence_k_dim].size
        kwargs[LanguageModelKwargs.position_ids] = self._position_ids[
            sequence_k - kwargs[TransformerKwargs.sequence_q_dim].size : sequence_k
        ].unsqueeze(int(kwargs[TransformerKwargs.sequence_first]))

    def preprocess_meta(self, kwargs: dict):
        # Position embeddings will be broadcast.
        sequence_q_dim = kwargs[TransformerKwargs.sequence_q_dim]
        kwargs[LanguageModelKwargs.position_ids] = TensorMeta.from_dims(
            (
                (sequence_q_dim, self._scalar_dim)
                if kwargs[TransformerKwargs.sequence_first]
                else (self._scalar_dim, sequence_q_dim)
            ),
            tensor_name=LanguageModelKwargs.position_ids,
            dtype=torch.int64,
        )
