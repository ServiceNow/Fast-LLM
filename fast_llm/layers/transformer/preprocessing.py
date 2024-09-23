import logging

import torch

from fast_llm.engine.config_utils.tensor_space import DefaultDimNames, TensorDim, TensorSpace
from fast_llm.functional.rotary import get_rotary_frequencies
from fast_llm.layers.transformer.config import TransformerConfig, TransformerDimNames, TransformerKwargs
from fast_llm.tensor import TensorMeta

logger = logging.getLogger(__name__)


class RotaryEmbeddingPreprocessor:
    _scalar_dim: TensorDim
    _kv_channels_dim: TensorDim
    _rotary_embedding_frequencies: torch.Tensor
    _mask: torch.Tensor
    _mask_value: torch.Tensor
    _tensor_cache_max_sequence_length: int = -1

    def __init__(
        self,
        config: TransformerConfig,
        tensor_space: TensorSpace,
    ):
        self._config = config
        assert self._config.use_rotary_position_embeddings
        self._tensor_space = tensor_space
        self._distributed_config = self._tensor_space.distributed_config
        self._scalar_dim = self._tensor_space.get_tensor_dim(DefaultDimNames.scalar)
        self._kv_channels_dim = self._tensor_space.get_tensor_dim(TransformerDimNames.kv_channels)

    def create_tensors(self, sequence_length: int):
        if sequence_length <= self._tensor_cache_max_sequence_length:
            return
        self._tensor_cache_max_sequence_length = sequence_length

        self._rotary_embedding_frequencies = get_rotary_frequencies(
            sequence_length,
            self._config.kv_channels,
            self._config.rotary_position_embedding_scale,
            complex_format=self._config.complex_rotary_embeddings,
            device=self._tensor_space.distributed.device,
        )

    def preprocess(self, kwargs: dict):
        sequence_k = kwargs[TransformerKwargs.sequence_k_dim].size
        kwargs[TransformerKwargs.rotary_freq_q] = self._rotary_embedding_frequencies[
            :, sequence_k - kwargs[TransformerKwargs.sequence_q_dim].size : sequence_k
        ]
        kwargs[TransformerKwargs.rotary_freq_k] = self._rotary_embedding_frequencies[:, :sequence_k]

    def preprocess_meta(self, kwargs: dict):
        kwargs[TransformerKwargs.rotary_freq_q] = TensorMeta.from_dims(
            (
                self._scalar_dim,
                kwargs[TransformerKwargs.sequence_q_dim],
                self._scalar_dim,
                self._kv_channels_dim,
            ),
            tensor_name=TransformerKwargs.rotary_freq_q,
        )
        kwargs[TransformerKwargs.rotary_freq_k] = TensorMeta.from_dims(
            (
                self._scalar_dim,
                kwargs[TransformerKwargs.sequence_q_dim],
                self._scalar_dim,
                self._kv_channels_dim,
            ),
            tensor_name=TransformerKwargs.rotary_freq_k,
        )


class BackupAttentionPreprocessor:
    _scalar_dim: TensorDim
    _kv_channels_dim: TensorDim
    _rotary_embedding_frequencies: torch.Tensor
    _mask: torch.Tensor
    _mask_value: torch.Tensor
    _tensor_cache_max_sequence_length: int = -1

    def __init__(
        self,
        config: TransformerConfig,
        tensor_space: TensorSpace,
    ):
        self._config = config
        self._tensor_space = tensor_space
        self._distributed_config = self._tensor_space.distributed_config
        assert not self._config.do_use_flash_attention(self._distributed_config)
        self._scalar_dim = self._tensor_space.get_tensor_dim(DefaultDimNames.scalar)

    def create_tensors(self, sequence_length: int):
        if sequence_length <= self._tensor_cache_max_sequence_length:
            return
        self._tensor_cache_max_sequence_length = sequence_length

        self._mask = torch.ones(
            (sequence_length, sequence_length),
            dtype=torch.bool,
            device=self._tensor_space.distributed.device,
        ).tril_()
        if self._config.window_size is not None:
            self._mask.triu_(-self._config.window_size + 1)
        self._mask_value = torch.full(
            [],
            torch.finfo(self._distributed_config.training_dtype.torch).min,
            dtype=self._distributed_config.training_dtype.torch,
            device=self._tensor_space.distributed.device,
        )

    def preprocess(self, kwargs: dict):
        sequence_k = kwargs[TransformerKwargs.sequence_k_dim].size
        kwargs[TransformerKwargs.attention_mask] = self._mask[
            None, None, sequence_k - kwargs[TransformerKwargs.sequence_q_dim].size : sequence_k, None, :sequence_k
        ]
        kwargs[TransformerKwargs.attention_mask_value] = self._mask_value

    def preprocess_meta(self, kwargs: dict):
        kwargs[TransformerKwargs.attention_mask] = TensorMeta.from_dims(
            (
                self._scalar_dim,
                self._scalar_dim,
                kwargs[TransformerKwargs.sequence_q_dim],
                self._scalar_dim,
                kwargs[TransformerKwargs.sequence_k_dim],
            ),
            tensor_name=TransformerKwargs.attention_mask,
            dtype=torch.bool,
        )
        kwargs[TransformerKwargs.attention_mask_value] = TensorMeta.from_dims(
            (self._scalar_dim,),
            tensor_name=TransformerKwargs.attention_mask_value,
            dtype=self._tensor_space.distributed_config.training_dtype.torch,
        )
