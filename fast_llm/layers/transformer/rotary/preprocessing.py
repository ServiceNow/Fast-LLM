import typing

import torch

from fast_llm.engine.base_model.config import Preprocessor
from fast_llm.engine.config_utils.tensor_space import DefaultDimNames, TensorDim, TensorSpace
from fast_llm.layers.transformer.config import TransformerDimNames, TransformerKwargs
from fast_llm.layers.transformer.rotary.config import DefaultRotaryConfig
from fast_llm.tensor import TensorMeta


class RotaryEmbeddingPreprocessor(Preprocessor):
    _scalar_dim: TensorDim
    _kv_channels_dim: TensorDim
    _rotary_embedding_frequencies: torch.Tensor
    _mask: torch.Tensor
    _mask_value: torch.Tensor
    _tensor_cache_max_sequence_length: int = -1

    def __init__(
        self,
        config: DefaultRotaryConfig,
        tensor_space: TensorSpace,
    ):
        self._config = config
        self._tensor_space = tensor_space
        self._distributed_config = self._tensor_space.distributed_config
        self._scalar_dim = self._tensor_space[DefaultDimNames.scalar]
        self._kv_channels_dim = self._tensor_space[TransformerDimNames.kv_channels]

    def preprocess(self, batch, kwargs: dict[str, typing.Any]) -> None:
        self._create_tensors(kwargs[TransformerKwargs.sequence_length])
        sequence_k = kwargs[TransformerKwargs.sequence_k_dim].size
        kwargs[TransformerKwargs.rotary_freq_q] = self._rotary_embedding_frequencies[
            :, sequence_k - kwargs[TransformerKwargs.sequence_q_dim].size : sequence_k
        ]
        kwargs[TransformerKwargs.rotary_freq_k] = self._rotary_embedding_frequencies[:, :sequence_k]

    def preprocess_meta(self, kwargs: dict[str, typing.Any]) -> None:
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

    def _create_tensors(self, sequence_length: int) -> None:
        if sequence_length <= self._tensor_cache_max_sequence_length:
            return
        self._tensor_cache_max_sequence_length = sequence_length

        self._rotary_embedding_frequencies = self._config.get_frequencies(
            sequence_length,
            self._kv_channels_dim.global_size,
            device=self._tensor_space.distributed.device,
        )
