import typing

import torch

from fast_llm.engine.base_model.base_model import Layer
from fast_llm.engine.config_utils.tensor_space import TensorSpace
from fast_llm.layers.audio_encoder.config import AudioEncoderConfig, AudioEncoderDimNames
from fast_llm.layers.transformer.config import AudioTransformerKwargs
from fast_llm.tensor import ParameterMeta, TensorMeta, init_normal_


class AudioConv(Layer):
    def __init__(self, config: AudioEncoderConfig, tensor_space: TensorSpace):
        super().__init__()
        self._tensor_space = tensor_space
        self.dropout_p = config.encoder_dropout

        # TODO Toby: lr_scale
        self.conv1_weight = ParameterMeta.from_dims(
            (
                self._tensor_space.get_tensor_dim(AudioEncoderDimNames.out_channels),
                self._tensor_space.get_tensor_dim(AudioEncoderDimNames.in_channels),
                self._tensor_space.get_tensor_dim(AudioEncoderDimNames.kernel_size),
            ),
            init_method=init_normal_(),
        )
        self.conv1_stride = 1  # TODO: parameterize?

        self.conv2_weight = ParameterMeta.from_dims(
            (
                self._tensor_space.get_tensor_dim(AudioEncoderDimNames.out_channels),
                self._tensor_space.get_tensor_dim(AudioEncoderDimNames.out_channels),
                self._tensor_space.get_tensor_dim(AudioEncoderDimNames.kernel_size),
            ),
            init_method=init_normal_(),
        )
        self.conv2_stride = 2  # TODO: parameterize?

        if config.conv_bias:
            self.conv1_bias = ParameterMeta.from_dims(
                (self._tensor_space.get_tensor_dim(AudioEncoderDimNames.out_channels),), init_method=init_normal_()
            )
            self.conv2_bias = ParameterMeta.from_dims(
                (self._tensor_space.get_tensor_dim(AudioEncoderDimNames.out_channels),), init_method=init_normal_()
            )
        else:
            self.conv1_bias = None
            self.conv2_bias = None

        self.positional_embeddings = ParameterMeta.from_dims(
            (
                self._tensor_space.get_tensor_dim(AudioEncoderDimNames.max_source_positions),
                self._tensor_space.get_tensor_dim(AudioEncoderDimNames.out_channels),
            ),
            init_method=init_normal_(),
        )

    def forward(
        self,
        input_: torch.Tensor,
        kwargs: dict[str, typing.Any],
        losses: dict[str, typing.Any] | None = None,
        metrics: dict | None = None,
    ) -> torch.Tensor:
        hidden_dims = kwargs[AudioTransformerKwargs.hidden_dims]  # TODO: check seq q
        if isinstance(input_, TensorMeta):
            return TensorMeta.from_dims(hidden_dims, tensor_name="audio conv output", dtype=input_.dtype)

        # TODO: check how to best cast dtype
        input_ = input_.to(self.conv1_weight.dtype)

        input_ = torch.nn.functional.conv1d(
            input_, self.conv1_weight, self.conv1_bias, stride=self.conv1_stride, padding=1
        )
        input_ = torch.nn.functional.gelu(input_)
        input_ = torch.nn.functional.conv1d(
            input_, self.conv2_weight, self.conv2_bias, stride=self.conv2_stride, padding=1
        )
        input_ = torch.nn.functional.gelu(input_)

        audio_embeddings = input_.permute(0, 2, 1)
        audio_embeddings = audio_embeddings + self.positional_embeddings
        audio_embeddings = torch.nn.functional.dropout(audio_embeddings, p=self.dropout_p, training=self.training)

        return audio_embeddings.contiguous()
