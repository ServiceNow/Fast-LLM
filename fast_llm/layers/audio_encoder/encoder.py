import typing

import torch

from fast_llm.engine.base_model.base_model import Layer
from fast_llm.engine.config_utils.tensor_space import TensorSpace
from fast_llm.layers.audio_encoder.config import AudioEncoderConfig, AudioEncoderDimNames, AudioEncoderKwargs
from fast_llm.tensor import ParameterMeta, TensorMeta, init_normal_


class AudioConv(Layer):
    def __init__(self, config: AudioEncoderConfig, tensor_space: TensorSpace):
        super().__init__()
        self._tensor_space = tensor_space
        # TODO Toby: lr_scale
        self.conv1_weight = ParameterMeta.from_dims(
            (
                self._tensor_space.get_tensor_dim(AudioEncoderDimNames.out_channels),
                self._tensor_space.get_tensor_dim(AudioEncoderDimNames.in_channels),
                self._tensor_space.get_tensor_dim(AudioEncoderDimNames.kernel_size),
            ),
            init_method=init_normal_(),
        )
        self.conv1_stride = 1

        self.conv2_weight = ParameterMeta.from_dims(
            (
                self._tensor_space.get_tensor_dim(AudioEncoderDimNames.in_channels),  # in/out channels are the same
                self._tensor_space.get_tensor_dim(AudioEncoderDimNames.in_channels),
                self._tensor_space.get_tensor_dim(AudioEncoderDimNames.kernel_size),
            ),
            init_method=init_normal_(),
        )
        self.conv2_stride = 2

        if config.conv_bias:
            self.bias = ParameterMeta.from_dims(
                (self._tensor_space.get_tensor_dim(AudioEncoderDimNames.out_channels),)
            )
        else:
            self.bias = None

    def forward(
        self,
        input_: torch.Tensor,
        kwargs: dict[str, typing.Any],
        losses: dict[str, typing.Any] | None = None,
        metrics: dict | None = None,
    ) -> torch.Tensor:
        hidden_dims = kwargs[AudioEncoderKwargs.hidden_dims]
        if isinstance(input_, TensorMeta):
            return TensorMeta.from_dims(hidden_dims, tensor_name="audio conv output", dtype=input_.dtype)
        input_ = torch.nn.functional.conv1d(input_, self.conv1_weight, self.bias, stride=self.conv1_stride)
        input_ = torch.nn.functional.gelu(input_)
        input_ = torch.nn.functional.conv1d(input_, self.conv2_weight, self.bias, stride=self.conv2_stride)
        input_ = torch.nn.functional.gelu(input_)

        # TODO Toby: add learned position embeddings and dropout
        audio_embeddings = audio_embeddings.reshape(*(x.size for x in hidden_dims))

        return audio_embeddings
