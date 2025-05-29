import typing

import torch

from fast_llm.engine.base_model.base_model import Layer
from fast_llm.engine.config_utils.tensor_space import TensorSpace
from fast_llm.functional.triton.mlp import torch_mlp_activation
from fast_llm.layers.audio_encoder.config import AudioEncoderConfig, AudioEncoderDimNames
from fast_llm.layers.common.linear import Linear
from fast_llm.layers.transformer.config import TransformerDimNames, TransformerKwargs
from fast_llm.tensor import TensorMeta, init_normal_


class AudioAdapter(Layer):
    """
    Vision adapter layer for the LLM.
    """

    def __init__(self, config: AudioEncoderConfig, tensor_space: TensorSpace):
        super().__init__()
        audio_hidden_dim = tensor_space.get_tensor_dim(AudioEncoderDimNames.out_channels)
        input_dim = tensor_space.get_tensor_dim(AudioEncoderDimNames.adapter_input)
        self._activation_type = config.adapter_activation_type
        self._use_adapter_bias = config.adapter_bias
        self.lr_scale = config.adapter_lr_scale

        self.norm_1 = config.transformer.normalization.get_layer(audio_hidden_dim)
        self.norm_1.lr_scale = self.lr_scale
        self.norm_2 = config.transformer.normalization.get_layer(
            tensor_space.get_tensor_dim(AudioEncoderDimNames.adapter_size)
        )
        self.norm_2.lr_scale = self.lr_scale

        # TODO Soham: Make them OutputParallelLinear instead? How would this work with parallelism?
        self.layer_1 = Linear(
            input_dim,
            tensor_space.get_tensor_dim(AudioEncoderDimNames.adapter_size),
            bias=self._use_adapter_bias,
            weight_init_method=init_normal_(),
            bias_init_method=init_normal_(),
            lr_scale=self.lr_scale,
        )
        self.layer_2 = Linear(
            tensor_space.get_tensor_dim(AudioEncoderDimNames.adapter_size),
            tensor_space.get_tensor_dim(TransformerDimNames.hidden),
            bias=self._use_adapter_bias,
            weight_init_method=init_normal_(),
            bias_init_method=init_normal_(),
            lr_scale=self.lr_scale,
        )

        self.aud_downsampling_k = config.aud_downsampling_k

    def forward(
        self,
        input_: torch.Tensor,
        kwargs: dict[str, typing.Any],
        losses: dict[str, typing.Any] | None = None,
        metrics: dict[str, typing.Any] | None = None,
    ) -> torch.Tensor:
        if isinstance(input_, TensorMeta):
            return TensorMeta.from_dims(
                kwargs[TransformerKwargs.hidden_dims],
                tensor_name="Audio adapter output",
                dtype=input_.dtype,
            )
        input_ = self.norm_1(input_)
        batch_size, seq_len, dim = input_.size()

        # Check if sequence length is divisible by downsampling rate.
        if seq_len % self.aud_downsampling_k != 0:
            # If not divisible, trim the end of the sequence.
            trimmed_seq_len = seq_len - (seq_len % self.aud_downsampling_k)
            input_ = input_[:, :trimmed_seq_len, :]
            seq_len = trimmed_seq_len

        # Reshape: group every k frames together (concatenate along feature dimension).
        new_seq_len = seq_len // self.aud_downsampling_k
        input_ = input_.contiguous().view(batch_size, new_seq_len, dim * self.aud_downsampling_k)

        res = self.layer_2(
            self.norm_2(
                torch_mlp_activation(input_=self.layer_1(input_), gated=False, activation_type=self._activation_type)
            )
        )
        return res
