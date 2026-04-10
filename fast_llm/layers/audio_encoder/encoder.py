import typing

import torch

from fast_llm.engine.base_model.base_model import Layer
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.layers.audio_encoder.config import AudioEncoderConfig, AudioEncoderDimNames, AudioKwargs
from fast_llm.engine.config_utils.initialization import init_normal_
from fast_llm.tensor import ParameterMeta, TensorMeta


class AudioConv(Layer):
    """
    Dual conv1d front-end for the Whisper-style audio encoder.

    Input:  (batch, num_mel_bins, time)  — mel spectrogram
    Output: (batch, time', hidden_size)  — downsampled frame embeddings
    where time' ≈ time / 2 due to the stride-2 second convolution.
    """

    def __init__(self, config: AudioEncoderConfig, distributed_config: DistributedConfig | None = None):
        super().__init__(distributed_config or DistributedConfig())
        self.dropout_p = config.encoder_dropout
        self._conv_lr_scale = config.conv_lr_scale
        self._pos_emb_lr_scale = config.pos_emb_lr_scale

        out_channels = TensorDim(AudioEncoderDimNames.out_channels, config.hidden_size)
        in_channels = TensorDim(AudioEncoderDimNames.in_channels, config.num_mel_bins)
        kernel_size = TensorDim(AudioEncoderDimNames.kernel_size, config.kernel_size)
        max_positions = TensorDim(AudioEncoderDimNames.max_source_positions, config.max_source_positions)

        self.conv1_weight = ParameterMeta.from_dims(
            (out_channels, in_channels, kernel_size),
            init_method=init_normal_(),
            lr_scale=self._conv_lr_scale,
        )
        self.conv1_stride = 1

        self.conv2_weight = ParameterMeta.from_dims(
            (out_channels, out_channels, kernel_size),
            init_method=init_normal_(),
            lr_scale=self._conv_lr_scale,
        )
        self.conv2_stride = 2

        if config.conv_bias:
            self.conv1_bias = ParameterMeta.from_dims(
                (out_channels,),
                init_method=init_normal_(),
                lr_scale=self._conv_lr_scale,
            )
            self.conv2_bias = ParameterMeta.from_dims(
                (out_channels,),
                init_method=init_normal_(),
                lr_scale=self._conv_lr_scale,
            )
        else:
            self.conv1_bias = None
            self.conv2_bias = None

        self.positional_embeddings = ParameterMeta.from_dims(
            (max_positions, out_channels),
            init_method=init_normal_(),
            lr_scale=self._pos_emb_lr_scale,
        )
        # Store for shape-inference in TensorMeta branch
        self._out_channels_dim = out_channels

    def forward(
        self,
        input_: torch.Tensor,
        kwargs: dict[str, typing.Any],
        losses: dict[str, typing.Any] | None = None,
        metrics: dict | None = None,
    ) -> torch.Tensor:
        audio_tokens_dim = kwargs.get(AudioKwargs.audio_tokens)
        if isinstance(input_, TensorMeta):
            if audio_tokens_dim is not None:
                return TensorMeta.from_dims(
                    (audio_tokens_dim, self._out_channels_dim),
                    tensor_name="audio conv output",
                    dtype=input_.dtype,
                )
            # Fallback: pass through with hidden dim only (batch dim unknown at meta time)
            return TensorMeta.from_dims(
                (self._out_channels_dim,),
                tensor_name="audio conv output",
                dtype=input_.dtype,
            )

        # Read mel spectrogram from kwargs (set by AudioPreprocessor.preprocess()).
        # Shape: (N_clips, n_mel_bins, mel_frames)
        audio_mel = kwargs[AudioKwargs.audio_mel]
        audio_mel = audio_mel.to(self.conv1_weight.dtype)

        audio_mel = torch.nn.functional.conv1d(
            audio_mel, self.conv1_weight, self.conv1_bias, stride=self.conv1_stride, padding=1
        )
        audio_mel = torch.nn.functional.gelu(audio_mel)
        audio_mel = torch.nn.functional.conv1d(
            audio_mel, self.conv2_weight, self.conv2_bias, stride=self.conv2_stride, padding=1
        )
        audio_mel = torch.nn.functional.gelu(audio_mel)

        # (N_clips, hidden, T) → (N_clips, T, hidden)
        audio_embeddings = audio_mel.permute(0, 2, 1)
        N_clips, T, _ = audio_embeddings.shape
        assert T <= self.positional_embeddings.size(0), (
            f"Audio conv output length {T} exceeds max_source_positions "
            f"{self.positional_embeddings.size(0)}. Ensure aud_padding_duration * sr / hop / conv_stride "
            f"<= max_source_positions, or that over-long audio is filtered before reaching the encoder."
        )
        audio_embeddings = audio_embeddings + self.positional_embeddings[:T]
        audio_embeddings = torch.nn.functional.dropout(audio_embeddings, p=self.dropout_p, training=self.training)

        # Store shape info for AudioAdapter to use when reshaping back.
        kwargs[AudioKwargs.audio_num_clips] = N_clips
        kwargs[AudioKwargs.audio_conv_len] = T

        # Flatten to (N_clips * T, hidden) for sequence-first transformer block processing.
        # TODO: When both vision and audio are active, clips from different modalities share
        #   the same flat sequence — cross-clip attention is technically incorrect but acceptable
        #   as an initial implementation when only audio is used.
        return audio_embeddings.reshape(N_clips * T, -1).contiguous()
