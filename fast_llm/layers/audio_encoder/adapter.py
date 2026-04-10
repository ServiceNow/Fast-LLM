import typing

import torch

from fast_llm.engine.base_model.base_model import Layer
from fast_llm.engine.config_utils.initialization import init_normal_, init_zeros_
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.functional.triton.mlp import torch_mlp_activation
from fast_llm.layers.audio_encoder.config import AudioEncoderConfig, AudioEncoderDimNames, AudioKwargs
from fast_llm.layers.block.config import BlockKwargs
from fast_llm.layers.common.linear.linear import Linear
from fast_llm.tensor import ParameterMeta, TensorMeta


class AudioAdapter(Layer):
    """
    Audio adapter: groups k consecutive encoder frames, then projects to LM hidden size.

    Downsampling: (batch, T, hidden) → (batch, T/k, hidden * k) → (batch, T/k, lm_hidden)
    """

    def __init__(
        self,
        config: AudioEncoderConfig,
        audio_hidden_dim: TensorDim,
        output_dim: TensorDim,
        distributed_config: DistributedConfig | None = None,
    ):
        super().__init__(distributed_config or DistributedConfig())
        self._activation_type = config.adapter_activation_type
        self._use_adapter_bias = config.adapter_bias
        self._lr_scale = config.adapter_lr_scale
        self.aud_downsampling_k = config.aud_downsampling_k

        adapter_input_dim = TensorDim(
            AudioEncoderDimNames.adapter_input,
            audio_hidden_dim.size * config.aud_downsampling_k,
        )
        adapter_size_dim = TensorDim(AudioEncoderDimNames.adapter_size, config.adapter_size)

        self.norm_1 = config.normalization.get_layer(audio_hidden_dim, peft=None)
        self.norm_1.lr_scale = self._lr_scale
        self.norm_2 = config.normalization.get_layer(adapter_size_dim, peft=None)
        self.norm_2.lr_scale = self._lr_scale

        weight_1 = ParameterMeta.from_dims(
            (adapter_size_dim, adapter_input_dim),
            init_method=init_normal_(),
            lr_scale=self._lr_scale,
        )
        bias_1 = ParameterMeta.from_dims(
            (adapter_size_dim,), init_method=init_zeros_, lr_scale=self._lr_scale
        ) if self._use_adapter_bias else None
        self.layer_1 = Linear(weight_1, bias_1)

        weight_2 = ParameterMeta.from_dims(
            (output_dim, adapter_size_dim),
            init_method=init_normal_(),
            lr_scale=self._lr_scale,
        )
        bias_2 = ParameterMeta.from_dims(
            (output_dim,), init_method=init_zeros_, lr_scale=self._lr_scale
        ) if self._use_adapter_bias else None
        self.layer_2 = Linear(weight_2, bias_2)

    def forward(
        self,
        input_: torch.Tensor,
        kwargs: dict[str, typing.Any],
        losses: dict[str, typing.Any] | None = None,
        metrics: dict[str, typing.Any] | None = None,
    ) -> torch.Tensor:
        if isinstance(input_, TensorMeta):
            hidden_token_dim = kwargs.get(BlockKwargs.hidden_token_dim)
            if hidden_token_dim is not None:
                return TensorMeta.from_dims(
                    (hidden_token_dim,),
                    tensor_name="Audio adapter output",
                    dtype=input_.dtype,
                )
            return TensorMeta.from_dims((), tensor_name="Audio adapter output", dtype=input_.dtype)

        # input_ arrives as (N_clips * T, hidden) — flatten from AudioConv + transformer blocks.
        # Reshape to (N_clips, T, hidden) for per-clip grouping.
        N_clips = kwargs[AudioKwargs.audio_num_clips]
        N_total, hidden = input_.shape
        T = N_total // N_clips
        input_ = input_.view(N_clips, T, hidden)

        input_ = self.norm_1(input_)
        batch_size, seq_len, dim = input_.size()

        # Trim to multiple of downsampling_k if needed
        if seq_len % self.aud_downsampling_k != 0:
            trimmed_seq_len = seq_len - (seq_len % self.aud_downsampling_k)
            input_ = input_[:, :trimmed_seq_len, :]
            seq_len = trimmed_seq_len

        # Group k frames: (N_clips, T, dim) → (N_clips, T/k, dim*k)
        new_seq_len = seq_len // self.aud_downsampling_k
        input_ = input_.contiguous().view(batch_size, new_seq_len, dim * self.aud_downsampling_k)

        layer1_res = torch_mlp_activation(
            input_=self.layer_1(input_), gated=False, activation_type=self._activation_type
        )
        layer1_res_dropout = torch.nn.functional.dropout(layer1_res, 0.1)
        layer1_res_norm = self.norm_2(layer1_res_dropout)
        output = self.layer_2(layer1_res_norm)  # (N_clips, T/k, lm_hidden)

        # Flatten to (N_clips * T/k, lm_hidden) for sequence-first LM injection.
        return output.reshape(batch_size * new_seq_len, -1).contiguous()
