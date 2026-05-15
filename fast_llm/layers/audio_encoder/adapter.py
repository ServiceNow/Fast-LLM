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
from fast_llm.layers.common.normalization.config import NoNormalizationConfig
from fast_llm.tensor import ParameterMeta, TensorMeta


class AudioAdapter(Layer):
    """
    Audio adapter / projector: groups k consecutive encoder frames, then projects to LM hidden size.

    Pipeline:
      (N_clips, T, audio_hidden)
        → stack k frames → (N_clips, T/k, audio_hidden * k)
        → norm_1        (projector pre-norm; receives Ultravox ``ln_pre`` weights,
                         identity-init for Whisper-only checkpoints)
        → layer_1       (audio_hidden*k → adapter_size [* 2 if gated])
        → activation    (gated chunks → adapter_size)
        → dropout       (adapter_dropout; 0 for Ultravox, 0.1 for legacy Whisper/Ayra)
        → norm_2        (projector mid-norm; receives Ultravox ``ln_mid`` /
                         Ayra ``encoder_projector.layer_norm``)
        → layer_2       (adapter_size → lm_hidden)
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
        self._gated = config.adapter_gated
        self._use_adapter_bias = config.adapter_bias
        self._dropout_p = config.adapter_dropout
        self._lr_scale = config.adapter_lr_scale
        self.aud_downsampling_k = config.aud_downsampling_k

        adapter_input_dim = TensorDim(
            AudioEncoderDimNames.adapter_input,
            audio_hidden_dim.size * config.aud_downsampling_k,
        )
        adapter_size_dim = TensorDim(AudioEncoderDimNames.adapter_size, config.adapter_size)
        # When gated, layer_1 emits 2× the adapter size; the activation halves it back.
        layer_1_output_dim = TensorDim(
            AudioEncoderDimNames.adapter_size + ("_gated" if self._gated else ""),
            config.adapter_size * (2 if self._gated else 1),
        )

        # norm_1: projector pre-norm, dim = audio_hidden * k (post-stack).
        # None = NoNormalization (Whisper / Ayra have no projector pre-norm; their
        # encoder.layer_norm now lives in AudioEncoder.final_norm).
        pre_norm_config = config.adapter_pre_normalization or NoNormalizationConfig()
        self.norm_1 = pre_norm_config.get_layer(adapter_input_dim, peft=None)
        self.norm_1.lr_scale = self._lr_scale
        # norm_2: projector mid-norm, dim = adapter_size. Inherits from
        # ``audio_encoder.normalization`` when ``adapter_mid_normalization`` is unset.
        mid_norm_config = config.adapter_mid_normalization or config.normalization
        self.norm_2 = mid_norm_config.get_layer(adapter_size_dim, peft=None)
        self.norm_2.lr_scale = self._lr_scale

        weight_1 = ParameterMeta.from_dims(
            (layer_1_output_dim, adapter_input_dim),
            init_method=init_normal_(),
            lr_scale=self._lr_scale,
        )
        bias_1 = ParameterMeta.from_dims(
            (layer_1_output_dim,), init_method=init_zeros_, lr_scale=self._lr_scale
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

        batch_size, seq_len, dim = input_.size()

        # Trim to multiple of downsampling_k if needed
        if seq_len % self.aud_downsampling_k != 0:
            trimmed_seq_len = seq_len - (seq_len % self.aud_downsampling_k)
            input_ = input_[:, :trimmed_seq_len, :]
            seq_len = trimmed_seq_len

        # Group k frames: (N_clips, T, dim) → (N_clips, T/k, dim*k)
        new_seq_len = seq_len // self.aud_downsampling_k
        input_ = input_.contiguous().view(batch_size, new_seq_len, dim * self.aud_downsampling_k)

        # Projector pre-norm operates over the stacked dim (matches HF Ultravox ln_pre).
        input_ = self.norm_1(input_)

        layer1_res = torch_mlp_activation(
            input_=self.layer_1(input_), gated=self._gated, activation_type=self._activation_type
        )
        if self._dropout_p > 0:
            layer1_res = torch.nn.functional.dropout(layer1_res, self._dropout_p, training=self.training)
        layer1_res_norm = self.norm_2(layer1_res)
        output = self.layer_2(layer1_res_norm)  # (N_clips, T/k, lm_hidden)

        # Trim each clip to its actual token count before flattening.
        # When audio clips are shorter than the padded length (always true for "longest"
        # padding with variable-length clips, and for "max_length" when clips are shorter
        # than aud_padding_duration), the flat output contains extra padded tokens per clip.
        # embedding.py injects via input_[:sum(actual_n_i)], which would take tokens from
        # the wrong clips if we don't trim first.
        audio_token_lens = kwargs.get(AudioKwargs.audio_token_lens)
        if audio_token_lens is not None and audio_token_lens.sum().item() < batch_size * new_seq_len:
            return torch.cat(
                [output[i, :n] for i, n in enumerate(audio_token_lens.tolist())]
            ).contiguous()

        # All clips fill their full padded slot — no trimming needed.
        return output.reshape(batch_size * new_seq_len, -1).contiguous()
