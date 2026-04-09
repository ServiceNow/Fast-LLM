import math
import typing

import numpy as np
import torch
from transformers import WhisperFeatureExtractor

from fast_llm.engine.base_model.config import Preprocessor
from fast_llm.engine.config_utils.tensor_space import TensorSpace
from fast_llm.layers.audio_encoder.config import AudioEncoderConfig, AudioEncoderKwargs


def get_num_audio_tokens(
    sizes,
    aud_padding_duration,
    aud_sampling_rate,
    aud_downsampling_k,
    audio_start_token,
    audio_end_token,
    audio_padding: str = "max_length",
):
    if len(sizes) == 0:  # sample has no audio
        return np.array(sizes), False

    to_filter = False
    if aud_padding_duration > 0:
        raw_audio_seq_length = aud_padding_duration * aud_sampling_rate
        to_filter = bool(np.any(sizes > raw_audio_seq_length))  # filter sample where any audio is too long
        if audio_padding == "max_length":
            # All audios billed as max length for sequence packing (original behaviour)
            sizes = sizes.copy()  # original is read-only
            sizes.fill(raw_audio_seq_length)
        # "longest": use actual sizes; filter still applied above

    # account for mel spectrogram hop, conv2 stride, and adapter downsampling k
    audio_token_size_arr = sizes // 160 // (2 * aud_downsampling_k)

    if audio_start_token is not None:
        audio_token_size_arr += 1
    if audio_end_token is not None:
        audio_token_size_arr += 1
    return audio_token_size_arr, to_filter


def apply_audio_padding(audio, aud_padding_duration, aud_sampling_rate):
    if len(audio) == 0:
        return audio
    # TODO Toby: check 2d
    padded_audio = []
    if aud_padding_duration > 0:
        raw_audio_seq_length = aud_padding_duration * aud_sampling_rate
        for aud in audio:
            padded = np.pad(aud, (0, raw_audio_seq_length - len(aud)), mode="constant", constant_values=0)
            padded_audio.append(padded)
        return padded_audio
    else:
        return audio


class AudioPreprocessor(Preprocessor):
    def __init__(
        self,
        config: AudioEncoderConfig,
        tensor_space: TensorSpace,
        aud_padding_duration: int | None = None,
        audio_padding: str | None = None,
    ):
        self._config = config
        self._tensor_space = tensor_space
        self._distributed_config = self._tensor_space.distributed_config
        # aud_padding_duration and audio_padding default to AudioEncoderConfig values.
        # Explicit kwargs can override (e.g. to pass BatchConfig.aud_padding_duration at runtime).
        self._aud_padding_duration = aud_padding_duration if aud_padding_duration is not None else config.aud_padding_duration
        self._audio_padding = audio_padding if audio_padding is not None else config.audio_padding

        self.feature_extractor = WhisperFeatureExtractor(sampling_rate=self._config.aud_sampling_rate)

        # self.mel_transform = MelSpectrogram(
        #     sample_rate=self._config.aud_sampling_rate,
        #     n_fft=400,
        #     win_length=400,
        #     hop_length=160,
        #     n_mels=80,
        #     f_min=0.0,
        #     f_max=8000.0,
        #     mel_scale="slaney",
        #     norm="slaney",
        #     center=True,
        #     power=2.0,
        # )

    def preprocess_meta(self, kwargs: dict[str, typing.Any]) -> None:
        # kwargs[AudioEncoderKwargs.audio_mel_meta] = TensorMeta.from_dims(
        #     (
        #         TensorDim(
        #             VisionTransformerDimNames.batch,
        #             kwargs[TransformerKwargs.micro_batch_size] * kwargs[TransformerKwargs.sequence_q_dim].size,
        #         ),
        #         TensorDim(VisionEncoderDimNames.in_channels, 3),
        #         TensorDim(VisionEncoderDimNames.patch_size, kwargs[VisionEncoderKwargs.patch_size]),
        #         TensorDim(VisionEncoderDimNames.patch_size, kwargs[VisionEncoderKwargs.patch_size]),
        #     ),
        #     dtype=self._distributed_config.training_dtype.torch,
        # )
        pass

    def preprocess(self, tokens, kwargs: dict[str, typing.Any]) -> None:
        _HOP_LENGTH = 160   # Whisper default mel hop length
        _CONV_STRIDE = 2    # AudioConv conv2 stride
        k = self._config.aud_downsampling_k

        audio_mel = []
        audio_token_lens: list[int] = []

        if AudioEncoderKwargs.audio in kwargs:
            audio_raw = kwargs[AudioEncoderKwargs.audio]
            flattened_audio = [audio_arr for sequence in audio_raw for audio_arr in sequence]

            # Compute actual LLM token count from raw audio lengths (before padding)
            for aud in flattened_audio:
                actual_mel_frames = len(aud) // _HOP_LENGTH
                audio_token_lens.append(actual_mel_frames // _CONV_STRIDE // k)

            # Determine padding target for mel extraction
            if self._audio_padding == "longest":
                max_length = max(len(a) for a in flattened_audio)
            else:
                max_length = self._aud_padding_duration * self._config.aud_sampling_rate

            for audio in flattened_audio:
                audio_mel.append(
                    self.feature_extractor(
                        audio,
                        sampling_rate=self._config.aud_sampling_rate,
                        return_tensors="pt",
                        max_length=max_length,
                        truncation=True,
                        padding="max_length",
                    )["input_features"]
                )
            audio_mel = torch.stack(audio_mel, dim=0).squeeze(1)
            curr_size = audio_mel.size(0)
        else:
            audio_mel = torch.tensor(audio_mel, dtype=torch.float32)
            curr_size = 0

        # Always pad to at least 1 for constant tensor shape when there is no audio
        max_pad = max(1, curr_size)
        padding_size = max_pad - curr_size
        if padding_size > 0:
            padding = torch.zeros(
                padding_size,
                self.feature_extractor.feature_size,
                audio_mel.shape[-1] if curr_size > 0 else self.feature_extractor.nb_max_frames,
                dtype=audio_mel.dtype,
            )
            audio_mel = torch.cat((audio_mel, padding), dim=0)

        audio_mel = audio_mel.to(self._tensor_space.distributed.device)
        kwargs[AudioEncoderKwargs.audio_mel] = audio_mel

        if audio_token_lens:
            kwargs[AudioEncoderKwargs.audio_token_lens] = torch.tensor(
                audio_token_lens, dtype=torch.int32, device=self._tensor_space.distributed.device
            )
