import math
import typing

import numpy as np
import torch
from transformers import WhisperFeatureExtractor

from fast_llm.engine.base_model.config import Preprocessor
from fast_llm.engine.config_utils.tensor_space import TensorSpace
from fast_llm.layers.audio_encoder.config import AudioEncoderConfig, AudioEncoderKwargs


def get_num_audio_tokens(
    sizes, aud_padding_duration, aud_sampling_rate, aud_downsampling_k, audio_start_token, audio_end_token
):
    if len(sizes) == 0:  # sample has no audio
        return np.array(sizes), False
    to_filter = False
    # account for padding
    if aud_padding_duration > 0:
        raw_audio_seq_length = aud_padding_duration * aud_sampling_rate
        sizes = sizes.copy()  # original is read-only
        to_filter = bool(np.any(sizes > raw_audio_seq_length))  # filter sample where any audio is too long
        sizes.fill(raw_audio_seq_length)  # set all audio sizes to padded amount

    # account for mel spectogram, convolution, downsampling k
    audio_token_size_arr = sizes // 160  # default hop length TODO Toby: check divisible?
    audio_token_size_arr = audio_token_size_arr // (
        2 * aud_downsampling_k
    )  # convolution (2 stride) * downsampling  TODO Toby: make configurable convolution

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
    def __init__(self, config: AudioEncoderConfig, tensor_space: TensorSpace):
        self._config = config
        self._tensor_space = tensor_space
        self._distributed_config = self._tensor_space.distributed_config

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
        # check if audio is in batch
        audio_mel = []
        if AudioEncoderKwargs.audio in kwargs:
            # print("Preprocessing Contains Audio")
            audio_raw = kwargs[AudioEncoderKwargs.audio]
            flattened_audio = [
                audio_arr for sequence in audio_raw for audio_arr in sequence
            ]  # flatten in the batch dimension
            # print("Preprocessing Flattened Audio: ", flattened_audio)

            for audio in flattened_audio:
                audio_mel.append(
                    self.feature_extractor(
                        audio,
                        sampling_rate=self._config.aud_sampling_rate,
                        return_tensors="pt",
                        max_length=30 * self._config.aud_sampling_rate,
                        device=self._tensor_space.distributed.device,
                    )["input_features"]
                )
            audio_mel = torch.stack(audio_mel, dim=0).squeeze(1)
            curr_size = audio_mel.size(0)
        else:
            # print("Preprocessing No Audio")
            audio_mel = torch.tensor(audio_mel, dtype=torch.float32)
            curr_size = 0
        
        # print("Preprocessing Audio Mel Raw: ", audio_mel)

        # compute max pad
        max_pad = math.ceil(
            kwargs["sequence_length"] / (kwargs["audio_encoder_sequence_length"] // self._config.aud_downsampling_k)
        )
        max_pad = 1
        max_pad = max(max_pad, curr_size)

        # add padding
        padding_size = max_pad - curr_size
        if padding_size > 0:
            padding = torch.zeros(
                padding_size,
                self.feature_extractor.feature_size,
                self.feature_extractor.nb_max_frames,
                dtype=audio_mel.dtype,
                device=audio_mel.device,
            )
            audio_mel = torch.cat((audio_mel, padding), dim=0)

        # print("Preprocessing Audio Mel Final: ", audio_mel)

        # move to device
        audio_mel = audio_mel.to(self._tensor_space.distributed.device)
        kwargs[AudioEncoderKwargs.audio_mel] = audio_mel

        # # set attention mask # TODO Toby: fix backup attention
        # sequence_k = kwargs[self._transformer_kwargs.sequence_k_dim].size
        # sequence_q = kwargs[self._transformer_kwargs.sequence_q_dim].size
        # kwargs[self._transformer_kwargs.attention_mask] = self._mask[
        #     None, None, sequence_k - sequence_q : sequence_k, None, :sequence_k
        # ]
        # kwargs[self._transformer_kwargs.attention_mask_value] = self._mask_value
        # audio_mel = torch.rand(len(flattened_audio), 80, 3000)
