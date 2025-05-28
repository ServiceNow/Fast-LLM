import typing

import torch
from torchaudio.transforms import MelSpectrogram

from fast_llm.engine.base_model.config import Preprocessor
from fast_llm.engine.config_utils.tensor_space import TensorSpace
from fast_llm.layers.audio_encoder.config import AudioEncoderConfig, AudioEncoderKwargs

# from transformers import WhisperFeatureExtractor


class AudioPreprocessor(Preprocessor):
    def __init__(self, config: AudioEncoderConfig, tensor_space: TensorSpace):
        self._config = config
        self._tensor_space = tensor_space
        self._distributed_config = self._tensor_space.distributed_config

        # self.feature_extractor = WhisperFeatureExtractor(sampling_rate=self._config.aud_sampling_rate)

        self.mel_transform = MelSpectrogram(
            sample_rate=self._config.aud_sampling_rate,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=80,
            f_min=0.0,
            f_max=8000.0,
            mel_scale="slaney",
            norm="slaney",
            center=True,
            power=2.0,
        )

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
        audio_raw = kwargs[AudioEncoderKwargs.audio]
        flattened_audio = [audio_arr for sequence in audio_raw for audio_arr in sequence]
        flattened_audio_tensor = torch.stack(flattened_audio, dim=0)
        # audio_inputs = self.feature_extractor(audio_raw, sampling_rate=16000, return_tensors="pt")
        self.mel_transform.to(self._tensor_space.distributed.device)

        audio_mel = self.mel_transform(flattened_audio_tensor)
        audio_mel = audio_mel[:, :, :-1]  # TODO Toby: check this!

        # # set attention mask # TODO Toby: fix backup attention
        # sequence_k = kwargs[self._transformer_kwargs.sequence_k_dim].size
        # sequence_q = kwargs[self._transformer_kwargs.sequence_q_dim].size
        # kwargs[self._transformer_kwargs.attention_mask] = self._mask[
        #     None, None, sequence_k - sequence_q : sequence_k, None, :sequence_k
        # ]
        # kwargs[self._transformer_kwargs.attention_mask_value] = self._mask_value

        kwargs[AudioEncoderKwargs.audio_mel] = audio_mel
