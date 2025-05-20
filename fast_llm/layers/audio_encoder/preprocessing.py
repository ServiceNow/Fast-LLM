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
        pass

    def preprocess(self, tokens, kwargs: dict[str, typing.Any]) -> None:
        audio_raw = kwargs[AudioEncoderKwargs.audio]
        # audio_inputs = self.feature_extractor(audio_raw, sampling_rate=16000, return_tensors="pt")
        self.mel_transform.to(self._tensor_space.distributed.device)

        audio_mel = []
        for batch in audio_raw:
            batch_stacked = torch.stack(batch).unsqueeze(1)
            audio_mel.append(self.mel_transform(batch_stacked))
        kwargs[AudioEncoderKwargs.audio_mel] = torch.cat(audio_mel)
