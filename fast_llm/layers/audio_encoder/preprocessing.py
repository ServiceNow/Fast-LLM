import logging
import typing

import numpy as np
import torch
from transformers import WhisperFeatureExtractor

logger = logging.getLogger(__name__)

from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.layers.attention.config import AttentionKwargs, MixerKwargs
from fast_llm.layers.audio_encoder.config import AudioEncoderConfig, AudioEncoderKwargs
from fast_llm.layers.block.config import BlockKwargs


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
        to_filter = bool(np.any(sizes > raw_audio_seq_length))
        if audio_padding == "max_length":
            # All audios billed as max length for sequence packing (original behaviour)
            sizes = sizes.copy()  # original is read-only
            sizes.fill(raw_audio_seq_length)
        # "longest": use actual sizes; filter still applied above

    # Account for mel spectrogram hop, conv2 stride, and adapter downsampling k
    audio_token_size_arr = sizes // 160 // (2 * aud_downsampling_k)

    if audio_start_token is not None:
        audio_token_size_arr += 1
    if audio_end_token is not None:
        audio_token_size_arr += 1
    return audio_token_size_arr, to_filter


def apply_audio_padding(audio, aud_padding_duration, aud_sampling_rate):
    if len(audio) == 0:
        return audio
    padded_audio = []
    if aud_padding_duration > 0:
        raw_audio_seq_length = aud_padding_duration * aud_sampling_rate
        for aud in audio:
            padded = np.pad(aud, (0, raw_audio_seq_length - len(aud)), mode="constant", constant_values=0)
            padded_audio.append(padded)
        return padded_audio
    else:
        return audio


class AudioPreprocessor:
    """
    Converts raw audio waveforms to mel spectrograms and populates kwargs for the audio encoder.

    Accepts an optional ``device`` argument (torch.device or str) to control where output
    tensors are placed. When None, tensors are kept on CPU — the training framework will
    move them as needed.
    """

    def __init__(
        self,
        config: AudioEncoderConfig,
        device=None,
        aud_padding_duration: int | None = None,
        audio_padding: str | None = None,
    ):
        self._config = config
        self._device = device  # torch.device | str | None
        # Allow runtime overrides (e.g. from BatchConfig)
        self._aud_padding_duration = aud_padding_duration if aud_padding_duration is not None else config.aud_padding_duration
        self._audio_padding = audio_padding if audio_padding is not None else config.audio_padding

        self.feature_extractor = WhisperFeatureExtractor(sampling_rate=self._config.aud_sampling_rate)

    def preprocess_meta(self, kwargs: dict[str, typing.Any]) -> None:
        pass

    def preprocess(self, tokens, kwargs: dict[str, typing.Any]) -> None:
        _HOP_LENGTH = 160   # Whisper default mel hop length
        _CONV_STRIDE = 2    # AudioConv conv2 stride
        k = self._config.aud_downsampling_k

        audio_mel = []
        audio_token_lens: list[int] = []

        if AudioEncoderKwargs.audio in kwargs:
            audio_raw = kwargs[AudioEncoderKwargs.audio]
            logger.debug(
                "AudioPreprocessor: audio_raw type=%s len=%d, first element type=%s shape=%s",
                type(audio_raw),
                len(audio_raw),
                type(audio_raw[0]) if len(audio_raw) > 0 else "N/A",
                getattr(audio_raw[0], "shape", "N/A") if len(audio_raw) > 0 else "N/A",
            )
            # audio_raw is a flat list[torch.Tensor] — one 1-D waveform per clip.
            # (AudioBatch.to_kwargs returns samples already flattened across documents.)
            flattened_audio = list(audio_raw)

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

        if self._device is not None:
            audio_mel = audio_mel.to(self._device)
        kwargs[AudioEncoderKwargs.audio_mel] = audio_mel

        if audio_token_lens:
            device_kwargs = {"device": self._device} if self._device is not None else {}
            kwargs[AudioEncoderKwargs.audio_token_lens] = torch.tensor(
                audio_token_lens, dtype=torch.int32, **device_kwargs
            )

        # Set up attention kwargs for the audio transformer blocks.
        # AudioConv applies two conv1d layers (stride-1 then stride-2), so the
        # output length per clip is: T = (mel_frames - 1) // 2 + 1
        # (kernel=3, padding=1, stride=2 for conv2; stride=1 for conv1).
        mel_frames = audio_mel.shape[2]
        N_clips = audio_mel.shape[0]
        T = (mel_frames - 1) // 2 + 1  # frames per clip after conv downsampling
        N_total = N_clips * T
        device = audio_mel.device

        # Override token-dimension kwargs to reflect audio sequence sizes.
        # These were copied from the LM kwargs but must be replaced with audio dims
        # so that the attention layers inside the audio transformer see the correct shapes.
        audio_token_dim = TensorDim("audio_token", N_total)
        kwargs[BlockKwargs.token_dim] = audio_token_dim
        kwargs[BlockKwargs.hidden_token_dim] = audio_token_dim
        kwargs[BlockKwargs.sequence_k_dim] = audio_token_dim
        kwargs[BlockKwargs.key_value_token_dim] = audio_token_dim
        kwargs[BlockKwargs.num_tokens] = N_total
        kwargs[BlockKwargs.sequence_length] = N_total

        # Document indices: each clip is a separate document, preventing cross-clip attention.
        doc_indices = torch.arange(N_clips, dtype=torch.int32, device=device).repeat_interleave(T)
        kwargs[MixerKwargs.document_index_q] = doc_indices
        kwargs[MixerKwargs.document_index_k] = doc_indices

        # Cumulative sequence lengths for flash attention (one document per clip).
        cu_seqlens = torch.arange(0, N_total + T, T, dtype=torch.int32, device=device)
        kwargs[MixerKwargs.cu_seqlens_q] = cu_seqlens
        kwargs[MixerKwargs.cu_seqlens_k] = cu_seqlens
        kwargs[MixerKwargs.max_seqlen_q] = T
        kwargs[MixerKwargs.max_seqlen_k] = T
