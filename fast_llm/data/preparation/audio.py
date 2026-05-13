import typing

from fast_llm.config import Config, Field, FieldHint, check_field, config_class
from fast_llm.utils import Assert

# Fixed constants from the Whisper mel-spectrogram front-end and the AudioConv dual-conv stack.
# These must match the values hardcoded in fast_llm/layers/audio_encoder/preprocessing.py and encoder.py.
AUDIO_HOP_LENGTH: int = 160   # mel-spec hop length (samples per mel frame)
AUDIO_CONV_STRIDE: int = 2    # AudioConv conv2 stride (downsampling factor)


@config_class()
class AudioPreparationConfig(Config):
    """
    Configuration for audio placeholder token insertion during dataset preparation.

    The preparator inserts ``num_audio_encoder_tokens`` placeholder tokens (token ID
    ``-100``) immediately after each audio position in the token sequence.  The audio
    encoder then overwrites these placeholders with its output embeddings during training.

    Any "audio start" / "audio end" wrap tokens are the chat template's
    responsibility: emit them as literal vocabulary tokens around the audio
    marker in the rendered text and they will tokenize like any other token,
    landing immediately adjacent to the ``-100`` placeholder block.

    All fields must match the corresponding fields in ``AudioEncoderConfig``.
    """

    aud_downsampling_k: int = Field(
        default=2,
        desc="Audio adapter downsampling factor. Must match audio_encoder.aud_downsampling_k.",
        hint=FieldHint.core,
        valid=check_field(Assert.geq, 1),
    )

    def num_audio_encoder_tokens(self, num_samples: int) -> int:
        """
        Return the number of ``-100`` placeholder tokens inserted into the LM
        token stream for a single clip.

        Args:
            num_samples: Length of the raw waveform in samples.
        """
        return num_samples // AUDIO_HOP_LENGTH // AUDIO_CONV_STRIDE // self.aud_downsampling_k

    @classmethod
    def get_patches_from_audio(
        cls,
        audio_clips: "list[typing.Any]",
        config: "AudioPreparationConfig",
        data_type: "typing.Any",
    ) -> "list[int]":
        """Return the number of placeholder tokens for each clip (list parallel to audio_clips)."""
        import numpy as np

        return [
            config.num_audio_encoder_tokens(len(np.asarray(clip["array"], dtype=np.float32)))
            for clip in audio_clips
        ]
