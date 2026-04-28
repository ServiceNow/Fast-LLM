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

    The preparator inserts ``num_audio_tokens`` placeholder tokens (token ID ``-100``)
    immediately after each audio position in the token sequence.  The audio encoder
    then overwrites these placeholders with its output embeddings during training.

    All fields must match the corresponding fields in ``AudioEncoderConfig``.
    """

    aud_downsampling_k: int = Field(
        default=2,
        desc="Audio adapter downsampling factor. Must match audio_encoder.aud_downsampling_k.",
        hint=FieldHint.core,
        valid=check_field(Assert.geq, 1),
    )
    audio_start_token: int | None = Field(
        default=None,
        desc="Token ID prepended to each audio clip output. Must match audio_encoder.audio_start_token.",
        hint=FieldHint.optional,
    )
    audio_end_token: int | None = Field(
        default=None,
        desc="Token ID appended to each audio clip output. Must match audio_encoder.audio_end_token.",
        hint=FieldHint.optional,
    )

    def num_audio_encoder_tokens(self, num_samples: int) -> int:
        """
        Return the number of audio encoder output slots for a single clip.

        This is the count of ``-100`` placeholder tokens inserted into the token sequence.
        It does *not* include ``audio_start_token`` or ``audio_end_token``, which are
        inserted as real vocabulary token IDs alongside the placeholders.

        Args:
            num_samples: Length of the raw waveform in samples.
        """
        return num_samples // AUDIO_HOP_LENGTH // AUDIO_CONV_STRIDE // self.aud_downsampling_k

    def num_audio_tokens(self, num_samples: int) -> int:
        """
        Return the total number of LM token slots produced by a single audio clip,
        including ``audio_start_token`` and ``audio_end_token`` if configured.

        Use this for sequence-length accounting.  Use ``num_audio_encoder_tokens``
        when you need only the placeholder (``-100``) count.

        Args:
            num_samples: Length of the raw waveform in samples.
        """
        n = self.num_audio_encoder_tokens(num_samples)
        if self.audio_start_token is not None:
            n += 1
        if self.audio_end_token is not None:
            n += 1
        return n

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
            config.num_audio_tokens(len(np.asarray(clip["array"], dtype=np.float32)))
            for clip in audio_clips
        ]
