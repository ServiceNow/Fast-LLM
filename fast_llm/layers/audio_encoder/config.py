import enum
import typing

from fast_llm.config import Field, FieldHint, check_field, config_class, skip_valid_if_none
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.functional.config import ActivationType
from fast_llm.layers.block.config import BlockConfig, BlockKwargs, BlockSequenceConfig
from fast_llm.layers.common.normalization.config import NormalizationConfig
from fast_llm.layers.language_model.config import LanguageModelConfig
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    from fast_llm.layers.audio_encoder.audio_encoder import AudioEncoder
    from fast_llm.layers.audio_encoder.audio_model import AudioMultiModalModel


class AudioKwargs(BlockKwargs):
    # Audio data passed from the data pipeline
    audio = "audio"
    audio_mel = "audio_mel"
    audio_positions = "audio_positions"
    audio_token_lens = "audio_token_lens"  # int32 tensor [N_audio] — actual LLM token count per clip
    # Shape dim for TensorMeta inference in the audio encoder
    audio_tokens = "audio_tokens"
    # Runtime shape info set by AudioConv.forward() for AudioAdapter
    audio_num_clips = "audio_num_clips"  # int: total number of clips (including padding)
    audio_conv_len = "audio_conv_len"    # int: frames per clip after conv downsampling


class AudioEncoderDimNames:
    """String name constants for TensorDim objects used inside the audio encoder."""

    in_channels = "audio_in_channels"
    out_channels = "audio_out_channels"
    kernel_size = "audio_kernel_size"
    adapter_input = "audio_adapter_input"
    adapter_size = "audio_adapter_size"
    max_source_positions = "audio_max_source_positions"


# Keep for backwards-compat in data pipeline helpers that reference these constants.
class AudioEncoderKwargs(AudioKwargs):
    pass


class AudioEncoderType(str, enum.Enum):
    none = "none"
    whisper = "whisper"


@config_class()
class AudioEncoderConfig(BlockConfig):
    _abstract = False

    encoder_type: AudioEncoderType = Field(
        default=AudioEncoderType.none,
        desc="Type of the audio encoder. Choices: none, whisper.",
        hint=FieldHint.architecture,
    )

    # --- Transformer encoder (replaces flat AudioTransformerConfig) ---
    encoder: BlockSequenceConfig = Field(
        desc="Configuration for the audio transformer blocks (bidirectional, causal=False).",
        hint=FieldHint.architecture,
    )
    hidden_size: int = Field(
        default=1024,
        desc="Hidden dimension of the audio encoder (input and output of the transformer).",
        hint=FieldHint.architecture,
        valid=check_field(Assert.gt, 0),
    )
    normalization: NormalizationConfig = Field(
        desc="Configuration for the normalization layers inside the audio adapter.",
        hint=FieldHint.architecture,
    )

    # --- Conv encoder ---
    conv_bias: bool = Field(
        default=True,
        desc="Whether to use bias in the convolutional layer.",
        hint=FieldHint.optional,
    )
    encoder_dropout: float = Field(
        default=0.0,
        desc="Dropout for encoder.",
        hint=FieldHint.core,
    )
    kernel_size: int = Field(
        default=3,
        desc="Encoder convolution layer kernel size.",
        hint=FieldHint.core,
    )
    conv_lr_scale: float | None = Field(
        default=None,
        desc="Custom learning rate scale for the convolutional layer weights.",
        hint=FieldHint.feature,
        valid=skip_valid_if_none(check_field(Assert.geq, 0)),
    )
    pos_emb_lr_scale: float | None = Field(
        default=None,
        desc="Custom learning rate scale for the position embedding layer weights.",
        hint=FieldHint.feature,
        valid=skip_valid_if_none(check_field(Assert.geq, 0)),
    )

    # --- Adapter ---
    adapter_size: int = Field(
        default=5120,
        desc="Intermediate size for the adapter linear layers (2-layer MLP).",
        hint=FieldHint.core,
    )
    adapter_activation_type: ActivationType = Field(
        default=ActivationType.gelu,
        desc="The intermediate activation type for the audio adapter. Default: GeLU.",
        hint=FieldHint.core,
    )
    adapter_bias: bool = Field(
        default=True,
        desc="Whether to use bias in the adapter layer.",
        hint=FieldHint.optional,
    )
    adapter_lr_scale: float | None = Field(
        default=None,
        desc="Custom learning rate scale for the adapter weights.",
        hint=FieldHint.feature,
        valid=skip_valid_if_none(check_field(Assert.geq, 0)),
    )
    aud_downsampling_k: int = Field(
        default=5,
        desc="Audio downsampling k parameter (groups k consecutive frames in the adapter).",
        hint=FieldHint.feature,
    )

    # --- Mel-spectrogram / audio settings ---
    num_mel_bins: int = Field(
        default=80,
        desc="Number of bins for mel spectrogram.",
        hint=FieldHint.core,
    )
    aud_sampling_rate: int = Field(
        default=16000,
        desc="Audio sampling rate to use.",
        hint=FieldHint.feature,
    )
    aud_padding_duration: int = Field(
        default=30,
        desc="Audio padding duration in seconds, used by the preprocessor to determine mel spectrogram length.",
        hint=FieldHint.feature,
    )
    audio_padding: str = Field(
        default="max_length",
        desc='"max_length" pads all audio to aud_padding_duration. "longest" pads to the longest in the batch.',
        hint=FieldHint.feature,
    )
    max_source_positions: int = Field(
        default=1500,
        desc=(
            "Max audio positions after conv downsampling "
            "(= aud_padding_duration * aud_sampling_rate / hop / conv_stride). "
            "Default 1500 = 30s × 16 kHz / 160 hop / 2 stride."
        ),
        hint=FieldHint.architecture,
    )

    # --- Special tokens ---
    audio_start_token: int | None = Field(
        default=None,
        desc="Token id for audio start.",
        hint=FieldHint.optional,
    )
    audio_end_token: int | None = Field(
        default=None,
        desc="Token id for audio end.",
        hint=FieldHint.optional,
    )

    @property
    def layer_class(self) -> "type[AudioEncoder]":
        from fast_llm.layers.audio_encoder.audio_encoder import AudioEncoder

        return AudioEncoder

    @property
    def enabled(self) -> bool:
        return self.encoder_type != AudioEncoderType.none

    def make_audio_hidden_dim(self) -> TensorDim:
        """Return a TensorDim for the audio encoder hidden dimension."""
        return TensorDim("audio_hidden", self.hidden_size)


@config_class()
class AudioMultiModalModelConfig(LanguageModelConfig):
    """
    Language model configuration extended with an audio encoder.

    Parallel to ``VisionMultiModalModelConfig`` in ``layers/vision/config.py``.
    When combined with ``GPTBaseModelConfig`` (via ``MultiModalBaseModelConfig``),
    this allows training a language model with audio input.
    """

    audio_encoder: AudioEncoderConfig = Field(
        hint=FieldHint.architecture,
        desc="Configuration for the audio encoder.",
    )

    @property
    def layer_class(self) -> "type[AudioMultiModalModel]":
        from fast_llm.layers.audio_encoder.audio_model import AudioMultiModalModel

        return AudioMultiModalModel
