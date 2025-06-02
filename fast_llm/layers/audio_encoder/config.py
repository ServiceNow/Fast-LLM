import enum

from fast_llm.config import Field, FieldHint, check_field, config_class, skip_valid_if_none
from fast_llm.engine.base_model.config import BaseModelConfig
from fast_llm.engine.config_utils.tensor_space import TensorDim, TensorSpace
from fast_llm.functional.config import ActivationType
from fast_llm.layers.transformer.config import AudioTransformerConfig
from fast_llm.utils import Assert


class AudioEncoderDimNames:
    in_channels = "audio_in_channels"
    out_channels = "audio_out_channels"
    kernel_size = "audio_kernel_size"
    adapter_input = "audio_adapter_input"
    adapter_size = "audio_adapter_size"
    audio_channels = "audio_kv_channels"
    max_source_positions = "audio_max_source_positions"


class AudioEncoderKwargs:
    audio = "audio"
    audio_mel = "audio_mel"
    audio_positions = "audio_positions"

    kv_channels = "audio_kv_channels"  # TODO: check this
    out_channels = "audio_out_channels"
    hidden_dims = "audio_hidden_dims"

    # TODO: used for backup attention
    sequence_length = "audio_sequence_length"
    sequence_k_dim = "audio_sequence_k_dim"
    sequence_q_dim = "audio_sequence_q_dim"


class AudioEncoderType(str, enum.Enum):
    none = "none"
    whisper = "whisper"


@config_class()
class AudioEncoderConfig(BaseModelConfig):
    _abstract = False

    type: AudioEncoderType = Field(
        default=AudioEncoderType.none,
        desc="Type of the audio encoder. Choices: none, whisper.",
        hint=FieldHint.architecture,
    )
    transformer: AudioTransformerConfig = Field(
        default_factory=AudioTransformerConfig,
        desc="Configuration for the audio transformer architecture.",
        hint=FieldHint.core,
    )

    # encoder configs
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

    # adapter configs
    adapter_size: int = Field(
        default=5120,
        desc="Intermediate size for the adapter linear layers. Assuming 2 linear layers",
        hint=FieldHint.core,
    )
    adapter_activation_type: ActivationType = Field(
        default=ActivationType.gelu,
        desc="The intermediate activation type for multi-modal adapter. Default: GeLU.",
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

    # audio configs
    num_mel_bins: int = Field(
        default=80,
        desc="Number of bins for mel spectogram.",
        hint=FieldHint.core,
    )
    aud_downsampling_k: int = Field(
        default=5,
        desc="Audio downsampling k parameter.",
        hint=FieldHint.feature,
    )
    aud_sampling_rate: int = Field(
        default=16000,
        desc="Audio sampling rate to use.",
        hint=FieldHint.feature,
    )

    # audio start/end tokens
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

    def setup_tensor_space(self, tensor_space: TensorSpace):
        tensor_space.add_tensor_dim(TensorDim(AudioEncoderDimNames.in_channels, self.num_mel_bins))
        tensor_space.add_tensor_dim(TensorDim(AudioEncoderDimNames.out_channels, self.transformer.hidden_size))
        tensor_space.add_tensor_dim(TensorDim(AudioEncoderDimNames.kernel_size, self.kernel_size))
        tensor_space.add_tensor_dim(
            TensorDim(AudioEncoderDimNames.adapter_input, self.transformer.hidden_size * self.aud_downsampling_k)
        )
        tensor_space.add_tensor_dim(TensorDim(AudioEncoderDimNames.adapter_size, self.adapter_size))
        tensor_space.add_tensor_dim(
            TensorDim(AudioEncoderDimNames.max_source_positions, 1500)
        )  # TODO: configure later

        tensor_space.add_tensor_dim(
            TensorDim(
                AudioEncoderDimNames.audio_channels,
                self.transformer.hidden_size // self.transformer.num_attention_heads,
            )
        )
        self.transformer.setup_tensor_space(tensor_space)

    @property
    def enabled(self) -> bool:
        return self.type != AudioEncoderType.none
