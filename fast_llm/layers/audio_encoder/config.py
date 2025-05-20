import enum

from fast_llm.config import Field, FieldHint, config_class
from fast_llm.engine.base_model.config import BaseModelConfig
from fast_llm.engine.config_utils.tensor_space import TensorDim, TensorSpace
from fast_llm.functional.config import ActivationType
from fast_llm.layers.transformer.config import AudioTransformerConfig


class AudioEncoderDimNames:
    in_channels = "audio_in_channels"
    out_channels = "audio_out_channels"
    kernel_size = "audio_kernel_size"
    adapter_size = "audio_adapter_size"
    audio_channels = "audio_kv_channels"


class AudioTransformerDimNames:
    # A set of common tensor dim names packed into a namespace.
    # Input dimensions (variable)
    # TODO: Does batch belong here?
    batch = "audio_batch"
    # TODO: Distinguish micro-sequence?
    sequence_q = "audio_sequence_q"
    sequence_q_tp = "audio_sequence_q_tp"
    sequence_k = "audio_sequence_k"
    hidden = "audio_hidden"
    # Self-attention dimensions
    head_groups = "audio_head_groups"
    group_heads = "audio_group_heads"
    key_and_value = "audio_key_value"
    kv_channels = "audio_kv_channels"
    composite_heads = "audio_composite_heads"
    composite_query = "audio_composite_query"
    composite_key_value = "audio_composite_key_value"
    composite_dense = "audio_composite_dense"
    # MLP dimensions
    mlp = "audio_mlp"
    gate_and_up = "audio_gate_and_up"
    composite_gated_mlp = "audio_composite_gated_mlp"
    experts = "audio_experts"
    top_experts = "audio_top_experts"
    shared_experts = "audio_shared_experts"
    unshared_experts = "audio_unshared_experts"
    composite_expert_mlp = "audio_composite_expert_mlp"
    composite_gated_expert_mlp = "audio_composite_gated_expert_mlp"
    composite_shared_expert_mlp = "audio_composite_shared_expert_mlp"
    composite_gated_shared_expert_mlp = "audio_composite_gated_shared_expert_mlp"


class AudioEncoderKwargs:
    audio = "audio"
    audio_mel = "audio_mel"
    audio_positions = "audio_positions"
    kv_channels = "audio_kv_channels"
    hidden_dims = "audio_hidden_dims"


class AudioEncoderType(str, enum.Enum):
    none = "none"
    whisper = "whisper"


# # TODO Toby: do we need all of them?
class AudioTransformerKwargs:
    rotary_freq_q = "audio_rotary_freq_q"
    rotary_freq_k = "audio_rotary_freq_k"
    attention_mask = "audio_attention_mask"
    attention_mask_value = "audio_attention_mask_value"
    sequence_lengths = "audio_sequence_lengths"
    cu_seqlens_q = "audio_cu_seqlens_q"
    cu_seqlens_k = "audio_cu_seqlens_k"
    max_seqlen_q = "audio_max_seqlen_q"
    max_seqlen_k = "audio_max_seqlen_k"
    # TODO: Review these
    presents = "audio_presents"
    past_key_values = "audio_past_key_values"
    sequence_first = "audio_sequence_first"
    hidden_dims = "audio_hidden_dims"
    sequence_q_dim = "audio_sequence_q_dim"
    sequence_k_dim = "audio_sequence_k_dim"
    sequence_length = "audio_sequence_length"
    micro_batch_size = "audio_micro_batch_size"
    # TODO: Move
    grad_output = "audio_grad_output"
    patch_position_ids = "patch_position_ids"


@config_class()
class AudioEncoderConfig(BaseModelConfig):
    _abstract = False

    transformer: AudioTransformerConfig = Field(
        default_factory=AudioTransformerConfig,
        desc="Configuration for the audio transformer architecture.",
        hint=FieldHint.core,
    )
    type: AudioEncoderType = Field(
        default=AudioEncoderType.none,
        desc="Type of the audio encoder. Choices: none, whisper.",
        hint=FieldHint.architecture,
    )
    conv_bias: bool = Field(
        default=False,
        desc="Whether to use bias in the convolutional layer.",
        hint=FieldHint.optional,
    )
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

    def setup_tensor_space(self, tensor_space: TensorSpace):
        tensor_space.add_tensor_dim(TensorDim(AudioEncoderDimNames.out_channels, self.transformer.hidden_size))
        tensor_space.add_tensor_dim(TensorDim(AudioEncoderDimNames.adapter_size, self.adapter_size))
        tensor_space.add_tensor_dim(TensorDim(AudioEncoderDimNames.in_channels))
        # TODO Soham: add a check for presence of kv channels parameter (head_dim)
        tensor_space.add_tensor_dim(
            TensorDim(
                AudioEncoderDimNames.kv_channels, self.transformer.hidden_size // self.transformer.num_attention_heads
            )
        )
        self.transformer.setup_tensor_space(tensor_space, type="audio")

    @property
    def enabled(self) -> bool:
        return self.type != AudioEncoderType.none
