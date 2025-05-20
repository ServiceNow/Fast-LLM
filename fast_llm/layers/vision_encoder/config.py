import enum

from fast_llm.config import Config, Field, FieldHint, config_class
from fast_llm.engine.base_model.config import BaseModelConfig
from fast_llm.engine.config_utils.tensor_space import TensorDim, TensorSpace
from fast_llm.functional.config import ActivationType
from fast_llm.layers.common.config import NormalizationConfig
from fast_llm.layers.transformer.config import VisionTransformerConfig


class VisionEncoderDimNames:
    in_channels = "vision_in_channels"
    out_channels = "vision_out_channels"
    adapter_size = "vision_adapter_size"
    patch_size = "vision_patch_size"
    kv_channels = "vision_kv_channels"


class VisionEncoderKwargs:
    patch_size = "patch_size"
    images = "images"
    image_patches = "image_patches"
    image_positions = "image_positions"
    image_size = "image_size"
    image_sizes = "image_sizes"
    image_mean = "image_normalization_mean"
    image_std = "image_normalization_std"
    image_rescale_factor = "image_rescale_factor"
    rope_theta = "vit_rope_theta"
    rotary_inv_freq = "vit_rotary_inv_freq"
    kv_channels = "vit_kv_channels"
    max_image_tokens = "max_image_tokens"
    patch_embeddings = "patch_embeddings"
    hidden_dims = "vit_hidden_dims"
    image_patches_meta = "vit_image_patches_meta"
    out_channels = "vit_out_channels"


@config_class()
class ImageNormalizationConfig(Config):
    mean_r: float = Field(
        default=0.48145466,
        desc="Mean value for the red channel in the image normalization process.",
        hint=FieldHint.optional,
    )
    mean_g: float = Field(
        default=0.4578275,
        desc="Mean value for the green channel in the image normalization process.",
        hint=FieldHint.optional,
    )
    mean_b: float = Field(
        default=0.40821073,
        desc="Mean value for the blue channel in the image normalization process.",
        hint=FieldHint.optional,
    )
    std_r: float = Field(
        default=0.26862954,
        desc="Standard deviation value for the red channel in the image normalization process.",
        hint=FieldHint.optional,
    )
    std_g: float = Field(
        default=0.26130258,
        desc="Standard deviation value for the green channel in the image normalization process.",
        hint=FieldHint.optional,
    )
    std_b: float = Field(
        default=0.27577711,
        desc="Standard deviation value for the blue channel in the image normalization process.",
        hint=FieldHint.optional,
    )
    rescale_factor: float = Field(
        default=255.0,
        desc="Rescale factor for the image normalization process.",
        hint=FieldHint.optional,
    )


class VisionEncoderType(str, enum.Enum):
    none = "none"
    pixtral = "pixtral"


@config_class()
class VisionEncoderConfig(BaseModelConfig):
    _abstract = False

    type: VisionEncoderType = Field(
        default=VisionEncoderType.none,
        desc="Type of the vision encoder. Choices: none, pixtral.",
        hint=FieldHint.architecture,
    )
    transformer: VisionTransformerConfig = Field(
        default_factory=VisionTransformerConfig,
        desc="Configuration for the vision transformer architecture.",
        hint=FieldHint.core,
    )
    patch_size: int = Field(
        default=16,
        desc="Patch size for the image encoder.",
        hint=FieldHint.core,
    )
    conv_bias: bool = Field(
        default=False,
        desc="Whether to use bias in the convolutional layer.",
        hint=FieldHint.optional,
    )
    patch_norm: NormalizationConfig = Field(
        default_factory=NormalizationConfig,
        desc="Configuration for the normalization layers applied to the image patches.",
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
    image_normalization: ImageNormalizationConfig = Field(
        default_factory=ImageNormalizationConfig,
        desc="Configuration for the normalization layers applied to the image patches.",
        hint=FieldHint.optional,
    )

    def setup_tensor_space(self, tensor_space: TensorSpace):
        tensor_space.add_tensor_dim(TensorDim(VisionEncoderDimNames.out_channels, self.transformer.hidden_size))
        tensor_space.add_tensor_dim(TensorDim(VisionEncoderDimNames.adapter_size, self.adapter_size))
        tensor_space.add_tensor_dim(TensorDim(VisionEncoderDimNames.patch_size, self.patch_size))
        tensor_space.add_tensor_dim(TensorDim(VisionEncoderDimNames.in_channels, 3))
        # TODO Soham: add a check for presence of kv channels parameter (head_dim)
        tensor_space.add_tensor_dim(
            TensorDim(
                VisionEncoderDimNames.kv_channels, self.transformer.hidden_size // self.transformer.num_attention_heads
            )
        )
        self.transformer.setup_tensor_space(tensor_space)

    @property
    def enabled(self) -> bool:
        return self.type != VisionEncoderType.none
