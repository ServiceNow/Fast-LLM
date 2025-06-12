import enum

from fast_llm.config import Config, Field, FieldHint, check_field, config_class, skip_valid_if_none
from fast_llm.engine.base_model.config import BaseModelConfig
from fast_llm.engine.config_utils.tensor_space import TensorDim, TensorSpace
from fast_llm.functional.config import ActivationType
from fast_llm.layers.common.config import NormalizationConfig
from fast_llm.layers.transformer.config import TransformerConfig
from fast_llm.utils import Assert


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
    max_image_size = "max_image_size"
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
    # TODO: better name? normalization, patch size, adapter can change based on implementation, no standard way currently.
    pixtral = "pixtral"


@config_class(registry=True)
class VisionEncoderConfig(BaseModelConfig):
    _abstract = False

    type: VisionEncoderType = Field(
        default=VisionEncoderType.none,
        desc="Type of the vision encoder. Choices: none, pixtral.",
        hint=FieldHint.architecture,
    )
    transformer: TransformerConfig = Field(
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
    adapter_bias: bool = Field(
        default=True,
        desc="Whether to use bias in the adapter linear layer.",
        hint=FieldHint.optional,
    )
    image_normalization: ImageNormalizationConfig = Field(
        desc="Configuration for the normalization layers applied to the image patches.",
        hint=FieldHint.optional,
    )
    image_break_token: int | None = Field(
        default=None,
        desc="Token id to separate image rows. If None, no token id is applied.",
        hint=FieldHint.optional,
    )
    image_end_token: int | None = Field(
        default=None,
        desc="Token id to indicate the end of an image. If None, no token id is applied.",
        hint=FieldHint.optional,
    )
    adapter_lr_scale: float | None = Field(
        default=None,
        desc="Custom learning rate scale for the adapter weights.",
        hint=FieldHint.feature,
        valid=skip_valid_if_none(check_field(Assert.geq, 0)),
    )
    conv_lr_scale: float | None = Field(
        default=None,
        desc="Custom learning rate scale for the convolutional layer weights.",
        hint=FieldHint.feature,
        valid=skip_valid_if_none(check_field(Assert.geq, 0)),
    )

    def setup_tensor_space(self, tensor_space: TensorSpace):
        tensor_space.add_tensor_dim(TensorDim(VisionEncoderDimNames.out_channels, self.transformer.hidden_size))
        tensor_space.add_tensor_dim(TensorDim(VisionEncoderDimNames.adapter_size, self.adapter_size))
        tensor_space.add_tensor_dim(TensorDim(VisionEncoderDimNames.patch_size, self.patch_size))
        tensor_space.add_tensor_dim(TensorDim(VisionEncoderDimNames.in_channels, 3))
        self.transformer.setup_tensor_space(tensor_space)

    @property
    def enabled(self) -> bool:
        return self.type != VisionEncoderType.none


for name in VisionEncoderType:
    # We need this because we are using the reserved field name `type`.
    # TODO: Implement proper dynamic typing.
    VisionEncoderConfig.register_subclass(name.value, VisionEncoderConfig)
