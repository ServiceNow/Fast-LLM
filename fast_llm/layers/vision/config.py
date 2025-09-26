import enum
import typing

from fast_llm.config import Config, Field, FieldHint, config_class
from fast_llm.engine.base_model.config import BaseModelConfig
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.layers.block.config import BlockConfig, BlockSequenceConfig
from fast_llm.layers.common.linear.config import Convolution2DConfig
from fast_llm.layers.common.normalization.config import NormalizationConfig
from fast_llm.layers.decoder.config import MLPBaseConfig

if typing.TYPE_CHECKING:
    pass


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


@config_class()
class PatchConvolutionConfig(BlockConfig):
    _abstract = False
    convolution: Convolution2DConfig = Field(
        desc="Configuration for the 2d convolution.",
        hint=FieldHint.architecture,
    )
    normalization: NormalizationConfig = Field(
        desc="Configuration for the normalization layer.",
        hint=FieldHint.architecture,
    )
    patch_size: int = Field(
        default=16,
        desc="Size of image patches, in pixels (width and height).",
        hint=FieldHint.core,
    )
    input_channels: int = Field(
        default=3,
        desc="Number of pixel channels (usually 3).",
        hint=FieldHint.feature,
    )


@config_class(registry=True)
class VisionEncoderConfig(BaseModelConfig):
    _abstract = False
    patch_convolution_layer: PatchConvolutionConfig = Field(
        desc="Configuration for the patch convolution layer.",
        hint=FieldHint.architecture,
    )
    adapter_layer: MLPBaseConfig = Field(
        desc="Configuration for the adapter layer.",
        hint=FieldHint.architecture,
    )
    decoder: BlockSequenceConfig = Field(
        desc="Configuration for the vision decoder.",
        hint=FieldHint.architecture,
    )

    type: VisionEncoderType = Field(
        default=VisionEncoderType.none,
        desc="Type of the vision encoder. Choices: none, pixtral.",
        hint=FieldHint.architecture,
    )
    # transformer: TransformerConfig = Field(
    #    desc="Configuration for the vision transformer architecture.",
    #    hint=FieldHint.core,
    # )
    # patch_size: int = Field(
    #    default=16,
    #    desc="Patch size for the image encoder.",
    #    hint=FieldHint.core,
    # )
    # conv_bias: bool = Field(
    #    default=False,
    #    desc="Whether to use bias in the convolutional layer.",
    #    hint=FieldHint.optional,
    # )
    # patch_norm: NormalizationConfig = Field(
    #    desc="Configuration for the normalization layers applied to the image patches.",
    #    hint=FieldHint.optional,
    # )
    # adapter_size: int = Field(
    #    default=5120,
    #    desc="Intermediate size for the adapter linear layers. Assuming 2 linear layers",
    #    hint=FieldHint.core,
    # )
    # adapter_activation_type: ActivationType = Field(
    #    default=ActivationType.gelu,
    #    desc="The intermediate activation type for multi-modal adapter. Default: GeLU.",
    #    hint=FieldHint.core,
    # )
    # adapter_bias: bool = Field(
    #    default=True,
    #    desc="Whether to use bias in the adapter linear layer.",
    #    hint=FieldHint.optional,
    # )
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
    # adapter_lr_scale: float | None = Field(
    #    default=None,
    #    desc="Custom learning rate scale for the adapter weights.",
    #    hint=FieldHint.feature,
    #    valid=skip_valid_if_none(check_field(Assert.geq, 0)),
    # )
    # conv_lr_scale: float | None = Field(
    #    default=None,
    #    desc="Custom learning rate scale for the convolutional layer weights.",
    #    hint=FieldHint.feature,
    #    valid=skip_valid_if_none(check_field(Assert.geq, 0)),
    # )
    # adapter_init_method_std: float = Field(
    #    default=None,
    #    desc="Standard deviation for the normal initialization of the adapter weights. Default: adapter_size ** -0.5.",
    #    hint=FieldHint.optional,
    #    valid=check_field(Assert.geq, 0),
    # )

    def _validate(self) -> None:
        with self._set_implicit_default():
            if self.adapter_init_method_std is None:
                self.adapter_init_method_std = self.adapter_size**-0.5
        super()._validate()

    def setup_tensor_space(self, tensor_space: "TensorSpace"):
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
