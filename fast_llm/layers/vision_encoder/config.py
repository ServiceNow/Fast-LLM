from fast_llm.config import Field, FieldHint, config_class
from fast_llm.engine.base_model.config import BaseModelArchitectureConfig
from fast_llm.engine.config_utils.tensor_space import TensorDim, TensorSpace
from fast_llm.functional.config import ActivationType


class VisionEncoderDimNames:
    out_channels = "vision_out_channels"
    intermediate_size = "vision_intermediate_size"
    patch_height = "vision_patch_height"
    patch_width = "vision_patch_width"


@config_class()
class PatchConvConfig(BaseModelArchitectureConfig):
    _abstract = False
    """
    Configuration class for the convolution layers to apply on the image patches
    """
    in_channels: int = Field(
        default=3,
        desc="Number of input channels for the convolution layers. Typically 3 for RGB images.",
        hint=FieldHint.optional,
    )
    bias: bool = Field(
        default=False, desc="Whether to use a bias term in the convolution layers.", hint=FieldHint.optional
    )
    height: int = Field(
        default=16,
        desc="Height of the image patches considered as tokens",
    )
    width: int | None = Field(
        default=16,
        desc="Width of the image patches considered as tokens",
    )


@config_class()
class VisionEncoderArchitectureConfig(BaseModelArchitectureConfig):
    _abstract = False
    """
    Configuration class for the vision encoder, which transforms images into embeddings
    """
    path: str | None = Field(default=None, desc="Path to a pretrained vision encoder model.", hint=FieldHint.optional)
    hidden_size: int = Field(
        default=1024, desc="The size of the hidden layers in the transformer model.", hint=FieldHint.optional
    )
    intermediate_size: int = Field(
        default=4096,
        desc="The size of the intermediate (feed-forward) layers in the transformer model.",
        hint=FieldHint.optional,
    )
    num_hidden_layers: int = Field(
        default=24, desc="The number of hidden layers in the transformer model.", hint=FieldHint.optional
    )
    num_attention_heads: int = Field(
        default=16, desc="The number of attention heads for the multi-head attention layers.", hint=FieldHint.optional
    )
    num_channels: int = Field(
        default=3, desc="Number of channels in the input image, typically 3 for RGB.", hint=FieldHint.optional
    )
    image_size: int = Field(
        default=1024, desc="The size of the input images (assumed square).", hint=FieldHint.optional
    )
    patch_size: int = Field(default=16, desc="The size of the image patches to be encoded.", hint=FieldHint.optional)
    hidden_act: str = Field(
        default="gelu", desc="The activation function used in the hidden layers.", hint=FieldHint.optional
    )
    attention_dropout: float = Field(
        default=0.0, desc="The dropout probability for attention layers.", hint=FieldHint.optional
    )
    rope_theta: float = Field(
        default=10000.0, desc="The base value for rotary position embeddings.", hint=FieldHint.optional
    )
    initializer_range: float = Field(
        default=0.02, desc="The standard deviation of the normal initializer.", hint=FieldHint.optional
    )


@config_class()
class VisionArchitectureConfig(BaseModelArchitectureConfig):
    _abstract = False

    encoder: VisionEncoderArchitectureConfig = Field(
        default_factory=VisionEncoderArchitectureConfig,
        desc="Configuration for the vision encoder that transforms images into embeddings.",
        hint=FieldHint.optional,
    )
    adapter_size: int = Field(
        default=5120,
        desc="Intermediate size for the adapter linear layers. Assuming 2 linear layers",
        hint=FieldHint.optional,
    )
    adapter_activation_type: ActivationType = Field(
        default=ActivationType.gelu,
        desc="The intermediate activation type for multi-modal adapter. Default: GeLU.",
        hint=FieldHint.core,
    )

    def setup_tensor_space(self, tensor_space: TensorSpace):
        tensor_space.add_tensor_dim(TensorDim(VisionEncoderDimNames.out_channels, self.encoder.hidden_size))
        tensor_space.add_tensor_dim(TensorDim(VisionEncoderDimNames.intermediate_size, self.adapter_size))
        tensor_space.add_tensor_dim(TensorDim(VisionEncoderDimNames.patch_height, self.encoder.patch_size))
        tensor_space.add_tensor_dim(TensorDim(VisionEncoderDimNames.patch_width, self.encoder.patch_size))
        # tensor_space.add_tensor_dim(
        #     CompositeTensorDim(VisionEncoderDimNames.)
        # )

    # patch_convolution: PatchConvConfig = Field(
    #     default_factory=PatchConvConfig,
    #     desc="Configuration for the convolution layers applied to the image patches.",
    #     hint=FieldHint.optional
    # )
    # normalization: NormalizationArchitectureConfig = Field(
    #     default_factory=NormalizationArchitectureConfig,
    #     desc="Configuration for the normalization layers applied to the image patches.",
    #     hint=FieldHint.optional
    # )
    # transformer: TransformerArchitectureConfig = Field(
    #     default_factory=TransformerArchitectureConfig,
    #     desc="Configuration for the transformer layers applied to the image patches.",
    #     hint=FieldHint.optional
    # )
    # patch_rotary: RotaryArchitectureConfig = Field(
    #     default_factory=RotaryArchitectureConfig,
    #     desc="Configuration for the rotary positional embeddings applied to the image patches.",
    #     hint=FieldHint.optional
    # )
