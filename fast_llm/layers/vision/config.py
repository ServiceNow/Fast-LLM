import functools
import typing

from fast_llm.config import Config, Field, FieldHint, check_field, config_class
from fast_llm.layers.block.config import BlockConfig, BlockKwargs, BlockSequenceConfig
from fast_llm.layers.common.linear.config import AffineLinearConfig
from fast_llm.layers.common.normalization.config import NormalizationConfig
from fast_llm.layers.decoder.config import MLPBaseConfig
from fast_llm.layers.language_model.config import LanguageModelConfig
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    from fast_llm.layers.vision.embeddings import PatchEmbeddings
    from fast_llm.layers.vision.vision_encoder import VisionEncoder, VisionMultiModalModel


class VisionKwargs(BlockKwargs):
    patch_positions = "patch_positions"


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


@config_class()
class PatchEmbeddingsConfig(BlockConfig):
    _abstract = False
    patch_embeddings: AffineLinearConfig = Field(
        desc="Configuration for the patch embedding layer.",
        hint=FieldHint.architecture,
    )
    normalization: NormalizationConfig = Field(
        desc="Configuration for the normalization layer.",
        hint=FieldHint.architecture,
    )
    patch_height: int = Field(
        default=16,
        desc="Height of image patches, in pixels.",
        hint=FieldHint.core,
    )
    patch_width: int = Field(
        default=16,
        desc="Width of image patches, in pixels.",
        hint=FieldHint.core,
    )
    full_precision_residual: bool = Field(
        default=False,
        desc="Store the residuals for the model in full precision (`optimization_dtype`).",
        hint=FieldHint.stability,
    )

    @functools.cached_property
    def input_channels(self) -> int:
        # Number of input channels. Currently hard-coded to 3 (RGB).
        return 3

    @property
    def layer_class(self) -> "type[PatchEmbeddings]":
        from fast_llm.layers.vision.embeddings import PatchEmbeddings

        return PatchEmbeddings


@config_class(registry=True)
class VisionEncoderConfig(BlockConfig):
    _abstract = False
    # TODO: ====== Rename to patch_embeddings? ======
    embeddings: PatchEmbeddingsConfig = Field(
        desc="Configuration for the patch convolution layer.",
        hint=FieldHint.architecture,
    )
    # TODO: Should use varlen mixer, 2d rotary, non-causal. Enforce?
    encoder: BlockSequenceConfig = Field(
        desc="Configuration for the vision decoder.",
        hint=FieldHint.architecture,
    )
    adapter: MLPBaseConfig = Field(
        desc="Configuration for the adapter layer.",
        hint=FieldHint.architecture,
    )
    hidden_size: int = Field(
        default=1024,
        desc="Size of the vision encoder main hidden dimension.",
        hint=FieldHint.architecture,
        valid=check_field(Assert.gt, 0),
    )

    @property
    def layer_class(self) -> "type[VisionEncoder]":
        from fast_llm.layers.vision.vision_encoder import VisionEncoder

        return VisionEncoder


@config_class()
class VisionMultiModalModelConfig(LanguageModelConfig):
    vision_encoder: VisionEncoderConfig = Field(
        hint=FieldHint.architecture,
        desc="Configuration for the vision encoder.",
    )
    image_token_index: int | None = Field(
        default=None,
        hint=FieldHint.optional,
        desc="Index of the image token. Unused, but required for Hugging Face conversion.",
    )

    @property
    def layer_class(self) -> "type[VisionMultiModalModel]":
        from fast_llm.layers.vision.vision_encoder import VisionMultiModalModel

        return VisionMultiModalModel
