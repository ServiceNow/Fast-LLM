import enum
import pathlib

from fast_llm.config import Config, Field, FieldHint, check_field, config_class
from fast_llm.utils import Assert


class MultiprocessingContext(str, enum.Enum):
    # Fast but risk of segfaults due to interactions with triton
    # (for example https://github.com/openai/triton/issues/2088).
    fork = "fork"
    # Safe but much slower.
    spawn = "spawn"


TokenizerFromFile = "TokenizerFromFile"


@config_class()
class TokenizerConfig(Config):
    """
    Configuration for the tokenizer.
    The tokenizer is needed for FIM and dataset preparation.
    """

    format: str = Field(
        default="TokenizerFromFile",
        desc="Unused.",
        hint=FieldHint.deprecated,
        valid=check_field(Assert.eq, TokenizerFromFile),
    )
    path: pathlib.Path | None = Field(
        default=None,
        desc="Path to the tokenizer file.",
        hint=FieldHint.core,
    )


@config_class()
class ImageProcessorConfig(Config):
    """
    Configuration for the image processor
    """

    # Defaults taken from [pixtral](https://github.com/huggingface/transformers/blob/794fde7b1c3d041519fc28ea3e1461b0cfcad4e7/src/transformers/models/pixtral/image_processing_pixtral.py#L201)
    patch_size: list[int] = Field(
        default_factory=lambda: [16, 16],
        desc="Size for each path extracted from the image. Each patch corresponds to a token for the transformer",
        hint=FieldHint.optional,
    )
    max_height: int = Field(
        default=1024,
        desc="Maximum height of the image. Image will be resized if larger",
        hint=FieldHint.optional,
    )
    max_width: int = Field(
        default=1024,
        desc="Maximum width of the image. Image will be resized if larger",
        hint=FieldHint.optional,
    )
    mean: list[float] = Field(
        default_factory=lambda: [0.48145466, 0.4578275, 0.40821073],
        desc="Mean RGB values for pixel normalization",
        hint=FieldHint.optional,
    )
    std: list[float] = Field(
        default_factory=lambda: [0.26862954, 0.26130258, 0.27577711],
        desc="Standard deviation RGB values for pixel normalization",
        hint=FieldHint.optional,
    )
    rescale_factor: float = Field(
        default=255.0,
        desc="Diminisher factor for pixel normalization",
        hint=FieldHint.optional,
    )


@config_class()
class MultiModalProcessorConfig(Config):
    """
    Wrapper config that stores the `ImageProcessorConfig` and `TokenizerConfig`
    """

    tokenizer: TokenizerConfig = Field(
        default_factory=TokenizerConfig,
        desc="Configuration for the tokenizer.",
        hint=FieldHint.core,
    )
    image_processor: ImageProcessorConfig = Field(
        default_factory=ImageProcessorConfig,
        desc="Configuration for the image processor.",
        hint=FieldHint.core,
    )
