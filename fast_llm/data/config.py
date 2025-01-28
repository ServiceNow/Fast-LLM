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


class SpecialTokensMode(str, enum.Enum):
    tokenizer_default = "tokenizer_default"
    bos_only = "bos_only"
    eos_only = "eos_only"
    bos_eos = "bos_eos"


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
    special_tokens_mode: SpecialTokensMode = Field(
        default=SpecialTokensMode.bos_only,
        desc="Special tokens configuration.",
        hint=FieldHint.core,
    )
