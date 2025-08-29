import logging
import typing

from fast_llm.layers.attention.config import AttentionConfig, TransformerConfig
from fast_llm.layers.block.block import Block

logger = logging.getLogger(__name__)


class TransformerBlock[ConfigType: TransformerConfig](Block[ConfigType]):
    # TODO: Standardize to `mixer`
    _mixer_module_name: typing.ClassVar[str] = "self_attn"

    @property
    def _mixer_config(self) -> AttentionConfig:
        return self._config.mixer
