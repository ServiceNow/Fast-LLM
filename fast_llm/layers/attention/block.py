import functools
import logging
import typing

from fast_llm.layers.attention.attention import Attention
from fast_llm.layers.attention.config import AttentionConfig, TransformerConfig
from fast_llm.layers.block.block import Block

logger = logging.getLogger(__name__)


class TransformerBlock[ConfigType: TransformerConfig](Block[ConfigType]):
    # TODO: Standardize to `mixer`
    _mixer_module_name: typing.ClassVar[str] = "self_attn"

    @functools.cached_property
    def _mixer_class(self) -> type[Attention]:
        return Attention

    @property
    def _mixer_config(self) -> AttentionConfig:
        return self._config
