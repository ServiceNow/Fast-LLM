import logging
import typing

from fast_llm.layers.block.block import Block, BlockLayer
from fast_llm.layers.transformer.attention import Attention
from fast_llm.layers.transformer.config import TransformerConfig

logger = logging.getLogger(__name__)


class TransformerBlock[ConfigType: TransformerConfig](Block[ConfigType]):
    # TODO: Standardize to `mixer`
    _mixer_module_name: typing.ClassVar[str] = "self_attn"
    _config: TransformerConfig

    def _create_mixer(self) -> BlockLayer:
        return Attention(
            self._config, self._distributed_config, self._hidden_dim, self._block_index, f"{self._name} attn"
        )
