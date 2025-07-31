import logging
import typing

from fast_llm.engine.config_utils.tensor_space import TensorSpace
from fast_llm.layers.block.block import Block, BlockLayer
from fast_llm.layers.transformer.attention import Attention
from fast_llm.layers.transformer.config import TransformerConfig

logger = logging.getLogger(__name__)


class TransformerBlock[ConfigType: TransformerConfig](Block[ConfigType]):
    _name = "Transformer layer"
    # TODO: Standardize to `mixer`
    _mixer_module_name: typing.ClassVar[str] = "self_attn"
    _config: TransformerConfig

    def __init__(self, config: ConfigType, tensor_space: TensorSpace, block_index: int, return_input: bool = False):
        super().__init__(config, tensor_space, block_index, return_input)

    def _create_mixer(self) -> BlockLayer:
        return Attention(self._config, self._tensor_space, self._block_index)
