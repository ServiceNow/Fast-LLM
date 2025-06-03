import logging
import typing

from fast_llm.engine.base_model.base_model import Layer
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.multi_stage.fast_llm_model import FastLLMModel
from fast_llm.layers.language_model.embedding import LanguageModelEmbedding
from fast_llm.layers.language_model.head import LanguageModelHead
from fast_llm.layers.transformer.transformer import BaseBlock
from fast_llm.models.gpt.model import GPTBaseModel
from fast_llm.models.hybrid.config import HybridBaseModelConfig, HybridModelConfig

logger = logging.getLogger(__name__)


class HybridBaseModel[ConfigType: HybridBaseModelConfig](GPTBaseModel[ConfigType]):
    """
    A hybrid model that can interleave Transformer, Mamba and other blocks.
    """

    config_class: typing.ClassVar[type[HybridBaseModelConfig]] = HybridBaseModelConfig
    _is_setup: bool = False

    def __init__(
        self,
        config: HybridBaseModelConfig,
        distributed_config: DistributedConfig,
    ):

        super().__init__(config, distributed_config)

    def get_output_layers(self) -> list[Layer]:
        """
        Get the output layers of the model.
        This includes the language model head and any additional heads specified in the configuration.
        """
        layers = [LanguageModelHead(self._config, self._tensor_space, prediction_distance=0)]

        if self._config.prediction_heads > 1:
            block_name = self._config.default_mtp_type
            assert block_name in self._config.blocks, f"Block {block_name} not found in config"
            BLOCK_CLS = self._config.blocks[block_name].block_class
            for i in range(1, self._config.prediction_heads):
                layers.append(
                    BLOCK_CLS(
                        self._config.blocks[block_name],
                        self._tensor_space,
                        layer_index=len(self._config.hybrid_block_layout),
                        return_input=i != self._config.prediction_heads - 1,
                        block_name=block_name,
                    )
                )
                layers.append(LanguageModelHead(self._config, self._tensor_space, prediction_distance=i))

        return layers

    def get_layers(self) -> list[Layer]:
        """
        Create a list of layers for the model, interleaving Transformer and Mamba blocks
        according to the block pattern.
        """
        layers = [LanguageModelEmbedding(self._config, self._tensor_space)]

        # Create blocks according to pattern
        for i, block_name in enumerate(self._config.hybrid_block_layout):
            BLOCK_CLS: BaseBlock = self._config.blocks[block_name].block_class
            layers.append(
                BLOCK_CLS(
                    self._config.blocks[block_name],
                    self._tensor_space,
                    layer_index=i + 1,
                    return_input=(
                        i == len(self._config.hybrid_block_layout) - 1 and self._config.prediction_heads > 1
                    ),
                    block_name=block_name,
                )
            )
        layers += self.get_output_layers()

        return layers


class HybridModel[ConfigType: HybridModelConfig](FastLLMModel[ConfigType]):
    """
    A hybrid model that combines Transformer and SSM blocks.
    """

    config_class: typing.ClassVar[type[HybridModelConfig]] = HybridModelConfig
    base_model_class: typing.ClassVar[type[HybridBaseModel]] = HybridBaseModel
