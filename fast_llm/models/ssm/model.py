import logging
import typing

from fast_llm.engine.base_model.base_model import Layer
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.multi_stage.fast_llm_model import FastLLMModel
from fast_llm.layers.language_model.embedding import LanguageModelEmbedding
from fast_llm.layers.language_model.head import LanguageModelHead
from fast_llm.layers.ssm.discrete_mamba2 import DiscreteMamba2
from fast_llm.layers.ssm.llamba_block import LlambaBlock
from fast_llm.layers.ssm.mamba_layer import MambaLayer
from fast_llm.layers.transformer.transformer import TransformerLayer
from fast_llm.models.gpt.model import GPTBaseModel
from fast_llm.models.ssm.config import HybridSSMBaseModelConfig, HybridSSMModelConfig

logger = logging.getLogger(__name__)


class HybridSSMBaseModel[ConfigType: HybridSSMBaseModelConfig](GPTBaseModel[ConfigType]):
    """
    A hybrid model that interleaves Transformer and Mamba blocks.
    Right now only LlambaBlock is supported.
    AS for the mixer, transformer uses MHA. For the LLlambaBlock we support Mamba1 and descrete mamba2.
    """

    config_class: typing.ClassVar[type[HybridSSMBaseModelConfig]] = HybridSSMBaseModelConfig
    _is_setup: bool = False

    def __init__(
        self,
        config: HybridSSMBaseModelConfig,
        distributed_config: DistributedConfig,
    ):
        self.SSM_BLOCK_CLS = LlambaBlock  # TODO: extend to other block types if needed
        super().__init__(config, distributed_config)

    def get_layers(self) -> list[Layer]:
        """
        Create a list of layers for the model, interleaving Transformer and Mamba blocks
        according to the block pattern.
        """
        layers = [LanguageModelEmbedding(self._config, self._tensor_space)]

        # Create blocks according to pattern
        for i, block_type in enumerate(self._config.hybrid_block_layout):
            if block_type == "t":
                # Transformer block
                layers.append(
                    TransformerLayer(
                        self._config.transformer,
                        self._tensor_space,
                        layer_index=i + 1,
                    )
                )
            elif block_type == "m2":
                mamba_block = self.SSM_BLOCK_CLS(
                    config_transformer=self._config.transformer,
                    config_ssm=self._config.ssm,
                    mixer_cls=DiscreteMamba2,
                    layer_index=i + 1,
                    tensor_space=self._tensor_space,
                )
                layers.append(mamba_block)

            elif block_type == "m":
                # Create Mamba block
                mamba_block = self.SSM_BLOCK_CLS(
                    config_transformer=self._config.transformer,
                    config_ssm=self._config.ssm,
                    mixer_cls=MambaLayer,
                    layer_index=i + 1,
                    tensor_space=self._tensor_space,
                )
                layers.append(mamba_block)

            else:
                raise ValueError(f"Invalid block type: {block_type}. Must be 't' or 'm' or 'm2'")

        # Add the language model head
        layers.append(LanguageModelHead(self._config, self._tensor_space, prediction_distance=0))

        return layers


class HybridSSMModel[ConfigType: HybridSSMModelConfig](FastLLMModel[ConfigType]):
    """
    A hybrid model that combines Transformer and SSM blocks.
    """

    config_class: typing.ClassVar[type[HybridSSMModelConfig]] = HybridSSMModelConfig
    base_model_class: typing.ClassVar[type[HybridSSMBaseModel]] = HybridSSMBaseModel
