import logging
import typing

from fast_llm.engine.base_model.base_model import Layer
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.layers.language_model.embedding import LanguageModelEmbedding
from fast_llm.layers.language_model.head import LanguageModelHead
from fast_llm.layers.ssm.discrete_mamba2 import DiscreteMamba2
from fast_llm.layers.ssm.llamba_block import LlambaBlock
from fast_llm.layers.ssm.mamba_layer import MambaLayer
from fast_llm.layers.transformer.transformer import TransformerLayer
from fast_llm.models.gpt.model import GPTBaseModel, GPTModel
from fast_llm.models.ssm.config import HybridSSMBaseModelConfig, HybridSSMModelConfig, SSMBlockType

logger = logging.getLogger(__name__)


class HybridSSMBaseModel[ConfigType: HybridSSMBaseModelConfig](GPTBaseModel[ConfigType]):
    """
    A hybrid model that interleaves Transformer and Mamba blocks.
    Right now only LlambaBlock is supported.
    As for the mixer, transformer uses MHA. For the LlambaBlock we support Mamba1 and discrete mamba2.
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

    def get_output_layers(self) -> list[Layer]:
        """
        Get the output layers of the model.
        This includes the language model head and any additional heads specified in the configuration.
        """
        layers = [LanguageModelHead(self._config, self._tensor_space, prediction_distance=0)]

        if self._config.prediction_heads > 1:
            block_type = self._config.default_mtp_type or self._config.hybrid_block_layout[-1]
            for i in range(1, self._config.prediction_heads):
                if block_type == SSMBlockType.transformer.value:
                    layers.append(
                        TransformerLayer(
                            self._config.transformer,
                            self._tensor_space,
                            layer_index=len(self._config.hybrid_block_layout),
                            return_input=i != self._config.prediction_heads - 1,
                        )
                    )
                elif block_type == SSMBlockType.mamba2_discrete.value:
                    mamba_block = self.SSM_BLOCK_CLS(
                        config_transformer=self._config.transformer,
                        config_ssm=self._config.ssm,
                        mixer_cls=DiscreteMamba2,
                        layer_index=len(self._config.hybrid_block_layout),
                        tensor_space=self._tensor_space,
                        return_input=i != self._config.prediction_heads - 1,
                    )
                    layers.append(mamba_block)
                elif block_type == SSMBlockType.mamba.value:
                    mamba_block = self.SSM_BLOCK_CLS(
                        config_transformer=self._config.transformer,
                        config_ssm=self._config.ssm,
                        mixer_cls=MambaLayer,
                        layer_index=len(self._config.hybrid_block_layout),
                        tensor_space=self._tensor_space,
                        return_input=i != self._config.prediction_heads - 1,
                    )
                    layers.append(mamba_block)
                else:
                    raise ValueError(f"Invalid block type: {block_type}. Must be {SSMBlockType.__members__}")
                layers.append(LanguageModelHead(self._config, self._tensor_space, prediction_distance=i))

        return layers

    def get_layers(self) -> list[Layer]:
        """
        Create a list of layers for the model, interleaving Transformer and Mamba blocks
        according to the block pattern.
        """
        layers = [LanguageModelEmbedding(self._config, self._tensor_space)]

        # Create blocks according to pattern
        for i, block_type in enumerate(self._config.hybrid_block_layout):
            if block_type == SSMBlockType.transformer:
                # Transformer block
                layers.append(
                    TransformerLayer(
                        self._config.transformer,
                        self._tensor_space,
                        layer_index=i + 1,
                        return_input=(
                            i == len(self._config.hybrid_block_layout) - 1 and self._config.prediction_heads > 1
                        ),
                    )
                )
            elif block_type == SSMBlockType.mamba2_discrete:
                mamba_block = self.SSM_BLOCK_CLS(
                    config_transformer=self._config.transformer,
                    config_ssm=self._config.ssm,
                    mixer_cls=DiscreteMamba2,
                    layer_index=i + 1,
                    tensor_space=self._tensor_space,
                    return_input=(
                        i == len(self._config.hybrid_block_layout) - 1 and self._config.prediction_heads > 1
                    ),
                )
                layers.append(mamba_block)

            elif block_type == SSMBlockType.mamba:
                # Create Mamba block
                mamba_block = self.SSM_BLOCK_CLS(
                    config_transformer=self._config.transformer,
                    config_ssm=self._config.ssm,
                    mixer_cls=MambaLayer,
                    layer_index=i + 1,
                    tensor_space=self._tensor_space,
                    return_input=(
                        i == len(self._config.hybrid_block_layout) - 1 and self._config.prediction_heads > 1
                    ),
                )
                layers.append(mamba_block)
            else:
                raise ValueError(f"Invalid block type: {block_type}. Must be {SSMBlockType.__members__}")

        # Add the output layers
        layers += self.get_output_layers()

        return layers


class HybridSSMModel[ConfigType: HybridSSMModelConfig](GPTModel[ConfigType]):
    """
    A hybrid model that combines Transformer and SSM blocks.
    """

    config_class: typing.ClassVar[type[HybridSSMModelConfig]] = HybridSSMModelConfig
    base_model_class: typing.ClassVar[type[HybridSSMBaseModel]] = HybridSSMBaseModel
