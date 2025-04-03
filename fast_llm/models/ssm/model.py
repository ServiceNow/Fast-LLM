import logging
import typing
from functools import partial

from lamba_block import LambaBlock

from fast_llm.engine.base_model.base_model import Layer
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.multi_stage.fast_llm_model import FastLLMModel
from fast_llm.layers.language_model.embedding import LanguageModelEmbedding
from fast_llm.layers.language_model.head import LanguageModelHead
from fast_llm.layers.ssm.discrete_mamba2 import DiscreteMamba2 as Mamba2Layer
from fast_llm.layers.ssm.mamba_layer import MambaLayer
from fast_llm.layers.transformer.transformer import TransformerLayer
from fast_llm.models.gpt.model import GPTBaseModel
from fast_llm.models.ssm.config import HybridBaseModelConfig, HybridModelConfig

# try:
#     from ops.triton.layernorm import RMSNorm
# except ImportError:
#     RMSNorm = None


logger = logging.getLogger(__name__)


# Wrap LambaBlock to match Layer interface
class LambaBlockWrapper(Layer):
    def __init__(self, mamba_block):
        super().__init__()
        self.mamba_block = mamba_block

    def forward(self, input_, kwargs, losses=None, metrics=None):
        # Extract residual from kwargs if available
        residual = kwargs.get("residual", None)
        inference_params = kwargs.get("inference_params", None)

        # Call LambaBlock
        hidden_states, new_residual = self.mamba_block(input_, residual=residual, inference_params=inference_params)

        # Store residual for next layer
        kwargs["residual"] = new_residual

        return hidden_states


class HybridBaseModel[ConfigType: HybridBaseModelConfig](GPTBaseModel[ConfigType]):
    """
    A hybrid model that interleaves Transformer and Mamba blocks.
    """

    config_class: typing.ClassVar[type[HybridBaseModelConfig]] = HybridBaseModelConfig
    _is_setup: bool = False

    def __init__(
        self,
        config: HybridBaseModelConfig,
        distributed_config: DistributedConfig,
    ):
        super().__init__(config, distributed_config)

        # Validate block pattern length
        if len(config.block_pattern) != config.transformer.num_layers:
            raise ValueError(
                f"Block pattern length ({len(config.block_pattern)}) must match "
                f"number of layers ({config.transformer.num_layers})"
            )

        # Validate block pattern values
        for block_type in config.block_pattern:
            if block_type not in ["t", "m"]:
                raise ValueError(f"Invalid block type: {block_type}. Must be 't' or 'm'")

    def get_layers(self) -> list[Layer]:
        """
        Create a list of layers for the model, interleaving Transformer and Mamba blocks
        according to the block pattern.
        """
        layers = [LanguageModelEmbedding(self._config, self._tensor_space)]

        # Create blocks according to pattern
        for i, block_type in enumerate(self._config.block_pattern):
            if block_type == "t":
                # Transformer block
                layers.append(
                    TransformerLayer(
                        self._config.transformer,
                        self._tensor_space,
                        layer_index=i + 1,
                    )
                )
            else:  # block_type == 'm'

                # Create Mamba block
                mixer_cls = (
                    partial(MambaLayer, layer_idx=i + 1)
                    if not self._config.ssm.use_mamba2
                    else partial(Mamba2Layer, layer_idx=i)
                )
                mamba_block = LambaBlock(
                    self._config,
                    mixer_cls=mixer_cls,
                    layer_index=i + 1,
                    tensor_space=self._tensor_space,
                )

                # layers.append(LambaBlockWrapper(mamba_block))
                layers.append(mamba_block)

        # Add the language model head
        layers.append(LanguageModelHead(self._config, self._tensor_space))

        return layers


class HybridModel[ConfigType: HybridModelConfig](FastLLMModel[ConfigType]):
    """
    A hybrid model that combines Transformer and Mamba blocks.
    """

    config_class: typing.ClassVar[type[HybridModelConfig]] = HybridModelConfig
    base_model_class: typing.ClassVar[type[HybridBaseModel]] = HybridBaseModel
