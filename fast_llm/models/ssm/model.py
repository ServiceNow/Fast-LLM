import logging
import typing
from functools import partial

import torch
import torch.nn as nn

from fast_llm.data.data.gpt.data import GPTBatch
from fast_llm.engine.base_model.base_model import BaseModel, Layer, LossDef
from fast_llm.engine.config_utils.tensor_space import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig, DistributedDimNames, PhaseType
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.multi_stage.fast_llm_model import FastLLMModel
from fast_llm.engine.schedule.config import BatchConfig
from fast_llm.layers.language_model.config import LanguageModelKwargs, LanguageModelLossNames
from fast_llm.layers.language_model.embedding import WORD_EMBEDDINGS_WEIGHT, LanguageModelEmbedding
from fast_llm.layers.language_model.head import LanguageModelHead
from fast_llm.layers.language_model.preprocessing import PositionEmbeddingPreprocessor
from fast_llm.layers.transformer.config import (
    RoutingType,
    TransformerDimNames,
    TransformerKwargs,
    TransformerLossNames,
)
from fast_llm.layers.transformer.preprocessing import BackupAttentionPreprocessor, RotaryEmbeddingPreprocessor
from fast_llm.layers.transformer.transformer import TransformerLayer
from fast_llm.layers.ssm.mamba_block import MambaBlock
from fast_llm.layers.ssm.mamba_layer import MambaLayer
from fast_llm.models.gpt.config import GPTBaseModelConfig, GPTModelConfig
from fast_llm.models.gpt.megatron import get_init_megatron
from fast_llm.models.gpt.model import GPTBaseModel, GPTModel
from fast_llm.tensor import ParameterMeta, TensorMeta
from fast_llm.utils import Assert

try:
    from ops.triton.layernorm import RMSNorm
except ImportError:
    RMSNorm = None

logger = logging.getLogger(__name__)


class HybridModelConfig(GPTBaseModelConfig):
    """Configuration for a hybrid model with both Transformer and Mamba blocks."""
    
    # Define the pattern of blocks: 't' for transformer, 'm' for mamba
    block_pattern: list[str] = []
    
    # Mamba configuration parameters
    mamba_expansion_factor: int = 2
    mamba_state_size: int = 16
    mamba_conv_dimension: int = 4
    mamba_rms_norm: bool = True
    mamba_residual_in_fp32: bool = True
    mamba_fused_add_norm: bool = False
    mamba_layernorm_epsilon: float = 1e-5
    
    def __post_init__(self):
        super().__post_init__()
        
        # If block pattern is empty, default to all transformer layers
        if not self.block_pattern:
            self.block_pattern = ['t'] * self.transformer.num_layers


class HybridBaseModel(GPTBaseModel[HybridModelConfig]):
    """
    A hybrid model that interleaves Transformer and Mamba blocks.
    """

    config_class: typing.ClassVar[type[HybridModelConfig]] = HybridModelConfig
    _is_setup: bool = False

    def __init__(
        self,
        config: HybridModelConfig,
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
            if block_type not in ['t', 'm']:
                raise ValueError(f"Invalid block type: {block_type}. Must be 't' or 'm'")

    def get_layers(self) -> list[Layer]:
        """
        Create a list of layers for the model, interleaving Transformer and Mamba blocks
        according to the block pattern.
        """
        layers = [LanguageModelEmbedding(self._config, self._tensor_space)]
        
        # Create norm class for Mamba blocks
        norm_cls = partial(
            nn.LayerNorm if not self._config.mamba_rms_norm else RMSNorm, 
            eps=self._config.mamba_layernorm_epsilon
        )
        
        # Create blocks according to pattern
        for i, block_type in enumerate(self._config.block_pattern):
            if block_type == 't':
                # Transformer block
                layers.append(
                    TransformerLayer(
                        self._config.transformer,
                        self._tensor_space,
                        layer_index=i + 1,
                    )
                )
            else:  # block_type == 'm'
                # Create Mamba config from model config
                from fast_llm.layers.ssm.config import MambaConfig
                mamba_config = MambaConfig(
                    hidden_size=self._config.transformer.hidden_size,
                    expansion_factor=self._config.mamba_expansion_factor,
                    state_size=self._config.mamba_state_size,
                    conv_dimension=self._config.mamba_conv_dimension,
                    rms_norm=self._config.mamba_rms_norm,
                    residual_in_fp32=self._config.mamba_residual_in_fp32,
                    fused_add_norm=self._config.mamba_fused_add_norm,
                    layernorm_epsilon=self._config.mamba_layernorm_epsilon,
                    add_bias_linear=self._config.transformer.add_linear_biases,
                )
                
                # Create Mamba block
                mixer_cls = partial(MambaLayer, layer_idx=i)
                mamba_block = MambaBlock(
                    mamba_config,
                    mixer_cls=mixer_cls,
                    norm_cls=norm_cls,
                    fused_add_norm=mamba_config.fused_add_norm,
                    residual_in_fp32=mamba_config.residual_in_fp32,
                )
                
                # Wrap MambaBlock to match Layer interface
                class MambaBlockWrapper(Layer):
                    def __init__(self, mamba_block):
                        super().__init__()
                        self.mamba_block = mamba_block
                        
                    def forward(self, input_, kwargs, losses=None, metrics=None):
                        # Extract residual from kwargs if available
                        residual = kwargs.get('residual', None)
                        inference_params = kwargs.get('inference_params', None)
                        
                        # Call MambaBlock
                        hidden_states, new_residual = self.mamba_block(
                            input_, 
                            residual=residual,
                            inference_params=inference_params
                        )
                        
                        # Store residual for next layer
                        kwargs['residual'] = new_residual
                        
                        return hidden_states
                
                layers.append(MambaBlockWrapper(mamba_block))
        
        # Add the language model head
        layers.append(LanguageModelHead(self._config, self._tensor_space))
        
        return layers


class HybridModel(FastLLMModel[HybridModelConfig]):
    """
    A hybrid model that combines Transformer and Mamba blocks.
    """
    config_class: typing.ClassVar[type[HybridModelConfig]] = HybridModelConfig
    base_model_class: typing.ClassVar[type[HybridBaseModel]] = HybridBaseModel