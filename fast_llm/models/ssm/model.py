import logging
import typing

from fast_llm.layers.attention.block import TransformerBlock
from fast_llm.layers.ssm.block import SSMBlock
from fast_llm.models.gpt.model import GPTBaseModel, GPTInferenceRunner, GPTModel
from fast_llm.models.ssm.config import HybridSSMBaseModelConfig, HybridSSMModelConfig, SSMBlockType

logger = logging.getLogger(__name__)


class HybridSSMBaseModel[ConfigType: HybridSSMBaseModelConfig](GPTBaseModel[ConfigType]):
    """
    A hybrid model that interleaves Transformer and Mamba blocks.
    Right now only LlambaBlock is supported.
    As for the mixer, transformer uses MHA. For the LlambaBlock we support Mamba1 and discrete mamba2.
    """

    def _get_block(
        self,
        block_index: int,
        name: str,
        return_input: bool = False,
    ):
        if block_index > self._config.transformer.num_layers:
            # MTP block
            block_type = self._config.default_mtp_type or self._config.hybrid_block_layout[-1]
        else:
            # Decoder block
            block_type = self._config.hybrid_block_layout[block_index - 1]

        lr_scale = (
            None
            if self._config.transformer.per_layer_lr_scale is None
            else self._config.transformer.per_layer_lr_scale[block_index]
        )

        if block_type == SSMBlockType.transformer:
            return TransformerBlock(
                self._config.transformer,
                self._distributed_config,
                self._hidden_dim,
                block_index,
                name,
                lr_scale,
                return_input,
            )
        else:
            return SSMBlock(
                self._config.transformer,
                self._config.ssm,
                self._distributed_config,
                self._hidden_dim,
                block_index,
                name,
                lr_scale,
                return_input,
            )


class HybridSSMModel[ConfigType: HybridSSMModelConfig](GPTModel[ConfigType]):
    """
    A hybrid model that combines Transformer and SSM blocks.
    """

    base_model_class: typing.ClassVar[type[HybridSSMBaseModel]] = HybridSSMBaseModel


class HybridSSMInferenceRunner(GPTInferenceRunner):
    model_class: typing.ClassVar[type[HybridSSMModel]] = HybridSSMModel
