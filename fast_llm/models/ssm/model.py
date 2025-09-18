import logging
import typing

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

        if block_type == SSMBlockType.transformer:
            block_config = self._config.transformer
        else:
            block_config = self._config.transformer.from_dict(self._config.transformer, {"mixer": self._config.ssm})

        return block_config.get_layer(
            self._distributed_config,
            hidden_dim=self._hidden_dim,
            lr_scale=None,
            peft=self._config.peft,
            return_input=return_input,
        )


class HybridSSMModel[ConfigType: HybridSSMModelConfig](GPTModel[ConfigType]):
    """
    A hybrid model that combines Transformer and SSM blocks.
    """

    base_model_class: typing.ClassVar[type[HybridSSMBaseModel]] = HybridSSMBaseModel


class HybridSSMInferenceRunner(GPTInferenceRunner):
    model_class: typing.ClassVar[type[HybridSSMModel]] = HybridSSMModel
