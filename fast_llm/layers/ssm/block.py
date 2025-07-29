from fast_llm.engine.config_utils.tensor_space import TensorSpace
from fast_llm.layers.block.block import Block
from fast_llm.layers.block.config import BlockConfig
from fast_llm.layers.block.mixer import Mixer
from fast_llm.layers.ssm.config import SSMConfig


# TODO: Sort out configs.
class SSMBlock[ConfigType: BlockConfig](Block[BlockConfig]):
    """
    A transformer-like decoder block with a SSM mixer, see https://arxiv.org/abs/2502.14458
    """

    _name = "Llamba block"

    def __init__(
        self,
        config: BlockConfig,
        ssm_config: SSMConfig,
        tensor_space: TensorSpace,
        mixer_cls: type[Mixer],
        block_index: int,
        return_input: bool = False,
    ):
        self._ssm_config = ssm_config
        self._mixer_cls = mixer_cls
        super().__init__(config, tensor_space, block_index, return_input)

    def _create_mixer(self) -> Mixer:
        return self._mixer_cls(
            self._ssm_config,
            tensor_space=self._tensor_space,
            block_index=self._block_index,
            block_config=self._config,
        )
