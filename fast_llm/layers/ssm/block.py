from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.layers.block.block import Block, BlockLayer
from fast_llm.layers.block.config import BlockConfig
from fast_llm.layers.ssm.config import SSMConfig


# TODO: Sort out configs.
class SSMBlock[ConfigType: BlockConfig](Block[ConfigType]):
    """
    A transformer-like decoder block with a SSM mixer, see https://arxiv.org/abs/2502.14458
    """

    _name = "Llamba block"

    def __init__(
        self,
        config: ConfigType,
        ssm_config: SSMConfig,
        distributed_config: DistributedConfig,
        hidden_dim: TensorDim,
        mixer_cls: type[BlockLayer],
        block_index: int,
        name: str,
        return_input: bool = False,
    ):
        self._ssm_config = ssm_config
        self._mixer_cls = mixer_cls
        super().__init__(config, distributed_config, hidden_dim, block_index, name, return_input)

    def _create_mixer(self) -> BlockLayer:
        return self._mixer_cls(
            self._ssm_config,
            self._config,
            self._distributed_config,
            self._hidden_dim,
            self._block_index,
            f"{self._name} mixer",
        )
