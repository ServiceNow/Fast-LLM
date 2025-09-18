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

    def __init__(
        self,
        config: ConfigType,
        ssm_config: SSMConfig,
        distributed_config: DistributedConfig,
        hidden_dim: TensorDim,
        block_index: int,
        name: str,
        lr_scale: float | None,
        mixer_class: type[BlockLayer],
        return_input: bool = False,
    ):
        self._ssm_config = ssm_config
        self._mixer_class = mixer_class
        super().__init__(config, distributed_config, hidden_dim, block_index, name, lr_scale, return_input)

    @property
    def _mixer_config(self) -> SSMConfig:
        return self._ssm_config
