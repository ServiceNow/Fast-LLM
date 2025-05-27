import typing

from fast_llm.layers.ssm.discrete_mamba2 import DiscreteMamba2
from fast_llm.layers.ssm.mamba_layer import MambaLayer
from fast_llm.layers.transformer.transformer import BaseBlock

if typing.TYPE_CHECKING:
    from fast_llm.engine.config_utils.tensor_space import TensorSpace
    from fast_llm.models.hybrid.config import MambaBlockConfig


class LlambaBlock(BaseBlock):
    """
    A transformer-like decoder block with a discrete Mamba 2 mixer, see https://arxiv.org/abs/2502.14458
    """

    _mixer_module_name = "mixer"

    def __init__(
        self,
        config: "MambaBlockConfig",
        tensor_space: "TensorSpace",
        layer_index: int,
        block_name: str = "",
        return_input: bool = False,
    ):
        super().__init__(config, tensor_space, layer_index, block_name, return_input)

    def _create_mixer(self):
        self.mixer = DiscreteMamba2(
            self._config, layer_index=self._layer_index, tensor_space=self._tensor_space, name=self.block_name
        )


class LlambaOneBlock(BaseBlock):
    """
    A transformer-like decoder block with a Mamba 1 mixer.
    """

    _mixer_module_name = "mamba1"

    def __init__(
        self,
        config: "MambaBlockConfig",
        tensor_space: "TensorSpace",
        layer_index: int,
        block_name: str = "",
        return_input: bool = False,
    ):
        super().__init__(config, tensor_space, layer_index, block_name, return_input)

    def _create_mixer(self):
        self.mixer = MambaLayer(
            self._config, layer_index=self._layer_index, tensor_space=self._tensor_space, name=self.block_name
        )
