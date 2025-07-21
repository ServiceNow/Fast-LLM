import typing

from fast_llm.layers.transformer.transformer import BaseBlock, Mixer

if typing.TYPE_CHECKING:
    from fast_llm.engine.config_utils.tensor_space import TensorSpace
    from fast_llm.layers.ssm.config import SSMConfig
    from fast_llm.layers.transformer.config import TransformerConfig


class LlambaBlock(BaseBlock):
    """
    A transformer-like decoder block with a SSM mixer, see https://arxiv.org/abs/2502.14458
    """

    _name = "Llamba block"

    def __init__(
        self,
        transformer_config: "TransformerConfig",
        ssm_config: "SSMConfig",
        tensor_space: "TensorSpace",
        mixer_cls: type[Mixer],
        layer_index: int,
        return_input: bool = False,
    ):
        self._debug_mode = self._config_ssm.debug_ssm
        super().__init__(transformer_config, tensor_space, layer_index, return_input)
        self.mixer = mixer_cls(ssm_config, layer_idx=self._layer_index, tensor_space=self._tensor_space)

    def get_mixer(self) -> Mixer:
        return self.mixer
