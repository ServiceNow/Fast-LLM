import typing

from fast_llm.layers.transformer.transformer import BaseBlock

if typing.TYPE_CHECKING:
    from fast_llm.engine.config_utils.tensor_space import TensorSpace
    from fast_llm.layers.ssm.config import SSMConfig
    from fast_llm.layers.transformer.config import TransformerConfig


class LlambaBlock(BaseBlock):
    """
    A transformer-like decoder block with a SSM mixer, see https://arxiv.org/abs/2502.14458
    """

    _name = "Llamba block"
    _mixer_module_name = "mixer"

    def __init__(
        self,
        config_transformer: "TransformerConfig",
        config_ssm: "SSMConfig",
        tensor_space: "TensorSpace",
        mixer_cls,
        layer_index: int,
        return_input: bool = False,
    ):
        self.mixer_cls = mixer_cls
        self._config_ssm = config_ssm
        self._debug_mode = self._config_ssm.debug_ssm
        super().__init__(config_transformer, tensor_space, layer_index, return_input)

    def _create_mixer(self):
        self.mixer = self.mixer_cls(self._config_ssm, layer_idx=self._layer_index, tensor_space=self._tensor_space)
