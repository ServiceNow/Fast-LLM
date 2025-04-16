import typing

from fast_llm.layers.transformer.transformer import BaseBlock

if typing.TYPE_CHECKING:
    from fast_llm.engine.config_utils.tensor_space import TensorSpace
    from fast_llm.layers.ssm.config import SSMLayerConfig
    from fast_llm.layers.transformer.config import TransformerConfig


class LllambaBlock(BaseBlock):
    """
    A transformer-like decoder block with a SSM mixer, see https://arxiv.org/abs/2502.14458
    """

    name = "Lllamba block"
    _mixer_module_name = "ssm_mixer"

    def __init__(
        self,
        config_transformer: "TransformerConfig",
        config_ssm: "SSMLayerConfig",
        tensor_space: "TensorSpace",
        mixer_cls,
        layer_index: int,
        return_input: bool = False,
    ):

        super().__init__(config_transformer, tensor_space, layer_index, return_input)
        self._config_ssm = config_ssm
        self._debug_mode = self._config_ssm.debug_ssm
        self.mixer_cls = mixer_cls

    def _create_mixer(self):
        self.ssm_mixer = self.mixer_cls(self._config_ssm, layer_idx=self._layer_index, tensor_space=self._tensor_space)
