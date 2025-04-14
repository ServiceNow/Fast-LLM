import typing

import torch

from fast_llm.layers.transformer.transformer import BaseBlock

if typing.TYPE_CHECKING:
    from fast_llm.engine.config_utils.tensor_space import TensorSpace
    from fast_llm.layers.ssm.config import SSMLayerConfig
    from fast_llm.layers.transformer.config import TransformerConfig


class LambaBlock(BaseBlock):
    def __init__(
        self,
        config_transformer: "TransformerConfig",
        config_ssm: "SSMLayerConfig",
        tensor_space: "TensorSpace",
        mixer_cls,
        layer_index: int,
        return_input: bool = False,
    ):

        super().__init__(
            config_transformer, tensor_space, layer_index, return_input, name="Lamba block", mixer_name="SSM"
        )
        self._config_ssm = config_ssm
        self._debug_mode = self._config_ssm.debug_ssm
        self.mixer = mixer_cls(self._config_ssm, layer_idx=layer_index, tensor_space=tensor_space)

    def mixer_forward(
        self, hidden_states: torch.Tensor, kwargs: dict[str, typing.Any]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.mixer(hidden_states, **kwargs), None
