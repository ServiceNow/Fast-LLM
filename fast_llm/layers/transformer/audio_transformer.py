import torch

from fast_llm.engine.config_utils.tensor_space import TensorDim, TensorSpace
from fast_llm.layers.transformer.config import AudioTransformerDimNames, AudioTransformerKwargs, TransformerConfig
from fast_llm.layers.transformer.transformer import TransformerLayer
from fast_llm.tensor import TensorMeta


class AudioTransformerLayer(TransformerLayer):
    """
    A audio transformer layer to encode image patches
    """

    def __init__(
        self,
        config: TransformerConfig,
        tensor_space: TensorSpace,
        layer_index: int,
        return_input: bool = False,
    ):
        super().__init__(config, tensor_space, layer_index, return_input)

        hidden_dim = self._tensor_space.get_tensor_dim(AudioTransformerDimNames.hidden)

        # use regular layernorm (not rms norm)
        self.norm_1 = self._config.normalization.get_layer(hidden_dim)
        self.norm_2 = self._config.normalization.get_layer(hidden_dim)

        self.norm_1 = self._config.peft.apply_other(self.norm_1)
        self.norm_2 = self._config.peft.apply_other(self.norm_2)

    @property
    def name(self) -> str:
        return f"Audio transformer layer {self._layer_index}"

    def _get_meta(self, tensor: torch.Tensor, name: str, kwargs: dict):
        dims = kwargs[AudioTransformerKwargs.hidden_dims]
        if self._return_input:
            dims = (TensorDim("stacked_input_output", 2),) + dims
        return TensorMeta.from_dims(dims, tensor_name=f"{self.name} {name}", dtype=tensor.dtype)
