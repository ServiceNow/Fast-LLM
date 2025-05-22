import torch

from fast_llm.engine.config_utils.tensor_space import TensorDim
from fast_llm.layers.transformer.config import VisionTransformerKwargs
from fast_llm.layers.transformer.transformer import TransformerLayer
from fast_llm.tensor import TensorMeta


class VisionTransformerLayer(TransformerLayer):
    _name: str = "Vision transformer layer"

    def _get_meta(self, tensor: torch.Tensor, name: str, kwargs: dict):
        dims = kwargs[VisionTransformerKwargs.hidden_dims]
        if self._return_input:
            dims = (TensorDim("stacked_input_output", 2),) + dims
        return TensorMeta.from_dims(dims, tensor_name=f"{self.name} {name}", dtype=tensor.dtype)
