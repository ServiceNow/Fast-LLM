import torch

from fast_llm.engine.config_utils.tensor_space import TensorSpace
from fast_llm.layers.common.linear import Linear
from fast_llm.layers.transformer.config import TransformerDimNames
from fast_llm.layers.vision_encoder.config import VisionEncoderDimNames
from fast_llm.tensor import init_normal_


class VisionAdapter(torch.nn.Module):
    """
    Vision adapter layer for the LLM.
    """

    def __init__(self, intermediate_size: int, tensor_space: TensorSpace, name: str = "vision_adapter"):
        super().__init__()
        self._name = name
        input_dim = tensor_space.get_tensor_dim(VisionEncoderDimNames.out_channels)
        self.layer_1 = Linear(
            input_dim,
            tensor_space.get_tensor_dim(VisionEncoderDimNames.intermediate_size),
            bias=True,
            weight_init_method=init_normal_(),
            bias_init_method=init_normal_(),
        )
        self.layer_2 = Linear(
            tensor_space.get_tensor_dim(VisionEncoderDimNames.intermediate_size),
            tensor_space.get_tensor_dim(TransformerDimNames.hidden),
            bias=True,
            weight_init_method=init_normal_(),
            bias_init_method=init_normal_(),
        )

    def forward(self, input_: torch.Tensor):
        return self.layer_2(self.layer_1(input_))
