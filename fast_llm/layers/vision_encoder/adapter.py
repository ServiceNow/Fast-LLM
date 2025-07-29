import typing

import torch

from fast_llm.engine.base_model.base_model import Layer
from fast_llm.engine.config_utils.tensor_space import TensorSpace
from fast_llm.functional.triton.mlp import torch_mlp_activation
from fast_llm.layers.common.linear import Linear
from fast_llm.layers.transformer.config import TransformerDimNames, TransformerKwargs
from fast_llm.layers.vision_encoder.config import VisionEncoderConfig, VisionEncoderDimNames
from fast_llm.tensor import TensorMeta, init_normal_


class VisionAdapter(Layer):
    """
    Vision adapter layer that projects vision encoder features into the language model token embeddings.
    """

    def __init__(self, config: VisionEncoderConfig, tensor_space: TensorSpace):
        super().__init__()
        input_dim = tensor_space.get_tensor_dim(VisionEncoderDimNames.out_channels)
        self._activation_type = config.adapter_activation_type
        self.layer_1 = Linear(
            input_dim,
            tensor_space.get_tensor_dim(VisionEncoderDimNames.adapter_size),
            bias=True,
            weight_init_method=init_normal_(),
            bias_init_method=init_normal_(),
        )
        self.layer_2 = Linear(
            tensor_space.get_tensor_dim(VisionEncoderDimNames.adapter_size),
            tensor_space.get_tensor_dim(TransformerDimNames.hidden),
            bias=True,
            weight_init_method=init_normal_(),
            bias_init_method=init_normal_(),
        )

    def forward(
        self,
        input_: torch.Tensor,
        kwargs: dict[str, typing.Any],
        losses: dict[str, typing.Any] | None = None,
        metrics: dict[str, typing.Any] | None = None,
    ) -> torch.Tensor:
        if isinstance(input_, TensorMeta):
            return TensorMeta.from_dims(
                kwargs[TransformerKwargs.hidden_dims],
                tensor_name="Vision adapter output",
                dtype=input_.dtype,
            )
        return self.layer_2(
            torch_mlp_activation(input_=self.layer_1(input_), gated=False, activation_type=self._activation_type)
        )
