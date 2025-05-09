import torch

from fast_llm.engine.config_utils.tensor_space import TensorDim, TensorSpace
from fast_llm.layers.transformer.config import TransformerConfig
from fast_llm.layers.transformer.transformer import TransformerLayer
from fast_llm.layers.vision_encoder.config import VisionTransformerDimNames, VisionTransformerKwargs
from fast_llm.tensor import TensorMeta


class VisionTransformerLayer(TransformerLayer):
    """
    A vision transformer layer to encode image patches
    """

    def __init__(
        self,
        config: TransformerConfig,
        tensor_space: TensorSpace,
        layer_index: int,
        return_input: bool = False,
    ):
        super().__init__(config, tensor_space, layer_index, return_input)

        hidden_dim = self._tensor_space.get_tensor_dim(VisionTransformerDimNames.hidden)
        self.norm_1 = self._config.normalization.get_layer(hidden_dim)
        self.norm_2 = self._config.normalization.get_layer(hidden_dim)

        self.norm_1 = self._config.peft.apply_other(self.norm_1)
        self.norm_2 = self._config.peft.apply_other(self.norm_2)

    @property
    def name(self) -> str:
        return f"Vision transformer layer {self._layer_index}"

    def _get_meta(self, tensor: torch.Tensor, name: str, kwargs: dict):
        dims = kwargs[VisionTransformerKwargs.hidden_dims]
        if self._return_input:
            dims = (TensorDim("stacked_input_output", 2),) + dims
        return TensorMeta.from_dims(dims, tensor_name=f"{self.name} {name}", dtype=tensor.dtype)

    # TODO Soham: remove this since we only need to call the parent method
    # def forward(
    #     self,
    #     input_: torch.Tensor,
    #     kwargs: dict[str, typing.Any],
    #     losses: dict[str, typing.Any] | None = None,
    #     metrics: dict[str, typing.Any] | None = None,
    # ) -> torch.Tensor:
    #     if isinstance(input_, TensorMeta):
    #         return self._get_meta(input_, "output", kwargs)
    #     # Hack for now to compute the patch embeddings
    #     kwargs[VisionTransformerKwargs.patch_embeddings] = super().forward(
    #         kwargs.pop(VisionTransformerKwargs.patch_embeddings), kwargs, losses, metrics
    #     )
    #     return input_
