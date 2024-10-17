import logging
import copy
import torch
from torch import nn

from fast_llm.layers.common.linear import Linear
from fast_llm.layers.transformer.config import TransformerDimNames, TransformerKwargs
from fast_llm.layers.multimodal_model.config import MultimodalModelBaseConfig, MultimodalModelDimNames, MultimodalModelKwargs
from fast_llm.layers.language_model.config import LanguageModelBaseConfig
from fast_llm.tensor import ParameterMeta, TensorMeta, TensorSpace, TensorDim, init_normal_

logger = logging.getLogger(__name__)

class Adapter(torch.nn.Module):
    
    # Ensure the layer is on its own stage.
    layer_count: float = 1000.0
    
    def __init__(
        self,
        config: LanguageModelBaseConfig,
        tensor_space: TensorSpace,
    ):
        super(Adapter, self).__init__()
        self._distributed_config = tensor_space.distributed_config
        self._tensor_space = tensor_space
        self._residual_dtype = (
            self._distributed_config.optimization_dtype
            if config.transformer.full_precision_residual
            else self._distributed_config.training_dtype
        ).torch

        in_dim = self._tensor_space.get_tensor_dim(MultimodalModelDimNames.image_encoder_hidden_size)
        out_dim = self._tensor_space.get_tensor_dim(TransformerDimNames.hidden)

        self.dropout = nn.Dropout(p=0.1)
        self.adapter_fc = Linear(
            in_dim,
            out_dim,
            bias=True,
            weight_init_method=init_normal_(std=config.transformer.init_method_std),
        )

    def _forward(self, input_: torch.Tensor, losses: dict | None = None, metrics: dict | None = None):
        hidden_states = self.dropout(input_)
        out = self.adapter_fc(hidden_states)

        return out.to(dtype=self._residual_dtype)

    def forward(self, input_, kwargs, losses: dict | None = None, metrics: dict | None = None):
        if isinstance(input_, TensorMeta):
            return TensorMeta.from_dims(
                kwargs[MultimodalModelKwargs.adapter_hidden_dims],
                tensor_name="Adapter output",
                dtype=self._residual_dtype,
            )
        
        return self._forward(input_)