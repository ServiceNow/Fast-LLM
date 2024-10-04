import logging

import torch

from fast_llm.core.distributed import set_generator
from fast_llm.engine.base_model.base_model import Layer
from fast_llm.engine.config_utils.run import log_pipeline_parallel_main_rank
from fast_llm.engine.config_utils.tensor_space import TensorSpace
from fast_llm.layers.transformer.attention import Attention
from fast_llm.layers.transformer.config import TransformerConfig, TransformerDimNames, TransformerKwargs
from fast_llm.layers.transformer.mixture_of_experts import MixtureOfExpertMLP
from fast_llm.layers.transformer.mlp import MLP
from fast_llm.logging import log_distributed_grad, log_distributed_tensor, log_memory_usage
from fast_llm.tensor import TensorMeta

logger = logging.getLogger(__name__)


class TransformerLayer(Layer):
    """
    A transformer decoder layer.
    """

    def __init__(
        self,
        config: TransformerConfig,
        tensor_space: TensorSpace,
        layer_index: int,
    ):
        super().__init__()
        self._config = config
        self._tensor_space = tensor_space
        self._dropout_p = self._config.hidden_dropout

        self._layer_index = layer_index
        self._debug_mode = self._config.debug_transformer or self._config.debug_transformer_memory

        hidden_dim = self._tensor_space.get_tensor_dim(TransformerDimNames.hidden)
        self.norm_1 = self._config.normalization.get_layer(hidden_dim)
        self.norm_2 = self._config.normalization.get_layer(hidden_dim)

        self.self_attn = Attention(self._config, self._tensor_space, layer_index)

        self.mlp = (MixtureOfExpertMLP if self._config.num_experts > 1 else MLP)(
            self._config, self._tensor_space, f"{self.name} mlp"
        )

    @torch.compile
    def _bias_dropout_add(self, input_, bias, residual):
        if bias is not None:
            input_ = input_ + bias
        return residual + torch.dropout(input_, self._dropout_p, self.training)

    @property
    def name(self):
        return f"Transformer layer {self._layer_index}"

    def _get_meta(self, tensor: torch.Tensor, name: str, kwargs: dict):
        return TensorMeta.from_dims(
            kwargs[TransformerKwargs.hidden_dims], tensor_name=f"{self.name} {name}", dtype=tensor.dtype
        )

    def _debug_log(self, tensor: torch.Tensor | None, name: str, kwargs: dict, *, bias=None):
        if self._config.debug_transformer_memory:
            log_pipeline_parallel_main_rank(lambda: log_memory_usage(f"{self.name} {name}", str))
        if self._config.debug_transformer and tensor is not None:
            # TODO: Local vs global
            log_distributed_tensor(
                "",
                tensor if bias is None else tensor + bias,
                level=self._config.debug_transformer,
                meta=self._get_meta(tensor, name, kwargs),
                distributed=self._tensor_space.distributed,
            )
            log_distributed_grad(
                "",
                tensor,
                level=self._config.debug_transformer,
                meta=self._get_meta(tensor, name + " grad", kwargs),
                distributed=self._tensor_space.distributed,
            )

    def forward(self, input_: torch.Tensor, kwargs: dict, losses: dict | None = None, metrics: dict | None = None):
        if isinstance(input_, TensorMeta):
            return self._get_meta(input_, "output", kwargs)
        generator = (
            self._tensor_space.distributed.tp_generator
            if self._tensor_space.distributed_config.sequence_tensor_parallel
            else self._tensor_space.distributed.pp_generator
        )
        if self._debug_mode:
            self._debug_log(None, "Begin", kwargs)
        hidden_states = self.norm_1(input_)
        if self._debug_mode:
            self._debug_log(hidden_states, "Norm 1", kwargs)
        hidden_states, bias = self.self_attn(hidden_states, kwargs)
        if self._debug_mode:
            self._debug_log(hidden_states, "Attn output", kwargs, bias=bias)
        with set_generator(generator):
            input_ = self._bias_dropout_add(hidden_states, bias, input_)
        if self._debug_mode:
            self._debug_log(input_, "Attn residual", kwargs)
        hidden_states = self.norm_2(input_)
        if self._debug_mode:
            self._debug_log(hidden_states, "Norm 2", kwargs)
        hidden_states, bias = self.mlp(hidden_states, kwargs, losses, metrics)
        if self._debug_mode:
            self._debug_log(hidden_states, "MLP output", kwargs, bias=bias)
        with set_generator(generator):
            hidden_states = self._bias_dropout_add(hidden_states, bias, input_)
        if self._debug_mode:
            self._debug_log(None, "MLP residual", kwargs, bias=bias)
        return hidden_states
