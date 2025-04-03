import typing

import torch

from fast_llm.engine.base_model.base_model import Layer
from fast_llm.engine.config_utils.tensor_space import TensorSpace
from fast_llm.layers.transformer.config import TransformerDimNames
from fast_llm.layers.transformer.mlp import MLP
from fast_llm.logging import log_distributed_grad, log_distributed_tensor
from fast_llm.models.ssm.config import HybridBaseModelConfig
from fast_llm.tensor import TensorMeta


class LambaBlock(Layer):
    def __init__(self, config: HybridBaseModelConfig, mixer_cls, layer_index: int, tensor_space: TensorSpace):

        super().__init__()
        self._layer_index = layer_index
        self._config = config
        self._tensor_space = tensor_space
        self._ssm_config = config.ssm

        self._debug_mode = self._ssm_config.debug_ssm
        self.mixer = mixer_cls(self._ssm_config, layer_idx=layer_index, tensor_space=tensor_space)

        hidden_dim = self._tensor_space.get_tensor_dim(TransformerDimNames.hidden)
        self.norm_1 = self._config.transformer.normalization.get_layer(hidden_dim)
        self.norm_2 = self._config.transformer.normalization.get_layer(hidden_dim)

        self.mlp = MLP(self._config.transformer, self._tensor_space, f"{self.name} mlp")

    def _get_meta(self, tensor: TensorMeta, name: str):
        return TensorMeta.from_dims(tensor.dims, tensor_name=f"{self.name} {name}", dtype=tensor.dtype)

    @property
    def name(self) -> str:
        return f"Lamba block {self._layer_index}"

    def _debug_log(self, tensor: torch.Tensor | None, name: str, kwargs: dict[str, typing.Any], *, bias=None) -> None:
        if self._debug_mode and tensor is not None:
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

    def forward(
        self,
        input_: torch.Tensor,
        kwargs: dict[str, typing.Any],
        losses: dict[str, typing.Any] | None = None,
        metrics: dict[str, typing.Any] | None = None,
    ):
        if isinstance(input_, TensorMeta):
            return self._get_meta(input_, "output")

        residual = input_
        input_ = self.norm_1(input_)
        if self._debug_mode:
            self._debug_log(input_, "Norm 1", kwargs)

        from_shared_proj = kwargs.get("from_shared_proj", False)
        inference_params = kwargs.get("inference_params", None)

        mixer_outputs = self.mixer(input_, from_shared_proj=from_shared_proj, inference_params=inference_params)

        hidden_states = mixer_outputs + residual

        residual = hidden_states
        hidden_states = self.norm_2(hidden_states)
        if self._debug_mode:
            self._debug_log(hidden_states, "Norm 2", kwargs)

        hidden_states, bias = self.mlp(hidden_states, kwargs, losses, metrics)
        hidden_states = residual + hidden_states

        return hidden_states
