from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from fast_llm.engine.config_utils.tensor_space import TensorDim, TensorSpace
from fast_llm.layers.common.normalization import LayerNorm, RMSNorm
from fast_llm.layers.ssm.config import MambaConfig
from fast_llm.tensor import TensorMeta

try:
    from ops.triton.layernorm import layer_norm_fn, rms_norm_fn
except ImportError:
    layer_norm_fn, rms_norm_fn = None, None


class MambaBlock(nn.Module):
    def __init__(
        self,
        config: MambaConfig,
        mixer_cls,
        layer_index: int,
        tensor_space: TensorSpace,
        norm_cls=LayerNorm,
    ):

        super().__init__()
        self._layer_index = layer_index
        self.config = config
        self.residual_in_fp32 = config.residual_in_fp32
        self.fused_add_norm = config.fused_add_norm
        self.mixer = mixer_cls(config, layer_idx=layer_index, tensor_space=tensor_space)
        if config.use_module_layernorm and not config.rms_norm:
            self.norm = norm_cls
        else:
            self.norm = norm_cls(TensorDim("D_model", config.hidden_size))
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"
            # assert config.num_mem_heads == 0, 'args.num_mem_heads > 0 only supports fused_add_norm=False' # TODO: ehat are mem_heads? They are implemented in the Zamba code

    def _get_meta(self, tensor: TensorMeta, name: str):
        return TensorMeta.from_dims(tensor.dims, tensor_name=f"{self.name} {name}", dtype=tensor.dtype)

    @property
    def name(self) -> str:
        return f"Mamba block {self._layer_index}"

    def forward(
        self,
        hidden_states: Tensor,
        from_shared_proj: Optional[Tensor] = None,
        from_tf: Optional[Tensor] = None,
        residual: Optional[Tensor] = None,
        inference_params=None,
    ):
        if isinstance(hidden_states, TensorMeta):
            return self._get_meta(hidden_states, "output"), self._get_meta(hidden_states, "residual")
        if not self.fused_add_norm:

            residual = (hidden_states + residual) if residual is not None else hidden_states
            if from_tf is not None:
                hidden_states = self.norm((residual + from_tf).to(dtype=self.norm.weight.dtype))
            else:
                hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))

            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:

            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias if hasattr(self.norm, "bias") else None,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm._eps,
            )

        hidden_states = self.mixer(hidden_states, from_shared_proj=from_shared_proj, inference_params=inference_params)

        return hidden_states, residual
