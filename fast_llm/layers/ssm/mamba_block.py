import math
from typing import Optional, Union
import re
from contextlib import nullcontext
from abc import ABC, abstractmethod
from dataclasses import dataclass
import functools
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
except ImportError:
    selective_scan_fn, mamba_inner_fn = None, None

try:
    from ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from ops.triton.layernorm import layer_norm_fn, rms_norm_fn
except ImportError:
    layer_norm_fn, rms_norm_fn = None, None

from fast_llm.layers.common.normalization import LayerNorm, RMSNorm
from mamba_layer import MambaLayer
from fast_llm.layers.ssm.config import MambaConfig
from fast_llm.engine.config_utils.tensor_space import TensorDim

class MambaBlock(nn.Module):
    def __init__(
        self, config: MambaConfig, mixer_cls, norm_cls=LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):

        super().__init__()
        self.config = config
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(config)
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

    def forward(
        self, hidden_states: Tensor,  from_shared_proj: Optional[Tensor] = None, from_tf: Optional[Tensor] = None, residual: Optional[Tensor] = None, inference_params=None, attention_mask=None
    ):
        
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
                self.norm.bias if hasattr(self.norm, 'bias') else None,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm._eps,
            )
            
        hidden_states = self.mixer(hidden_states, from_shared_proj=from_shared_proj, inference_params=inference_params)
        
        return hidden_states , residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)