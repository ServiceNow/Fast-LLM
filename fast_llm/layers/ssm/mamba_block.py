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
    from ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from mamba_layer import MambaLayer
from mamba2_layer import Mamba2Layer
from fast_llm.layers.ssm.config import MambaConfig


class MambaBlock(nn.Module):
    def __init__(
        self, config: MambaConfig, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):

        super().__init__()
        self.config = config
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(config)
        if config.use_module_layernorm and not config.rms_norm:
            self.norm = norm_cls
        else:
            self.norm = norm_cls(config.hidden_size)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"
            assert config.num_mem_heads == 0, 'args.num_mem_heads > 0 only supports fused_add_norm=False'

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
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
            
        hidden_states = self.mixer(hidden_states, from_shared_proj=from_shared_proj, inference_params=inference_params)
        
        return hidden_states , residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

def create_block(config, layer_idx):
    factory_kwargs = {}
    
    norm_cls = partial(nn.LayerNorm if not config.rms_norm else RMSNorm, eps=config.layernorm_epsilon)
    
    if (not config.layer_mapping) or config.layer_mapping[layer_idx-1][0] == 'r' or config.layer_mapping[layer_idx-1][0] == 'g':
        if (not config.layer_mapping) or len(config.layer_mapping[layer_idx-1]) == 1:
            if 'm' in config.layer_mapping:
                mixer_cls = partial(Mamba2Layer, layer_idx=layer_idx, **factory_kwargs)
            else:
                mixer_cls = partial(MambaLayer, layer_idx=layer_idx, **factory_kwargs)
            block = MambaBlock(
                config,
                mixer_cls=mixer_cls,
                norm_cls=norm_cls,
                fused_add_norm=config.fused_add_norm,
                residual_in_fp32=config.residual_in_fp32,
            )
    elif config.layer_mapping[layer_idx-1][0] == 'm': 
        
        mixer_cls = partial(Mamba2Layer, layer_idx=layer_idx, **factory_kwargs)
        block = MambaBlock(
            config,
            mixer_cls=mixer_cls,
            norm_cls=norm_cls,
            fused_add_norm=config.fused_add_norm,
            residual_in_fp32=config.residual_in_fp32,
            )
    return block

# class MambaDecoder(nn.Module):
#     def __init__(
#         self,
#         config: MambaConfig,
#         post_layer_norm=True,
#         pre_process=True,
#         post_process=True,
#     ):
#         super().__init__()

#         self.config: MambaConfig = config

#         self.post_layer_norm = post_layer_norm
#         self.pre_process = pre_process
#         self.post_process = post_process

#         self.input_tensor = None

#         self.checkpoint_core_block = self.config.recompute_granularity == 'selective'

#         self.num_layers_per_pipeline_rank = (
#             self.config.num_layers
#         )
        
#         self.layer_mapping = config.layer_mapping

#         self.use_mem_rope = config.use_mem_rope
        

#         self._build_layers()

#     def _build_layers(self):
#         num_layers_to_build = self.num_layers_per_pipeline_rank
#         self.layers = torch.nn.ModuleList([create_block(self.config, i + 1) for i in range(num_layers_to_build)])
#         if self.config.num_mem_heads > 0:
#             blocks = []
#             for _ in range(self.config.num_mem_blocks):
#                 blocks.append(create_block(self.config, layer_idx=-1))
#             self.blocks = torch.nn.ModuleList(blocks)
        
#             self.block_map = torch.nn.ModuleList([
#                 nn.Linear(self.config.hidden_size, self.config.hidden_size, bias = self.config.add_bias_linear) if (i%2 == 1 if (self.layer_mapping is None) else self.layer_mapping[i] == 'g') else nn.Identity() for i in range(self.config.num_layers)]) 
#             if self.use_mem_rope:
#                 self.rotary_pos_emb = RotaryEmbedding(
#                         2 * self.config.hidden_size // self.config.num_mem_heads, rotary_percent=1.0, seq_len_interpolation_factor=None
#                     )

#         if self.config.use_low_rank_mamba_proj:
#             blocks = []
#             d_inner = self.config.expansion_factor * self.config.hidden_size
#             nheads = d_inner // self.config.mamba_headdim
#             d_in_proj = 2 * d_inner + 2 * self.config.mamba_ngroups * self.config.state_size + nheads
#             for _ in range(self.config.num_shared_mamba_proj):
#                 blocks.append(nn.Linear(self.config.hidden_size, d_in_proj, bias = self.config.add_bias_linear))
#             self.in_projs = torch.nn.ModuleList(blocks)


#         if self.post_process and self.post_layer_norm:
#             self.final_layernorm = RMSNorm(self.config.hidden_size, eps=self.config.layernorm_epsilon, dtype=torch.float32)
            

#     def _get_layer(self, layer_number):
#         return self.layers[layer_number]

#     def _checkpointed_forward(self, hidden_states, residual, inference_params):
#         """Forward method with activation checkpointing."""

#         def custom(start, end):
#             def custom_forward(*args, **kwargs):
#                 x, residual, *args = args
#                 for index in range(start, end):
#                     layer = self._get_layer(index)
#                     x, residual = layer(x, residual, *args, **kwargs)
#                 return x, residual

#             return custom_forward

#         if self.config.recompute_method == 'uniform':
#             l = 0
#             while l < self.num_layers_per_pipeline_rank:
#                 hidden_states,residual = tensor_parallel.checkpoint(
#                     custom(l, l + self.config.recompute_num_layers),
#                     self.config.distribute_saved_activations,
#                     hidden_states,
#                     residual,
#                     inference_params,
#                 )

#                 l += self.config.recompute_num_layers

#         elif self.config.recompute_method == 'block':
#             for l in range(self.num_layers_per_pipeline_rank):
#                 if l < self.config.recompute_num_layers:
#                     hidden_states, residual = tensor_parallel.checkpoint(
#                         custom(l, l + 1),
#                         self.config.distribute_saved_activations,
#                         hidden_states,
#                         residual,
#                         inference_params,
#                     )
#                 else:
#                     hidden_states = custom(l, l + 1)(hidden_states, residual, inference_params)
#         else:
#             raise ValueError("Invalid activation recompute method.")

#         return hidden_states, residual

#     def set_input_tensor(self, input_tensor):
#         """Set input tensor to be used instead of forward()'s input.

#         When doing pipeline parallelism the input from the previous
#         stage comes from communication, not from the input, so the
#         model's forward_step_func won't have it. This function is thus
#         used by internal code to bypass the input provided by the
#         forward_step_func"""
#         self.input_tensor = input_tensor

#     def forward(self, hidden_states, residual = None, inference_params=None, attention_mask=None):

#         if not self.pre_process:
#             hidden_states = self.input_tensor
            
#         rng_context = nullcontext()
#         fp8_context = nullcontext()

#         with rng_context and fp8_context:
#             residual = None
#             x_orig = torch.clone(hidden_states)
#             from_tf = None
#             block_count = 0
#             rotary_pos_emb=None
#             if self.use_mem_rope:
#                 if inference_params is not None and inference_params.sequence_len_offset > 0:
#                     rotary_seq_len = inference_params.max_sequence_length
#                 else:
#                     rotary_seq_len = hidden_states.shape[1]
#                 rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)
#             for i, layer in enumerate(self.layers):
#                 if self.config.num_mem_heads > 0:
#                     if (i%2 == 1 if (self.layer_mapping is None) else self.layer_mapping[i] == 'g'):
#                         from_tf = self.block_map[i](
#                             self.blocks[block_count % self.config.num_mem_blocks](
#                                 hidden_states, residual, x_orig, inference_params=inference_params, attention_mask = attention_mask, rotary_pos_emb=rotary_pos_emb, forward_layer_idx=block_count
#                             )
#                         )
#                         block_count += 1
#                     else:
#                         from_tf, _ = (None, None)
#                 from_shared_proj = None
#                 if self.config.use_low_rank_mamba_proj:
#                     from_shared_proj = self.in_projs[i % self.config.num_shared_mamba_proj](hidden_states)
#                 hidden_states, residual = layer(
#                     hidden_states=hidden_states,
#                     from_shared_proj=from_shared_proj,
#                     from_tf=from_tf,
#                     residual = residual,
#                     inference_params=inference_params,
#                     attention_mask = attention_mask,
#                 )

#         if self.post_process and self.post_layer_norm:
#             if not self.config.fused_add_norm:
#                 residual = (hidden_states + residual) if residual is not None else hidden_states
#                 hidden_states = self.final_layernorm(residual.to(dtype=self.final_layernorm.weight.dtype))
#             else:
#                 fused_add_norm_fn = rms_norm_fn if isinstance(self.final_layernorm, RMSNorm) else layer_norm_fn
#                 hidden_states = fused_add_norm_fn(
#                     hidden_states,
#                     self.final_layernorm.weight,
#                     self.final_layernorm.bias,
#                     eps=self.final_layernorm.eps,
#                     residual=residual,
#                     prenorm=False,
#                     residual_in_fp32=self.residual_in_fp32,
#                 )

#         return hidden_states
    