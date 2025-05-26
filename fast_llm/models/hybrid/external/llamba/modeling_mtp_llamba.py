# Copyright (c) 2024, Kevin Li, Aviv Bick.

import json
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from mamba_ssm.utils.generation import GenerationMixin
from torch import Tensor, nn
from transformers.activations import ACT2FN
from transformers.utils.generic import ModelOutput

from .configuration_mtp_llamba import MTPLlambaConfig as LlambaConfig
from .discrete_mamba2 import DiscreteMamba2


class LlamaRMSNorm(nn.Module):
    """LlamaRMSNorm (taken from transformers.models.llama.modeling_llama.LlamaRMSNorm)."""

    def __init__(self, hidden_size, eps=1e-6, factory_kwargs=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, **factory_kwargs))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: torch.Tensor of shape (batch_size, seq_len, hidden_size).

        Returns:
            torch.Tensor of shape (batch_size, seq_len, hidden_size).
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        """Set the extra representation of the module."""
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class LlamaMLP(nn.Module):
    """LlamaMLP (taken from transformers.models.llama.modeling_llama.LlamaMLP)."""

    def __init__(self, hidden_size, intermediate_size, bias, act_fn, factory_kwargs=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias, **factory_kwargs)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias, **factory_kwargs)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias, **factory_kwargs)
        self.act_fn = ACT2FN[act_fn]

    def forward(self, x):
        """
        Args:
            x: torch.Tensor of shape (batch_size, seq_len, hidden_size).

        Returns:
            torch.Tensor of shape (batch_size, seq_len, hidden_size).
        """
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


@dataclass
class CustomMambaCausalLMOutput(ModelOutput):
    """Custom output class for MambaLMHeadModel."""

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    all_hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    last_hidden_state: Optional[torch.FloatTensor] = None


class MTPLlambaLMHeadModel(nn.Module, GenerationMixin, PyTorchModelHubMixin):
    """MambaLM model with a language modeling head on top (linear layer)."""

    def __init__(self, config, initializer_cfg=None, device=None, dtype=None, **kwargs) -> None:
        super().__init__()

        # Load config
        if not isinstance(config, LlambaConfig):
            config = LlambaConfig(**config)
        self.config = config

        # Factory kwargs
        factory_kwargs = {"device": device, "dtype": dtype}

        # Pad vocab size to be a multiple of pad_vocab_size_multiple
        vocab_size = config.vocab_size
        pad_vocab_size_multiple = config.pad_vocab_size_multiple
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        self.config.vocab_size = vocab_size

        # Mixer model
        self.backbone = MixerModel(
            input_size=vocab_size,
            config=self.config,
            initializer_cfg=initializer_cfg,
            **factory_kwargs,
        )

        # MTP heads
        self.mtp_heads = nn.ModuleList(
            [
                Block(
                    config=config,
                    factory_kwargs=factory_kwargs,
                    layer_idx=layer_idx,
                ).to(device)
                for layer_idx in range(config.n_layer, config.n_layer + config.prediction_heads - 1)
            ]
        )

        self.mtp_norms = nn.ModuleList(
            [
                LlamaRMSNorm(config.d_model, eps=config.norm_epsilon, factory_kwargs=factory_kwargs)
                for _ in range(config.prediction_heads - 1)
            ]
        )
        # LM head
        if not self.config.tie_embeddings:
            self.lm_head = nn.Linear(
                in_features=self.config.d_model,
                out_features=self.config.vocab_size,
                bias=self.config.lm_head_bias,
                **factory_kwargs,
            )
        else:
            self.lm_head = lambda x: x @ self.backbone.embedding.weight.t()

    def allocate_inference_cache(self, *args, **kwargs):
        """Allocate inference cache for the model."""

        mtps = {
            i + self.config.n_layer: layer.allocate_inference_cache(*args, **kwargs)
            for i, layer in enumerate(self.mtp_heads)
        }
        return {**self.backbone.allocate_inference_cache(*args, **kwargs), **mtps}

    def forward(
        self,
        input_ids,
        position_ids=None,
        return_hidden_states=False,
        return_logits=True,
        inference_params=None,
        num_last_tokens=0,
    ):
        """
        Args:
            input_ids: torch.Tensor of shape (batch_size, seq_len),
            position_ids: torch.Tensor of shape (batch_size, seq_len), optional, not used (just for compatibility),
            return_hidden_states: bool, optional,
            return_logits: bool, optional, whether to compute the logits with the LM head,
            inference_params: dict, optional, the model's inference cache,
            num_last_tokens: int, optional. If > 0, only return the logits for the last n tokens.

        Returns:
            CustomMambaCausalLMOutput.

        """
        outputs = self.backbone(
            input_ids,
            return_hidden_states=return_hidden_states,
            inference_params=inference_params,
            position_ids=position_ids,
        )

        # MTP heads processing
        latents = []
        hidden_states = outputs["last_hidden_state"]
        hidden_states_before_last = outputs["hidden_state_before_last"]

        # last layer already has layer norm applied
        latents.append(hidden_states)

        # Process through MTP heads
        for i, mtp_head in enumerate(self.mtp_heads):
            mtp_outputs = mtp_head(
                hidden_states_before_last,
                inference_params=inference_params,
                position_ids=position_ids,
            )
            mtp_hidden_states = mtp_outputs["hidden_states"]
            latents.append(self.mtp_norms[i](mtp_hidden_states))

        # Stack the latents to get (batch_size, seq_len, num_prediction_heads, hidden_size)
        stacked_latents = torch.stack(latents, dim=-2)

        if return_logits:
            if isinstance(self.lm_head, nn.Linear):
                # Apply lm_head to each prediction head's output
                logits = self.lm_head(stacked_latents).float()
            else:
                # Using the tied embedding weights
                logits = self.lm_head(stacked_latents)

            outputs["logits"] = logits if num_last_tokens == 0 else logits[:, -num_last_tokens:]
        else:
            outputs["logits"] = None

        return CustomMambaCausalLMOutput(
            loss=None,
            logits=outputs["logits"],
            all_hidden_states=outputs["all_hidden_states"],
            last_hidden_state=stacked_latents,
        )

    def save_pretrained(self, save_directory):
        """
        Minimal implementation of save_pretrained for MambaLMHeadModel.
        Save the model and its configuration file to a directory.
        """
        # Ensure save_directory exists
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)

        # Save the configuration of the model
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f)


class MixerModel(nn.Module):
    """Mixer model with a stack of Mixer layers."""

    def __init__(self, input_size, config=None, device=None, dtype=None, **kwargs) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(input_size, self.config.d_model, **factory_kwargs)

        self.layers = nn.ModuleList(
            [
                Block(
                    config=config,
                    factory_kwargs=factory_kwargs,
                    layer_idx=i,
                ).to(device)
                for i in range(self.config.n_layer)
            ]
        )

        self.final_layernorm = LlamaRMSNorm(
            hidden_size=self.config.d_model,
            eps=self.config.norm_epsilon,
            factory_kwargs=factory_kwargs,
        )

        return

    def allocate_inference_cache(self, *args, **kwargs):
        """Allocate inference cache for the model."""
        return {i: layer.allocate_inference_cache(*args, **kwargs) for i, layer in enumerate(self.layers)}

    def forward(
        self,
        input_ids,
        return_hidden_states=False,
        inference_params=None,
        position_ids=None,
    ):
        """Run the model."""
        # Start running the layers
        hidden_states = self.embedding(input_ids)

        # Initialize outputs
        outputs = {
            "last_hidden_state": None,
            "hidden_state_before_last": None,
            "all_hidden_states": (hidden_states,) if return_hidden_states else (),
        }

        # Run the layers
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                inference_params=inference_params,
                position_ids=position_ids,
            )
            if layer == self.layers[-1]:
                outputs["hidden_state_before_last"] = hidden_states
            # Record outputs
            hidden_states = layer_outputs["hidden_states"]
            if return_hidden_states:
                outputs["all_hidden_states"] += (hidden_states,)

        # Last layer, apply layer norm
        outputs["last_hidden_state"] = self.final_layernorm(hidden_states)
        return outputs


class Block(nn.Module):
    """
    Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection.

    This Block has a slightly different structure compared to a regular
    prenorm Transformer block.
    The standard block is: LN -> MHA/MLP -> Add.
    [Ref: https://arxiv.org/abs/2002.04745]
    Here we have: Add -> LN -> Mixer, returning both
    the hidden_states (output of the mixer) and the residual.
    This is purely for performance reasons, as we can fuse add and LayerNorm.
    The residual needs to be provided (except for the very first block).
    """

    def __init__(self, config, factory_kwargs, layer_idx, **kwargs):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Mixer
        self.mixer = DiscreteMamba2(
            d_model=self.config.d_model,
            layer_idx=layer_idx,
            **config.ssm_cfg,
            **factory_kwargs,
        )

        # Other components
        self.input_layernorm = LlamaRMSNorm(hidden_size=self.config.d_model, eps=1e-5, factory_kwargs=factory_kwargs)
        self.post_attention_layernorm = LlamaRMSNorm(
            hidden_size=self.config.d_model, eps=1e-5, factory_kwargs=factory_kwargs
        )
        self.mlp = LlamaMLP(
            hidden_size=self.config.d_model,
            **config.mlp_cfg,
            factory_kwargs=factory_kwargs,
        )

    def forward(
        self,
        hidden_states: Tensor,
        inference_params=None,
        **kwargs,
    ):
        """
        Pass the input through the encoder layer.

        Args:
            hidden_states: torch.Tensor of shape (batch_size, seq_len, hidden_size),
            inference_params: dict, optional,

        Returns:
            dict with keys:
                hidden_states: torch.Tensor of shape (batch_size, seq_len, hidden_size),
                mamba_hidden_states: torch.Tensor of shape (batch_size, seq_len, hidden_size),
                transfer_matrix: torch.Tensor of shape (batch_size, seq_len, seq_len).
        """
        outputs = {}

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Apply Mixer
        mixer_outputs = self.mixer(
            hidden_states,
            inference_params=inference_params,
        )

        hidden_states = mixer_outputs["hidden_states"].to(residual.dtype) + residual

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs["hidden_states"] = hidden_states

        return outputs

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        """Allocate inference cache for the model."""
        if getattr(self.mixer, "allocate_inference_cache", None) is None:
            return
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
