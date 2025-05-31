# Copyright (c) 2024, Kevin Li, Aviv Bick.

import json
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from einops import rearrange, repeat
from mamba_ssm.ops.triton.selective_state_update import selective_state_update
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from torch import Tensor, nn
from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.utils.generic import ModelOutput

from .configuration_mtp_llamba import MTPLlambaConfig as LlambaConfig


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


class MTPLlambaLMHeadModel(PreTrainedModel):  # PyTorchModelHubMixin removed for now
    """MambaLM model with a language modeling head on top (linear layer)."""

    config_class = LlambaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def __init__(self, config, initializer_cfg=None, device=None, dtype=None, **kwargs) -> None:
        super().__init__(config)

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
        return_all_prediction_heads=False,
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
        if return_all_prediction_heads:
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
        else:
            assert len(latents) == 1
            stacked_latents = latents[0]

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


def segsum(x):
    """More stable segment sum calculation."""
    # [1, 2, 3]
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    # [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    # [[0, 0, 0], [2, 0, 0], [3, 3, 0]]
    x_segsum = torch.cumsum(x, dim=-2)
    # [[0, 0, 0], [2, 0, 0], [5, 3, 0]]
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def materialize_mixer(A_log, B, C, D):
    """
    Since the transfer matrix will be equated to the attention matrix,
    we need to support the form: torch.matmul(attn_weights, value_states).
    Thus, y = torch.matmul(T, X)
    Arguments:
        A_log: (batch, length, n_heads)
        B: (batch, length, n_heads, d_state)
        C: (batch, length, n_heads, d_state)
    Return:
        T: (batch, n_heads, length, length)
    """
    batch_size, length, n_heads, d_state = B.shape
    assert A_log.shape == (batch_size, length, n_heads)
    assert B.shape == C.shape == (batch_size, length, n_heads, d_state)

    # Compute:
    A_log = rearrange(-F.softplus(A_log), "b l h -> b h l")
    powers = torch.exp(segsum(A_log))
    T = torch.einsum("blhn,bshn,bhls->bhsl", C, B, powers)

    # Add D:
    if D is not None:
        T[:, :, torch.arange(length), torch.arange(length)] += D.view(1, n_heads, 1)

    T = rearrange(T, "b h z l -> b h l z")
    return T


class DiscreteMamba2(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=64,
        n_qk_heads=32,
        n_v_heads=32,
        d_conv=4,
        expand=1,
        activation="identity",
        bias=False,
        conv_bias=True,
        chunk_size=128,
        layer_idx=None,
        device=None,
        dtype=None,
        d_inner=None,
        **kwargs,  # Absorb kwarg for general module
    ):
        """
        See the class .kernel.SSKernel for the kernel constructor which accepts kernel_args.
        Relevant options that are worth considering and tuning include "mode" + "measure", "dt_min", "dt_max", "lr"

        Other options are all experimental and should not need to be configured
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = self.expand * self.d_model if d_inner is None else d_inner
        self.n_qk_heads = n_qk_heads
        self.n_v_heads = n_v_heads
        self.headdim = self.d_inner // self.n_v_heads
        assert self.n_v_heads == self.d_inner // self.headdim
        assert self.d_inner % self.headdim == 0
        assert self.n_v_heads % self.n_qk_heads == 0
        self.activation = activation
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx
        self.bias = bias
        self.kwargs = kwargs

        # Projections
        self.in_proj = nn.Linear(
            self.d_model,
            2 * self.d_inner + 2 * self.n_qk_heads * self.d_state + self.n_v_heads,
            bias=bias,
            **factory_kwargs,
        )
        self.z_bias = (
            nn.Parameter(torch.zeros(self.d_inner, **factory_kwargs)) if not bias else 0
        )  # make sure z_bias always exists

        # Convolutional layer
        conv_dim = self.d_inner + 2 * self.n_qk_heads * self.d_state
        self.conv_bias = conv_bias
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        # Activation after conv
        if self.activation == "identity":
            self.act = nn.Identity()
        elif self.activation in ["silu", "swish"]:
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation {self.activation}")

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.n_v_heads, **factory_kwargs))
        self.D._optim = {"weight_decay": 0.0}

        # out_proj
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    @property
    def d_output(self):
        return self.d_model

    @property
    def state_to_tensor(self):
        return self.layer.state_to_tensor

    def forward(self, u, return_mixer_matrix=False, inference_params=None, **kwargs):
        """
        u: (B, L, D)
        Returns: same shape as u
        """
        outputs = {}
        # assert state is None
        batch, seqlen, dim = u.shape

        state = None
        if inference_params is not None:
            state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # States are updated inplace
                u = u.squeeze(1) if len(u.shape) == 3 else u
                out, _ = self.step(u, state)
                out = out.unsqueeze(1) if len(u.shape) == 2 else out
                return {"hidden_states": out}

        # Hacky way to initialize state during inference
        chunk_size = self.chunk_size if state is None else seqlen

        # Pad input to nearest multiple of chunklen
        padded_len = (1 + (seqlen - 1) // chunk_size) * chunk_size
        u = F.pad(u, (0, 0, 0, padded_len - seqlen))

        # Project input
        xBCzA_log = self.in_proj(u)
        xBC, z, A_log = torch.split(
            xBCzA_log,
            [
                self.d_inner + 2 * self.n_qk_heads * self.d_state,
                self.d_inner,
                self.n_v_heads,
            ],
            dim=-1,
        )

        if state is not None:
            # If we just take xBC[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
            # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
            xBC_t = rearrange(xBC[:, :seqlen, :], "b l d -> b d l")
            state["conv"].copy_(F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0)))  # Update state (B D W)

        # Convolutional layer
        xBC = self.convolutional_forward(xBC, padded_len)

        x, B, C = torch.split(
            xBC,
            [
                self.d_inner,
                self.n_qk_heads * self.d_state,
                self.n_qk_heads * self.d_state,
            ],
            dim=-1,
        )

        x = rearrange(x, "b l (h n) -> b l h n", h=self.n_v_heads)
        B = rearrange(B, "b l (h n) -> b l h n", h=self.n_qk_heads)
        C = rearrange(C, "b l (h n) -> b l h n", h=self.n_qk_heads)

        # SSM forward
        result = mamba_chunk_scan_combined(
            x=x / F.softplus(A_log).to(x.dtype).unsqueeze(-1),
            dt=A_log,
            dt_softplus=True,
            A=-torch.ones(self.n_v_heads, device=A_log.device),
            B=B,
            C=C,
            chunk_size=chunk_size,
            # initial_states=(state["ssm"] if state is not None else None), # currently not supported by mamba_ssm.utils.generation
            return_final_states=(state is not None),
        )

        if state is not None:
            y, ssm_state = result
            state["ssm"].copy_(ssm_state)
        else:
            y = result

        Du = torch.einsum("h,blhp->blhp", self.D, x)
        y = rearrange(y + Du, "b l h p -> b l (h p)")

        # Norm and gate
        out = self.out_proj(y * F.silu(z + self.z_bias))
        outputs["hidden_states"] = out[:, :seqlen, :]

        if return_mixer_matrix:
            outputs["transfer_matrix"] = materialize_mixer(A_log=A_log, B=B, C=C, D=self.D)[..., :seqlen, :seqlen]
        return outputs

    def step(self, u, state, **kwargs):
        """
        u: (B D)
        state: dict of states
        Returns: same shape as u
        """

        # Project input
        xBCzA_log = self.in_proj(u.squeeze(1))
        xBC, z, A_log = torch.split(
            xBCzA_log,
            [
                self.d_inner + 2 * self.n_qk_heads * self.d_state,
                self.d_inner,
                self.n_v_heads,
            ],
            dim=-1,
        )

        xBC, conv_state = self.convolutional_step(xBC, state["conv"])
        state["conv"].copy_(conv_state)  # update state in place

        x, B, C = torch.split(
            xBC,
            [
                self.d_inner,
                self.n_qk_heads * self.d_state,
                self.n_qk_heads * self.d_state,
            ],
            dim=-1,
        )

        x = rearrange(x, "b (h s) -> b h s", h=self.n_v_heads)
        B = rearrange(B, "b (h s) -> b h s", h=self.n_qk_heads)
        C = rearrange(C, "b (h s) -> b h s", h=self.n_qk_heads)

        state["ssm"] = state["ssm"].to(x.dtype)
        zeros = torch.zeros((self.n_v_heads, self.headdim), device=A_log.device).to(dtype=x.dtype)
        ones = torch.ones((self.n_v_heads, self.headdim, self.d_state), device=A_log.device).to(dtype=x.dtype)
        y = selective_state_update(
            x=x / F.softplus(A_log).to(x.dtype).unsqueeze(-1),
            dt=repeat(A_log, "b h -> b h p", p=self.headdim),
            dt_softplus=True,
            A=-ones,
            B=B,
            C=C,
            state=state["ssm"],  # will be updated in place
            dt_bias=zeros,
            D=zeros,
        )

        y = y + self.D[:, None] * x
        y = rearrange(y, "b h p -> b (h p)")

        # Norm and gate
        out = self.out_proj(y * F.silu(z + self.z_bias))

        return out, state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.in_proj.weight.device
        # conv_state:
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size,
            self.d_conv,
            self.conv1d.weight.shape[0],
            device=device,
            dtype=conv_dtype,
        ).transpose(1, 2)
        # ssm_state:
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size,
            self.n_v_heads,
            self.headdim,
            self.d_state,
            device=device,
            dtype=ssm_dtype,
        )
        return {"conv": conv_state, "ssm": ssm_state}

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        """
        conv_state: (batch, d_conv, conv1d.weight.shape[0])
        ssm_state: (batch, n_qk_heads, headdim, d_state)
        """
        assert self.layer_idx is not None
        # Allocate memory if not exists
        if self.layer_idx not in inference_params.key_value_memory_dict:
            inference_params.key_value_memory_dict[self.layer_idx] = self.allocate_inference_cache(
                batch_size, inference_params.max_seqlen, dtype=torch.float32
            )
        # Get states
        states = inference_params.key_value_memory_dict[self.layer_idx]
        if initialize_states:
            states["conv"].zero_()
            states["ssm"].zero_()
        return states

    def convolutional_forward(self, xBC, padded_len):
        if causal_conv1d_fn is None or self.activation not in [
            "silu",
            "swish",
            "identity",
        ]:
            xBC = self.act(self.conv1d(xBC.transpose(1, 2))[..., :padded_len].transpose(1, 2))
        else:
            xBC = causal_conv1d_fn(
                xBC.transpose(1, 2),
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                activation=None if self.activation == "identity" else self.activation,
            ).transpose(1, 2)
        return xBC

    def convolutional_step(self, xBC, conv_state):
        # Convolutional layer
        conv_state = conv_state.to(xBC.dtype)
        if causal_conv1d_update:
            xBC = causal_conv1d_update(
                xBC,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation if self.activation != "identity" else None,
            )
        else:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = xBC
            xBC = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv_bias:
                xBC = xBC + self.conv1d.bias
            xBC = self.act(xBC).to(xBC.dtype)  # Some activations change dtype

        return xBC, conv_state
