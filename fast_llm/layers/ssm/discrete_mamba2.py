import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

from fast_llm.engine.config_utils.tensor_space import TensorDim, TensorSpace
from fast_llm.layers.common.linear import Linear
from fast_llm.layers.ssm.config import MambaConfig, SSMDimNames
from fast_llm.layers.ssm.mamba_layer import kaiming_init
from fast_llm.tensor import ParameterMeta, init_ones_, init_zeros_

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None


try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

"""
This is adapted fropm https://github.com/cartesia-ai/edge/blob/main/cartesia-pytorch/cartesia_pytorch/Llamba/mixers/discrete_mamba2.py
"""


class DiscreteMamba2(nn.Module):
    """DiscreteMamba2 (taken github.com/goombalab/phi-mamba.git)."""

    def __init__(
        self,
        config: MambaConfig,
        layer_idx: int,
        tensor_space: TensorSpace,
    ):
        """
        See the class .kernel.SSKernel for the kernel constructor which accepts kernel_args.
        TODO: check what this comment means
        Relevant options that are worth considering and tuning include "mode" + "measure", "dt_min", "dt_max", "lr".

        Other options are all experimental and should not need to be configured.
        """
        factory_kwargs = {"device": "meta"}  # , "dtype": torch.bfloat16}
        super().__init__()
        self.config: MambaConfig = config
        bias = config.add_bias_linear
        self.layer_idx = layer_idx

        td_inner = tensor_space.get_tensor_dim(SSMDimNames.d_inner)
        td_state = tensor_space.get_tensor_dim(SSMDimNames.d_state)
        td_model = tensor_space.get_tensor_dim(SSMDimNames.d_model)
        td_conv = tensor_space.get_tensor_dim(SSMDimNames.d_conv)
        td_n_qk_heads = tensor_space.get_tensor_dim(SSMDimNames.n_qk_heads)
        td_n_v_heads = tensor_space.get_tensor_dim(SSMDimNames.n_v_heads)
        td_conv_kernel = tensor_space.get_tensor_dim(SSMDimNames.d_conv_kernel)
        td_inner_proj = tensor_space.get_tensor_dim(SSMDimNames.d_inner_proj)

        self.d_model = td_model.size
        self.d_inner = td_inner.size
        self.d_state = td_state.size
        self.chunk_size = config.chunk_size
        self.n_qk_heads = td_n_qk_heads.size
        self.n_v_heads = td_n_v_heads.size
        self.conv_kernel_size = td_conv_kernel.size

        self.activation = config.activation
        if self.activation == "silu":
            self.act = nn.SiLU()
        elif self.activation == "identity":
            self.act = nn.Identity()
        else:
            raise ValueError(f"Activation {self.activation} not supported")

        # TODO: double check innitializations
        # Projections
        self.in_proj = Linear(
            td_model, td_inner_proj, bias=bias, weight_init_method=kaiming_init(td_model.size, td_inner_proj.size)
        )
        self.z_bias = (
            ParameterMeta(
                torch.zeros(td_inner.size, **factory_kwargs),
                dims=(td_inner,),
                weight_decay=False,
                init_method=init_zeros_,
            )
            if not bias
            else 0.0
        )

        # Convolutional layer
        self.conv1d_weight = ParameterMeta(
            torch.empty(td_conv.size, 1, td_conv_kernel.size, **factory_kwargs),
            dims=(td_conv, TensorDim("1", 1), td_conv_kernel),
            init_method=kaiming_init(td_conv.size, td_conv_kernel.size),
        )
        self.conv1d_bias = ParameterMeta(
            torch.empty(td_conv.size, **factory_kwargs),
            dims=(td_conv,),
            init_method=kaiming_init(td_conv.size, 1),
        )

        # D "skip" parameter
        self.D = ParameterMeta(
            torch.empty(td_n_qk_heads.size, **factory_kwargs),
            dims=(td_n_qk_heads,),
            weight_decay=False,
            init_method=init_ones_,
        )

        # out_proj
        self.out_proj = Linear(
            td_inner,
            td_model,
            bias=bias,
            weight_init_method=kaiming_init(td_inner.size, td_model.size),
        )

    @property
    def d_output(self):
        """Returns the output dimension of the model."""
        return self.d_model

    @property
    def state_to_tensor(self):
        """Returns the state of the model as a tensor."""
        return self.layer.state_to_tensor

    def forward(self, u, inference_params=None, **kwargs):
        """
        Args:
            u: (B, L, D),
            inference_params: dict.

        Returns:
            outputs: dict.
            outputs["hidden_states"]: (B, L, D).
            outputs["state"]: inference cache.
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
            state["conv"].copy_(F.pad(xBC_t, (self.conv_kernel_size - xBC_t.shape[-1], 0)))  # Update state (B D W)

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

        # TODO: since we do not support inference for now, we only return the hidden states for now.
        return outputs["hidden_states"].contiguous()

    def convolutional_forward(self, xBC, padded_len):
        """Convolutional layer forward pass for the full sequence."""
        if causal_conv1d_fn is None or self.activation not in [
            "silu",
            "swish",
            "identity",
        ]:
            raise NotImplementedError("Only support causal_conv1d_fn kernel for now")
            # xBC = self.act(self.conv1d(xBC.transpose(1, 2))[..., :padded_len].transpose(1, 2))
        else:
            xBC = causal_conv1d_fn(
                xBC.transpose(1, 2),
                rearrange(self.conv1d_weight, "d 1 w -> d w"),
                self.conv1d_bias,
                activation=None if self.activation == "identity" else self.activation,
            ).transpose(1, 2)
        return xBC

    # TODO: this will be added for inference
    # def step(self, u, state, **kwargs):
    #     """
    #     Args:
    #         u: (B, D),
    #         state: dict.

    #     Returns:
    #         out: (B, D),
    #         state: dict.

    #     """
    #     # Project input
    #     xBCzA_log = self.in_proj(u)
    #     xBC, z, A_log = torch.split(
    #         xBCzA_log,
    #         [
    #             self.d_inner + 2 * self.n_qk_heads * self.d_state,
    #             self.d_inner,
    #             self.n_v_heads,
    #         ],
    #         dim=-1,
    #     )

    #     xBC, conv_state = self.convolutional_step(xBC, state["conv"])
    #     state["conv"].copy_(conv_state)  # update state in place

    #     x, B, C = torch.split(
    #         xBC,
    #         [
    #             self.d_inner,
    #             self.n_qk_heads * self.d_state,
    #             self.n_qk_heads * self.d_state,
    #         ],
    #         dim=-1,
    #     )

    #     x = rearrange(x, "b (h s) -> b h s", h=self.n_v_heads)
    #     B = rearrange(B, "b (h s) -> b h s", h=self.n_qk_heads)
    #     C = rearrange(C, "b (h s) -> b h s", h=self.n_qk_heads)

    #     state["ssm"] = state["ssm"].to(x.dtype)
    #     zeros = torch.zeros((self.n_v_heads, self.headdim), device=A_log.device).to(dtype=x.dtype)
    #     ones = torch.ones((self.n_v_heads, self.headdim, self.d_state), device=A_log.device).to(
    #         dtype=x.dtype
    #     )
    #     y = selective_state_update(
    #         x=x / F.softplus(A_log).to(x.dtype).unsqueeze(-1),
    #         dt=repeat(A_log, "b h -> b h p", p=self.headdim),
    #         dt_softplus=True,
    #         A=-ones,
    #         B=B,
    #         C=C,
    #         state=state["ssm"],  # will be updated in place
    #         dt_bias=zeros,
    #         D=zeros,
    #     )

    #     y = y + self.D[:, None] * x
    #     y = rearrange(y, "b h p -> b (h p)")

    #     # Norm and gate
    #     out = self.out_proj(y * F.silu(z + self.z_bias))

    #     return out, state

    # def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
    #     """Allocate memory for inference cache."""
    #     device = self.in_proj.weight.device
    #     # conv_state:
    #     conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
    #     conv_state = torch.zeros(
    #         batch_size,
    #         self.d_conv,
    #         self.conv1d.weight.shape[0],
    #         device=device,
    #         dtype=conv_dtype,
    #     ).transpose(1, 2)
    #     # ssm_state:
    #     ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
    #     ssm_state = torch.zeros(
    #         batch_size,
    #         self.n_v_heads,
    #         self.headdim,
    #         self.d_state,
    #         device=device,
    #         dtype=ssm_dtype,
    #     )
    #     return {"conv": conv_state, "ssm": ssm_state}

    # def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
    #     """
    #     Get states from cache.

    #     conv_state: (batch, d_conv, conv1d.weight.shape[0])
    #     ssm_state: (batch, n_qk_heads, headdim, d_state)
    #     """
    #     assert self.layer_idx is not None
    #     # Allocate memory if not exists
    #     if self.layer_idx not in inference_params.key_value_memory_dict:
    #         inference_params.key_value_memory_dict[self.layer_idx] = self.allocate_inference_cache(
    #             batch_size, inference_params.max_seqlen, dtype=torch.float32
    #         )
    #     # Get states
    #     states = inference_params.key_value_memory_dict[self.layer_idx]
    #     if initialize_states:
    #         states["conv"].zero_()
    #         states["ssm"].zero_()
    #     return states

    # def convolutional_step(self, xBC, conv_state):
    #     """Convolutional layer forward pass for a single step."""
    #     conv_state = conv_state.to(xBC.dtype)
    #     if causal_conv1d_update:
    #         xBC = causal_conv1d_update(
    #             xBC,
    #             conv_state,
    #             rearrange(self.conv1d.weight, "d 1 w -> d w"),
    #             self.conv1d.bias,
    #             self.activation if self.activation != "identity" else None,
    #         )
    #     else:
    #         conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
    #         conv_state[:, :, -1] = xBC
    #         xBC = torch.sum(
    #             conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1
    #         )  # (B D)
    #         if self.conv_bias:
    #             xBC = xBC + self.conv1d.bias
    #         xBC = self.act(xBC).to(xBC.dtype)  # Some activations change dtype

    #     return xBC, conv_state
