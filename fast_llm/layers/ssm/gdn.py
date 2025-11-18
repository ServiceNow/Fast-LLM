import logging
import typing

import torch
import torch.nn.functional as F

from fast_llm.engine.base_model.config import ResourceUsageConfig
from fast_llm.engine.config_utils.initialization import LambdaInitializer, init_normal_, init_ones_
from fast_llm.engine.config_utils.tensor_dim import CompositeTensorDim, ConcatenatedTensorDim, TensorDim
from fast_llm.engine.distributed.config import DistributedConfig, DistributedDimNames
from fast_llm.layers.block.config import BlockKwargs
from fast_llm.layers.common.peft.config import PeftConfig
from fast_llm.layers.decoder.block import BlockWithBias
from fast_llm.layers.ssm.config import GatedDeltaNetConfig
from fast_llm.tensor import ParameterMeta, TensorMeta
from fast_llm.utils import div

logger = logging.getLogger(__name__)

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
except ImportError:
    chunk_gated_delta_rule = None


is_fast_path_available = chunk_gated_delta_rule is not None


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def torch_recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    use_qk_l2norm_in_kernel: bool,
) -> torch.Tensor:
    """
    Simplified gated Delta rule used during training.
    Args expect tensors shaped as (batch, heads, seq, dim) except for g/beta which are (batch, heads, seq).
    """
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1)
        key = _l2norm(key, dim=-1)

    query = query.to(torch.float32)
    key = key.to(torch.float32)
    value = value.to(torch.float32)
    beta = beta.to(torch.float32)
    g = g.to(torch.float32)

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    state = torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, device=key.device, dtype=key.dtype)
    outputs = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim, device=value.device, dtype=value.dtype)

    for idx in range(sequence_length):
        q_t = query[:, :, idx]
        k_t = key[:, :, idx]
        v_t = value[:, :, idx]
        g_t = g[:, :, idx].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, idx].unsqueeze(-1)
        state = state * g_t
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        outputs[:, :, idx] = (state * q_t.unsqueeze(-1)).sum(dim=-2)

    return outputs.to(initial_dtype), state


def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = (
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    )

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = (
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    )
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    # for each chunk
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


# class _GatedRMSNorm(torch.nn.Module):
#     def __init__(self, hidden_size: int, eps: float):
#         super().__init__()
#         self.weight = torch.nn.Parameter(torch.ones(hidden_size))
#         self.eps = eps

#     def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
#         dtype = hidden_states.dtype
#         hidden_states = hidden_states.to(torch.float32)
#         variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
#         hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
#         hidden_states = self.weight * hidden_states.to(dtype)
#         hidden_states = hidden_states * F.silu(gate.to(torch.float32))
#         return hidden_states.to(dtype)


class GatedDeltaNet[ConfigType: GatedDeltaNetConfig](BlockWithBias[ConfigType]):
    _config: ConfigType

    def __init__(
        self,
        config: ConfigType,
        distributed_config: DistributedConfig,
        *,
        hidden_dim: TensorDim,
        lr_scale: float | None,
        peft: PeftConfig | None,
        return_bias: bool = True,
    ):
        super().__init__(
            config, distributed_config, hidden_dim=hidden_dim, lr_scale=lr_scale, peft=peft, return_bias=return_bias
        )
        self._parallel_dim = self._distributed_config.get_distributed_dim(DistributedDimNames.tensor)
        self._value_heads_dim = TensorDim(
            "gdn_value_heads", self._config.value_heads, self._parallel_dim if self._config.value_heads > 1 else None
        )
        self._key_heads_dim = TensorDim(
            "gdn_key_heads", self._config.key_heads, self._parallel_dim if self._config.key_heads > 1 else None
        )
        self._value_head_dim = TensorDim("gdn_value_head_dim", self._config.value_head_dim)
        self._key_head_dim = TensorDim("gdn_key_head_dim", self._config.key_head_dim)
        self._local_value_heads = self._value_heads_dim.size
        self._local_key_heads = self._key_heads_dim.size
        self._value_heads_per_key = div(self._local_value_heads, max(self._local_key_heads, 1))

        query_dim = CompositeTensorDim("gdn_query", (self._key_heads_dim, self._key_head_dim))
        key_dim = CompositeTensorDim("gdn_key", (self._key_heads_dim, self._key_head_dim))
        value_dim = CompositeTensorDim("gdn_value", (self._value_heads_dim, self._value_head_dim))
        z_dim = CompositeTensorDim("gdn_z", (self._value_heads_dim, self._value_head_dim))
        qkvz_dim = ConcatenatedTensorDim("gdn_qkvz", (query_dim, key_dim, value_dim, z_dim))
        ba_dim = ConcatenatedTensorDim(
            "gdn_ba",
            (
                CompositeTensorDim("gdn_beta", (self._value_heads_dim,)),
                CompositeTensorDim("gdn_alpha", (self._value_heads_dim,)),
            ),
        )

        qkv_channels_dim = ConcatenatedTensorDim("gdn_qkv", (query_dim, key_dim, value_dim))

        self.in_proj_qkvz = self._config.qkv_projection_layer.get_layer(
            hidden_dim,
            qkvz_dim,
            default_weight_initialization=init_normal_(std=self._hidden_size**-0.5),
            default_add_bias=False,
            sequence_parallel=self._sequence_parallel,
            lr_scale=self._lr_scale,
            peft=self._peft,
        )
        self.in_proj_ba = self._config.ba_projection_layer.get_layer(
            hidden_dim,
            ba_dim,
            default_weight_initialization=init_normal_(std=self._hidden_size**-0.5),
            default_add_bias=False,
            sequence_parallel=self._sequence_parallel,
            lr_scale=self._lr_scale,
            peft=self._peft,
        )
        self.convolution = self._config.convolution_layer.get_layer(
            qkv_channels_dim,
            default_add_bias=False,
            default_activation=self._config.activation,
            lr_scale=self._lr_scale,
            peft=self._peft,
        )
        self.out_proj = self._config.output_layer.get_layer(
            value_dim,
            hidden_dim,
            default_weight_initialization=init_normal_(std=self._hidden_size**-0.5),
            default_add_bias=False,
            sequence_parallel=self._sequence_parallel,
            lr_scale=self._lr_scale,
            peft=self._peft,
        )
        self.dt_bias: ParameterMeta = self._config.dt_bias_weight.get_parameter(
            (self._value_heads_dim,),
            default_initialization=init_ones_,
            lr_scale=self._lr_scale,
            peft=self._peft,
        )
        self.A_log: ParameterMeta = self._config.a_log_weight.get_parameter(
            (self._value_heads_dim,),
            default_initialization=LambdaInitializer(
                lambda _, tensor, generator: tensor.uniform_(0, 16, generator=generator).log_()
            ),
            lr_scale=self._lr_scale,
            peft=self._peft,
        )
        self.norm = self._config.normalization.get_layer(
            self._value_head_dim, lr_scale=self._lr_scale, peft=self._peft
        )
        # _GatedRMSNorm(self._config.value_head_dim, self._config.norm_epsilon)
        self._use_qk_l2norm = self._config.use_qk_l2norm

        self._value_dim = value_dim
        self._query_dim = query_dim
        self.chunk_gated_delta_rule = chunk_gated_delta_rule or torch_chunk_gated_delta_rule

        if not is_fast_path_available:
            logger.warning(
                "Fast paths for GatedDeltaNet are not available. Please ensure that 'causal_conv1d' and 'fla' are properly installed."
            )

    def _reshape_heads(self, tensor: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
        batch, seq, _ = tensor.shape
        return tensor.view(batch, seq, num_heads, head_dim)

    def _forward(
        self,
        input_: torch.Tensor,
        kwargs: dict[str, typing.Any],
        losses: dict[str, typing.Any] | None = None,
        metrics: dict[str, typing.Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        sequence_first = kwargs[BlockKwargs.sequence_first]
        if sequence_first:
            hidden_states = input_.transpose(0, 1)
        else:
            hidden_states = input_

        batch_size, sequence_length, _ = hidden_states.shape
        qkvz = self.in_proj_qkvz(hidden_states)
        ba = self.in_proj_ba(hidden_states)
        key_size = self._query_dim.size
        value_size = self._value_dim.size
        query, key, value, z = torch.split(qkvz, (key_size, key_size, value_size, value_size), dim=-1)
        beta, alpha = torch.split(ba, (self._local_value_heads, self._local_value_heads), dim=-1)

        query = self._reshape_heads(query, self._local_key_heads, self._config.key_head_dim)
        key = self._reshape_heads(key, self._local_key_heads, self._config.key_head_dim)
        value = self._reshape_heads(value, self._local_value_heads, self._config.value_head_dim)
        z = self._reshape_heads(z, self._local_value_heads, self._config.value_head_dim)

        mixed_qkv = torch.cat(
            (
                query.reshape(batch_size, sequence_length, -1),
                key.reshape(batch_size, sequence_length, -1),
                value.reshape(batch_size, sequence_length, -1),
            ),
            dim=-1,
        )
        mixed_qkv = mixed_qkv.transpose(1, 2)
        mixed_qkv = self.convolution(mixed_qkv)
        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            (
                self._local_key_heads * self._config.key_head_dim,
                self._local_key_heads * self._config.key_head_dim,
                self._local_value_heads * self._config.value_head_dim,
            ),
            dim=-1,
        )
        query = self._reshape_heads(query, self._local_key_heads, self._config.key_head_dim)
        key = self._reshape_heads(key, self._local_key_heads, self._config.key_head_dim)
        value = self._reshape_heads(value, self._local_value_heads, self._config.value_head_dim)

        beta = beta.view(batch_size, sequence_length, self._local_value_heads).sigmoid()
        alpha = alpha.view(batch_size, sequence_length, self._local_value_heads)
        dt_bias = self.dt_bias.to(hidden_states.dtype)
        a_log = self.A_log.to(hidden_states.dtype)
        g = -torch.exp(a_log) * F.softplus(alpha + dt_bias)

        if self._value_heads_per_key > 1:
            query = query.repeat_interleave(self._value_heads_per_key, dim=2)
            key = key.repeat_interleave(self._value_heads_per_key, dim=2)

        core_attn_out, _ = self.chunk_gated_delta_rule(
            query.permute(0, 2, 1, 3),
            key.permute(0, 2, 1, 3),
            value.permute(0, 2, 1, 3),
            g=g.permute(0, 2, 1),
            beta=beta.permute(0, 2, 1),
            use_qk_l2norm_in_kernel=self._use_qk_l2norm,
        )

        core_attn_out = core_attn_out.permute(0, 2, 1, 3).reshape(
            batch_size, sequence_length, -1, self._config.value_head_dim
        )
        z = z.reshape(batch_size, sequence_length, -1, self._config.value_head_dim)
        norm_input = core_attn_out.reshape(-1, self._config.value_head_dim)
        norm_gate = z.reshape(-1, self._config.value_head_dim)
        norm_out = self.norm(norm_input, norm_gate).view(batch_size, sequence_length, -1)
        output = self.out_proj(norm_out)

        if sequence_first:
            output = output.transpose(0, 1)
        return output

    def get_compute_usage(self, input_: TensorMeta, kwargs: dict[str, typing.Any], config: ResourceUsageConfig) -> int:
        # return (
        #     self.in_proj_qkvz.get_compute_usage(input_, config)
        #     + self.in_proj_ba.get_compute_usage(input_, config)
        #     + self.out_proj.get_compute_usage(input_, config)
        # )
        raise NotImplementedError()
