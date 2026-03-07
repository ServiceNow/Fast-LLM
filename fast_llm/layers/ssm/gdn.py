import logging
import typing

import torch

from fast_llm.engine.base_model.config import ResourceUsageConfig
from fast_llm.engine.config_utils.initialization import LambdaInitializer, init_normal_, init_ones_
from fast_llm.engine.config_utils.tensor_dim import CompositeTensorDim, ConcatenatedTensorDim, TensorDim
from fast_llm.engine.distributed.config import DistributedConfig, DistributedDimNames
from fast_llm.functional.config import ActivationType
from fast_llm.layers.attention.config import MixerKwargs
from fast_llm.layers.attention.preprocessing import preprocess_for_varlen
from fast_llm.layers.common.peft.config import PeftConfig
from fast_llm.layers.decoder.block import BlockWithBias
from fast_llm.layers.ssm.config import GatedDeltaNetConfig
from fast_llm.tensor import ParameterMeta, TensorMeta
from fast_llm.utils import div

logger = logging.getLogger(__name__)

try:
    from causal_conv1d import causal_conv1d_fn as _causal_conv1d_fn  # noqa

    _causal_conv1d_available = torch.cuda.is_available()
except (ImportError, RuntimeError):
    _causal_conv1d_available = False


try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    _fast_gdn_available = torch.cuda.is_available()
except (ImportError, RuntimeError):
    _fast_gdn_available = False


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


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
    cu_seqlens=None,
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
    query = torch.nn.functional.pad(query, (0, 0, 0, pad_size))
    key = torch.nn.functional.pad(key, (0, 0, 0, pad_size))
    value = torch.nn.functional.pad(value, (0, 0, 0, pad_size))
    beta = torch.nn.functional.pad(beta, (0, pad_size))
    g = torch.nn.functional.pad(g, (0, pad_size))
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


class GatedDeltaNet[ConfigType: GatedDeltaNetConfig](BlockWithBias[ConfigType]):
    """
    Follows implementation here: https://github.com/huggingface/transformers/blob/a5c903f877fda21e739027eed133e03162eb7712/src/transformers/models/qwen3_next/modeling_qwen3_next.py#L593
    - For tensor parallel implementtion (no sequnece prallel): we scatter teh heads accross ranks.
    - Sequence Tensor parallel: in_proj_qkvz all reduces across sequence dim. --> each rank performs work on full sequence but only a subset of heads (standrd TP).

    Note, Qwen3_Next follows a different layout, where gdn_qkvz is assumed to be layed out as [h0: Q,K,V,Z][h1: Q,K,V,Z][h2: Q,K,V,Z]
    Here we follow a more natural layout for gdn_qkvz: [Q_all_heads | K_all_heads | V_all_heads | Z_all_heads]. If we want to apply MIL init here it should be easier like this.

    """

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
        # for Qwen's layour use soemthing like this instead:
        # n_vheads_per_k_head = self._config.value_heads // self._config.key_heads
        # head_size = 2 * self._config.key_head_dim + 2 * self._config.value_head_dim * n_vheads_per_k_head
        # n_heads = self._config.key_heads
        # qkvz_dim = TensorDim(e
        #     "gdn_qkvz",
        #     n_heads * head_size,
        #     self._parallel_dim if n_heads > 1 else None,
        # )
        ba_dim = ConcatenatedTensorDim(
            "gdn_ba",
            (
                CompositeTensorDim("gdn_beta", (self._value_heads_dim,)),
                CompositeTensorDim("gdn_alpha", (self._value_heads_dim,)),
            ),
        )
        # for Qwen's layour use something like this instead:
        # ba_dim = TensorDim(
        #     "gdn_ba",
        #     2 * self._config.value_heads,
        #     self._parallel_dim if 2 * self._config.value_heads > 1 else None,
        # )

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
            default_activation=ActivationType.silu,
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

        if _fast_gdn_available:
            self.chunk_gated_delta_rule = chunk_gated_delta_rule
        else:
            logger.warning(
                "Fast implementation for GatedDeltaNet is not available. Please ensure that 'fla' is properly installed."
            )
            self.chunk_gated_delta_rule = torch_chunk_gated_delta_rule

    def _forward(
        self,
        input_: torch.Tensor,
        kwargs: dict[str, typing.Any],
        losses: dict[str, typing.Any] | None = None,
        metrics: dict[str, typing.Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        - we flatten batch + seq
        - forward as packed sequence, i.e. BS = 1, cu_seqlens and seq_idx created in the preprocessing step must reflect this (these are None if cross_document_attention is True)
        - scatter results back to B/T x T/B x D
        - note, if there are padding tokens they are not treated in a special way here.
            They are
             - assumed to be ignored later in the loss calculation and
             - are assumed to be always on the right and, hence, will be reflected in seq_idx and cu_seqlens (i.e. treated as a seperate packed sequence?)
        -
        """

        # in sequence parallel TP the input here is already scattered across sequence dimension
        # TODO: fuse some of the reshapes into rearranges
        hidden_states = input_

        # TODO: ====== Merge qkvz and ba ======
        projected_states_qkvz = self.in_proj_qkvz(hidden_states)  # bs/seq x seq_len/bs x (qkvz)
        projected_states_ba = self.in_proj_ba(hidden_states)  # bs/seq x seq_len/bs x (b a)

        query_key_value, z = torch.split(
            projected_states_qkvz,
            [
                2 * self._local_key_heads * self._config.key_head_dim
                + self._local_value_heads * self._config.value_head_dim,
                self._local_value_heads * self._config.value_head_dim,
            ],
            dim=-1,
        )

        # Move sequence dim to last so the convolution acts on it, add pretend batch dimension.
        # sequence, qkv_total -> 1, qkv_total, sequence
        query_key_value = query_key_value.unsqueeze(0).transpose(1, 2)
        query_key_value = self.convolution(
            query_key_value,
            document_index=kwargs[MixerKwargs.document_index_q].unsqueeze(0),
            lengths=[length for lengths in kwargs[MixerKwargs.lengths] for length in lengths],
        )
        # 1, qkv_total, sequence -> 1, sequence, qkv_total
        query_key_value = query_key_value.transpose(1, 2)
        query, key, value = torch.split(
            query_key_value,
            [
                self._local_key_heads * self._config.key_head_dim,
                self._local_key_heads * self._config.key_head_dim,
                self._local_value_heads * self._config.value_head_dim,
            ],
            dim=-1,
        )

        # 1, sequence, heads, head_dim
        query = query.unflatten(-1, (self._local_key_heads, self._config.key_head_dim))
        key = key.unflatten(-1, (self._local_key_heads, self._config.key_head_dim))
        value = value.unflatten(-1, (self._local_value_heads, self._config.value_head_dim))

        if self._value_heads_per_key > 1:
            query = query.repeat_interleave(self._value_heads_per_key, dim=2)
            key = key.repeat_interleave(self._value_heads_per_key, dim=2)

        beta, alpha = torch.split(projected_states_ba, [self._local_value_heads, self._local_value_heads], dim=-1)

        out, _ = self.chunk_gated_delta_rule(
            query,
            key,
            value,
            g=self._calculate_g(alpha).unsqueeze(0),
            beta=beta.sigmoid().unsqueeze(0),
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=kwargs[MixerKwargs.cu_seqlens_q],
        )
        out = out.squeeze(0)
        out = self.norm(out, z.reshape_as(out))
        return self.out_proj(out.flatten(-2))

    @torch.compile
    def _calculate_g(self, alpha: torch.Tensor) -> torch.Tensor:
        return -self.A_log.float().exp() * torch.nn.functional.softplus(alpha.float() + self.dt_bias)

    def preprocess(self, kwargs: dict[str, typing.Any]) -> None:
        preprocess_for_varlen(
            kwargs,
            kwargs[MixerKwargs.device] if MixerKwargs.device in kwargs else self._distributed.device,
            return_cu_seqlens=True,
            return_seq_idx=True,
        )

    def get_compute_usage(self, input_: TensorMeta, kwargs: dict[str, typing.Any], config: ResourceUsageConfig) -> int:
        raise NotImplementedError()
