import logging
import typing

import torch
import torch.nn.functional as F
from einops import rearrange

from fast_llm.engine.base_model.config import ResourceUsageConfig
from fast_llm.engine.config_utils.initialization import LambdaInitializer, init_normal_, init_ones_
from fast_llm.engine.config_utils.tensor_dim import CompositeTensorDim, ConcatenatedTensorDim, TensorDim
from fast_llm.engine.distributed.config import DistributedConfig, DistributedDimNames
from fast_llm.functional.config import ActivationType
from fast_llm.layers.attention.config import MixerKwargs
from fast_llm.layers.attention.preprocessing import preprocess_for_varlen
from fast_llm.layers.block.config import BlockKwargs
from fast_llm.layers.common.peft.config import PeftConfig
from fast_llm.layers.decoder.block import BlockWithBias
from fast_llm.layers.ssm.config import GatedDeltaNetConfig
from fast_llm.tensor import ParameterMeta, TensorMeta
from fast_llm.utils import div

logger = logging.getLogger(__name__)

try:
    from causal_conv1d import causal_conv1d_fn as _causal_conv1d_fn  # noqa
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    _fast_path_available = torch.cuda.is_available()
except (ImportError, RuntimeError):
    _fast_path_available = False


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

        self.chunk_gated_delta_rule = chunk_gated_delta_rule or torch_chunk_gated_delta_rule

        if not _fast_path_available:
            logger.warning(
                "Fast paths for GatedDeltaNet are not available. Please ensure that 'causal_conv1d' and 'fla' are properly installed."
            )

    def fix_query_key_value_ordering(self, mixed_qkvz, mixed_ba):
        """
        Derives `query`, `key` and `value` tensors from `mixed_qkvz` and `mixed_ba`.
        Replaces fix_query_key_value_ordering from Qwen due to layout differences.
        """

        local_qkv_sizes = (
            self._local_key_heads * self._config.key_head_dim,
            self._local_key_heads * self._config.key_head_dim,
            self._local_value_heads * self._config.value_head_dim,
            self._local_value_heads * self._config.value_head_dim,
        )
        query, key, value, z = torch.split(mixed_qkvz, local_qkv_sizes, dim=-1)
        query = query.reshape(*query.shape[:-1], self._local_key_heads, self._config.key_head_dim)
        key = key.reshape(*key.shape[:-1], self._local_key_heads, self._config.key_head_dim)
        value = value.reshape(*value.shape[:-1], self._local_value_heads, self._config.value_head_dim)
        z = z.reshape(*z.shape[:-1], self._local_value_heads, self._config.value_head_dim)

        beta, alpha = torch.split(
            mixed_ba,
            (self._local_value_heads, self._local_value_heads),
            dim=-1,
        )
        beta = beta.reshape(*beta.shape[:-1], self._local_value_heads)
        alpha = alpha.reshape(*alpha.shape[:-1], self._local_value_heads)
        return query, key, value, z, beta, alpha

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

        sequence_first = kwargs[BlockKwargs.sequence_first]
        # in sequence parallel TP the input here is already scattered across sequence dimension
        # TODO: fuse soome of the reshapes into rearranges
        hidden_states = input_

        projected_states_qkvz = self.in_proj_qkvz(hidden_states)  # bs/seq x seq_len/bs x (qkvz)
        projected_states_ba = self.in_proj_ba(hidden_states)  # bs/seq x seq_len/bs x (b a)
        if sequence_first:
            projected_states_qkvz = projected_states_qkvz.transpose(0, 1)
            projected_states_ba = projected_states_ba.transpose(0, 1)

        batch_size, sequence_length = projected_states_qkvz.shape[:2]

        # note: to support var len training (packing) we need to flatten hidden states to batch_size = 1
        # this is does not seem to be required by causal_conv1d_fn, but it it required by chunked_gdn_rule: https://github.com/fla-org/flash-linear-attention/blob/71260ecd573cfaaa94305b726465143199e99734/fla/ops/gated_delta_rule/chunk.py#L299
        # similarly to kimi linear and to SHortCOnv from fla, we pass it flattened tro conv_1d as well, i.e. see https://github.com/fla-org/flash-linear-attention/blob/71260ecd573cfaaa94305b726465143199e99734/fla/modules/convolution.py#L914
        query, key, value, z, beta, alpha = self.fix_query_key_value_ordering(
            projected_states_qkvz, projected_states_ba
        )
        query, key, value = (x.reshape(x.shape[0], x.shape[1], -1) for x in (query, key, value))

        mixed_qkv = torch.cat((query, key, value), dim=-1)
        mixed_qkv = rearrange(mixed_qkv, "b s ... -> (b s) ...").unsqueeze(0)  # 1 s d
        mixed_qkv = rearrange(mixed_qkv, "b t d -> b d t")  # mixed_qkv.transpose(1, 2)
        # conv func. gets sequence dim as last dim, see https://github.com/Dao-AILab/causal-conv1d/blob/22a4577d8ace9d5703daea91a7fb56695492152b/causal_conv1d/causal_conv1d_interface.py#L110
        mixed_qkv = self.convolution(mixed_qkv, seq_idx=kwargs[MixerKwargs.seq_idx].unsqueeze(0))
        mixed_qkv = rearrange(mixed_qkv, "b d t -> b t d")  # mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            (
                self._local_key_heads * self._config.key_head_dim,
                self._local_key_heads * self._config.key_head_dim,
                self._local_value_heads * self._config.value_head_dim,
            ),
            dim=-1,
        )
        query = query.reshape(query.shape[0], query.shape[1], -1, self._config.key_head_dim)
        key = key.reshape(key.shape[0], key.shape[1], -1, self._config.key_head_dim)
        value = value.reshape(value.shape[0], value.shape[1], -1, self._config.value_head_dim)

        beta = beta.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(alpha.float() + self.dt_bias)

        beta = rearrange(beta, "b s ... -> (b s) ...").unsqueeze(0)
        g = rearrange(g, "b s ... -> (b s) ...").unsqueeze(0)

        if self._value_heads_per_key > 1:
            query = query.repeat_interleave(self._value_heads_per_key, dim=2)
            key = key.repeat_interleave(self._value_heads_per_key, dim=2)

        core_attn_out, _ = self.chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=kwargs[MixerKwargs.cu_seqlens_q],
        )

        z_shape_og = z.shape
        core_attn_out = rearrange(core_attn_out.squeeze(0), "(b s) ... -> b s ...", b=batch_size, s=sequence_length)

        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1)
        if sequence_first:
            core_attn_out = core_attn_out.transpose(0, 1)
        output = self.out_proj(core_attn_out)

        return output

    def preprocess(self, kwargs: dict[str, typing.Any]) -> None:
        preprocess_for_varlen(
            kwargs,
            kwargs[MixerKwargs.device] if MixerKwargs.device in kwargs else self._distributed.device,
            return_cu_seqlens=True,
            return_seq_idx=True,
        )

    def get_compute_usage(self, input_: TensorMeta, kwargs: dict[str, typing.Any], config: ResourceUsageConfig) -> int:
        raise NotImplementedError()
