import logging
import typing

import torch
import torch.nn.functional

from fast_llm.engine.base_model.config import ResourceUsageConfig
from fast_llm.engine.config_utils.initialization import LambdaInitializer, init_normal_, init_ones_
from fast_llm.engine.config_utils.tensor_dim import CompositeTensorDim, TensorDim
from fast_llm.engine.distributed.config import DistributedConfig, DistributedDimNames
from fast_llm.functional.config import ActivationType
from fast_llm.layers.attention.config import MixerKwargs
from fast_llm.layers.common.peft.config import PeftConfig
from fast_llm.layers.decoder.block import BlockWithBias
from fast_llm.layers.ssm.config import KimiDeltaAttentionConfig
from fast_llm.tensor import ParameterMeta, TensorMeta

logger = logging.getLogger(__name__)

try:
    from fla.ops.kda import chunk_kda
    from fla.ops.kda.gate import fused_kda_gate

    _kda_available = torch.cuda.is_available()
except (ImportError, RuntimeError):
    _kda_available = False


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def torch_kda_gate(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Pure PyTorch backup for fused_kda_gate."""
    num_heads, head_dim = g.shape[-2:]
    g = g.float()
    if dt_bias is not None:
        g = g + dt_bias.view(num_heads, head_dim)
    return (-A_log.view(num_heads, 1).float().exp() * torch.nn.functional.softplus(g)).to(output_dtype)


@torch.compile
def _torch_chunk_kda_single(
    q: torch.Tensor,  # batch, sequence, heads, head_dim
    k: torch.Tensor,  # batch, sequence, heads, head_dim
    v: torch.Tensor,  # batch, sequence, heads, head_dim
    g: torch.Tensor,  # batch, sequence, heads, head_dim (log decay rates per dim)
    beta: torch.Tensor,  # batch, sequence, heads (write gate strengths)
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,  # batch, heads, head_dim, head_dim
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    input_dtype = q.dtype

    if use_qk_l2norm_in_kernel:
        q = _l2norm(q, dim=-1, eps=1e-6)
        k = _l2norm(k, dim=-1, eps=1e-6)

    # Transpose to head-first layout and upcast for numerical stability.
    # batch, sequence, heads, dim -> batch, heads, sequence, dim
    q, k, v, g = (x.transpose(1, 2).contiguous().to(torch.float32) for x in (q, k, v, g))
    beta = beta.transpose(1, 2).contiguous().to(torch.float32)

    batch_size, num_heads, sequence_length, head_dim = q.shape

    # Pad sequence length to a multiple of chunk_size.
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    q = torch.nn.functional.pad(q, (0, 0, 0, pad_size))
    k = torch.nn.functional.pad(k, (0, 0, 0, pad_size))
    v = torch.nn.functional.pad(v, (0, 0, 0, pad_size))
    g = torch.nn.functional.pad(g, (0, 0, 0, pad_size))
    beta = torch.nn.functional.pad(beta, (0, pad_size))
    padded_sequence_length = sequence_length + pad_size
    num_chunks = padded_sequence_length // chunk_size

    q = q * (head_dim**-0.5)

    # Reshape to chunks: (batch, heads, num_chunks, chunk_size, head_dim)
    q, k, v, g = (x.reshape(batch_size, num_heads, num_chunks, chunk_size, head_dim) for x in (q, k, v, g))
    # beta: (batch, heads, num_chunks, chunk_size)
    beta = beta.reshape(batch_size, num_heads, num_chunks, chunk_size)

    # Cumulative sum of log-decays within each chunk (over the position dimension).
    g = g.cumsum(dim=-2)  # batch, heads, num_chunks, chunk_size, head_dim

    # Build the per-chunk intra-sequence delta-rule transform matrix A.
    # decay_matrix[..., c, i, d] = exp(g[c, d] - g[i, d]) — decay from position i to position c.
    # g.unsqueeze(-2): (batch, heads, num_chunks, chunk_size, 1, head_dim) — "c" positions
    # g.unsqueeze(-3): (batch, heads, num_chunks, 1, chunk_size, head_dim) — "i" positions
    decay_matrix = (g.unsqueeze(-2) - g.unsqueeze(-3)).exp()
    # intra_chunk_A[..., c, i] = sum_d(k[c, d] * k[i, d] * decay_matrix[c, i, d])
    intra_chunk_A = (k.unsqueeze(-2) * k.unsqueeze(-3) * decay_matrix).sum(-1)
    # Multiply each row c by beta[c] (write gate applied before delta-rule correction).
    intra_chunk_A = intra_chunk_A * beta.unsqueeze(-1)
    # Mask upper triangular (including diagonal) and flip sign for delta-rule update.
    upper_triangular_mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0
    )
    intra_chunk_A = -intra_chunk_A.masked_fill(upper_triangular_mask, 0)
    # Iterative delta-rule refinement.
    for chunk_pos in range(1, chunk_size):
        row = intra_chunk_A[..., chunk_pos, :chunk_pos].clone()
        above = intra_chunk_A[..., :chunk_pos, :chunk_pos].clone()
        intra_chunk_A[..., chunk_pos, :chunk_pos] = row + (row.unsqueeze(-1) * above).sum(-2)
    # Add identity and multiply each column i by beta[i] (write gate applied after correction).
    intra_chunk_A = (
        intra_chunk_A + torch.eye(chunk_size, dtype=intra_chunk_A.dtype, device=q.device)
    ) * beta.unsqueeze(-2)

    # Precompute per-chunk write keys and corrected values for the recurrent state update.
    # intra_chunk_w[..., c, d] = sum_i(A[c, i] * exp(g[i, d]) * k[i, d])
    intra_chunk_w = intra_chunk_A @ (g.exp() * k)  # batch, heads, num_chunks, chunk_size, head_dim
    # intra_chunk_u[..., c, d] = sum_i(A[c, i] * v[i, d])
    intra_chunk_u = intra_chunk_A @ v  # batch, heads, num_chunks, chunk_size, head_dim

    # Precompute intra-chunk causal attention scores.
    # intra_chunk_attn[..., c, j] = sum_d(q[c, d] * k[j, d] * decay_matrix[c, j, d])
    intra_chunk_attn = (q.unsqueeze(-2) * k.unsqueeze(-3) * decay_matrix).sum(-1)
    causal_mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
    intra_chunk_attn = intra_chunk_attn.masked_fill(causal_mask, 0)

    if initial_state is None:
        recurrent_state = torch.zeros(batch_size, num_heads, head_dim, head_dim, device=q.device, dtype=q.dtype)
    else:
        recurrent_state = initial_state.to(q)

    output = torch.zeros_like(intra_chunk_u)
    for chunk_index in range(num_chunks):
        q_chunk = q[:, :, chunk_index]  # batch, heads, chunk_size, head_dim
        k_chunk = k[:, :, chunk_index]
        u_chunk = intra_chunk_u[:, :, chunk_index]
        w_chunk = intra_chunk_w[:, :, chunk_index]
        g_chunk = g[:, :, chunk_index]
        attn_chunk = intra_chunk_attn[:, :, chunk_index]

        # Remove the state's contribution from the intra-chunk corrected values.
        value_corrected = u_chunk - w_chunk @ recurrent_state  # batch, heads, chunk_size, head_dim
        # Cross-chunk contribution: queries attend to the recurrent state via the cumulative decay.
        cross_chunk_output = (q_chunk * g_chunk.exp()) @ recurrent_state
        output[:, :, chunk_index] = cross_chunk_output + attn_chunk @ value_corrected

        # Decay the state by the cumulative log-decay at the last position in the chunk.
        last_g = g_chunk[:, :, -1]  # batch, heads, head_dim
        recurrent_state = recurrent_state * last_g.exp().unsqueeze(-1)  # broadcast over value dim
        # Write new key-value associations, weighted by the decay from each position to the chunk end.
        inter_chunk_decay = (last_g.unsqueeze(-2) - g_chunk).exp()  # batch, heads, chunk_size, head_dim
        recurrent_state = recurrent_state + (inter_chunk_decay * k_chunk).transpose(-1, -2) @ value_corrected

    if not output_final_state:
        recurrent_state = None

    # Remove padding and restore sequence-first layout.
    output = output.reshape(batch_size, num_heads, padded_sequence_length, head_dim)
    output = output[:, :, :sequence_length]
    output = output.transpose(1, 2).contiguous().to(input_dtype)
    return output, recurrent_state


def torch_chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if cu_seqlens is None:
        return _torch_chunk_kda_single(
            q,
            k,
            v,
            g,
            beta,
            chunk_size=chunk_size,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
    sequence_boundaries = cu_seqlens.tolist()
    outputs = []
    for seq_start, seq_end in zip(sequence_boundaries, sequence_boundaries[1:]):
        out, _ = _torch_chunk_kda_single(
            q[:, seq_start:seq_end],
            k[:, seq_start:seq_end],
            v[:, seq_start:seq_end],
            g[:, seq_start:seq_end],
            beta[:, seq_start:seq_end],
            chunk_size=chunk_size,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        outputs.append(out)
    return torch.cat(outputs, dim=1), None


class KimiDeltaAttention[ConfigType: KimiDeltaAttentionConfig](BlockWithBias[ConfigType]):
    """
    Implementation of the Kimi Delta Attention mixer.
    Reference Implementation: https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/main/modeling_kimi.py
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
        self._heads_dim = TensorDim(
            "kda_heads", self._config.heads, self._parallel_dim if self._config.heads > 1 else None
        )
        self._head_dim = TensorDim("kda_head_dim", self._config.head_dim)
        self._projection_dim = CompositeTensorDim("kda_projection", (self._heads_dim, self._head_dim))
        self._local_heads = self._heads_dim.size
        self._projection_size = self._projection_dim.size

        init = init_normal_(std=self._hidden_size**-0.5)
        self.q_proj = self._config.q_projection_layer.get_layer(
            hidden_dim,
            self._projection_dim,
            default_weight_initialization=init,
            default_add_bias=False,
            sequence_parallel=self._sequence_parallel,
            lr_scale=self._lr_scale,
            peft=self._peft,
        )
        self.k_proj = self._config.k_projection_layer.get_layer(
            hidden_dim,
            self._projection_dim,
            default_weight_initialization=init,
            default_add_bias=False,
            sequence_parallel=self._sequence_parallel,
            lr_scale=self._lr_scale,
            peft=self._peft,
        )
        self.v_proj = self._config.v_projection_layer.get_layer(
            hidden_dim,
            self._projection_dim,
            default_weight_initialization=init,
            default_add_bias=False,
            sequence_parallel=self._sequence_parallel,
            lr_scale=self._lr_scale,
            peft=self._peft,
        )

        self.q_conv = self._config.convolution_layer.get_layer(
            self._projection_dim,
            default_add_bias=False,
            default_activation=ActivationType.silu,
            lr_scale=self._lr_scale,
            peft=self._peft,
        )
        self.k_conv = self._config.convolution_layer.get_layer(
            self._projection_dim,
            default_add_bias=False,
            default_activation=ActivationType.silu,
            lr_scale=self._lr_scale,
            peft=self._peft,
        )
        self.v_conv = self._config.convolution_layer.get_layer(
            self._projection_dim,
            default_add_bias=False,
            default_activation=ActivationType.silu,
            lr_scale=self._lr_scale,
            peft=self._peft,
        )

        self.f_a_proj = self._config.f_a_projection_layer.get_layer(
            hidden_dim,
            self._head_dim,
            default_weight_initialization=init,
            default_add_bias=False,
            sequence_parallel=False,  # self._sequence_parallel,
            lr_scale=self._lr_scale,
            peft=self._peft,
        )
        self.f_b_proj = self._config.f_b_projection_layer.get_layer(
            self._head_dim,
            self._projection_dim,
            default_weight_initialization=init,
            default_add_bias=False,
            sequence_parallel=self._sequence_parallel,
            lr_scale=self._lr_scale,
            peft=self._peft,
        )
        self.g_a_proj = self._config.g_a_projection_layer.get_layer(
            hidden_dim,
            self._head_dim,
            default_weight_initialization=init,
            default_add_bias=False,
            sequence_parallel=False,  # self._sequence_parallel,
            lr_scale=self._lr_scale,
            peft=self._peft,
        )
        self.g_b_proj = self._config.g_b_projection_layer.get_layer(
            self._head_dim,
            self._projection_dim,
            default_weight_initialization=init,
            default_add_bias=False,
            sequence_parallel=self._sequence_parallel,
            lr_scale=self._lr_scale,
            peft=self._peft,
        )

        self.beta_proj = self._config.beta_projection_layer.get_layer(
            hidden_dim,
            self._heads_dim,
            default_weight_initialization=init,
            default_add_bias=False,
            sequence_parallel=self._sequence_parallel,
            lr_scale=self._lr_scale,
            peft=self._peft,
        )
        self.o_proj = self._config.output_projection_layer.get_layer(
            self._projection_dim,
            hidden_dim,
            default_weight_initialization=init,
            default_add_bias=False,
            sequence_parallel=self._sequence_parallel,
            lr_scale=self._lr_scale,
            peft=self._peft,
        )

        self.dt_bias: ParameterMeta = self._config.dt_bias_weight.get_parameter(
            (self._projection_dim,),
            default_initialization=init_ones_,
            lr_scale=self._lr_scale,
            peft=self._peft,
        )
        self.A_log: ParameterMeta = self._config.a_log_weight.get_parameter(
            (self._heads_dim,),
            default_initialization=LambdaInitializer(
                lambda _, tensor, generator: tensor.uniform_(1, 16, generator=generator).log_()
            ),
            lr_scale=self._lr_scale,
            peft=self._peft,
        )
        self.norm = self._config.normalization.get_layer(
            self._head_dim,
            lr_scale=self._lr_scale,
            peft=self._peft,
        )

        if _kda_available and distributed_config.use_cuda:
            self._chunk_kda = chunk_kda
            self._kda_gate = fused_kda_gate
        else:
            logger.warning(
                "Fast implementation for KimiDeltaAttention is not available. "
                "Please ensure that 'fla-core' is properly installed."
            )
            self._chunk_kda = torch_chunk_kda
            self._kda_gate = torch_kda_gate

    def _forward(
        self,
        input_: torch.Tensor,
        kwargs: dict[str, typing.Any],
        losses: dict[str, typing.Any] | None = None,
        metrics: dict[str, typing.Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Same as in gdn, the idea is to always do forward pass in a packed way, which is required for varlen support.
        """

        # TODO: Merge q,k,v into a single tensor?
        q = self.q_proj(input_)
        k = self.k_proj(input_)
        v = self.v_proj(input_)

        document_index = kwargs[MixerKwargs.document_index_q].unsqueeze(0)
        lengths = kwargs[MixerKwargs.lengths]
        # Move sequence dim to last so the convolution acts on it, add pretend batch dimension.
        q = (
            self.q_conv(q.unsqueeze(0).transpose(1, 2), document_index=document_index, lengths=lengths)
            .transpose(1, 2)
            .unflatten(-1, (self._local_heads, self._config.head_dim))
        )
        k = (
            self.k_conv(k.unsqueeze(0).transpose(1, 2), document_index=document_index, lengths=lengths)
            .transpose(1, 2)
            .unflatten(-1, (self._local_heads, self._config.head_dim))
        )
        v = (
            self.v_conv(v.unsqueeze(0).transpose(1, 2), document_index=document_index, lengths=lengths)
            .transpose(1, 2)
            .unflatten(-1, (self._local_heads, self._config.head_dim))
        )

        g_kernel = (
            self.f_b_proj(self.f_a_proj(input_)).unsqueeze(0).unflatten(-1, (self._local_heads, self._config.head_dim))
        )
        g_kernel = self._kda_gate(g_kernel, self.A_log.float(), dt_bias=self.dt_bias)

        out, _ = self._chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g_kernel,
            beta=torch.sigmoid(self.beta_proj(input_).float()).unsqueeze(0),
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=kwargs[MixerKwargs.cu_seqlens_q],
        )
        out = out.to(input_.dtype).squeeze(0)

        g_out = self.g_b_proj(self.g_a_proj(input_))
        out = self.norm(out, g_out.view_as(out))
        return self.o_proj(out.flatten(-2))

    def get_compute_usage(self, input_: TensorMeta, kwargs: dict[str, typing.Any], config: ResourceUsageConfig) -> int:
        raise NotImplementedError()

    def get_preprocessing_config(self) -> dict[str, typing.Any]:
        return {"return_cumulative_sequence_lengths": True, "return_document_index": True}
