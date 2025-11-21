import logging
import typing

import torch

from fast_llm.engine.base_model.config import ResourceUsageConfig
from fast_llm.engine.config_utils.initialization import LambdaInitializer, init_normal_, init_ones_
from fast_llm.engine.config_utils.tensor_dim import CompositeTensorDim, TensorDim
from fast_llm.engine.distributed.config import DistributedConfig, DistributedDimNames
from fast_llm.layers.block.config import BlockKwargs
from fast_llm.layers.common.peft.config import PeftConfig
from fast_llm.layers.decoder.block import BlockWithBias
from fast_llm.layers.ssm.config import KimiDeltaAttentionConfig
from fast_llm.tensor import ParameterMeta, TensorMeta

logger = logging.getLogger(__name__)

try:
    from fla.ops.kda import chunk_kda
    from fla.ops.kda.gate import fused_kda_gate
except ImportError:
    chunk_kda = None
    fused_kda_gate = None


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
        if chunk_kda is None or fused_kda_gate is None:
            raise ImportError(
                "KimiDeltaAttention requires the `fla-core` package. "
                "Please install it with `pip install -U fla-core`."
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
            default_activation=self._config.convolution_layer.activation,
            lr_scale=self._lr_scale,
            peft=self._peft,
        )
        self.k_conv = self._config.convolution_layer.get_layer(
            self._projection_dim,
            default_add_bias=False,
            default_activation=self._config.convolution_layer.activation,
            lr_scale=self._lr_scale,
            peft=self._peft,
        )
        self.v_conv = self._config.convolution_layer.get_layer(
            self._projection_dim,
            default_add_bias=False,
            default_activation=self._config.convolution_layer.activation,
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

    def _apply_conv(self, tensor: torch.Tensor, conv: torch.nn.Module) -> torch.Tensor:
        """
        Applies convolution.
        Note, in the reference code they use Short Convolution from flash-linear-attention/fla/modules/convolution.py, but that one kjust uses causal_conv1danyways.
        TODO: make sure varlen is supported correctly.
        """
        tensor = tensor.transpose(1, 2).contiguous()
        tensor = conv(tensor)
        return tensor.transpose(1, 2).contiguous()

    def _reshape_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.contiguous()
        # since head_dim is the same vor k,q and v
        # same as rearrange(v, '... (h d) -> ... h d', d=self.head_dim)
        return tensor.view(tensor.shape[0], tensor.shape[1], self._local_heads, self._config.head_dim)

    def _forward(
        self,
        input_: torch.Tensor,
        kwargs: dict[str, typing.Any],
        losses: dict[str, typing.Any] | None = None,
        metrics: dict[str, typing.Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # TODO: make sure varlen is supported
        # TODO: make sure we dont need to mask padding tokens in training
        sequence_first = kwargs[BlockKwargs.sequence_first]
        hidden_states = input_

        residual_dtype = hidden_states.dtype

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        if sequence_first:
            q = q.transpose(0, 1)
            k = k.transpose(0, 1)
            v = v.transpose(0, 1)

        q = self._apply_conv(q, self.q_conv)
        k = self._apply_conv(k, self.k_conv)
        v = self._apply_conv(v, self.v_conv)

        if sequence_first:
            _, batch_size, _ = hidden_states.shape
            sequence_length = q.size(1)
            # hidden_states = gather_op(hidden_states, self._distributed.tensor_group, dim=0, async_op=False).transpose(
            #     0, 1
            # )
            # hidden_states = hidden_states.transpose(0, 1)
        else:
            batch_size, sequence_length, _ = hidden_states.shape

        g_kernel = self.f_b_proj(self.f_a_proj(hidden_states))
        if sequence_first:
            g_kernel = g_kernel.transpose(0, 1)
        g_kernel = fused_kda_gate(g_kernel, self.A_log, self._config.head_dim, g_bias=self.dt_bias)

        beta = torch.sigmoid(self.beta_proj(hidden_states).float())
        q = self._reshape_heads(q)
        k = self._reshape_heads(k)
        v = self._reshape_heads(v)
        if sequence_first:
            beta = beta.transpose(0, 1)

        # need to install nightly triton for now
        attn_out, _ = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g_kernel,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=None,
        )

        attn_out = attn_out.to(residual_dtype)
        attn_out = self._reshape_heads(attn_out)

        g_out = self._reshape_heads(self.g_b_proj(self.g_a_proj(hidden_states)))  # bs x seq x n_local_heads x head dim

        attn_out = attn_out.reshape(-1, self._config.head_dim)
        g_out = g_out.reshape(-1, self._config.head_dim)
        attn_out = self.norm(attn_out, g_out)
        attn_out = attn_out.view(batch_size, sequence_length, self._projection_size)
        if sequence_first:
            attn_out = attn_out.transpose(0, 1)
        attn_out = self.o_proj(attn_out)

        return attn_out

    def get_compute_usage(self, input_: TensorMeta, kwargs: dict[str, typing.Any], config: ResourceUsageConfig) -> int:
        raise NotImplementedError()
