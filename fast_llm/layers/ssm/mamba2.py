import logging
import typing

import torch

from fast_llm.engine.base_model.config import ResourceUsageConfig
from fast_llm.engine.config_utils.initialization import init_normal_, init_ones_, init_uniform_centered_
from fast_llm.engine.config_utils.tensor_dim import CompositeTensorDim, ConcatenatedTensorDim, TensorDim
from fast_llm.engine.distributed.config import DistributedConfig, DistributedDimNames
from fast_llm.functional.config import ActivationType
from fast_llm.layers.block.block import BlockLayer
from fast_llm.layers.block.config import BlockDimNames, BlockKwargs
from fast_llm.layers.common.peft.config import PeftConfig
from fast_llm.layers.ssm.config import Mamba2Config, init_a, init_dtprojbias
from fast_llm.tensor import TensorMeta
from fast_llm.utils import div

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn  # noqa

    _mamba_available = True
except (ImportError, RuntimeError):
    _mamba_available = False

logger = logging.getLogger(__name__)


class Mamba2[ConfigType: Mamba2Config](BlockLayer[ConfigType]):
    """
    This code is adapted from https://github.com/jxiw/M1/blob/537a1ca5407a786a99dc6c721873493cf8750d5e/mamba/hybrid_mamba_layer.py
    """

    _mixer_name: typing.ClassVar[str] = "mamba_2"

    def __init__(
        self,
        config: ConfigType,
        distributed_config: DistributedConfig,
        *,
        hidden_dim: TensorDim,
        lr_scale: float | None,
        peft: PeftConfig | None,
    ):
        super().__init__(
            config,
            distributed_config,
            hidden_dim=hidden_dim,
            lr_scale=lr_scale,
            peft=peft,
        )

        num_heads = div(self._config.d_inner, self._config.state_size)
        num_head_groups = div(self._config.d_xb, self._config.state_size)

        state_dim = TensorDim("state", self._config.state_size)

        head_groups_dim = TensorDim(
            "head_groups", num_head_groups, self._distributed_config.get_distributed_dim(DistributedDimNames.tensor)
        )
        group_heads_dim = TensorDim("group_heads", div(num_heads, num_head_groups))

        heads_dim = CompositeTensorDim("heads", (head_groups_dim, group_heads_dim))

        inner_dim = CompositeTensorDim("inner", (head_groups_dim, group_heads_dim, state_dim))
        xb_dim = CompositeTensorDim("xb", (head_groups_dim, state_dim))

        # DT projection
        dt_rank_dim = TensorDim("dt_rank", self._config.dt_rank)

        inner_projection_dim = ConcatenatedTensorDim(
            "inner_projection",
            (inner_dim, xb_dim, xb_dim, inner_dim),
        )

        self._local_heads = heads_dim.size
        self._local_head_groups = head_groups_dim.size
        self._group_heads = div(self._local_heads, self._local_head_groups)
        self._local_inner_size = inner_dim.size
        self._local_xb_size = xb_dim.size
        convolution_dim = inner_dim if self._config.repeat_kv_before_conv else xb_dim

        self.convolution = self._config.convolution_layer.get_layer(
            convolution_dim,
            default_activation=ActivationType.silu,
            lr_scale=self._lr_scale,
            peft=self._peft,
        )
        # TODO: Use x_layer, b_layer, c_layer
        self.in_proj = self._config.z_layer.get_layer(
            hidden_dim,
            inner_projection_dim,
            default_weight_initialization=init_normal_(0, (2 / hidden_dim.global_size) ** 0.5),
            default_add_bias=self._config.add_linear_biases,
            sequence_parallel=self._sequence_parallel,
            lr_scale=self._lr_scale,
            peft=self._peft,
        )
        self.dt_in_proj = self._config.dt_input_layer.get_layer(
            hidden_dim,
            dt_rank_dim,
            default_weight_initialization=init_normal_(0, (2 / hidden_dim.global_size) ** 0.5),
            default_add_bias=self._config.add_linear_biases,
            lr_scale=self._lr_scale,
            peft=self._peft,
        )
        self.dt_proj = self._config.dt_layer.get_layer(
            dt_rank_dim,
            inner_dim,
            default_weight_initialization=init_uniform_centered_(self._config.dt_rank**-0.5),
            default_bias_initialization=init_dtprojbias(),
            sequence_parallel=self._sequence_parallel,
            lr_scale=self._lr_scale,
            peft=self._peft,
        )
        self.A_log = self._config.a_log_weight.get_parameter(
            (inner_dim, state_dim),
            default_initialization=init_a(self._config.state_size, self._config.d_inner),
            weight_decay=False,
            lr_scale=self._lr_scale,
            peft=self._peft,
        )
        # D "skip" parameter
        self.D = self._config.d_weight.get_parameter(
            (inner_dim,),
            default_initialization=init_ones_,
            weight_decay=False,
            lr_scale=self._lr_scale,
            peft=self._peft,
        )
        self.out_proj = self._config.output_layer.get_layer(
            inner_dim,
            hidden_dim,
            default_weight_initialization=init_normal_(0, (2 / self._config.d_inner) ** 0.5),
            default_add_bias=self._config.add_linear_biases,
            sequence_parallel=self._sequence_parallel,
            lr_scale=self._lr_scale,
            peft=self._peft,
        )

        # Debug dims
        self._xz_dims = (
            BlockDimNames.batch,
            inner_dim,
            BlockDimNames.sequence_q,
        )
        self._bc_dims = (
            BlockDimNames.batch,
            heads_dim,
            state_dim,
            BlockDimNames.sequence_q,
        )

    def forward(
        self,
        input_: torch.Tensor,
        kwargs: dict[str, typing.Any],
        losses: dict[str, typing.Any] | None = None,
        metrics: dict[str, typing.Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert _mamba_available

        # inner_projection : (batch/local_sequence, local_sequence/batch, hidden)
        #   -> (batch/sequence, sequence/batch, local_inner_projection)
        inner_projection = self.in_proj(input_)
        dt = self.dt_proj(self.dt_in_proj(input_))
        # Standardize to (batch, sequence, local_inner_projection)
        if kwargs[BlockKwargs.sequence_first]:
            inner_projection = inner_projection.transpose(0, 1)
            dt = dt.transpose(0, 1)

        sequence_length = inner_projection.size(1)

        z, x, b, c = torch.split(
            inner_projection,
            [self._local_inner_size, self._local_xb_size, self._local_xb_size, self._local_inner_size],
            dim=2,
        )

        # z: (batch, sequence, local_heads * state) -> (batch, local_heads * state, sequence)
        z = z.transpose(1, 2)

        # x: (batch, sequence, local_head_groups * state) -> (batch, local_heads * state, sequence)
        x = x.transpose(1, 2)
        if self._config.repeat_kv_before_conv:
            x = self.convolution(
                x.unflatten(1, (self._local_head_groups, self._config.state_size))
                .repeat_interleave(self._group_heads, 1, output_size=self._local_heads)
                .flatten(1, 2)
            )
        else:
            x = (
                self.convolution(x)
                .unflatten(1, (self._local_head_groups, self._config.state_size))
                .repeat_interleave(self._group_heads, 1, output_size=self._local_heads)
                .flatten(1, 2)
            )

        # b: (batch, sequence, local_head_groups * state) -> (batch, local_heads, state, sequence)
        b = (
            b.transpose(1, 2)
            .unflatten(1, (self._local_head_groups, self._config.state_size))
            .repeat_interleave(self._group_heads, 1, output_size=self._local_heads)
        )

        # c: (batch, sequence, heads * state) -> (batch, heads, state, sequence)
        c = c.transpose(1, 2).unflatten(1, (self._local_heads, self._config.state_size))

        # dt: (batch, sequence, heads * state) -> (batch, heads * state, sequence)
        dt = dt.transpose(1, 2)

        if self._debug.enabled:
            self._debug(z, "z", self._xz_dims, kwargs)
            self._debug(x, "x", self._xz_dims, kwargs)
            self._debug(b, "b", self._bc_dims, kwargs)
            self._debug(c, "c", self._bc_dims, kwargs)
            self._debug(dt, "dt", self._xz_dims, kwargs)

        y = selective_scan_fn(
            x,
            dt,
            -torch.exp(self.A_log.float()),
            b,
            c,
            self.D.float(),
            z,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )

        if self._debug.enabled:
            self._debug(y, "y", self._xz_dims, kwargs)

        # y: (batch, local_heads * state, sequence) -> (batch, sequence, local_heads * state)
        y = y.transpose(1, 2)[:, :sequence_length]
        if kwargs[BlockKwargs.sequence_first]:
            # TODO: Is contiguous needed?
            y = y.transpose(0, 1).contiguous()
        # (batch/sequence, sequence/batch, local_heads * state)
        #   -> (batch/local_sequence, local_sequence/batch, hidden)
        return self.out_proj(y)

    def get_compute_usage(self, input_: TensorMeta, kwargs: dict[str, typing.Any], config: ResourceUsageConfig) -> int:
        # TODO: Implement.
        raise NotImplementedError()
