import logging
import typing

import torch

from fast_llm.engine.config_utils.initialization import init_ones_, init_uniform_centered_
from fast_llm.engine.config_utils.tensor_space import DefaultDimNames, TensorDim, TensorSpace
from fast_llm.functional.config import ActivationType
from fast_llm.layers.block.block import BlockLayer
from fast_llm.layers.block.config import BlockConfig, BlockKwargs
from fast_llm.layers.common.linear import InputParallelLinear, Linear, OutputParallelLinear
from fast_llm.layers.ssm.config import SSMConfig, SSMDimNames
from fast_llm.layers.ssm.mamba_layer import init_A, init_dtprojbias, init_kaiming_
from fast_llm.tensor import ParameterMeta
from fast_llm.utils import Assert, div, get_lr_scale

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn  # noqa

    _mamba_available = True
except (ImportError, RuntimeError):
    _mamba_available = False

try:
    from causal_conv1d import causal_conv1d_fn as _causal_conv1d_fn  # noqa

    _causal_conv1d_available = True
except (ImportError, RuntimeError):
    _causal_conv1d_available = False

logger = logging.getLogger(__name__)


class Mamba2(BlockLayer):
    """
    This code is adapted from https://github.com/jxiw/M1/blob/537a1ca5407a786a99dc6c721873493cf8750d5e/mamba/hybrid_mamba_layer.py
    """

    _mixer_name: typing.ClassVar[str] = "mamba_2"

    _XZ_DIMS = (
        SSMDimNames.batch,
        SSMDimNames.composite_heads_and_head_dim,
        SSMDimNames.sequence_q,
    )
    _BC_DIMS = (
        SSMDimNames.batch,
        SSMDimNames.composite_heads,
        SSMDimNames.state,
        SSMDimNames.sequence_q,
    )

    def __init__(
        self,
        config: SSMConfig,
        tensor_space: TensorSpace,
        block_index: int,
        block_config: BlockConfig,
    ):
        super().__init__(tensor_space, block_index, debug_level=block_config.debug_transformer)
        self._config: SSMConfig = config
        Assert.eq(self._config.activation_type, ActivationType.silu)
        layer_lr_scale: float | None = (
            block_config.per_layer_lr_scale[block_index] if block_config.per_layer_lr_scale else None
        )
        lr_scale: float | tuple[float | None, ...] | None = get_lr_scale(self._config.mamba_lr_scale, layer_lr_scale)

        inner_dim: TensorDim = tensor_space[SSMDimNames.composite_heads_and_head_dim]
        xb_dim = tensor_space[SSMDimNames.composite_head_groups_and_state]
        hidden_dim: TensorDim = tensor_space[SSMDimNames.hidden]
        dt_rank_dim = tensor_space[SSMDimNames.dt_rank]

        self._local_heads = tensor_space[SSMDimNames.composite_heads].size
        self._local_head_groups = tensor_space[SSMDimNames.head_groups].size
        self._group_heads = div(self._local_heads, self._local_head_groups)
        self._local_inner_size = inner_dim.size
        self._local_xb_size = xb_dim.size

        conv1d_dim = inner_dim if self._config.repeat_kv_before_conv else xb_dim
        self.conv1d_weight = ParameterMeta.from_dims(
            (
                conv1d_dim,
                tensor_space[DefaultDimNames.scalar],
                tensor_space[SSMDimNames.convolution_kernel],
            ),
            init_method=init_uniform_centered_((conv1d_dim.global_size * self._config.conv_kernel_dimension) ** -0.5),
            lr_scale=lr_scale,
        )
        self.conv1d_bias = ParameterMeta.from_dims(
            (conv1d_dim,),
            init_method=init_uniform_centered_(self._config.conv_kernel_dimension**-0.5),
            lr_scale=lr_scale,
        )
        self.in_proj = OutputParallelLinear(
            hidden_dim,
            tensor_space[SSMDimNames.concatenated_inner_projection],
            bias=config.add_bias_linear,
            weight_init_method=init_kaiming_(block_config.hidden_size),
            sequence_parallel=self._sequence_parallel,
            lr_scale=lr_scale,
        )

        self.dt_in_proj = Linear(
            hidden_dim,
            dt_rank_dim,
            bias=config.add_bias_linear,
            weight_init_method=init_kaiming_(block_config.hidden_size),
            lr_scale=lr_scale,
        )
        self.dt_proj = OutputParallelLinear(
            dt_rank_dim,
            inner_dim,
            bias=False,
            # Initialize special dt projection to preserve variance at initialization
            weight_init_method=self._config.dt_init.get_init_method(
                self._config.dt_rank**-0.5 * self._config.dt_scale
            ),
            sequence_parallel=self._sequence_parallel,
            lr_scale=lr_scale,
        )
        # define bias outside the linear layer since it's also used in the selective_scan_fn
        self.dt_proj_bias = ParameterMeta.from_dims(
            (inner_dim,),
            init_method=init_dtprojbias(self._config.dt_max, self._config.dt_min, self._config.dt_init_floor),
            lr_scale=lr_scale,
        )
        self.A_log = ParameterMeta.from_dims(
            (inner_dim, tensor_space[SSMDimNames.state]),
            init_method=init_A(self._config.state_size, self._config.d_inner),
            lr_scale=lr_scale,
            weight_decay=False,
        )
        self.D = ParameterMeta.from_dims(
            (inner_dim,),
            weight_decay=False,
            init_method=init_ones_,
            lr_scale=lr_scale,
        )
        self.out_proj = InputParallelLinear(
            inner_dim,
            hidden_dim,
            bias=config.add_bias_linear,
            weight_init_method=init_kaiming_(self._config.d_inner),
            sequence_parallel=self._sequence_parallel,
            # TODO: lr_scale?
        )

    def forward(
        self,
        input_: torch.Tensor,
        kwargs: dict[str, typing.Any],
        losses: dict[str, typing.Any] | None = None,
        metrics: dict[str, typing.Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert _mamba_available
        assert _causal_conv1d_available

        # inner_projection : (batch/local_sequence, local_sequence/batch, hidden)
        #   -> (batch/sequence, sequence/batch, inner_projection)
        inner_projection = self.in_proj(input_)
        dt = self.dt_proj(self.dt_in_proj(input_)) + self.dt_proj_bias
        # Standardize to (batch, sequence, inner_projection)
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
            x = (
                x.unflatten(1, (self._local_head_groups, self._config.state_size))
                .repeat_interleave(self._group_heads, 1, output_size=self._local_heads)
                .flatten(1, 2)
            )
            x = _causal_conv1d_fn(x=x, weight=self.conv1d_weight.squeeze(1), bias=self.conv1d_bias, activation="silu")
        else:
            x = _causal_conv1d_fn(x=x, weight=self.conv1d_weight.squeeze(1), bias=self.conv1d_bias, activation="silu")
            x = (
                x.unflatten(1, (self._local_head_groups, self._config.state_size))
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
            self._debug(z, "z", self._XZ_DIMS, kwargs)
            self._debug(x, "x", self._XZ_DIMS, kwargs)
            self._debug(b, "b", self._BC_DIMS, kwargs)
            self._debug(c, "c", self._BC_DIMS, kwargs)
            self._debug(dt, "dt", self._XZ_DIMS, kwargs)

        y = selective_scan_fn(
            x,
            dt,
            -torch.exp(self.A_log.float()),
            b,
            c,
            self.D.float(),
            z,
            delta_bias=self.dt_proj_bias.float(),
            delta_softplus=True,
        )

        if self._debug_level:
            self._debug_log(y, "y", self._XZ_DIMS, kwargs)

        # y: (batch, local_heads * state, sequence) -> (batch, sequence, local_heads * state)
        y = y.transpose(1, 2)[:, :sequence_length]
        if kwargs[BlockKwargs.sequence_first]:
            # TODO: Is contiguous needed?
            y = y.transpose(0, 1).contiguous()
        # (batch/sequence, sequence/batch, local_heads * state)
        #   -> (batch/local_sequence, local_sequence/batch, hidden)
        return self.out_proj(y)
