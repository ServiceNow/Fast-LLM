import logging
import typing

import torch

from fast_llm.engine.config_utils.tensor_space import (
    CompositeTensorDim,
    ConcatenatedTensorDim,
    DefaultDimNames,
    TensorDim,
    TensorSpace,
)
from fast_llm.engine.distributed.config import DistributedDimNames
from fast_llm.functional.config import ActivationType
from fast_llm.layers.common.linear import InputParallelLinear, Linear, OutputParallelLinear
from fast_llm.layers.ssm.config import SSMConfig
from fast_llm.layers.ssm.mamba_layer import init_A, init_dtprojbias
from fast_llm.layers.transformer.config import TransformerConfig, TransformerDimNames, TransformerKwargs
from fast_llm.layers.transformer.transformer import Mixer
from fast_llm.tensor import ParameterMeta, init_kaiming_, init_ones_, init_uniform_centered_
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


class Mamba2(Mixer):
    """
    This code is adapted from https://github.com/jxiw/M1/blob/537a1ca5407a786a99dc6c721873493cf8750d5e/mamba/hybrid_mamba_layer.py
    """

    _mixer_name: typing.ClassVar[str] = "mamba_2"

    def __init__(
        self,
        config: SSMConfig,
        tensor_space: TensorSpace,
        block_index: int,
        transformer_config: TransformerConfig,
    ):
        super().__init__(tensor_space, block_index, debug_level=transformer_config.debug_transformer)
        self._config: SSMConfig = config
        Assert.eq(self._config.activation_type, ActivationType.silu)
        layer_lr_scale: float | None = config.per_layer_lr_scale[block_index] if config.per_layer_lr_scale else None
        lr_scale: float | tuple[float | None, ...] | None = get_lr_scale(self._config.mamba_lr_scale, layer_lr_scale)

        num_heads = div(self._config.d_inner, self._config.state_size)
        num_head_groups = div(self._config.d_xb, self._config.state_size)

        hidden_dim: TensorDim = tensor_space[TransformerDimNames.hidden]
        state_dim = TensorDim("state", self._config.state_size)

        head_groups_dim = TensorDim(
            "head_groups",
            num_head_groups,
            self._tensor_space.distributed_config.get_distributed_dim(DistributedDimNames.tensor),
        )
        group_heads_dim = TensorDim("group_heads", div(num_heads, num_head_groups))

        heads_dim = CompositeTensorDim("heads", (head_groups_dim, group_heads_dim))

        inner_dim = CompositeTensorDim("inner", (head_groups_dim, group_heads_dim, state_dim))
        xb_dim = CompositeTensorDim("xb", (head_groups_dim, state_dim))
        convolution_kernel_dim = TensorDim("convolution_kernel", self._config.conv_kernel_dimension)

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

        conv1d_dim = inner_dim if self._config.repeat_kv_before_conv else xb_dim
        self.conv1d_weight = ParameterMeta.from_dims(
            (
                conv1d_dim,
                tensor_space[DefaultDimNames.scalar],
                convolution_kernel_dim,
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
            inner_projection_dim,
            bias=config.add_bias_linear,
            weight_init_method=init_kaiming_(transformer_config.hidden_size),
            sequence_parallel=self._sequence_parallel,
            lr_scale=lr_scale,
        )

        self.dt_in_proj = Linear(
            hidden_dim,
            dt_rank_dim,
            bias=config.add_bias_linear,
            weight_init_method=init_kaiming_(transformer_config.hidden_size),
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
            (inner_dim, state_dim),
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
            lr_scale=lr_scale,
        )
        if self._debug_level:
            self._xz_dims = (
                TransformerDimNames.batch,
                inner_dim,
                TransformerDimNames.sequence_q,
            )
            self._bc_dims = (
                TransformerDimNames.batch,
                heads_dim,
                state_dim,
                TransformerDimNames.sequence_q,
            )

    def forward(self, input_: torch.Tensor, kwargs: dict[str, typing.Any]) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert _mamba_available
        assert _causal_conv1d_available

        # inner_projection : (batch/local_sequence, local_sequence/batch, hidden)
        #   -> (batch/sequence, sequence/batch, local_inner_projection)
        inner_projection = self.in_proj(input_)
        dt = self.dt_proj(self.dt_in_proj(input_)) + self.dt_proj_bias
        # Standardize to (batch, sequence, local_inner_projection)
        if kwargs[TransformerKwargs.sequence_first]:
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

        if self._debug_level:
            self._debug_log(z, "z", self._xz_dims, kwargs)
            self._debug_log(x, "x", self._xz_dims, kwargs)
            self._debug_log(b, "b", self._bc_dims, kwargs)
            self._debug_log(c, "c", self._bc_dims, kwargs)
            self._debug_log(dt, "dt", self._xz_dims, kwargs)

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
            self._debug_log(y, "y", self._xz_dims, kwargs)

        # y: (batch, local_heads * state, sequence) -> (batch, sequence, local_heads * state)
        y = y.transpose(1, 2)[:, :sequence_length]
        if kwargs[TransformerKwargs.sequence_first]:
            # TODO: Is contiguous needed?
            y = y.transpose(0, 1).contiguous()
        # (batch/sequence, sequence/batch, local_heads * state)
        #   -> (batch/local_sequence, local_sequence/batch, hidden)
        return self.out_proj(y)
