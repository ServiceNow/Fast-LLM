import torch

from fast_llm.engine.config_utils.tensor_space import TensorDim, TensorSpace
from fast_llm.functional.config import ActivationType
from fast_llm.layers.common.linear import InputParallelLinear, Linear, OutputParallelLinear
from fast_llm.layers.ssm.config import SSMConfig, SSMDimNames
from fast_llm.layers.ssm.discrete_mamba2 import bias_init_method
from fast_llm.layers.ssm.mamba_layer import init_A, init_dtprojbias
from fast_llm.layers.transformer.config import TransformerDimNames, TransformerKwargs
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


class Mamba2(Mixer):
    """
    This code is adapted from https://github.com/jxiw/M1/blob/537a1ca5407a786a99dc6c721873493cf8750d5e/mamba/hybrid_mamba_layer.py
    """

    def __init__(
        self,
        config: SSMConfig,
        layer_idx: int,
        tensor_space: TensorSpace,
    ):
        super().__init__()
        self._config: SSMConfig = config
        Assert.eq(self._config.activation_type, ActivationType.silu)
        layer_lr_scale: float | None = config.per_layer_lr_scale[layer_idx] if config.per_layer_lr_scale else None
        lr_scale: float | tuple[float | None, ...] | None = get_lr_scale(self._config.mamba_lr_scale, layer_lr_scale)

        inner_dim: TensorDim = tensor_space.get_tensor_dim(name=SSMDimNames.composite_heads_and_state)
        hidden_dim: TensorDim = tensor_space.get_tensor_dim(name=TransformerDimNames.hidden)
        dt_rank_dim = tensor_space.get_tensor_dim(name=SSMDimNames.dt_rank)

        self._head_groups = div(self._config.d_xb, self._config.state_size)
        self._heads = div(self._config.d_inner, self._config.state_size)
        self._group_heads = div(self._heads, self._head_groups)

        conv1d_dim = (
            inner_dim
            if self._config.repeat_kv_before_conv
            else tensor_space.get_tensor_dim(name=SSMDimNames.composite_head_groups_and_state)
        )
        self.conv1d_weight = ParameterMeta.from_dims(
            (conv1d_dim, tensor_space.get_tensor_dim(name=SSMDimNames.conv_kernel)),
            init_method=init_uniform_centered_((conv1d_dim.size * self._config.conv_kernel_dimension) ** -0.5),
            lr_scale=lr_scale,
        )
        self.conv1d_bias = ParameterMeta.from_dims(
            (conv1d_dim,), init_method=bias_init_method(self._config.conv_kernel_dimension**-0.5), lr_scale=lr_scale
        )
        self.in_proj = OutputParallelLinear(
            hidden_dim,
            tensor_space.get_tensor_dim(name=SSMDimNames.concatenated_inner_projection),
            bias=config.add_bias_linear,
            weight_init_method=init_kaiming_(hidden_dim.size),
            lr_scale=lr_scale,
        )

        self.dt_in_proj = Linear(
            hidden_dim,
            dt_rank_dim,
            bias=config.add_bias_linear,
            weight_init_method=init_kaiming_(hidden_dim.size),
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
            lr_scale=lr_scale,
        )
        # define bias outside the linear layer since its also used in the selective_scan_fn
        self.dt_proj_bias = ParameterMeta.from_dims(
            (inner_dim,),
            init_method=init_dtprojbias(self._config.dt_max, self._config.dt_min, self._config.dt_init_floor),
            lr_scale=lr_scale,
        )
        self.A_log = ParameterMeta.from_dims(
            (inner_dim, tensor_space.get_tensor_dim(name=SSMDimNames.state)),
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
        )

    def forward(self, hidden_states, kwargs):
        assert _mamba_available
        assert _causal_conv1d_available

        inner_projection = self.in_proj(hidden_states)
        dt = self.dt_in_proj(hidden_states)
        # Standardize to (batch, sequence, inner_projection)
        if kwargs[TransformerKwargs.sequence_first]:
            inner_projection = inner_projection.transpose(0, 1)
            dt = dt.transpose(0, 1)
        sequence_length = hidden_states.size(1)

        z, x, b, c = torch.split(
            inner_projection,
            [self._config.d_inner, self._config.d_xb, self._config.d_xb, self._config.d_inner],
            dim=2,
        )

        # z: (batch, sequence, heads * state) -> (batch, heads * state, sequence)
        z = z.transpose(1, 2)

        # x: (batch, sequence, head_groups * state) -> (batch, heads * state, sequence)
        x = x.transpose(1, 2)
        if self._config.repeat_kv_before_conv:
            x = (
                x.unflatten(1, (self._head_groups, self._config.state_size))
                .repeat_interleave(self._group_heads, 1, output_size=self._heads)
                .flatten(1, 2)
            )
            x = _causal_conv1d_fn(x=x, weight=self.conv1d_weight, bias=self.conv1d_bias, activation="silu")
        else:
            x = _causal_conv1d_fn(x=x, weight=self.conv1d_weight, bias=self.conv1d_bias, activation="silu")
            x = (
                x.unflatten(1, (self._head_groups, self._config.state_size))
                .repeat_interleave(self._group_heads, 1, output_size=self._heads)
                .flatten(1, 2)
            )

        # b: (batch, sequence, head_groups * state) -> (batch, heads, state, sequence)
        b = (
            b.transpose(1, 2)
            .unflatten(1, (self._head_groups, self._config.state_size))
            .repeat_interleave(self._group_heads, 1, output_size=self._heads)
        )

        # c: (batch, sequence, heads * state) -> (batch, heads, state, sequence)
        c = c.transpose(1, 2).unflatten(1, (self._heads, self._config.state_size))

        # dt: (batch, sequence, dt_rank) -> (batch, heads * state, sequence)
        dt = (self.dt_proj(dt) + self.dt_proj_bias).transpose(1, 2)

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

        # y: (batch, heads * state, sequence) -> out: (batch, sequence, hidden)
        out = self.out_proj(y.transpose(1, 2))[:, :sequence_length]
        if kwargs[TransformerKwargs.sequence_first]:
            out = out.transpose(0, 1)
        # TODO: Is contiguous needed?
        return out.contiguous(), None
