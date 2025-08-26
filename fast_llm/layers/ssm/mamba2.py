import inspect
import logging
import typing

import einops
import torch

from fast_llm.engine.config_utils.tensor_space import DefaultDimNames, TensorDim, TensorSpace
from fast_llm.functional.config import ActivationType
from fast_llm.layers.common.linear import InputParallelLinear, Linear, OutputParallelLinear
from fast_llm.layers.common.normalization import MambaRMSNormGated
from fast_llm.layers.ssm.config import SSMConfig, SSMDimNames, SSMKwargs
from fast_llm.layers.ssm.mamba_layer import init_A, init_dtprojbias
from fast_llm.layers.transformer.config import TransformerConfig, TransformerDimNames, TransformerKwargs
from fast_llm.layers.transformer.transformer import Mixer
from fast_llm.tensor import LambdaInitializer, ParameterMeta, init_kaiming_, init_ones_, init_uniform_centered_
from fast_llm.utils import Assert, div, get_lr_scale

_mamba_varlen = False
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn  # noqa
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

    _mamba_available = True
    sig = inspect.signature(selective_scan_fn)
    if "position_indices" in sig.parameters:
        _mamba_varlen = True
        logging.warning("Using selective_scan_fn from varlen_mamba that supports packing")
    else:
        _mamba_varlen = False
        logging.warning("Using selective_scan_fn from original mamba without packing support")
        # for training with packing install https://github.com/jxiw/varlen_mamba
        # see https://github.com/jxiw/M1/blob/main/HYBRID_PACK.md

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

    _XZ_DIMS = (
        TransformerDimNames.batch,
        SSMDimNames.composite_heads_and_head_dim,
        TransformerDimNames.sequence_q,
    )
    _BC_DIMS = (
        TransformerDimNames.batch,
        SSMDimNames.composite_heads,
        SSMDimNames.state,
        TransformerDimNames.sequence_q,
    )

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

        inner_dim: TensorDim = tensor_space[SSMDimNames.composite_heads_and_head_dim]
        xb_dim = tensor_space[SSMDimNames.composite_head_groups_and_head]
        hidden_dim: TensorDim = tensor_space[TransformerDimNames.hidden]
        dt_rank_dim = tensor_space[SSMDimNames.dt_rank]

        self._local_heads = tensor_space[SSMDimNames.composite_heads].size
        self._local_head_groups = tensor_space[SSMDimNames.head_groups].size
        self._group_heads = div(self._local_heads, self._local_head_groups)
        self._local_inner_size = inner_dim.size
        self._local_xb_size = xb_dim.size

        state_size = tensor_space[SSMDimNames.state].size
        div(self._local_inner_size, state_size)

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

    def forward(self, input_: torch.Tensor, kwargs: dict[str, typing.Any]) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Note, we are nto doing "read" sequence-tensor parallel trainign here, since inner_projection is gathered over all GPUS.
        This is also desired, since the currently used mamba kernel does not support STP.
        TODO: use correct kernel from Mamba2!
        """
        assert _mamba_available
        assert _causal_conv1d_available
        cu_seqlens = kwargs[SSMKwargs.cu_seqlens]
        seq_idx = kwargs[SSMKwargs.seq_idx]
        position_indices = kwargs[SSMKwargs.ssm_position_ids]

        # inner_projection : (batch/local_sequence, local_sequence/batch, hidden)
        #   -> (batch/sequence, sequence/batch, inner_projection)
        inner_projection = self.in_proj(input_)
        dt = self.dt_proj(self.dt_in_proj(input_)) + self.dt_proj_bias
        # Standardize to (batch, sequence, inner_projection)
        if kwargs[TransformerKwargs.sequence_first]:
            inner_projection = inner_projection.transpose(0, 1)
            dt = dt.transpose(0, 1)

        sequence_length = inner_projection.size(1)
        # is this like Mamba1, the conv is only on the x?
        z, x, b, c = torch.split(
            inner_projection,
            [self._local_inner_size, self._local_xb_size, self._local_xb_size, self._local_inner_size],
            dim=2,
        )

        # z: (batch, sequence, local_heads * state) -> (batch, local_heads * state, sequence)
        z = z.transpose(1, 2)

        # x: (batch, sequence, local_head_groups * state) -> (batch, local_heads * state, sequence)
        x = x.transpose(1, 2)
        # x: (batch, local_heads * state, sequence) -> (batch, local_head_per_groups, state, sequence)
        if self._config.repeat_kv_before_conv:
            x = (
                x.unflatten(1, (self._local_head_groups, self._config.state_size))
                .repeat_interleave(self._group_heads, 1, output_size=self._local_heads)
                .flatten(1, 2)
            )

        if cu_seqlens is not None:
            # from https://github.com/jxiw/M1/blob/d92b53faa640f8ebf624d3e9e771fe24648ef014/rl/verl/verl/models/mamba/hybrid_wrapper.py#L152
            x = _causal_conv1d_fn(
                x=x.transpose(1, 2).contiguous().transpose(1, 2),
                weight=self.conv1d_weight.squeeze(1),
                bias=self.conv1d_bias,
                seq_idx=seq_idx,
                activation="silu",
            )
        else:
            x = _causal_conv1d_fn(x=x, weight=self.conv1d_weight.squeeze(1), bias=self.conv1d_bias, activation="silu")

        if not self._config.repeat_kv_before_conv:
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
            self._debug_log(z, "z", self._XZ_DIMS, kwargs)
            self._debug_log(x, "x", self._XZ_DIMS, kwargs)
            self._debug_log(b, "b", self._BC_DIMS, kwargs)
            self._debug_log(c, "c", self._BC_DIMS, kwargs)
            self._debug_log(dt, "dt", self._XZ_DIMS, kwargs)

        if not _mamba_varlen:
            Assert.eq(cu_seqlens, None, msg="This version of Mamba2 does not support cu_seqlens, install verlen mamba")
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
        else:
            position_indices = position_indices if cu_seqlens is not None else None

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
                position_indices=position_indices,
            )

        if self._debug_level:
            self._debug_log(y, "y", self._XZ_DIMS, kwargs)

        # y: (batch, local_heads * state, sequence) -> (batch, sequence, local_heads * state)
        y = y.transpose(1, 2)[:, :sequence_length]
        if kwargs[TransformerKwargs.sequence_first]:
            # TODO: Is contiguous needed?
            y = y.transpose(0, 1).contiguous()
        # (batch/sequence, sequence/batch, local_heads * state)
        #   -> (batch/local_sequence, local_sequence/batch, hidden)
        return self.out_proj(y)


class NemotronHMamba2(Mixer):
    """
    This is the actual Mamab2, called NemotronHMamba2 for historical reasons.
    Decompesl, d_state and head_dim.
    Head dimention -- later head dimention means me project hidden statte into larger space (more channel mixing)
    Larger state size -- more temporar memory.

    This code is adapted from https://huggingface.co/nvidia/Nemotron-H-8B-Base-8K/blob/main/modeling_nemotron_h.py
    """

    _mixer_name: typing.ClassVar[str] = "mamba_2"

    _XZ_DIMS = (
        TransformerDimNames.batch,
        SSMDimNames.composite_heads_and_head_dim,
        TransformerDimNames.sequence_q,
    )
    _BC_DIMS = (
        TransformerDimNames.batch,
        SSMDimNames.composite_heads,
        SSMDimNames.state,
        TransformerDimNames.sequence_q,
    )

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

        inner_dim: TensorDim = tensor_space[SSMDimNames.composite_heads_and_head_dim]
        inner_dim_non_tp: TensorDim = tensor_space[SSMDimNames.composite_heads_and_head_dim_nontp]
        c_dim: TensorDim = tensor_space[SSMDimNames.composite_heads_and_state_dim]
        xb_dim = tensor_space[SSMDimNames.composite_head_groups_and_head]
        bb_dim = tensor_space[SSMDimNames.composite_head_groups_and_state]
        hidden_dim: TensorDim = tensor_space[TransformerDimNames.hidden]

        self._head_dim_size: TensorDim = tensor_space[SSMDimNames.head_dim].size
        self._local_heads = tensor_space[SSMDimNames.composite_heads].size
        self._local_head_groups = tensor_space[SSMDimNames.head_groups].size
        self._group_heads = div(self._local_heads, self._local_head_groups)
        Assert.eq(self._local_heads, self._local_head_groups * self._group_heads)

        self._local_inner_size = inner_dim.size
        self._local_c_size = c_dim.size

        Assert.eq(self._local_inner_size, self._head_dim_size * self._local_heads)
        self._local_xb_size = xb_dim.size  # x has head dim and is for each head group
        self._local_bb_size = bb_dim.size  # b has state dim and is for each head group
        Assert.eq(self._local_xb_size, self._head_dim_size * self._local_head_groups)
        Assert.eq(self._local_bb_size, self._config.state_size * self._local_head_groups)

        conv1d_dim = tensor_space[SSMDimNames.conv1d_dim]  # applied to xBC, so d_xb + d_bb + c_dim
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
            weight_init_method=init_kaiming_(transformer_config.hidden_size),
            sequence_parallel=self._sequence_parallel,
            lr_scale=lr_scale,
        )

        # project single number per head
        self.dt_in_proj = OutputParallelLinear(
            hidden_dim,
            tensor_space[SSMDimNames.composite_heads],
            bias=config.add_bias_linear,
            weight_init_method=init_kaiming_(transformer_config.hidden_size),
            sequence_parallel=self._sequence_parallel,
            lr_scale=lr_scale,
        )

        self.dt_proj_bias = ParameterMeta.from_dims(
            (tensor_space[SSMDimNames.composite_heads],),
            init_method=init_dtprojbias(self._config.dt_max, self._config.dt_min, self._config.dt_init_floor),
            lr_scale=lr_scale,
        )

        def init_A_uniform(A_init_range: tuple[float, float] = (1, 16)) -> LambdaInitializer:
            def init_(meta: ParameterMeta, tensor: torch.Tensor, generator: torch.Generator) -> None:  # noqa
                tensor.uniform_(*A_init_range).log_()

            return LambdaInitializer(init_, requires_global_initialization=True)

        self.A_log = ParameterMeta.from_dims(
            (tensor_space[SSMDimNames.composite_heads],),
            init_method=init_A_uniform(A_init_range=(1, 16)),
            lr_scale=lr_scale,
            weight_decay=False,
        )
        self.D = ParameterMeta.from_dims(
            (tensor_space[SSMDimNames.composite_heads],),  # can also be nheads x headim
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
        # TODO: this norm does nto support TP. So we need a workaround!
        self.norm = MambaRMSNormGated(
            inner_dim_non_tp,
            group_size=self._local_inner_size,
            eps=1e-5,
            lr_scale=lr_scale,
        )

    def forward(self, input_: torch.Tensor, kwargs: dict[str, typing.Any]) -> tuple[torch.Tensor, torch.Tensor | None]:
        """ """
        assert _mamba_available
        assert _causal_conv1d_available
        cu_seqlens = kwargs[SSMKwargs.cu_seqlens]
        seq_idx = kwargs[SSMKwargs.seq_idx]

        # inner_projection : (batch/local_sequence, local_sequence/batch, hidden)
        #   -> (batch/sequence, sequence/batch, inner_projection)
        inner_projection = self.in_proj(input_)
        dt = self.dt_in_proj(input_)  # bs, seq, heads
        # Standardize to (batch, sequence, inner_projection)
        if kwargs[TransformerKwargs.sequence_first]:
            inner_projection = inner_projection.transpose(0, 1)
            dt = dt.transpose(0, 1)
        # note: self.in_proj gathers full sequence length here
        sequence_length = inner_projection.size(1)

        z, xBC = torch.split(
            inner_projection,
            [self._local_inner_size, self._local_xb_size + self._local_bb_size + self._local_c_size],
            dim=2,
        )

        if cu_seqlens is not None:
            xBC = _causal_conv1d_fn(
                xBC.transpose(1, 2),
                weight=self.conv1d_weight.squeeze(1),
                bias=self.conv1d_bias,
                seq_idx=seq_idx,
                activation="silu",
            ).transpose(1, 2)
        else:
            xBC = _causal_conv1d_fn(
                x=xBC.transpose(1, 2), weight=self.conv1d_weight.squeeze(1), bias=self.conv1d_bias, activation="silu"
            ).transpose(1, 2)

        x, b, c = torch.split(xBC, [self._local_xb_size, self._local_bb_size, self._local_c_size], dim=-1)
        # simulate GQA by repeating heads in x,b, x -> v, B -> k, C -> q
        x = einops.rearrange(
            x, "b l (local_head_groups head_dim) -> b local_head_groups l head_dim", head_dim=self._head_dim_size
        )  # x is b x local_head_groups x l x head_dim
        b = einops.rearrange(
            b,
            "b l (local_head_groups state_size) -> b local_head_groups l state_size",
            state_size=self._config.state_size,
        )  # b is b x local_head_groups x l x state_size
        batch, num_key_value_heads, slen, head_dim = x.shape
        x = x[:, :, None, :, :].expand(batch, num_key_value_heads, self._group_heads, slen, head_dim)
        x = x.reshape(batch, num_key_value_heads * self._group_heads, slen, head_dim)
        b = b[:, :, None, :, :].expand(batch, num_key_value_heads, self._group_heads, slen, self._config.state_size)
        b = b.reshape(batch, num_key_value_heads * self._group_heads, slen, self._config.state_size)

        if self._debug_level:
            self._debug_log(z, "z", self._XZ_DIMS, kwargs)
            self._debug_log(x, "x", self._XZ_DIMS, kwargs)
            self._debug_log(b, "b", self._BC_DIMS, kwargs)
            self._debug_log(c, "c", self._BC_DIMS, kwargs)
            self._debug_log(dt, "dt", self._XZ_DIMS, kwargs)

        dt_limit_kwargs = (
            {}
        )  # can be used to set time-step limit as in https://huggingface.co/nvidia/Nemotron-H-8B-Base-8K/blob/main/modeling_nemotron_h.py#L424
        # c is b x seq x (heads * state)
        # b is b x heads x seq x state)
        # x is b x heads x seq x head_dim
        # note, we could used mamba_split_conv1d_scan_combined directly for training, however because of the GQA, we need to use the chunked version.
        y = mamba_chunk_scan_combined(
            einops.rearrange(x, "b g l p -> b l g p"),
            dt,
            A=-torch.exp(self.A_log.float()),
            B=einops.rearrange(b, "b g l n -> b l g n"),
            C=einops.rearrange(c, "b l (g n) -> b l g n", g=self._local_heads),
            chunk_size=self._config.chunk_size,
            D=self.D,
            z=None,
            dt_bias=self.dt_proj_bias,
            dt_softplus=True,
            seq_idx=seq_idx,  # assume this is used for packing
            cu_seqlens=cu_seqlens,  # assume this is used for packing, but maybe not needed at training
            **dt_limit_kwargs,
            return_final_states=False,
            return_varlen_states=False,
        )

        if self._debug_level:
            self._debug_log(y, "y", self._XZ_DIMS, kwargs)

        # y: (batch, local_heads * state, sequence) -> (batch, sequence, local_heads * state)
        y = y.view(batch, sequence_length, -1)

        if kwargs[TransformerKwargs.sequence_first]:
            # TODO: Is contiguous needed?
            y = y.transpose(0, 1).contiguous()
            z = z.transpose(0, 1).contiguous()
        # in tp need to to gather the y and z, cause norm does not
        # gate norm
        y = self.norm(y, gate=z)
        # (batch/sequence, sequence/batch, local_heads * state)
        #   -> (batch/local_sequence, local_sequence/batch, hidden)
        out = self.out_proj(y)
        return out
