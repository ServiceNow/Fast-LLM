import logging
import typing

import einops
import torch

from fast_llm.engine.config_utils.tensor_space import DefaultDimNames, TensorSpace
from fast_llm.functional.config import ActivationType
from fast_llm.layers.common.linear import InputParallelLinear, OutputParallelLinear
from fast_llm.layers.ssm.config import SSMConfig, SSMDimNames
from fast_llm.layers.transformer.config import TransformerConfig, TransformerDimNames, TransformerKwargs
from fast_llm.layers.transformer.transformer import Mixer
from fast_llm.tensor import ParameterMeta, init_kaiming_, init_ones_, init_uniform_centered_, init_zeros_
from fast_llm.utils import get_lr_scale

logger = logging.getLogger(__name__)


try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined as _mamba_chunk_scan_combined  # noqa

    _mamba_available = True
except (ImportError, RuntimeError):
    _mamba_available = False


try:
    from causal_conv1d import causal_conv1d_fn as _causal_conv1d_fn  # noqa

    _causal_conv1d_available = True
except (ImportError, RuntimeError):
    _causal_conv1d_available = False


class DiscreteMamba2(Mixer):
    """DiscreteMamba2 (This code is adapted from https://github.com/cartesia-ai/edge/blob/main/cartesia-pytorch/cartesia_pytorch/Llamba/mixers/discrete_mamba2.py)."""

    _mixer_name: typing.ClassVar[str] = "discrete_mamba_2"

    def __init__(
        self,
        config: SSMConfig,
        block_index: int,
        tensor_space: TensorSpace,
        transformer_config: TransformerConfig,
    ):
        super().__init__(tensor_space, block_index, debug_level=transformer_config.debug_transformer)
        self._config: SSMConfig = config
        layer_lr_scale = config.per_layer_lr_scale[block_index] if config.per_layer_lr_scale else None
        lr_scale = get_lr_scale(self._config.mamba_lr_scale, layer_lr_scale)

        inner_dim = tensor_space.get_tensor_dim(SSMDimNames.composite_heads_and_head_dim)
        hidden_dim = tensor_space.get_tensor_dim(TransformerDimNames.hidden)
        conv1d_dim = tensor_space.get_tensor_dim(SSMDimNames.concatenated_convolution)
        heads_dim = tensor_space.get_tensor_dim(SSMDimNames.composite_heads)

        # local_head_groups = head_groups / TP
        self._local_head_groups = tensor_space.get_tensor_dim(SSMDimNames.head_groups).size
        # local_heads = local_head_groups * group_heads
        self._local_heads = heads_dim.size
        # local_inner_size = local_heads * head_size
        self._local_inner_size = inner_dim.size
        # local_bc_size = local_head_groups * state
        self._local_bc_size = tensor_space.get_tensor_dim(SSMDimNames.composite_head_groups_and_state).size

        # TODO: double check initializations
        # Projections
        self.in_proj = OutputParallelLinear(
            hidden_dim,
            tensor_space.get_tensor_dim(name=SSMDimNames.concatenated_inner_projection),
            bias=config.add_bias_linear,
            weight_init_method=init_kaiming_(transformer_config.hidden_size),
            sequence_parallel=self._sequence_parallel,
            lr_scale=lr_scale,
        )
        if not config.add_bias_linear:
            self.z_bias = ParameterMeta.from_dims(
                (inner_dim,),
                weight_decay=False,
                init_method=init_zeros_,
                lr_scale=lr_scale,
            )
        self.conv1d_weight = ParameterMeta.from_dims(
            (
                conv1d_dim,
                tensor_space.get_tensor_dim(DefaultDimNames.scalar),
                tensor_space.get_tensor_dim(name=SSMDimNames.convolution_kernel),
            ),
            init_method=init_uniform_centered_((conv1d_dim.global_size * self._config.conv_kernel_dimension) ** -0.5),
            lr_scale=lr_scale,
        )
        self.conv1d_bias = ParameterMeta.from_dims(
            (conv1d_dim,),
            init_method=init_uniform_centered_(self._config.conv_kernel_dimension**-0.5),
            lr_scale=lr_scale,
        )
        # D "skip" parameter
        self.D = ParameterMeta.from_dims(
            (heads_dim,),
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

    def forward(self, input_: torch.Tensor, kwargs: dict[str, typing.Any]) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert _mamba_available

        sequence_length = kwargs[TransformerKwargs.sequence_q_dim].global_size

        # Pad input to nearest multiple of chunklen
        padded_length = (1 + (sequence_length - 1) // self._config.chunk_size) * self._config.chunk_size
        if padded_length != sequence_length:
            assert not kwargs[TransformerKwargs.sequence_first] and input_.size(1) == sequence_length
            input_ = torch.nn.functional.pad(input_, (0, 0, 0, padded_length - sequence_length))

        # inner_projection : (batch/local_or_padded_sequence, local_sequence/batch, hidden)
        #   -> (batch/local_or_padded_sequence, local_sequence/batch, inner_projection)
        # inner_projection: (batch, local_or_padded_sequence, hidden) -> (batch, padded_sequence, local_inner_size)
        inner_projection = self.in_proj(input_)
        # Standardize to (batch, padded_sequence, inner_projection)
        if kwargs[TransformerKwargs.sequence_first]:
            inner_projection = inner_projection.transpose(0, 1)

        print("QAIKOFNMJOWENM inner_projection", inner_projection.shape)
        xBC, z, A_log = torch.split(
            inner_projection,
            [
                self._local_inner_size + 2 * self._local_bc_size,
                self._local_inner_size,
                self._local_heads,
            ],
            dim=-1,
        )
        print("QAIKOFNMJOWENM xBC", xBC.shape, self._local_inner_size, self._local_bc_size)
        print("QAIKOFNMJOWENM z", z.shape)
        print("QAIKOFNMJOWENM A_log", A_log.shape)

        # Convolutional layer
        # xbc: (batch, padded_sequence, local_heads * head_size + 2 * local_head_groups * state)
        xBC = self.convolutional_forward(xBC, padded_length)

        x, B, C = torch.split(
            xBC,
            [
                self._local_inner_size,
                self._local_bc_size,
                self._local_bc_size,
            ],
            dim=-1,
        )

        # x: (batch, padded_sequence, local_heads * head_size) -> (batch, padded_sequence, local_heads, head_size)
        x = einops.rearrange(x, "b l (h n) -> b l h n", h=self._local_heads)

        # b,c: (batch, padded_sequence, local_head_groups * state) -> (batch, padded_sequence, local_head_groups, state)
        B = einops.rearrange(B, "b l (h n) -> b l h n", h=self._local_head_groups)
        C = einops.rearrange(C, "b l (h n) -> b l h n", h=self._local_head_groups)

        # SSM forward
        y = _mamba_chunk_scan_combined(
            x=self._apply_a_log(x, A_log),
            dt=A_log,
            dt_softplus=True,
            A=-torch.ones(self._local_heads, device=A_log.device),
            B=B,
            C=C,
            chunk_size=self._config.chunk_size,
            return_final_states=False,
        )
        Du = torch.einsum("h,blhp->blhp", self.D, x)

        # Norm and gate
        if not self._config.add_bias_linear:
            z = z + self.z_bias

        # y: (batch, padded_sequence, local_heads, head_size) -> (batch, sequence, local_heads * head_size)
        y = ((y + Du).flatten(2, 3) * torch.nn.functional.silu(z))[:, :sequence_length]
        if kwargs[TransformerKwargs.sequence_first]:
            # TODO: Is contiguous needed?
            y = y.transpose(0, 1).contiguous()
        # out_proj: (batch/sequence, sequence/batch, local_heads * head_size)
        #   -> (batch/local_sequence, local_sequence/batch, hidden)
        a, b = self.out_proj(y)
        logger.info(f"EKFBN y {y.shape}")
        logger.info(f"EKFBN a {a.shape}")
        return self.out_proj(y)

    @torch.compile
    def _apply_a_log(self, x: torch.Tensor, A_log: torch.Tensor) -> torch.Tensor:
        return x / torch.nn.functional.softplus(A_log).to(x.dtype).unsqueeze(-1)

    def convolutional_forward(self, xBC, padded_len):
        """Convolutional layer forward pass for the full sequence."""
        if _causal_conv1d_available and self._config.activation_type in (
            ActivationType.silu,
            ActivationType.identity,
        ):
            xBC = _causal_conv1d_fn(
                xBC.transpose(1, 2),
                self.conv1d_weight.squeeze(1),
                self.conv1d_bias,
                activation=(
                    None
                    if self._config.activation_type == ActivationType.identity
                    else self._config.activation_type.value
                ),
            ).transpose(1, 2)
        else:
            xBC = self._config.activation_type.activation_fn(
                torch.nn.functional.conv1d(
                    xBC.transpose(1, 2),
                    self.conv1d_weight,
                    bias=self.conv1d_bias,
                    groups=self.conv1d_weight.shape[0],
                    padding=self._config.conv_kernel_dimension - 1,
                )[..., :padded_len].transpose(1, 2)
            )
        return xBC
