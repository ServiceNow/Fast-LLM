import logging
import typing

import einops
import torch

from fast_llm.engine.config_utils.initialization import init_normal_, init_ones_, init_zeros_
from fast_llm.engine.config_utils.tensor_dim import CompositeTensorDim, ConcatenatedTensorDim, TensorDim
from fast_llm.engine.distributed.config import DistributedConfig, DistributedDimNames
from fast_llm.functional.config import ActivationType
from fast_llm.layers.block.block import BlockLayer
from fast_llm.layers.block.config import BlockConfig, BlockKwargs
from fast_llm.layers.ssm.config import DiscreteMamba2Config
from fast_llm.tensor import ParameterMeta
from fast_llm.utils import combine_lr_scales, div

logger = logging.getLogger(__name__)


try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined as _mamba_chunk_scan_combined  # noqa

    _mamba_available = True
except (ImportError, RuntimeError):
    _mamba_available = False


class DiscreteMamba2[ConfigType: DiscreteMamba2Config](BlockLayer[ConfigType]):
    """
    This code is adapted from https://github.com/cartesia-ai/edge/blob/main/cartesia-pytorch/cartesia_pytorch/Llamba/mixers/discrete_mamba2.py
    """

    _mixer_name: typing.ClassVar[str] = "discrete_mamba_2"

    def __init__(
        self,
        config: ConfigType,
        block_config: BlockConfig,
        distributed_config: DistributedConfig,
        hidden_dim: TensorDim,
        block_index: int,
        name: str,
        lr_scale: float | None,
    ):
        super().__init__(config, block_config, distributed_config, hidden_dim, block_index, name, lr_scale)
        state_dim = TensorDim("state", self._config.state_size)
        v_head_size_dim = TensorDim("v_head_size", div(self._config.d_inner, self._config.n_v_heads))

        head_groups_dim = TensorDim(
            "head_groups",
            self._config.n_qk_heads,
            self._distributed_config.get_distributed_dim(DistributedDimNames.tensor),
        )
        group_heads_dim = TensorDim("group_heads", div(self._config.n_v_heads, self._config.n_qk_heads))
        heads_dim = CompositeTensorDim("heads", (head_groups_dim, group_heads_dim))
        inner_dim = CompositeTensorDim("inner", (head_groups_dim, group_heads_dim, v_head_size_dim))
        bc_dim = CompositeTensorDim("bc", (head_groups_dim, state_dim))

        inner_projection_dim = ConcatenatedTensorDim(
            "inner_projection",
            (inner_dim, bc_dim, bc_dim, inner_dim, heads_dim),
        )
        convolution_dim = ConcatenatedTensorDim("convolution", (inner_dim, bc_dim, bc_dim))

        # local_head_groups = head_groups / TP
        self._local_head_groups = head_groups_dim.size
        # local_heads = local_head_groups * group_heads
        self._local_heads = heads_dim.size
        # local_inner_size = local_heads * head_size
        self._local_inner_size = inner_dim.size
        # local_bc_size = local_head_groups * state
        self._local_bc_size = bc_dim.size

        lr_scale = combine_lr_scales(self._lr_scale, self._config.mamba_lr_scale)

        # TODO: double check initializations
        # Projections

        # TODO: Use x_layer, b_layer, c_layer, a_log_layer
        self.in_proj = self._config.z_layer.get_layer(
            hidden_dim,
            inner_projection_dim,
            default_weight_initializer=init_normal_(0, (2 / self._config.d_inner) ** 0.5),
            default_add_bias=self._block_config.add_linear_biases,
            sequence_parallel=self._sequence_parallel,
            lr_scale=lr_scale,
        )
        if self.in_proj.bias is None:
            # TODO: Integrate to z_layer config?
            self.z_bias = ParameterMeta.from_dims(
                (inner_dim,),
                weight_decay=False,
                init_method=init_zeros_,
                lr_scale=lr_scale,
            )

        self.convolution = self._config.convolution_layer.get_layer(
            convolution_dim,
            default_activation=ActivationType.silu,
            lr_scale=lr_scale,
        )
        # D "skip" parameter
        self.D = self._config.d_weight.get_parameter(
            (heads_dim,),
            default_initializer=init_ones_,
            lr_scale=lr_scale,
            weight_decay=False,
        )
        self.out_proj = self._config.output_layer.get_layer(
            inner_dim,
            hidden_dim,
            default_weight_initializer=init_normal_(0, (2 / self._config.d_inner) ** 0.5),
            default_add_bias=self._block_config.add_linear_biases,
            sequence_parallel=self._sequence_parallel,
            lr_scale=lr_scale,
        )

    def forward(
        self,
        input_: torch.Tensor,
        kwargs: dict[str, typing.Any],
        losses: dict[str, typing.Any] | None = None,
        metrics: dict[str, typing.Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert _mamba_available

        sequence_length = kwargs[BlockKwargs.sequence_q_dim].global_size

        # Pad input to nearest multiple of chunklen
        padded_length = (1 + (sequence_length - 1) // self._config.chunk_size) * self._config.chunk_size
        if padded_length != sequence_length:
            assert not kwargs[BlockKwargs.sequence_first] and input_.size(1) == sequence_length
            input_ = torch.nn.functional.pad(input_, (0, 0, 0, padded_length - sequence_length))

        # -> (batch/padded_sequence, sequence/batch, local_inner_projection
        inner_projection = self.in_proj(input_)
        # Standardize to (batch, padded_sequence, local_inner_projection)
        if kwargs[BlockKwargs.sequence_first]:
            inner_projection = inner_projection.transpose(0, 1)

        xBC, z, A_log = torch.split(
            inner_projection,
            [
                self._local_inner_size + 2 * self._local_bc_size,
                self._local_inner_size,
                self._local_heads,
            ],
            dim=-1,
        )
        # Convolutional layer
        # xbc: (batch, padded_sequence, local_heads * head_size + 2 * local_head_groups * state)
        xBC = self.convolution(xBC.transpose(1, 2)).transpose(1, 2)

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
        if hasattr(self, "z_bias"):
            z = z + self.z_bias

        # y: (batch, padded_sequence, local_heads, head_size) -> (batch, sequence, local_heads * head_size)
        y = ((y + Du).flatten(2, 3) * torch.nn.functional.silu(z))[:, :sequence_length]
        if kwargs[BlockKwargs.sequence_first]:
            # TODO: Is contiguous needed?
            y = y.transpose(0, 1).contiguous()
        # out_proj: (batch/sequence, sequence/batch, local_heads * head_size)
        #   -> (batch/local_sequence, local_sequence/batch, hidden)
        return self.out_proj(y)

    @torch.compile
    def _apply_a_log(self, x: torch.Tensor, A_log: torch.Tensor) -> torch.Tensor:
        return x / torch.nn.functional.softplus(A_log).to(x.dtype).unsqueeze(-1)
