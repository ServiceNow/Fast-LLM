import logging
import typing

import torch

from fast_llm.engine.config_utils.initialization import init_normal_, init_ones_
from fast_llm.engine.config_utils.tensor_dim import CompositeTensorDim, ConcatenatedTensorDim, TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.functional.config import ActivationType
from fast_llm.layers.block.block import BlockLayer
from fast_llm.layers.block.config import BlockConfig, BlockKwargs
from fast_llm.layers.common.peft.config import PeftConfig
from fast_llm.layers.ssm.config import MambaConfig, init_a, init_dtprojbias
from fast_llm.utils import div

try:
    from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn as _mamba_inner_fn  # noqa

    _mamba_available = True
except (ImportError, RuntimeError):
    _mamba_available = False

logger = logging.getLogger(__name__)

"""
Note: this is mostly adapted from https://github.com/Zyphra/Zamba2, similar code is also in https://github.com/state-spaces/mamba.
For now it only supports training and not inference.
This works with triton 3.1.0
"""


class Mamba[ConfigType: MambaConfig](BlockLayer[ConfigType]):
    _mixer_name: typing.ClassVar[str] = "mamba"

    def __init__(
        self,
        config: ConfigType,
        block_config: BlockConfig,
        distributed_config: DistributedConfig,
        *,
        # TODO: Review `hidden_dim` and `block_index`
        hidden_dim: TensorDim,
        block_index: int,
        name: str,
        lr_scale: float | None,
        peft: PeftConfig | None,
    ):
        super().__init__(
            config,
            block_config,
            distributed_config,
            hidden_dim=hidden_dim,
            block_index=block_index,
            name=name,
            lr_scale=lr_scale,
            peft=peft,
        )
        assert self._distributed_config.tensor_parallel == 1, "Tensor-parallel not supported for Mamba"

        # Tensor dims:
        heads_dim = TensorDim("heads", div(self._config.d_inner, self._config.state_size))
        state_dim = TensorDim("state", self._config.state_size)
        inner_dim = CompositeTensorDim("inner", (heads_dim, state_dim))
        dt_rank_dim = TensorDim("dt_rank", self._config.dt_rank)
        inner_projection_dim = ConcatenatedTensorDim("inner_projection", (inner_dim, inner_dim))
        x_projection_dim = ConcatenatedTensorDim("x_projection", (dt_rank_dim, state_dim, state_dim))

        # TODO: Use x_layer
        self.in_proj = self._config.z_layer.get_layer(
            hidden_dim,
            inner_projection_dim,
            default_weight_initialization=init_normal_(0, (2 / hidden_dim.global_size) ** 0.5),
            default_add_bias=self._block_config.add_linear_biases,
            lr_scale=self._lr_scale,
            peft=self._peft,
        )
        self.convolution = self._config.convolution_layer.get_layer(
            inner_dim,
            default_weight_initialization=init_normal_(0, (2 / self._config.d_inner) ** 0.5),
            default_add_bias=False,
            default_activation=ActivationType.silu,
            lr_scale=self._lr_scale,
            peft=self._peft,
        )
        self.x_proj = self._config.x_projection_layer.get_layer(
            inner_dim,
            x_projection_dim,
            default_weight_initialization=init_normal_(0, (2 / self._config.d_inner) ** 0.5),
            lr_scale=self._lr_scale,
            peft=self._peft,
        )

        # TODO: the weights are initialized a bit differently here https://github.com/state-spaces/mamba/blob/0cce0fa645f100f00620ddf2333c2b7712abfdec/mamba_ssm/modules/mamba_simple.py#L82
        self.dt_proj = self._config.dt_layer.get_layer(
            dt_rank_dim,
            inner_dim,
            default_weight_initialization=init_normal_(0, (2 / self._config.d_inner) ** 0.5),
            default_bias_initialization=init_dtprojbias(),
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
            default_weight_initialization=init_normal_(0, (2 / hidden_dim.global_size) ** 0.5),
            default_add_bias=False,
            lr_scale=self._lr_scale,
            peft=self._peft,
        )

    def forward(
        self,
        input_: torch.Tensor,
        kwargs: dict[str, typing.Any],
        losses: dict[str, typing.Any] | None = None,
        metrics: dict[str, typing.Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert _mamba_available
        in_proj = self.in_proj(input_).permute((1, 2, 0) if kwargs[BlockKwargs.sequence_first] else (0, 2, 1))

        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        # If we wanbt to support inference, we would need to implement slow path here, see https://github.com/Zyphra/Zamba2/blob/1b182f40f2257f822cc06dd785df53d67d691a15/mamba_layer.py#L172s
        out = _mamba_inner_fn(
            in_proj,
            self.convolution.weight,
            self.convolution.bias,
            self.x_proj.weight,
            self.dt_proj.weight,
            self.out_proj.weight,
            self.out_proj.bias,  # is None here
            -torch.exp(self.A_log.float()),
            None,  # input-dependent B
            None,  # input-dependent C
            self.D.float(),
            delta_bias=None if self.dt_proj.bias is None else self.dt_proj.bias.float(),
            delta_softplus=True,
        )
        if kwargs[BlockKwargs.sequence_first]:
            out = out.transpose(0, 1)
        return out, None
