import typing

import torch

from fast_llm.core.ops import split
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig, DistributedDimNames
from fast_llm.layers.attention.config import AttentionKwargs
from fast_llm.layers.block.block import Block
from fast_llm.layers.common.peft.config import PeftConfig
from fast_llm.layers.vision.config import PatchConvolutionConfig
from fast_llm.tensor import TensorMeta


class PatchConvolution[ConfigType: PatchConvolutionConfig](Block[ConfigType]):
    _config: ConfigType

    def __init__(
        self,
        config: ConfigType,
        distributed_config: DistributedConfig,
        *,
        # TODO: Input or output dim?
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
        self._parallel_dim = self._distributed_config.get_distributed_dim(DistributedDimNames.tensor)

        self.convolution = self._config.convolution.get_layer(
            self._hidden_dim,
            TensorDim("input_channels", self._config.input_channels),
            TensorDim("patch_height", self._config.patch_height),
            TensorDim("patch_width", self._config.patch_width),
            stride=(self._config.patch_height, self._config.patch_width),
            default_add_bias=False,
            lr_scale=self._lr_scale,
            peft=self._peft,
        )
        self.normalization = self._config.normalization.get_layer(hidden_dim, lr_scale=self._lr_scale, peft=self._peft)

    def forward(
        self,
        input_: torch.Tensor,
        kwargs: dict[str, typing.Any],
        losses: dict[str, typing.Any] | None = None,
        metrics: dict | None = None,
    ) -> torch.Tensor:
        if isinstance(input_, TensorMeta):
            return TensorMeta.from_dims(
                input_.dims[:-1] + (self._hidden_dim,), tensor_name="patch conv output", dtype=input_.dtype
            )
        # TODO: Avoid padding
        input_ = self.convolution(input_)
        patch_embeddings = self.normalization(input_.flatten(1)).view_as(input_)

        # TODO: Permute earlier?
        if kwargs[AttentionKwargs.sequence_first]:
            patch_embeddings = patch_embeddings.permute(1, 0, 2).contiguous()
        if self._sequence_parallel:
            patch_embeddings = split(patch_embeddings, group=self._parallel_dim.group, dim=0)
        return patch_embeddings
