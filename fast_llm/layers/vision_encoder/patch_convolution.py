import typing

import torch

from fast_llm.core.ops import split
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.layers.attention.config import AttentionKwargs
from fast_llm.layers.block.block import Block
from fast_llm.layers.common.peft.config import PeftConfig
from fast_llm.layers.vision_encoder.config import PatchConvolutionConfig, VisionEncoderKwargs
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
        input_dim = TensorDim("input_channels", self._config.input_channels)
        patch_dim = TensorDim("patch", self._config.patch_size)

        self.convolution = self._config.convolution.get_layer(
            self._hidden_dim,
            input_dim,
            patch_dim,
            patch_dim,
            stride=(self._config.patch_size, self._config.patch_size),
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
            # TODO: Replace last dim instead
            hidden_dims = kwargs[VisionTransformerKwargs.hidden_dims]
            return TensorMeta.from_dims(hidden_dims, tensor_name="patch conv output", dtype=input_.dtype)
        micro_batch_size = kwargs[TransformerKwargs.micro_batch_size]
        sequence_length = kwargs[AttentionKwargs.sequence_length]
        out_channels = kwargs[VisionEncoderKwargs.out_channels]
        # TODO: Avoid padding
        reshape_dims = (micro_batch_size, sequence_length, out_channels)
        group = self._tensor_space.distributed.tensor_group

        input_ = self.convolution(input_)
        patch_embeddings = self.norm(input_.flatten(1))
        patch_embeddings = patch_embeddings.view(reshape_dims)

        # TODO: Sequence first"
        if sequence_first:
            patch_embeddings = patch_embeddings.permute(1, 0, 2).contiguous()
        if self._sequence_parallel:
            patch_embeddings = split(patch_embeddings, group=group, dim=0)
        return patch_embeddings
