import typing

import torch

from fast_llm.core.ops import split
from fast_llm.engine.base_model.base_model import Layer
from fast_llm.engine.config_utils.tensor_space import TensorSpace
from fast_llm.layers.transformer.config import TransformerKwargs, VisionTransformerKwargs
from fast_llm.layers.vision_encoder.config import VisionEncoderConfig, VisionEncoderDimNames, VisionEncoderKwargs
from fast_llm.tensor import ParameterMeta, TensorMeta, init_normal_


class PatchConv(Layer):
    def __init__(self, config: VisionEncoderConfig, tensor_space: TensorSpace):
        super().__init__()
        self._tensor_space = tensor_space
        self._distributed_config = tensor_space.distributed_config
        self._sequence_parallel = self._distributed_config.sequence_tensor_parallel
        self._lr_scale = config.adapter_lr_scale
        self.weight = ParameterMeta.from_dims(
            (
                self._tensor_space[VisionEncoderDimNames.out_channels],
                self._tensor_space[VisionEncoderDimNames.in_channels],
                self._tensor_space[VisionEncoderDimNames.patch_size],
                self._tensor_space[VisionEncoderDimNames.patch_size],
            ),
            init_method=init_normal_(),
            lr_scale=self._lr_scale,
        )
        if config.conv_bias:
            self.bias = ParameterMeta.from_dims(
                (self._tensor_space[VisionEncoderDimNames.out_channels],),
                init_method=init_normal_(),
                lr_scale=self._lr_scale,
            )
        else:
            self.bias = None
        self.norm = config.patch_norm.get_layer(tensor_space[VisionEncoderDimNames.out_channels])
        self.stride = config.patch_size

    def forward(
        self,
        input_: torch.Tensor,
        kwargs: dict[str, typing.Any],
        losses: dict[str, typing.Any] | None = None,
        metrics: dict | None = None,
    ) -> torch.Tensor:
        hidden_dims = kwargs[VisionTransformerKwargs.hidden_dims]
        if isinstance(input_, TensorMeta):
            return TensorMeta.from_dims(hidden_dims, tensor_name="patch conv output", dtype=input_.dtype)
        micro_batch_size = kwargs[TransformerKwargs.micro_batch_size]
        sequence_length = kwargs[TransformerKwargs.sequence_length]
        out_channels = kwargs[VisionEncoderKwargs.out_channels]
        reshape_dims = (micro_batch_size, sequence_length, out_channels)
        group = self._tensor_space.distributed.tensor_group
        input_ = torch.nn.functional.conv2d(input_, self.weight, self.bias, stride=self.stride)
        patch_embeddings = self.norm(input_.flatten(1))
        patch_embeddings = patch_embeddings.view(reshape_dims)
        if self._sequence_parallel:
            patch_embeddings = patch_embeddings.permute(1, 0, 2).contiguous()
            patch_embeddings = split(patch_embeddings, group=group, dim=0)
        return patch_embeddings
