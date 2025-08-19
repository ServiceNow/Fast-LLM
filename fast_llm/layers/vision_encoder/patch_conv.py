import typing

import torch

from fast_llm.core.ops import split
from fast_llm.engine.base_model.base_model import Layer
from fast_llm.engine.config_utils.tensor_space import TensorSpace
from fast_llm.layers.transformer.config import TransformerKwargs
from fast_llm.layers.vision_encoder.config import PixtralVisionEncoderConfig, VisionEncoderDimNames
from fast_llm.tensor import ParameterMeta, TensorMeta, init_normal_


class PatchConvolution(Layer):
    """
    A convolution layer applied to image patches to create embeddings for each patch. These embeddings are fed into the vision transformer.
    """

    def __init__(self, config: PixtralVisionEncoderConfig, tensor_space: TensorSpace):
        super().__init__()
        self._config = config
        self._tensor_space = tensor_space
        self._distributed_config = tensor_space.distributed_config
        self._sequence_tensor_parallel = self._distributed_config.sequence_tensor_parallel
        self.weight = ParameterMeta.from_dims(
            (
                self._tensor_space.get_tensor_dim(VisionEncoderDimNames.out_channels),
                self._tensor_space.get_tensor_dim(VisionEncoderDimNames.in_channels),
                self._tensor_space.get_tensor_dim(VisionEncoderDimNames.patch_size),
                self._tensor_space.get_tensor_dim(VisionEncoderDimNames.patch_size),
            ),
            init_method=init_normal_(),
            lr_scale=self._config.adapter_lr_scale,
        )
        if config.conv_bias:
            self.bias = ParameterMeta.from_dims(
                (self._tensor_space.get_tensor_dim(VisionEncoderDimNames.out_channels),),
                init_method=init_normal_(),
                lr_sclae=self._config.adapter_lr_scale,
            )
        else:
            self.bias = None
        self.normalization = config.patch_normalization.get_layer(
            tensor_space.get_tensor_dim(VisionEncoderDimNames.out_channels)
        )

    def forward(
        self,
        input_: torch.Tensor,
        kwargs: dict[str, typing.Any],
        losses: dict[str, typing.Any] | None = None,
        metrics: dict | None = None,
    ) -> torch.Tensor:
        hidden_dims = kwargs[TransformerKwargs.hidden_dims]
        if isinstance(input_, TensorMeta):
            return TensorMeta.from_dims(hidden_dims, tensor_name="patch conv output", dtype=input_.dtype)
        input_ = torch.nn.functional.conv2d(input_, self.weight, self.bias, stride=self._config.patch_size)
        patch_embeddings = self.normalization(input_.flatten(1)).view(
            kwargs[TransformerKwargs.batch_dim].size,
            kwargs[TransformerKwargs.sequence_q_dim].size,
            self._config.transformer.hidden_size,
        )
        if kwargs[TransformerKwargs.sequence_first]:
            patch_embeddings = patch_embeddings.permute(1, 0, 2).contiguous()
            if self._sequence_tensor_parallel:
                patch_embeddings = split(patch_embeddings, group=self._tensor_space.distributed.tensor_group, dim=0)
        return patch_embeddings
