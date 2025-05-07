import typing

import torch

from fast_llm.engine.base_model.base_model import Layer
from fast_llm.engine.config_utils.tensor_space import TensorDim, TensorSpace
from fast_llm.layers.vision_encoder.config import VisionEncoderConfig, VisionEncoderDimNames, VisionEncoderKwargs
from fast_llm.tensor import ParameterMeta, TensorMeta, init_normal_


def position_ids_in_meshgrid(patch_embeddings_list, max_size):
    positions = []
    for patch in patch_embeddings_list:
        height, width = patch.shape[-2:]
        mesh = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        h_grid, v_grid = torch.stack(mesh, dim=-1).reshape(-1, 2).chunk(2, -1)
        ids = h_grid * max_size + v_grid
        positions.append(ids[:, 0])
    return torch.cat(positions)


def generate_block_attention_mask(patch_embeds_list, tensor):
    dtype = tensor.dtype
    device = tensor.device
    seq_len = tensor.shape[1]
    d_min = torch.finfo(dtype).min
    causal_mask = torch.full((seq_len, seq_len), fill_value=d_min, dtype=dtype, device=device)

    block_end_idx = torch.tensor(patch_embeds_list).cumsum(-1)
    block_start_idx = torch.tensor([0] + patch_embeds_list[:-1]).cumsum(-1)
    for start, end in zip(block_start_idx, block_end_idx):
        causal_mask[start:end, start:end] = 0

    causal_mask = causal_mask[None, None, :, :].expand(tensor.shape[0], 1, -1, -1)
    return causal_mask


class PatchConv(Layer):
    def __init__(self, config: VisionEncoderConfig, tensor_space: TensorSpace):
        super().__init__()
        # TODO Soham: device=meta
        with torch.device("meta"):
            self.conv = torch.nn.Conv2d(
                in_channels=3,
                out_channels=config.transformer.hidden_size,
                kernel_size=config.patch_size,
                stride=config.patch_size,
                bias=False,
                dtype=tensor_space.distributed_config.training_dtype.torch,
            )
            self.conv.weight = ParameterMeta.from_dims(
                tuple(TensorDim(f"patch_conv_weight_{idx}", size) for idx, size in enumerate(self.conv.weight.shape)),
                init_method=init_normal_(),
            )
        self.norm = config.patch_norm.get_layer(tensor_space.get_tensor_dim(VisionEncoderDimNames.out_channels))

    def forward(
        self,
        input_: torch.Tensor,
        kwargs: dict[str, typing.Any],
        losses: dict[str, typing.Any] | None = None,
        metrics: dict | None = None,
    ) -> torch.Tensor:
        hidden_dims = kwargs[VisionEncoderKwargs.hidden_dims]
        if isinstance(input_, TensorMeta):
            return TensorMeta.from_dims(hidden_dims, tensor_name="patch conv output", dtype=input_.dtype)
        # we don't need images after this point
        # image_patches = kwargs.pop(VisionEncoderKwargs.image_patches)
        patch_embeddings = self.norm(self.conv(input_))
        patch_embeddings = patch_embeddings.reshape(*(x.size for x in hidden_dims))
        # Hack to pass patch embeddings to the next layer
        # kwargs[VisionEncoderKwargs.patch_embeddings] = patch_embeddings
        return patch_embeddings
