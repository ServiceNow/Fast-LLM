import math
import typing

import torch
import torchvision.transforms.v2.functional as F

from fast_llm.engine.config_utils.tensor_space import TensorSpace
from fast_llm.layers.transformer.config import TransformerKwargs
from fast_llm.layers.vision_encoder.config import (
    VisionEncoderArchitectureConfig,
    VisionEncoderKwargs,
    VisionTransformerKwargs,
)
from fast_llm.utils import div


def get_num_patches(height: int, width: int, patch_size: int) -> tuple[int, int]:
    """
    Calculate the number of patches in height and width dimensions.
    """
    return div(height, patch_size) * div(width, patch_size)


def get_resize_dims(height: int, width: int, max_height: int, max_width: int, patch_size: int) -> tuple[int, int]:
    """
    Calculate the new dimensions for resizing an image while maintaining the aspect ratio.
    If the image is larger than the max dimensions, it will be resized to fit within them.
    If the image is smaller, it will be resized to the nearest multiple of the patch size.
    """
    ratio = max(height / max_height, width / max_width)
    if ratio > 1:
        # Resize to fit within max dimensions
        height = int(height / ratio)
        width = int(width / ratio)
    return patch_size * math.ceil(height / patch_size), patch_size * math.ceil(width / patch_size)


def resize(image: torch.Tensor, max_height: int, max_width: int, patch_size: int) -> tuple[int, int]:
    resize_dims = get_resize_dims(image.size(1), image.size(2), max_height, max_width, patch_size=patch_size)
    # TODO: options for interpolation mode?
    return F.resize(image, size=resize_dims, interpolation=F.InterpolationMode.BICUBIC)


def normalize(image: torch.Tensor, mean: list[float], std: list[float]) -> torch.Tensor:
    """
    Normalize the image using the specified mean and standard deviation.
    """
    return F.normalize(image, mean=mean, std=std)


def pad(image: torch.Tensor, max_height, max_width) -> torch.Tensor:
    """
    Pad images on the right and bottom with 0s untitl max_height and max_width
    """
    width_padding = max(0, max_height - image.size(1))
    depth_padding = max(0, max_width - image.size(2))
    return F.pad(image, (0, 0, depth_padding, width_padding), 0)


def create_inv_freqs(rope_theta: int, kv_channels: int, image_size: int, patch_size: int) -> torch.Tensor:
    freqs = 1.0 / (rope_theta ** (torch.arange(0, kv_channels, 2).float() / kv_channels))
    max_patches_per_side = image_size // patch_size

    h = torch.arange(max_patches_per_side)
    w = torch.arange(max_patches_per_side)

    freqs_h = torch.outer(h, freqs[::2]).float()
    freqs_w = torch.outer(w, freqs[1::2]).float()
    inv_freq = torch.cat(
        [
            freqs_h[:, None, :].repeat(1, max_patches_per_side, 1),
            freqs_w[None, :, :].repeat(max_patches_per_side, 1, 1),
        ],
        dim=-1,
    ).reshape(-1, kv_channels // 2)

    return torch.cat((inv_freq, inv_freq), dim=-1)


def position_ids_in_meshgrid(image_sizes: list[torch.Tensor], max_size: int, patch_size: int) -> torch.Tensor:
    positions = []
    for h, w in image_sizes:
        patch_height = h // patch_size
        patch_width = w // patch_size
        mesh = torch.meshgrid(torch.arange(patch_height), torch.arange(patch_width), indexing="ij")
        h_grid, v_grid = torch.stack(mesh, dim=-1).reshape(-1, 2).chunk(2, -1)
        ids = h_grid * max_size + v_grid
        positions.append(ids[:, 0])
    return positions


def position_ids_in_meshgrid(height, width, max_size, patch_size) -> torch.Tensor:
    patch_height = height // patch_size
    patch_width = width // patch_size
    mesh = torch.meshgrid(torch.arange(patch_height), torch.arange(patch_width), indexing="ij")
    h_grid, v_grid = torch.stack(mesh, dim=-1).reshape(-1, 2).chunk(2, -1)
    ids = h_grid * max_size + v_grid
    return ids[:, 0]


class VisionPreprocessor:
    def __init__(self, config: VisionEncoderArchitectureConfig, tensor_space: TensorSpace):
        self._config = config
        self._tensor_space = tensor_space
        self._distributed_config = self._tensor_space.distributed_config

    def preprocess(self, kwargs: dict[str, typing.Any]) -> None:
        images = kwargs.get(VisionEncoderKwargs.images)
        im_height = kwargs.get(VisionEncoderKwargs.image_size)
        im_width = kwargs.get(VisionEncoderKwargs.image_size)
        patch_size = kwargs[VisionEncoderKwargs.patch_size]
        image_sizes = [
            [get_resize_dims(im.size(1), im.size(2), im_height, im_width, patch_size=patch_size) for im in ims]
            for ims in images
        ]
        kwargs[VisionEncoderKwargs.image_sizes] = image_sizes
        images = [
            [
                normalize(
                    resize(image, im_height, im_width, patch_size).to(
                        dtype=self._tensor_space.distributed_config.training_dtype.torch
                    )
                    / kwargs[VisionEncoderKwargs.image_rescale_factor],
                    mean=kwargs[VisionEncoderKwargs.image_mean],
                    std=kwargs[VisionEncoderKwargs.image_std],
                )
                for image in imgs
            ]
            for imgs in images
        ]
        # position_ids = position_ids_in_meshgrid(image_sizes, im_height, patch_size)
        patches = []
        patch_position_ids = []
        cu_seqlens = [0]
        max_seqlen = -1
        for imgs, sizes in zip(images, image_sizes):
            # TODO Soham: should this be micro_sequence_length?
            # sum(
            #     get_num_patches(*size, patch_size) for size in sizes
            # )
            seq_patches = []
            for image, size in zip(imgs, sizes):
                seqlen = get_num_patches(*size, patch_size)
                if seqlen > max_seqlen:
                    max_seqlen = seqlen
                cu_seqlens.append(cu_seqlens[-1] + seqlen)
                seq_patches.append(
                    torch.cat(
                        [
                            torch.nn.functional.unfold(image, kernel_size=patch_size, stride=patch_size).T.reshape(
                                -1, 3, patch_size, patch_size
                            ),
                        ]
                    )
                )
            padding_size = kwargs[TransformerKwargs.sequence_length] - cu_seqlens[-1]
            if padding_size > max_seqlen:
                max_seqlen = padding_size
            cu_seqlens.append(kwargs[TransformerKwargs.sequence_length])
            patches.append(
                torch.cat(
                    [
                        *seq_patches,
                        torch.zeros(padding_size, 3, patch_size, patch_size).to(
                            dtype=self._tensor_space.distributed_config.training_dtype.torch,
                            device=self._tensor_space.distributed.device,
                        ),
                    ]
                )
            )
            position_ids = torch.cat(
                [position_ids_in_meshgrid(*size, im_height // patch_size, patch_size) for size in sizes]
            ).to(device=self._tensor_space.distributed.device)
            # We pad at the end instead of padding at the position in meshgrid because flash attention does not support custom attention masks
            patch_position_ids.append(
                torch.cat(
                    [
                        position_ids,
                        torch.full((padding_size,), 0).to(device=self._tensor_space.distributed.device),
                    ]
                )
            )
            # TODO Soham: remove
            assert patches[-1].size(0) == kwargs[TransformerKwargs.sequence_length]
        patches = torch.cat(patches)
        patch_position_ids = torch.cat(patch_position_ids)
        kwargs[VisionEncoderKwargs.image_patches] = patches
        kwargs[VisionEncoderKwargs.rotary_inv_freq] = create_inv_freqs(
            kwargs[VisionEncoderKwargs.rope_theta],
            kwargs[VisionEncoderKwargs.kv_channels],
            im_height,
            patch_size,
        ).to(device=self._tensor_space.distributed.device)
        kwargs[VisionEncoderKwargs.max_image_tokens] = div(im_height * im_width, patch_size**2)
        kwargs[VisionTransformerKwargs.patch_position_ids] = patch_position_ids
        # TODO Soham: handle sequence data parallel
        kwargs[VisionTransformerKwargs.cu_seqlens_q] = torch.tensor(
            cu_seqlens, device=self._tensor_space.distributed.device, dtype=torch.int32
        )
        kwargs[VisionTransformerKwargs.cu_seqlens_k] = torch.tensor(
            cu_seqlens, device=self._tensor_space.distributed.device, dtype=torch.int32
        )
        kwargs[VisionTransformerKwargs.max_seqlen_q] = max_seqlen
        kwargs[VisionTransformerKwargs.max_seqlen_k] = max_seqlen
