import math
import typing

import torch
import torchvision.transforms.v2.functional as F

from fast_llm.engine.config_utils.tensor_space import TensorSpace
from fast_llm.layers.vision_encoder.config import VisionArchitectureConfig, VisionModelKwargs
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
    return (
        (int(height / ratio), int(width / ratio))
        if ratio > 1
        else (patch_size * math.ceil(height / patch_size), patch_size * math.ceil(width / patch_size))
    )


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


class VisionPreprocessor:
    def __init__(self, config: VisionArchitectureConfig, tensor_space: TensorSpace):
        self._config = config
        self._tensor_space = tensor_space
        self._distributed_config = self._tensor_space.distributed_config

    def preprocess(self, kwargs: dict[str, typing.Any]) -> None:
        images = kwargs.get("images")
        im_height = kwargs.get(VisionModelKwargs.image_size)
        im_width = kwargs.get(VisionModelKwargs.image_size)
        patch_size = kwargs[VisionModelKwargs.patch_size]
        image_sizes = [
            get_resize_dims(im.size(1), im.size(2), im_height, im_width, patch_size=patch_size) for im in images
        ]
        kwargs[VisionModelKwargs.image_sizes] = image_sizes
        images = [
            pad(
                normalize(
                    resize(image, im_height, im_width, patch_size) / kwargs[VisionModelKwargs.image_rescale_factor],
                    mean=kwargs[VisionModelKwargs.image_mean],
                    std=kwargs[VisionModelKwargs.image_std],
                ),
                max_height=im_height,
                max_width=im_width,
            )
            for image in images
        ]
        images = torch.stack(images, dim=0).to(
            # TODO Soham: is this needed?
            device=self._tensor_space.distributed.device,
            dtype=self._distributed_config.training_dtype.torch,
        )
        kwargs[VisionModelKwargs.images] = images
        kwargs[VisionModelKwargs.rotary_inv_freq] = create_inv_freqs(
            kwargs[VisionModelKwargs.rope_theta],
            kwargs[VisionModelKwargs.kv_channels],
            im_height,
            patch_size,
        ).to(device=self._tensor_space.distributed.device)
