import math

import torch
from torchvision.transforms.v2 import functional as F

from fast_llm.data.config import ImageProcessorConfig


class ImageProcessor:
    def __init__(self, config: ImageProcessorConfig):
        self.patch_size = config.patch_size
        self.mean = config.mean / config.rescale_factor
        self.std = config.std / config.rescale_factor
        self.max_height = config.max_height
        self.max_width = config.max_width
        assert (
            self.max_height % self.patch_size[0] == 0
        ), "max_height must be divisible by patch_size[0]. Found {max_height} and {self.patch_size[0]}"
        assert (
            self.max_width % self.patch_size[1] == 0
        ), "max_width must be divisible by patch_size[1]. Found {max_width} and {self.patch_size[1]}"

    def resize(self, image: torch.Tensor) -> torch.Tensor:
        # Resize the image to the specified size
        height = image.shape[0]
        width = image.shape[1]
        ratio = max(height / self.max_height, width / self.max_width)
        if ratio > 1:
            height = math.ceil(height / ratio)
            width = math.ceil(width / ratio)
        else:
            height = self.patch_size[0] * math.ceil(height / self.self.patch_size[0])
            width = self.patch_size[1] * math.ceil(width / self.patch_size[1])

        # TODO: options for interpolation mode
        return F.resize(image, size=(height, width), interpolation=F.InterpolationMode.BICUBIC)

    def normalize(self, image: torch.Tensor) -> torch.Tensor:
        # Normalize the image using the mean and std
        return F.normalize(image, mean=self.mean, std=self.std)

    def get_num_patches(self, image: torch.Tensor) -> torch.Tensor:
        return (image.size(0) // self.patch_size[0]) * (image.size(1) // self.patch_size[1])
