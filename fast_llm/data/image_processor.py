import math

import torch
from torchvision.transforms.v2 import functional as F

from fast_llm.data.config import ImageProcessorConfig


class ImageProcessor:
    def __init__(self, config: ImageProcessorConfig):
        self.patch_size = config.patch_size
        self.mean = [x / config.rescale_factor for x in config.mean]
        self.std = [x / config.rescale_factor for x in config.std]
        self.max_height = config.max_height
        self.max_width = config.max_width
        assert (
            self.max_height % self.patch_size[0] == 0
        ), "max_height must be divisible by patch_size[0]. Found {max_height} and {self.patch_size[0]}"
        assert (
            self.max_width % self.patch_size[1] == 0
        ), "max_width must be divisible by patch_size[1]. Found {max_width} and {self.patch_size[1]}"

    def resize(self, image):
        # Resize the image to the specified size
        # TODO Soham: resize for patches only during train?
        # TODO Soham: convert all images to tensor?
        # height = image.shape[0]
        # width = image.shape[1]
        height, width = self.get_resize_dims(image.shape[0], image.shape[1], self.max_height, self.max_width)

        # TODO: options for interpolation mode
        return F.resize(image, size=(height, width), interpolation=F.InterpolationMode.BICUBIC)

    # TODO Soham: move to utils
    @classmethod
    def get_resize_dims(height, width, max_height, max_width, patch_size: list[int]):
        ratio = max(height / max_height, width / max_width)
        return (
            (math.ceil(height / ratio), math.ceil(width / ratio))
            if ratio > 1
            else (patch_size[0] * math.ceil(height / patch_size[0]), patch_size[1] * math.ceil(width / patch_size[1]))
        )

    def normalize(self, image: torch.Tensor) -> torch.Tensor:
        # Normalize the image using the mean and std
        return F.normalize(image, mean=self.mean, std=self.std)

    @classmethod
    # TODO Soham: move to utils
    def get_num_patches(image: torch.Tensor, patch_size: list[int]) -> torch.Tensor:
        return (image.size[0] // patch_size[0]) * (image.size[1] // patch_size[1])
