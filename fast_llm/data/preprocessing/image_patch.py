import io
import math
import typing

from fast_llm.config import Config, Field, FieldHint, config_class
from fast_llm.utils import Assert, div

if typing.TYPE_CHECKING:
    import torch


@config_class()
class ImagePatchConfig(Config):
    """
    Configuration for the tokenizer.
    The tokenizer is needed for FIM and dataset preparation.
    """

    height: int = Field(
        default=16,
        desc="Height of the image patches, in pixels.",
        hint=FieldHint.core,
    )
    width: int = Field(
        default=16,
        desc="Height of the image patches, in pixels.",
        hint=FieldHint.core,
    )
    max_image_height: int | None = Field(
        default=None,
        desc="Maximum height of the complete image, in pixels."
        "If the original image is larger than this, it will be resized to this height.",
        hint=FieldHint.optional,
    )
    max_image_width: int | None = Field(
        default=None,
        desc="Maximum width of the complete image, in pixels."
        "If the original image is larger than this, it will be resized to this width.",
        hint=FieldHint.optional,
    )
    image_break_token: int | None = Field(
        default=None,
        desc="Add this token at the end of each row of image patches.",
        hint=FieldHint.optional,
    )
    image_end_token: int | None = Field(
        default=None,
        desc="Add this token after the last patch of each image."
        "If `image_break_token` is also defined, only `image_end_token` is added after the last row.",
        hint=FieldHint.optional,
    )

    # TODO: ====== Image type? =====
    def get_patches(self, image_bytes: typing.Any) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", list[int]]:
        import PIL.Image
        import torch

        # assume 3 channels (RGB) for all images
        with PIL.Image.open(io.BytesIO(image_bytes)) as image:
            if image.mode != "RGB":
                # Convert all images to RGB
                image = image.convert("RGB")
            image = torch.tensor(image).permute(2, 0, 1)  # HWC to CHW
            Assert.eq(image.dtype, torch.uint8)

        # Resize to a multiple of patch size smaller or equal to max size.
        image = self._resize(image)

        # Convert to patches.
        patches = torch.nn.functional.unfold(
            image, kernel_size=(self.height, self.width), stride=(self.height, self.width)
        ).T.reshape(-1, 3, self.height, self.width)

        num_patches_height = div(image.size(1), self.height)
        num_patches_width = div(image.size(2), self.width)
        position_ids = torch.arange(num_patches_height).repeat_interleave(num_patches_width) * div(
            self.max_image_width, self.width
        ) + torch.arange(num_patches_width).repeat(num_patches_height)

        token_map = torch.arange(0, num_patches_width * num_patches_height, dtype=torch.int64)
        if self.image_break_token is None:
            token_ids = [-100] * (num_patches_width * num_patches_height)
            if self.image_end_token is not None:
                token_ids.append(self.image_end_token)
        else:
            token_ids = ([-100] * num_patches_width + [self.image_break_token]) * num_patches_height
            token_map += torch.arange(num_patches_height).repeat_interleave(num_patches_width)
            if self.image_end_token is not None:
                token_ids[-1] = self.image_end_token

        return patches, position_ids, token_map, token_ids

    def _resize(self, image: "torch.Tensor") -> "torch.Tensor":
        """
        Calculate the new dimensions for resizing an image while maintaining the aspect ratio.
        If the image is larger than the max dimensions, it will be resized to fit within them.
        If the image is smaller, it will be resized to the nearest multiple of the patch size.
        """
        import torchvision.transforms.v2 as torchvision_transforms

        target_height, target_width = image.shape[1:]
        ratio = max(target_height / self.max_image_height, target_width / self.max_image_width, 1)
        target_height = self.height * math.ceil(target_height / self.height / ratio)
        target_width = self.width * math.ceil(target_width / self.width / ratio)

        # Cap the resizing to half of the current size as a workaround for large images
        # See pytorch issue: https://github.com/pytorch/pytorch/issues/103589
        while max(image.size(1) / target_height, image.size(2) / target_width) > 2:
            image = torchvision_transforms.functional.resize(
                image,
                size=(math.ceil(image.size(1) / 2), math.ceil(image.size(2) / 2)),
                interpolation=torchvision_transforms.InterpolationMode.BICUBIC,
            )

        # TODO: options for interpolation mode?
        return torchvision_transforms.functional.resize(
            image, size=(target_height, target_width), interpolation=torchvision_transforms.InterpolationMode.BICUBIC
        )


@config_class()
class ImageNormalizationConfig(Config):
    scale: float = Field(default=255.0)
    # Default values from OpenAI Clip.
    mean: tuple[float, float, float] = Field(default=(0.48145466, 0.4578275, 0.40821073))
    std: tuple[float, float, float] = Field(default=(0.26862954, 0.26130258, 0.27577711))

    def normalize(self, image: "torch.Tensor") -> "torch.Tensor":
        import torchvision.transforms.v2 as torchvision_transforms

        return torchvision_transforms.functional.normalize(image / self.scale, list(self.mean), list(self.std))
