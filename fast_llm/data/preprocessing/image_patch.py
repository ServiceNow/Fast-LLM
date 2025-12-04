import functools
import io
import math
import typing

from fast_llm.config import Config, Field, FieldHint, check_field, config_class
from fast_llm.engine.config_utils.data_type import DataType
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
        valid=check_field(Assert.gt, 0),
    )
    width: int = Field(
        default=16,
        desc="Height of the image patches, in pixels.",
        hint=FieldHint.core,
        valid=check_field(Assert.gt, 0),
    )
    do_resize: bool = Field(default=True, desc="Whether to resize the image.")
    max_image_height: int = Field(
        default=1024,
        desc="Maximum height of the complete image, in pixels."
        "If the original image is larger than this and resizing is enabled, it will be resized to this height.",
        hint=FieldHint.optional,
    )
    max_image_width: int = Field(
        default=1024,
        desc="Maximum width of the complete image, in pixels."
        "If the original image is larger than this and resizing is enabled, it will be resized to this width.",
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
    image_format: str = Field(
        default="bytes",
        desc="Format of the input images. 'bytes' expects raw image bytes, 'pil' expects PIL Image objects, "
        "'dict' expects a dictionary with a 'bytes' key containing the image bytes.",
        hint=FieldHint.optional,
    )

    @property
    def num_channels(self) -> int:
        # assume 3 channels (RGB) for all images
        return 3

    @functools.cached_property
    def max_patches_height(self) -> int:
        return div(self.max_image_height, self.height)

    @functools.cached_property
    def max_patches_width(self) -> int:
        return div(self.max_image_width, self.width)

    def _validate(self):
        super()._validate()
        Assert.gt(self.max_patches_height, 0)
        Assert.gt(self.max_patches_width, 0)

    def get_patches_from_images(
        self, images: list["torch.Tensor|bytes|dict"], token_data_type: DataType = DataType.int64
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", list["torch.Tensor"], list[int]]:
        import torch

        if len(images) > 0:
            image_patches, image_positions, image_token_maps, image_token_ids = zip(
                *(self._get_patches_from_image(image, token_data_type) for image in images)
            )
            return (
                torch.cat(image_patches),
                torch.cat(image_positions),
                torch.cat(image_token_maps),
                image_token_ids,
                [len(position_ids) for position_ids in image_positions],
            )
        else:
            # Return empty tensors of appropriate shapes and data types so we can concatenate with other documents.
            return (
                torch.empty(0, self.num_channels, self.height, self.width, dtype=torch.uint8),
                torch.empty(0, 2, dtype=torch.int64),
                torch.empty(0, dtype=torch.int64),
                [],
                [0],
            )

    def _get_patches_from_image(
        self, image: "torch.Tensor|bytes|dict", token_data_type: DataType = DataType.int64
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        import torch

        if not torch.is_tensor(image):
            import contextlib

            import numpy as np
            import PIL.Image
            import PIL.PngImagePlugin

            # Load the image based on format
            # Set a larger limit for decompression to handle images with large ICC profiles
            PIL.Image.MAX_IMAGE_PIXELS = None
            original_max_text_chunk = PIL.PngImagePlugin.MAX_TEXT_CHUNK
            PIL.PngImagePlugin.MAX_TEXT_CHUNK = 10 * (1024**2)  # 10 MB

            try:
                if self.image_format == "bytes":
                    image_ctx = PIL.Image.open(io.BytesIO(image))
                elif self.image_format == "pil":
                    image_ctx = contextlib.nullcontext(image)
                elif self.image_format == "dict":
                    image_bytes = image["bytes"]
                    image_ctx = PIL.Image.open(io.BytesIO(image_bytes))
                else:
                    raise ValueError(
                        f"Unsupported image_format: {self.image_format}. Must be 'bytes', 'pil', or 'dict'."
                    )
            finally:
                PIL.PngImagePlugin.MAX_TEXT_CHUNK = original_max_text_chunk

            # Convert to RGB and tensor
            with image_ctx as pil_image:
                if pil_image.mode != "RGB":
                    pil_image = pil_image.convert("RGB")
                image = torch.tensor(np.array(pil_image)).permute(2, 0, 1)  # HWC to CHW
                Assert.eq(image.dtype, torch.uint8)

        if self.do_resize:
            # Resize to a multiple of patch size smaller or equal to max size.
            image = self._resize(image)
        else:
            # Crop the image to ensure its shape is a multiple of the patch size.
            image = image[
                :, : image.size(1) - image.size(1) % self.height, : image.size(2) - image.size(2) % self.width
            ]

        num_patches_height = div(image.size(1), self.height)
        num_patches_width = div(image.size(2), self.width)
        # Convert to patches. (`torch.nn.functional.unfold` not supported for uint8.)
        patches = (
            image.reshape(self.num_channels, num_patches_height, self.height, num_patches_width, self.width)
            .permute(1, 3, 0, 2, 4)
            .flatten(0, 1)
        )

        positions = torch.stack(
            [
                torch.arange(num_patches_height).repeat_interleave(num_patches_width),
                torch.arange(num_patches_width).repeat(num_patches_height),
            ],
            1,
        )

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

        return patches, positions, token_map, torch.tensor(token_ids, dtype=token_data_type.torch)

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
