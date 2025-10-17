import math
import typing

import torch
import torchvision.transforms.v2 as torchvision_transforms

from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.layers.attention.config import AttentionKwargs
from fast_llm.layers.language_model.config import LanguageModelKwargs
from fast_llm.layers.vision.config import ImageNormalizationConfig, VisionEncoderConfig
from fast_llm.utils import div


def get_num_patches(height: int, width: int, patch_size: int) -> int:
    """
    Calculate the number of patches in height and width dimensions.
    """
    return div(height, patch_size) * div(width, patch_size)


def get_num_image_tokens(height: int, width: int, patch_size: int, image_break: bool, image_end: bool) -> int:
    """
    Calculate the number of image tokens.
    If image_break is True, we consider 1 additional token after every row of patches.
    """
    height_patches = div(height, patch_size)
    width_patches = div(width, patch_size)
    num_tokens = height_patches * width_patches
    if image_break:
        num_tokens += height_patches
    elif image_end:
        num_tokens += 1
    return num_tokens


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


def resize(image: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
    # cap the resizing to half of the current size as a workaround for large images
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


def position_ids_in_meshgrid(height, width, max_size, patch_size) -> torch.Tensor:
    patch_height = height // patch_size
    patch_width = width // patch_size
    return torch.arange(patch_height).repeat_interleave(patch_width) * max_size + torch.arange(patch_width).repeat(
        patch_height
    )


class VisionPreprocessor:
    def __init__(self, config: VisionEncoderConfig, distributed: Distributed):
        self._config = config
        self._distributed = distributed

    def preprocess(self, tokens, kwargs: dict[str, typing.Any]) -> None:
        max_image_size = kwargs.get(VisionEncoderKwargs.max_image_size)
        patch_size = self._config.patch_size
        image_sizes = []

        norm_config: ImageNormalizationConfig = kwargs["norm_config"]

        if LanguageModelKwargs.labels in kwargs:
            labels = kwargs[LanguageModelKwargs.labels]
            if (self._config.image_break_token is not None) or (self._config.image_end_token is not None):
                # If image break or end token is present, we need to replace image token ids to -100 in labels
                # TODO: avoid double cloning labels in case of loss masking spans?
                labels = labels.clone()
        patches = []
        patch_position_ids = []
        sequence_lengths = [0]
        max_sequence_length = -1

        for sample_index, (sample_images_, positions) in enumerate(
            zip(kwargs[VisionEncoderKwargs.images], kwargs.get(VisionEncoderKwargs.image_positions), strict=True)
        ):
            image_sizes.append(sample_image_sizes := [])

            sample_sequence_length = 0

            for image, position in zip(sample_images_, positions, strict=True):
                height, width = get_resize_dims(
                    image.size(1), image.size(2), max_image_size, max_image_size, patch_size=patch_size
                )

                sample_image_sizes.append((height, width))

                image = resize(image, height, width)

                # TODO: Normalize with constant dtype instead?
                image = image.to(dtype=self._distributed.config.training_dtype.torch)

                image = torchvision_transforms.functional.normalize(
                    image / norm_config.rescale_factor,
                    mean=[norm_config.mean_r, norm_config.mean_g, norm_config.mean_b],
                    std=[norm_config.std_r, norm_config.std_g, norm_config.std_b],
                )
                patches.extend(
                    torch.nn.functional.unfold(image, kernel_size=patch_size, stride=patch_size).T.reshape(
                        -1, 3, patch_size, patch_size
                    )
                )

                num_height_patches = div(height, patch_size)
                num_width_patches = div(width, patch_size)
                grid_height = torch.arange(num_height_patches).repeat_interleave(num_width_patches)
                grid_width = torch.arange(num_width_patches).repeat(num_height_patches)
                grid_height * div(max_image_size, patch_size) + grid_width
                patch_position_ids.append(grid_height * div(max_image_size, patch_size) + grid_width)

                if LanguageModelKwargs.labels in kwargs:
                    num_tokens = get_num_image_tokens(
                        height,
                        width,
                        patch_size=patch_size,
                        image_break=self._config.image_break_token is not None,
                        image_end=self._config.image_end_token is not None,
                    )
                    # set labels for image patches to -100
                    labels[sample_index, max(position - 1, 0) : position + num_tokens - 1] = -100

                sequence_lengths.append(sequence_length := num_height_patches * num_width_patches)
                if sequence_length > max_sequence_length:
                    max_sequence_length = sequence_length
                sample_sequence_length += sequence_length

            # TODO: No need for padding with varlen?
            padding_size = kwargs[AttentionKwargs.sequence_length] - sample_sequence_length
            if padding_size > max_sequence_length:
                max_sequence_length = padding_size
            sequence_lengths.append(padding_size)

            patches.append(
                torch.zeros(padding_size, 3, patch_size, patch_size).to(
                    dtype=self._tensor_space.distributed_config.training_dtype.torch,
                    device=self._tensor_space.distributed.device,
                ),
            )
            patch_position_ids.append(torch.full((padding_size,), 0, dtype=torch.int64))

        kwargs[VisionEncoderKwargs.image_sizes] = image_sizes
        kwargs[VisionEncoderKwargs.image_patches] = torch.cat(patches).to(device=self._distributed.device)
        kwargs[VisionTransformerKwargs.patch_position_ids] = torch.cat(patch_position_ids).to(
            device=self._distributed.device
        )
        kwargs[VisionEncoderKwargs.max_image_tokens] = div(max_image_size**2, patch_size**2)
        # sequence data parallel is not yet supported for images, so we use the same cu_seqlens for q and k
        kwargs[VisionTransformerKwargs.cu_seqlens_q] = torch.tensor(
            cu_seqlens, device=self._distributed.device, dtype=torch.int32
        )
        kwargs[VisionTransformerKwargs.cu_seqlens_k] = torch.tensor(
            cu_seqlens, device=self._distributed.device, dtype=torch.int32
        )
        kwargs[VisionTransformerKwargs.max_seqlen_q] = max_sequence_length
        kwargs[VisionTransformerKwargs.max_seqlen_k] = max_sequence_length
        if LanguageModelKwargs.labels in kwargs:
            kwargs[LanguageModelKwargs.labels] = labels

        # TODO: add proper preprocessing for attention-mask when not using flash attention
        # Following is just a dummy code to run the tests.
        kwargs[self._config.transformer._transformer_kwargs.attention_mask] = torch.ones(
            (1, 1, kwargs[AttentionKwargs.sequence_length], 1, kwargs[AttentionKwargs.sequence_length]),
            dtype=torch.bool,
            device=self._tensor_space.distributed.device,
        )
        kwargs[self._config.transformer._transformer_kwargs.attention_mask_value] = torch.full(
            [],
            torch.finfo(self._distributed.config.training_dtype.torch).min,
            dtype=self._distributed.config.training_dtype.torch,
            device=self._distributed.device,
        )
