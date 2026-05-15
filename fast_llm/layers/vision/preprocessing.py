"""
Vision preprocessing utilities: patch counting, image resizing, rotary inverse frequencies.

These are standalone helpers used by data preparation and model preprocessing.
The full VisionPreprocessor class (model-level) is deferred to step 2.3 when
AudioMultiModalModel / VisionMultiModalModel preprocessing is integrated.
"""

import math

import torch
import torchvision.transforms.v2.functional as F

from fast_llm.utils import div


def get_num_patches(height: int, width: int, patch_size: int) -> int:
    """Number of patches for an image of the given size."""
    return div(height, patch_size) * div(width, patch_size)


def get_num_image_tokens(height: int, width: int, patch_size: int, image_break: bool, image_end: bool) -> int:
    """
    Total number of LM tokens consumed by an image.

    With ``image_break=True`` a break token is inserted after every row of patches
    (Pixtral-style tiling). With ``image_end=True`` a single end token follows the image.
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
    Resize dimensions that fit within (max_height, max_width), rounded up to a multiple
    of patch_size while preserving aspect ratio.
    """
    ratio = max(height / max_height, width / max_width)
    if ratio > 1:
        height = int(height / ratio)
        width = int(width / ratio)
    return patch_size * math.ceil(height / patch_size), patch_size * math.ceil(width / patch_size)


def resize(image: torch.Tensor, max_height: int, max_width: int, patch_size: int) -> torch.Tensor:
    resize_dims = get_resize_dims(image.size(1), image.size(2), max_height, max_width, patch_size=patch_size)
    return F.resize(image, size=resize_dims, interpolation=F.InterpolationMode.BICUBIC)


def normalize(image: torch.Tensor, mean: list[float], std: list[float]) -> torch.Tensor:
    return F.normalize(image, mean=mean, std=std)


def pad(image: torch.Tensor, max_height: int, max_width: int) -> torch.Tensor:
    """Pad image on the right and bottom with zeros to (max_height, max_width)."""
    width_padding = max(0, max_height - image.size(1))
    depth_padding = max(0, max_width - image.size(2))
    return F.pad(image, (0, 0, depth_padding, width_padding), 0)


def create_inv_freqs(rope_theta: int, kv_channels: int, image_size: int, patch_size: int) -> torch.Tensor:
    """2D rotary inverse frequencies for Pixtral-style vision attention."""
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


def position_ids_in_meshgrid(height: int, width: int, max_size: int, patch_size: int) -> torch.Tensor:
    """Row-major 2D patch position ids for a single image."""
    patch_height = height // patch_size
    patch_width = width // patch_size
    mesh = torch.meshgrid(torch.arange(patch_height), torch.arange(patch_width), indexing="ij")
    h_grid, v_grid = torch.stack(mesh, dim=-1).reshape(-1, 2).chunk(2, -1)
    ids = h_grid * max_size + v_grid
    return ids[:, 0]


def generate_block_attention_mask(patch_embeds_list: list[int], tensor: torch.Tensor) -> torch.Tensor:
    """
    Block-diagonal attention mask for Pixtral tiled images.

    Each image attends only within its own patch block; no cross-image attention.
    Returns shape (batch, 1, seq_len, seq_len) filled with 0.0 or -inf.
    """
    dtype = tensor.dtype
    device = tensor.device
    seq_len = tensor.shape[1]
    d_min = torch.finfo(dtype).min
    causal_mask = torch.full((seq_len, seq_len), fill_value=d_min, dtype=dtype, device=device)

    block_end_idx = torch.tensor(patch_embeds_list).cumsum(-1)
    block_start_idx = torch.tensor([0] + patch_embeds_list[:-1]).cumsum(-1)
    for start, end in zip(block_start_idx, block_end_idx):
        causal_mask[start:end, start:end] = 0

    return causal_mask[None, None, :, :].expand(tensor.shape[0], 1, -1, -1)
