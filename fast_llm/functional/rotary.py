import torch

from fast_llm.utils import div


def convert_rotary_complex_to_real(tensor: torch.Tensor, kv_channels: int, dim: int) -> torch.Tensor:
    return tensor.unflatten(dim, (-1, div(kv_channels, 2), 2)).movedim(dim + 1, dim + 2).flatten(dim, dim + 2)


def convert_rotary_real_to_complex(tensor: torch.Tensor, kv_channels: int, dim: int) -> torch.Tensor:
    return tensor.unflatten(dim, (-1, 2, div(kv_channels, 2))).movedim(dim + 1, dim + 2).flatten(dim, dim + 2)


def apply_rotary_embeddings(tensor: torch.Tensor, rope_frequencies: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embeddings to a tensor:
    * Convert it to a complex, full-precision tensor
    * Multiply by the frequencies
    * Convert back tho the input format.
    # TODO: Full precision only needed for bfloat16? (Doesn't support complex numbers)
    # TODO: This could use torch compile, but it doesn't support complex tensors at the moment.
    """
    complex_tensor = torch.view_as_complex(tensor.to(torch.float32).view(*tensor.shape[:-1], -1, 2))
    return torch.view_as_real(complex_tensor * rope_frequencies).view_as(tensor).type_as(tensor)
