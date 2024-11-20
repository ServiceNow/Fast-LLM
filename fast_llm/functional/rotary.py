import math

import torch

from fast_llm.functional.config import RotaryScalingType
from fast_llm.utils import div


def convert_rotary_complex_to_real(tensor: torch.Tensor, kv_channels: int, dim: int):
    return tensor.unflatten(dim, (-1, div(kv_channels, 2), 2)).movedim(dim + 1, dim + 2).flatten(dim, dim + 2)


def convert_rotary_real_to_complex(tensor: torch.Tensor, kv_channels: int, dim: int):
    return tensor.unflatten(dim, (-1, 2, div(kv_channels, 2))).movedim(dim + 1, dim + 2).flatten(dim, dim + 2)


def apply_llama3_scaling(freqs: torch.Tensor) -> torch.Tensor:
    """
    Llama3 scaling: https://github.com/meta-llama/llama-models/blob/baf7b01b6e62bc7126c7b558d2b67d4533142680/models/llama3/reference_impl/model.py#L45-L67
    """
    # Values obtained from grid search
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # original llama3 length

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def get_rotary_frequencies(
    sequence_length,
    kv_channels,
    scale=-math.log(10000),
    *,
    complex_format: bool = True,
    rotary_scaling_type: RotaryScalingType = RotaryScalingType.none,
    device="cuda",
):
    # Calculate the complex frequencies (https://blog.eleuther.ai/rotary-embeddings/)
    # `exp(i * n * a) = cos(n * a) + i sin(n * a)`,
    # `a = theta ** - (2 * (channel // 2) / kv_channels)`,
    # where n is the position in the sequence.
    # We preform the calculation in high precision because it matters for rotary embeddings.
    positions = torch.arange(sequence_length, device=device, dtype=torch.float64)
    freqs = torch.exp(scale * torch.arange(0, 1, 2 / kv_channels, device=device, dtype=torch.float64))
    # Apply scaling
    if rotary_scaling_type == RotaryScalingType.llama3:
        freqs = apply_llama3_scaling(freqs)
    angles = torch.outer(positions, freqs)
    frequencies = torch.polar(torch.ones_like(angles), angles)[None, :, None, :].to(torch.complex64)
    if not complex_format:
        frequencies = convert_rotary_complex_to_real(
            torch.view_as_real(frequencies).flatten(-2), kv_channels, 3
        ).contiguous()
    return frequencies


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
