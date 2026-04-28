import dataclasses
import functools
import math

import pytest
import torch

from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.layers.attention.config import AttentionKwargs
from fast_llm.layers.attention.rotary.config import (
    DefaultRotaryConfig,
    Llama3RotaryConfig,
    ProportionalRotaryConfig,
    Rotary2DConfig,
    RotaryConfig,
    YarnRotaryConfig,
)
from fast_llm.layers.vision.config import VisionKwargs
from fast_llm.utils import Assert

_NUM_HEADS = 4
_GRID_WIDTH = 4  # Patch grid column count for 2D tests; _sequence_lengths must be divisible by this.


def _reference_rotary_1d_forward(
    tensor: torch.Tensor,
    angle_scales: torch.Tensor,
    positions: torch.Tensor,
    attention_factor: float = 1.0,
) -> torch.Tensor:
    """
    Independent reference for 1D RoPE in Fast-LLM's split-real layout.
    Pair k = (re_k, im_k) is rotated by angle positions[n] * angle_scales[k].

    tensor: (num_tokens, num_heads, head_size)
    angle_scales: (head_size // 2,) float64
    attention_factor: overall cos/sin scale (non-1 for YarnRotary)
    """
    angle_scales = angle_scales.to(positions.device)
    angles = torch.outer(positions.to(torch.float64), angle_scales)  # (num_tokens, head_size//2)
    cos = (attention_factor * torch.cos(angles)).to(tensor.dtype).unsqueeze(1)
    sin = (attention_factor * torch.sin(angles)).to(tensor.dtype).unsqueeze(1)
    half = tensor.shape[-1] // 2
    re, im = tensor[..., :half], tensor[..., half:]
    return torch.cat([re * cos - im * sin, im * cos + re * sin], dim=-1)


def _reference_rotary_2d_forward(
    tensor: torch.Tensor,
    patch_positions: torch.Tensor,
    theta: float,
    head_size: int,
) -> torch.Tensor:
    """
    Independent reference for 2D RoPE (Pixtral/Rotary2D style).
    theta^-arange(0,1,2/H).view(-1,2) gives per-pair [h_scale, w_scale]:
    column 0 (even-indexed) drives height rotation, column 1 (odd-indexed) drives width.
    First head_size//4 pairs rotate by h-position, last head_size//4 by w-position.

    tensor: (num_patches, num_heads, head_size)
    patch_positions: (num_patches, 2) int64 with (h, w) coordinates
    """
    arange = torch.arange(0, 1, 2 / head_size, dtype=torch.float64, device=patch_positions.device)
    freq_matrix = (theta**-arange).view(-1, 2)  # (head_size//4, 2)
    h_angles = torch.outer(patch_positions[:, 0].to(torch.float64), freq_matrix[:, 0])
    w_angles = torch.outer(patch_positions[:, 1].to(torch.float64), freq_matrix[:, 1])
    angles = torch.cat([h_angles, w_angles], dim=-1)  # (num_patches, head_size//2)
    cos = torch.cos(angles).to(tensor.dtype).unsqueeze(1)  # (num_patches, 1, head_size//2)
    sin = torch.sin(angles).to(tensor.dtype).unsqueeze(1)
    half = head_size // 2
    re, im = tensor[..., :half], tensor[..., half:]
    return torch.cat([re * cos - im * sin, im * cos + re * sin], dim=-1)


@dataclasses.dataclass
class RotaryTestConfig:
    name: str
    head_size: int
    rotary_type: str = "default"
    theta: float = 10000.0
    # proportional
    partial_rotary_factor: float = 1.0
    # llama3 and yarn
    scale_factor: float = 8.0
    original_context_length: int = 8192
    # llama3
    low_frequency_factor: float = 1.0
    high_frequency_factor: float = 4.0
    # yarn
    beta_fast: float = 32.0
    beta_slow: float = 1.0

    @property
    def attention_factor(self) -> float:
        if self.rotary_type == "yarn":
            return 0.1 * math.log(self.scale_factor) + 1.0
        return 1.0

    def get_rotary_config(self) -> RotaryConfig:
        if self.rotary_type == "default":
            return DefaultRotaryConfig(theta=self.theta)
        if self.rotary_type == "proportional":
            return ProportionalRotaryConfig(theta=self.theta, partial_rotary_factor=self.partial_rotary_factor)
        if self.rotary_type == "llama3":
            return Llama3RotaryConfig(
                theta=self.theta,
                scale_factor=self.scale_factor,
                low_frequency_factor=self.low_frequency_factor,
                high_frequency_factor=self.high_frequency_factor,
                original_context_length=self.original_context_length,
            )
        if self.rotary_type == "yarn":
            return YarnRotaryConfig(
                theta=self.theta,
                scale_factor=self.scale_factor,
                beta_fast=self.beta_fast,
                beta_slow=self.beta_slow,
                original_context_length=self.original_context_length,
            )
        if self.rotary_type == "2d":
            return Rotary2DConfig(theta=self.theta)
        raise ValueError(self.rotary_type)

    @functools.cached_property
    def reference_angle_scales(self) -> torch.Tensor:
        base = self.theta ** -torch.arange(0, 1, 2 / self.head_size, dtype=torch.float64)
        if self.rotary_type in ("default", "2d"):
            return base
        if self.rotary_type == "proportional":
            rotary_pairs = round(self.head_size * self.partial_rotary_factor) // 2
            nope_pairs = self.head_size // 2 - rotary_pairs
            if nope_pairs == 0:
                return base
            return torch.cat([base[:rotary_pairs], base.new_zeros(nope_pairs)])
        if self.rotary_type == "llama3":
            high_freq_wavelength = self.original_context_length / self.high_frequency_factor
            low_freq_wavelength = self.original_context_length / self.low_frequency_factor
            new_scales = []
            for scale in base.tolist():
                wavelength = 2 * math.pi / scale
                if wavelength < high_freq_wavelength:
                    new_scales.append(scale)
                elif wavelength > low_freq_wavelength:
                    new_scales.append(scale / self.scale_factor)
                else:
                    smooth = (self.original_context_length / wavelength - self.low_frequency_factor) / (
                        self.high_frequency_factor - self.low_frequency_factor
                    )
                    new_scales.append((1 - smooth) * scale / self.scale_factor + smooth * scale)
            return torch.tensor(new_scales, dtype=torch.float64)
        if self.rotary_type == "yarn":
            low = max(math.floor(self._yarn_correction(self.beta_fast)), 0)
            high = min(math.ceil(self._yarn_correction(self.beta_slow)), self.head_size - 1)
            if low == high:
                high += 0.001
            extrapolation_factor = torch.clamp(
                (torch.arange(self.head_size // 2, dtype=torch.float64) - low) / (high - low), 0, 1
            )
            return base / self.scale_factor * extrapolation_factor + base * (1 - extrapolation_factor)
        raise ValueError(self.rotary_type)

    def _yarn_correction(self, beta: float) -> float:
        return (
            self.head_size * math.log(self.original_context_length / (beta * 2 * math.pi)) / (2 * math.log(self.theta))
        )

    def make_preprocess_kwargs(self, num_tokens: int, device: torch.device) -> dict:
        if self.rotary_type == "2d":
            patch_positions = torch.tensor(
                [[i // _GRID_WIDTH, i % _GRID_WIDTH] for i in range(num_tokens)],
                dtype=torch.int64,
                device=device,
            )
            return {VisionKwargs.patch_positions: patch_positions, AttentionKwargs.device: device}
        return {
            AttentionKwargs.sequence_length: num_tokens,
            AttentionKwargs.sequence_k_dim: TensorDim("sequence_k", num_tokens),
            AttentionKwargs.token_dim: TensorDim("token", num_tokens),
            AttentionKwargs.device: device,
        }

    def reference_output(
        self, query: torch.Tensor, key: torch.Tensor, preprocess_kwargs: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.rotary_type == "2d":
            patch_positions = preprocess_kwargs[VisionKwargs.patch_positions]
            return (
                _reference_rotary_2d_forward(query, patch_positions, self.theta, self.head_size),
                _reference_rotary_2d_forward(key, patch_positions, self.theta, self.head_size),
            )
        positions = torch.arange(query.shape[0], dtype=torch.int64, device=query.device)
        return (
            _reference_rotary_1d_forward(query, self.reference_angle_scales, positions, self.attention_factor),
            _reference_rotary_1d_forward(key, self.reference_angle_scales, positions, self.attention_factor),
        )


_head_sizes = [16, 32, 64]

_rotary_test_configs: list[RotaryTestConfig] = []


def _add_configs(name: str, **kwargs) -> None:
    for head_size in _head_sizes:
        _rotary_test_configs.append(RotaryTestConfig(name=f"{name}_h{head_size}", head_size=head_size, **kwargs))


_add_configs("default")
_add_configs("default_big_theta", theta=500000.0)
_add_configs("llama3", rotary_type="llama3")
_add_configs("yarn", rotary_type="yarn")
_add_configs("2d", rotary_type="2d")

for _head_size in _head_sizes:
    for _factor in [0.25, 0.5, 0.75, 1.0]:
        if round(_head_size * _factor) % 2 == 0 and round(_head_size * _factor) > 0:
            _rotary_test_configs.append(
                RotaryTestConfig(
                    name=f"proportional_{int(_factor * 100)}pct_h{_head_size}",
                    head_size=_head_size,
                    rotary_type="proportional",
                    partial_rotary_factor=_factor,
                )
            )

_sequence_lengths = [8, 24]


@pytest.mark.parametrize(
    "sequence_length",
    [pytest.param(seq_len, id=str(seq_len)) for seq_len in _sequence_lengths],
)
@pytest.mark.parametrize(
    "config",
    [pytest.param(config, id=config.name) for config in _rotary_test_configs],
)
def test_rotary(config: RotaryTestConfig, sequence_length: int, testing_device) -> None:
    query = torch.randn(sequence_length, _NUM_HEADS, config.head_size, device=testing_device)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    rotary = config.get_rotary_config().get_layer(TensorDim("head_size", config.head_size))
    preprocess_kwargs = config.make_preprocess_kwargs(sequence_length, testing_device)
    rotary.preprocess(preprocess_kwargs)
    out_query, out_key_value = rotary.forward(query, torch.cat([key, value], dim=-2), preprocess_kwargs)
    out_key, out_value = out_key_value.chunk(2, dim=-2)

    expected_query, expected_key = config.reference_output(query, key, preprocess_kwargs)
    Assert.rms_close_relative(out_query, expected_query, 1e-5, 1e-7)
    Assert.rms_close_relative(out_key, expected_key, 1e-5, 1e-7)
    Assert.all_equal(out_value, value)
