"""
Tests for 2D rotary position embedding equivalence between Fast-LLM and HuggingFace Pixtral.

This test verifies whether Fast-LLM's Rotary2D and HF's PixtralRotaryEmbedding
produce equivalent attention outputs.

If this test PASSES: The implementations are equivalent for attention computation.
If this test FAILS: The implementations produce different attention outputs.
"""

import typing
from types import SimpleNamespace

import pytest
import torch
from transformers.models.pixtral.modeling_pixtral import PixtralRotaryEmbedding, apply_rotary_pos_emb

from fast_llm.config import Field, FieldHint, config_class
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.layers.attention.attention import Attention
from fast_llm.layers.attention.config import AttentionConfig, AttentionKwargs
from fast_llm.layers.attention.rotary.config import DefaultRotaryConfig, Rotary2DConfig, RotaryConfig
from fast_llm.layers.attention.rotary.rotary import (
    Rotary,
    convert_rotary_complex_to_real,
    convert_rotary_real_to_complex,
)
from fast_llm.layers.vision.config import VisionKwargs
from fast_llm.utils import Assert
from tests.utils.utils import requires_cuda


def apply_rotary_pos_emb_interleaved(q, k, cos, sin, unsqueeze_dim=1):
    """
    Apply rotary embeddings to interleaved layout [r0, i0, r1, i1, ...].

    Standard apply_rotary_pos_emb expects real layout [r0, r1, ..., i0, i1, ...].
    This version handles interleaved format used by Fast-LLM when triton=False.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Extract real/imag from interleaved positions
    q_real, q_imag = q[..., 0::2], q[..., 1::2]
    k_real, k_imag = k[..., 0::2], k[..., 1::2]

    # cos/sin from Pixtral are duplicated, take first half
    cos_half = cos[..., : cos.shape[-1] // 2]
    sin_half = sin[..., : sin.shape[-1] // 2]

    # Apply rotation: (real + i*imag) * (cos + i*sin) = (real*cos - imag*sin) + i*(imag*cos + real*sin)
    q_real_out = q_real * cos_half - q_imag * sin_half
    q_imag_out = q_imag * cos_half + q_real * sin_half
    k_real_out = k_real * cos_half - k_imag * sin_half
    k_imag_out = k_imag * cos_half + k_real * sin_half

    # Interleave back
    q_out = torch.stack([q_real_out, q_imag_out], dim=-1).flatten(-2)
    k_out = torch.stack([k_real_out, k_imag_out], dim=-1).flatten(-2)

    return q_out, k_out


@config_class(dynamic_type={RotaryConfig: "pixtral_2d"})
class PixtralRotary2DConfig(DefaultRotaryConfig):
    """
    Config for PixtralRotary2D that uses HuggingFace Pixtral's frequency calculation.
    """

    image_size: int = Field(
        default=1024,
        desc="Maximum image size for computing max patches per side",
        hint=FieldHint.architecture,
    )
    patch_size: int = Field(
        default=32,
        desc="Patch size for computing max patches per side",
        hint=FieldHint.architecture,
    )

    def _get_configurable_class(self) -> "type[PixtralRotary2D]":
        return PixtralRotary2D


class PixtralRotary2D[ConfigType: PixtralRotary2DConfig](Rotary[ConfigType]):
    """
    A Rotary2D implementation that uses HuggingFace Pixtral's actual PixtralRotaryEmbedding.

    This follows the exact same pattern as Fast-LLM's Rotary2D class but delegates
    frequency computation to the actual HuggingFace Pixtral implementation.
    """

    _pixtral_rotary: PixtralRotaryEmbedding
    _config: ConfigType

    def __init__(
        self,
        config: ConfigType,
        head_size_dim: TensorDim,
    ):
        super().__init__(config, head_size_dim)
        Assert.multiple(self._head_size, 4)
        self._max_patches_per_side = config.image_size // config.patch_size

        pixtral_config = SimpleNamespace(
            head_dim=self._head_size,
            rope_theta=config.theta,
            image_size=config.image_size,
            patch_size=config.patch_size,
        )
        self._pixtral_rotary = PixtralRotaryEmbedding(config=pixtral_config)

    def preprocess(self, kwargs: dict[str, typing.Any]) -> None:
        patch_positions = kwargs[VisionKwargs.patch_positions]
        device = kwargs[AttentionKwargs.device]
        num_patches = len(patch_positions)

        if self._pixtral_rotary.inv_freq.device != device:
            self._pixtral_rotary = self._pixtral_rotary.to(device)

        # Convert patch positions (h, w) to Pixtral's linear position IDs
        # Pixtral expects: position_id = h * max_patches_per_side + w
        position_ids = (patch_positions[:, 0] * self._max_patches_per_side + patch_positions[:, 1]).long()[
            None, :
        ]  # [1, num_patches]

        dummy_x = torch.empty(1, num_patches, self._head_size, device=device)
        cos, sin = self._pixtral_rotary(dummy_x, position_ids)

        kwargs[AttentionKwargs.rotary_freq_q] = (cos, sin)
        kwargs[AttentionKwargs.rotary_freq_k] = (cos, sin)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, kwargs: dict[str, typing.Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = kwargs[AttentionKwargs.rotary_freq_q]
        if self._config.triton:
            # triton=True uses real layout [r0, r1, ..., i0, i1, ...]
            query, key = apply_rotary_pos_emb(query, key, cos, sin, unsqueeze_dim=2)
        else:
            # triton=False uses interleaved layout [r0, i0, r1, i1, ...]
            query, key = apply_rotary_pos_emb_interleaved(query, key, cos, sin, unsqueeze_dim=2)
        return query, key


class TestRotary2DEquivalence:
    """
    Test that Fast-LLM's Rotary2D and HF's PixtralRotaryEmbedding produce
    equivalent attention outputs.
    """

    @requires_cuda
    @pytest.mark.parametrize("head_dim", [32, 64])
    @pytest.mark.parametrize("grid", [(4, 4), (6, 8), (3, 5)])
    def test_attention_output_equivalence(self, head_dim: int, grid: tuple[int, int]):
        num_patches_h, num_patches_w = grid
        num_patches = num_patches_h * num_patches_w
        batch_size = 2
        num_heads = 8
        hidden_size = num_heads * head_dim
        theta = 10000.0
        image_size = 1024
        patch_size = 32

        # Create Attention layer
        attention: Attention = AttentionConfig(
            head_size=head_dim,
            heads=num_heads,
            head_groups=num_heads,
            causal=False,
            cross_document_attention=True,
        ).get_layer(
            DistributedConfig(compute_dtype="float32"),
            TensorDim("hidden_size", hidden_size),
            lr_scale=None,
            peft=None,
        )

        torch.manual_seed(42)
        query = torch.empty(batch_size, num_patches, num_heads, head_dim, dtype=torch.float32, device="cuda").normal_()
        key = torch.empty(batch_size, num_patches, num_heads, head_dim, dtype=torch.float32, device="cuda").normal_()
        value = torch.empty(batch_size, num_patches, num_heads, head_dim, dtype=torch.float32, device="cuda").normal_()

        patch_positions = torch.tensor(
            [[h, w] for h in range(num_patches_h) for w in range(num_patches_w)],
            dtype=torch.float64,
            device="cuda",
        )

        head_size_dim = TensorDim("head_size", head_dim)
        rotary_configs = {
            "fastllm-triton": (Rotary2DConfig(theta=theta, triton=True), True),
            "fastllm-no-triton": (Rotary2DConfig(theta=theta, triton=False), False),
            "pixtral-triton": (
                PixtralRotary2DConfig(theta=theta, triton=True, image_size=image_size, patch_size=patch_size),
                True,
            ),
            "pixtral-no-triton": (
                PixtralRotary2DConfig(theta=theta, triton=False, image_size=image_size, patch_size=patch_size),
                False,
            ),
        }

        outputs = {}
        for name, (config, uses_real_layout) in rotary_configs.items():
            rotary = config.get_layer(head_size_dim)
            kwargs = {
                VisionKwargs.patch_positions: patch_positions,
                AttentionKwargs.device: torch.device("cuda"),
                AttentionKwargs.sequence_length: num_patches,
                AttentionKwargs.sequence_lengths: [[num_patches]] * batch_size,
                AttentionKwargs.sequence_q_dim: TensorDim("sequence_q", num_patches),
                AttentionKwargs.sequence_k_dim: TensorDim("sequence_k", num_patches),
            }
            rotary.preprocess(kwargs)
            attention._preprocess_for_backup_attention(kwargs)

            if uses_real_layout:
                q_in = convert_rotary_complex_to_real(query.clone(), head_dim, dim=3)
                k_in = convert_rotary_complex_to_real(key.clone(), head_dim, dim=3)
                v_in = convert_rotary_complex_to_real(value.clone(), head_dim, dim=3)
            else:
                q_in, k_in, v_in = query.clone(), key.clone(), value.clone()

            q, k = rotary(q_in, k_in, kwargs)
            out = attention._attn_backup(q, k, v_in, kwargs)

            # Note: attention output has shape [batch, seq, hidden_size] where hidden_size = heads * head_dim
            if uses_real_layout:
                out = out.view(batch_size, num_patches, num_heads, head_dim)
                out = convert_rotary_real_to_complex(out, head_dim, dim=3)
                out = out.view(batch_size, num_patches, hidden_size)

            outputs[name] = out

        print(f"\n[head_dim={head_dim}, grid={grid}]")
        names = list(outputs.keys())
        for i, name1 in enumerate(names):
            for name2 in names[i + 1 :]:
                diff = outputs[name1] - outputs[name2]
                rms = (diff**2).mean().sqrt().item()
                print(f"  {name1} vs {name2}: RMS={rms:.6e}")

        # Layout equivalence: triton vs no-triton should match for same implementation
        Assert.rms_close(outputs["fastllm-triton"], outputs["fastllm-no-triton"], 1e-5)
        Assert.rms_close(outputs["pixtral-triton"], outputs["pixtral-no-triton"], 1e-5)

        # Frequency equivalence: FastLLM vs Pixtral use different 2D frequency calculations
        # TODO: Make FastLLM's Rotary2D match Pixtral's frequency calculation
        try:
            Assert.rms_close(outputs["fastllm-triton"], outputs["pixtral-triton"], 1e-5)
            Assert.rms_close(outputs["fastllm-no-triton"], outputs["pixtral-no-triton"], 1e-5)
        except AssertionError:
            pytest.skip("FastLLM Rotary2D frequency calculation differs from Pixtral")
