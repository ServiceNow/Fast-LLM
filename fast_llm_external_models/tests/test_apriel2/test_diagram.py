"""Consolidated tests for the architecture diagram package.

Tests cover: model extraction, layout vocabulary, elements, detail panels,
and end-to-end diagram generation.
"""

from __future__ import annotations

import math
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import pytest
import svg as S

from fast_llm_external_models.apriel2.conversion.diagram import generate_diagram
from fast_llm_external_models.apriel2.conversion.diagram.elements import (
    ArchitectureOverview,
    Arrow,
    AttentionDetail,
    Box,
    DecoderBlock,
    DetailEnvelope,
    EnvelopeResult,
    ExternalLabel,
    GDNDetail,
    KDADetail,
    Label,
    MambaDetail,
    ValueLabel,
    MLPDetail,
    StochasticMixerPanel,
    Symbol,
    VisionEncoderColumn,
    connector_bezier,
    defs,
    detail_for_mixer,
    mixer_css_class,
)
from fast_llm_external_models.apriel2.conversion.diagram.layout import (
    Align,
    Aligned,
    AnchorSide,
    AspectFit,
    BBox,
    Background,
    Clamped,
    FixedSize,
    FixedWidth,
    HStack,
    LayoutRoot,
    MinSize,
    Offset,
    Padded,
    Renderable,
    Responsive,
    Size,
    Spacer,
    VStack,
    ZStack,
    anchor_point,
    detect_overlaps,
    render_brace,
    render_connector,
    resolve_overlaps,
)
from fast_llm_external_models.apriel2.conversion.diagram.model import (
    ArchitectureModel,
    AttentionDisplayConfig,
    BlockSpec,
    GDNDisplayConfig,
    KDADisplayConfig,
    MLPDisplayConfig,
    MambaDisplayConfig,
    MixerSpec,
    StochasticMixerSpec,
    VisionEncoderSpec,
    _run_length_encode,
    extract_model,
    mlp_label,
)
from fast_llm_external_models.apriel2.conversion.diagram.style import (
    Geometry,
    Palette,
    Theme,
    Typography,
)

# Default theme used throughout tests
TH = Theme()


# ═══════════════════════════════════════════════════════════════════════
# Fixture Configs
# ═══════════════════════════════════════════════════════════════════════


def _fixed_attention_config(num_blocks: int = 24) -> dict:
    """Minimal fixed-decoder attention config."""
    return {
        "model_type": "apriel2",
        "architectures": ["Apriel2ForCausalLM"],
        "hidden_size": 4096,
        "vocab_size": 32000,
        "tie_word_embeddings": False,
        "decoder": {
            "type": "fixed",
            "num_blocks": num_blocks,
            "block": {
                "mixer": {
                    "type": "attention",
                    "heads": 32,
                    "head_groups": 8,
                    "head_size": 128,
                },
                "mlp": {
                    "type": "mlp",
                    "intermediate_size": 11008,
                    "activation": "silu",
                    "gated": True,
                },
                "normalization": {"type": "rms_norm", "epsilon": 1e-5},
            },
        },
    }


def _hybrid_dil_config() -> dict:
    """Pattern decoder with 8+32+8 blocks."""
    return {
        "model_type": "apriel2",
        "architectures": ["Apriel2ForCausalLM"],
        "hidden_size": 5120,
        "vocab_size": 131072,
        "tie_word_embeddings": True,
        "decoder": {
            "type": "pattern",
            "num_blocks": 48,
            "pattern": ["attn"] * 8 + ["hybrid"] * 32 + ["attn"] * 8,
            "blocks": {
                "attn": {
                    "mixer": {"type": "attention", "heads": 40, "head_groups": 8, "head_size": 128},
                    "mlp": {"type": "mlp", "intermediate_size": 12800, "activation": "silu", "gated": True},
                    "normalization": {"type": "rms_norm"},
                },
                "hybrid": {
                    "mixer": {
                        "type": "stochastic",
                        "main_mixer_name": "attention",
                        "mixers": {
                            "attention": {"type": "attention", "heads": 40, "head_groups": 8, "head_size": 128},
                            "gdn": {
                                "type": "gdn",
                                "value_heads": 40,
                                "key_heads": 8,
                                "key_head_dim": 128,
                                "value_head_dim": 128,
                                "convolution_layer": {"kernel_size": 4},
                            },
                        },
                    },
                    "mlp": {"type": "mlp", "intermediate_size": 12800, "activation": "silu", "gated": True},
                    "normalization": {"type": "rms_norm"},
                },
            },
        },
    }


def _stochastic_supernet_config() -> dict:
    """Fixed decoder with stochastic supernet (all 4 mixer types)."""
    return {
        "model_type": "apriel2",
        "architectures": ["Apriel2ForCausalLM"],
        "hidden_size": 256,
        "vocab_size": 1000,
        "tie_word_embeddings": False,
        "decoder": {
            "type": "fixed",
            "num_blocks": 4,
            "block": {
                "mixer": {
                    "type": "stochastic",
                    "main_mixer_name": "attention",
                    "sampling_strategy": "uniform",
                    "mixers": {
                        "attention": {"type": "attention", "heads": 8, "head_groups": 4, "head_size": 32},
                        "sliding_window": {
                            "type": "attention",
                            "heads": 8,
                            "head_groups": 4,
                            "head_size": 32,
                            "window_size": 4096,
                        },
                        "gdn": {
                            "type": "gdn",
                            "value_heads": 8,
                            "key_heads": 4,
                            "key_head_dim": 32,
                            "value_head_dim": 32,
                            "convolution_layer": {"kernel_size": 4},
                        },
                        "kda": {
                            "type": "kda",
                            "heads": 8,
                            "head_dim": 32,
                            "convolution_layer": {"kernel_size": 4},
                        },
                    },
                },
                "mlp": {"type": "mlp", "intermediate_size": 512, "activation": "silu", "gated": True},
                "normalization": {"type": "rms_norm"},
            },
        },
    }


def _comprehensive_config() -> dict:
    """Comprehensive config with all 8 block types."""
    return {
        "model_type": "apriel2",
        "architectures": ["Apriel2ForCausalLM"],
        "hidden_size": 5120,
        "vocab_size": 131072,
        "tie_word_embeddings": False,
        "decoder": {
            "type": "pattern",
            "num_blocks": 48,
            "pattern": [
                "attn", "mamba", "gdn", "stoch_am", "swa", "stoch_sg", "kda", "attn",
                "stoch_ak", "mamba", "swa", "stoch_am", "gdn", "stoch_ak", "attn", "mamba",
                "stoch_am", "swa", "kda", "attn", "stoch_sg", "mamba", "stoch_ak", "swa",
                "attn", "gdn", "stoch_ak", "mamba", "swa", "stoch_am", "kda", "attn",
                "mamba", "stoch_sg", "swa", "stoch_ak", "attn", "gdn", "mamba", "stoch_ak",
                "stoch_am", "swa", "attn", "kda", "mamba", "stoch_sg", "swa", "attn",
            ],
            "blocks": {
                "attn": {
                    "mixer": {"type": "attention", "heads": 40, "head_groups": 8, "head_size": 128},
                    "mlp": {"type": "mlp", "intermediate_size": 12800, "activation": "silu", "gated": True},
                    "normalization": {"type": "rms_norm"},
                },
                "swa": {
                    "mixer": {"type": "attention", "heads": 40, "head_groups": 8, "head_size": 128, "window_size": 4096},
                    "mlp": {"type": "mlp", "intermediate_size": 12800, "activation": "silu", "gated": True},
                    "normalization": {"type": "rms_norm"},
                },
                "mamba": {
                    "mixer": {"type": "mamba", "d_state": 64, "d_conv": 4, "d_inner": 10240},
                    "mlp": {"type": "mlp", "intermediate_size": 12800, "activation": "silu", "gated": True},
                    "normalization": {"type": "rms_norm"},
                },
                "gdn": {
                    "mixer": {
                        "type": "gdn",
                        "value_heads": 40, "key_heads": 8, "key_head_dim": 128, "value_head_dim": 128,
                        "convolution_layer": {"kernel_size": 4},
                    },
                    "mlp": {"type": "mlp", "intermediate_size": 12800, "activation": "silu", "gated": True},
                    "normalization": {"type": "rms_norm"},
                },
                "kda": {
                    "mixer": {"type": "kda", "heads": 40, "head_dim": 128, "convolution_layer": {"kernel_size": 4}},
                    "mlp": {"type": "mlp", "intermediate_size": 12800, "activation": "silu", "gated": True},
                    "normalization": {"type": "rms_norm"},
                },
                "stoch_am": {
                    "mixer": {
                        "type": "stochastic",
                        "main_mixer_name": "attention",
                        "mixers": {
                            "attention": {"type": "attention", "heads": 40, "head_groups": 8, "head_size": 128},
                            "mamba": {"type": "mamba", "d_state": 64, "d_conv": 4, "d_inner": 10240},
                        },
                    },
                    "mlp": {"type": "mlp", "intermediate_size": 12800, "activation": "silu", "gated": True},
                    "normalization": {"type": "rms_norm"},
                },
                "stoch_sg": {
                    "mixer": {
                        "type": "stochastic",
                        "main_mixer_name": "swa",
                        "mixers": {
                            "swa": {
                                "type": "attention", "heads": 40, "head_groups": 8, "head_size": 128,
                                "window_size": 4096,
                            },
                            "gdn": {
                                "type": "gdn",
                                "value_heads": 40, "key_heads": 8, "key_head_dim": 128, "value_head_dim": 128,
                                "convolution_layer": {"kernel_size": 4},
                            },
                        },
                    },
                    "mlp": {"type": "mlp", "intermediate_size": 12800, "activation": "silu", "gated": True},
                    "normalization": {"type": "rms_norm"},
                },
                "stoch_ak": {
                    "mixer": {
                        "type": "stochastic",
                        "main_mixer_name": "attention",
                        "mixers": {
                            "attention": {"type": "attention", "heads": 40, "head_groups": 8, "head_size": 128},
                            "kda": {
                                "type": "kda", "heads": 40, "head_dim": 128,
                                "convolution_layer": {"kernel_size": 4},
                            },
                        },
                    },
                    "mlp": {"type": "mlp", "intermediate_size": 12800, "activation": "silu", "gated": True},
                    "normalization": {"type": "rms_norm"},
                },
            },
        },
    }


def _vision_config() -> dict:
    """Config with vision encoder."""
    return {
        "model_type": "apriel2",
        "architectures": ["Apriel2ForConditionalGeneration"],
        "hidden_size": 256,
        "vocab_size": 1000,
        "tie_word_embeddings": False,
        "decoder": {
            "type": "fixed",
            "num_blocks": 4,
            "block": {
                "mixer": {"type": "attention", "heads": 8, "head_groups": 4, "head_size": 32},
                "mlp": {"type": "mlp", "intermediate_size": 512, "activation": "silu", "gated": True},
                "normalization": {"type": "rms_norm"},
            },
        },
        "vision_encoder": {
            "hidden_size": 128,
            "embeddings": {"patch_height": 16, "patch_width": 16, "input_channels": 3},
            "encoder": {
                "type": "fixed",
                "num_blocks": 3,
                "block": {
                    "mixer": {"type": "attention", "heads": 4, "head_groups": 4, "head_size": 32, "causal": False},
                    "mlp": {"type": "mlp", "intermediate_size": 512, "activation": "silu", "gated": True},
                    "normalization": {"type": "rms_norm"},
                },
            },
            "adapter": {"type": "mlp", "intermediate_size": 256, "activation": "gelu", "gated": False},
        },
    }


def _gdn_config() -> dict:
    """Fixed decoder with GDN mixer."""
    return {
        "model_type": "apriel2",
        "architectures": ["Apriel2ForCausalLM"],
        "hidden_size": 2048,
        "vocab_size": 32000,
        "tie_word_embeddings": False,
        "decoder": {
            "type": "fixed",
            "num_blocks": 12,
            "block": {
                "mixer": {
                    "type": "gdn",
                    "value_heads": 32,
                    "key_heads": 8,
                    "key_head_dim": 64,
                    "value_head_dim": 64,
                    "convolution_layer": {"kernel_size": 4},
                },
                "mlp": {"type": "mlp", "intermediate_size": 5504, "activation": "silu", "gated": True},
                "normalization": {"type": "rms_norm"},
            },
        },
    }


def _kda_config() -> dict:
    """Fixed decoder with KDA mixer."""
    return {
        "model_type": "apriel2",
        "architectures": ["Apriel2ForCausalLM"],
        "hidden_size": 2048,
        "vocab_size": 32000,
        "tie_word_embeddings": False,
        "decoder": {
            "type": "fixed",
            "num_blocks": 8,
            "block": {
                "mixer": {
                    "type": "kda",
                    "heads": 16,
                    "head_dim": 128,
                    "convolution_layer": {"kernel_size": 4},
                },
                "mlp": {"type": "mlp", "intermediate_size": 5504, "activation": "silu", "gated": False},
                "normalization": {"type": "rms_norm"},
            },
        },
    }


# ═══════════════════════════════════════════════════════════════════════
# Model tests
# ═══════════════════════════════════════════════════════════════════════


class TestExtractModel:
    def test_extract_fixed_decoder(self):
        config = _fixed_attention_config(24)
        arch = extract_model(config)
        assert len(arch.block_groups) == 1
        assert len(arch.unique_block_specs) == 1
        assert arch.block_groups[0].count == 24
        assert arch.block_groups[0].range_label == "Blocks 0..23"

    def test_extract_pattern_decoder_runs(self):
        config = _hybrid_dil_config()
        arch = extract_model(config)
        assert len(arch.block_groups) == 3
        assert len(arch.unique_block_specs) == 2
        assert arch.block_groups[0].count == 8
        assert arch.block_groups[1].count == 32
        assert arch.block_groups[2].count == 8

    def test_extract_pattern_decoder_all_different(self):
        config = _comprehensive_config()
        arch = extract_model(config)
        assert len(arch.block_groups) == 48
        assert all(g.count == 1 for g in arch.block_groups)
        assert len(arch.unique_block_specs) == 8

    def test_extract_stochastic_mixer(self):
        config = _stochastic_supernet_config()
        arch = extract_model(config)
        assert len(arch.block_groups) == 1
        mixer = arch.block_groups[0].block_spec.mixer
        assert isinstance(mixer, StochasticMixerSpec)
        assert mixer.main_mixer_name == "attention"
        names = [name for name, _ in mixer.sub_mixers]
        assert "attention" in names
        assert "sliding_window" in names
        assert "gdn" in names
        assert "kda" in names

    def test_extract_vision_encoder(self):
        config = _vision_config()
        arch = extract_model(config)
        assert arch.vision_encoder is not None
        assert arch.vision_encoder.hidden_size == 128
        assert arch.vision_encoder.num_blocks == 3
        assert arch.vision_encoder.patch_size == (16, 16)

    def test_tie_word_embeddings(self):
        config = _hybrid_dil_config()
        arch = extract_model(config)
        assert arch.tie_word_embeddings is True

    def test_typed_display_configs(self):
        """Verify MixerSpec.display has correct types."""
        config = _fixed_attention_config(4)
        arch = extract_model(config)
        mixer = arch.block_groups[0].block_spec.mixer
        assert isinstance(mixer, MixerSpec)
        assert isinstance(mixer.display, AttentionDisplayConfig)
        assert mixer.display.heads == 32
        assert mixer.display.kv_heads == 8
        assert mixer.display.head_dim == 128

    def test_mlp_display_config(self):
        """Verify BlockSpec.mlp is an MLPDisplayConfig."""
        config = _fixed_attention_config(4)
        arch = extract_model(config)
        mlp_cfg = arch.block_groups[0].block_spec.mlp
        assert isinstance(mlp_cfg, MLPDisplayConfig)
        assert mlp_cfg.gated is True
        assert mlp_cfg.activation == "silu"
        assert mlp_cfg.intermediate_size == 11008

    def test_gdn_display_config(self):
        config = _gdn_config()
        arch = extract_model(config)
        mixer = arch.block_groups[0].block_spec.mixer
        assert isinstance(mixer, MixerSpec)
        assert isinstance(mixer.display, GDNDisplayConfig)
        assert mixer.display.value_heads == 32
        assert mixer.display.key_heads == 8
        assert mixer.display.conv_kernel == 4

    def test_kda_display_config(self):
        config = _kda_config()
        arch = extract_model(config)
        mixer = arch.block_groups[0].block_spec.mixer
        assert isinstance(mixer, MixerSpec)
        assert isinstance(mixer.display, KDADisplayConfig)
        assert mixer.display.heads == 16
        assert mixer.display.head_dim == 128
        assert mixer.display.conv_kernel == 4


class TestRunLengthEncode:
    def _make_spec(self, mixer_type: str) -> BlockSpec:
        return BlockSpec(
            mixer=MixerSpec(mixer_type=mixer_type, label=mixer_type),
            mlp=MLPDisplayConfig(),
            norm_type="RMSNorm",
        )

    def test_all_identical(self):
        spec = self._make_spec("attention")
        blocks = [(None, spec)] * 10
        groups = _run_length_encode(blocks)
        assert len(groups) == 1
        assert groups[0].count == 10

    def test_all_different(self):
        blocks = [(None, self._make_spec(f"type_{i}")) for i in range(5)]
        groups = _run_length_encode(blocks)
        assert len(groups) == 5
        assert all(g.count == 1 for g in groups)

    def test_runs(self):
        a = self._make_spec("attention")
        b = self._make_spec("gdn")
        blocks = [(None, a)] * 3 + [(None, b)] * 5 + [(None, a)] * 2
        groups = _run_length_encode(blocks)
        assert len(groups) == 3
        assert groups[0].count == 3
        assert groups[1].count == 5
        assert groups[2].count == 2

    def test_empty(self):
        assert _run_length_encode([]) == []


class TestMixerLabels:
    def test_attention_label(self):
        config = {"type": "attention", "heads": 32, "head_groups": 8, "head_size": 128}
        arch = extract_model({
            "hidden_size": 4096,
            "decoder": {"type": "fixed", "num_blocks": 1, "block": {
                "mixer": config, "mlp": {}, "normalization": {},
            }},
        })
        mixer = arch.block_groups[0].block_spec.mixer
        assert isinstance(mixer, MixerSpec)
        assert "Attention" in mixer.label
        assert "32" in mixer.label

    def test_sliding_window_label(self):
        config = {"type": "attention", "heads": 32, "head_groups": 8, "head_size": 128, "window_size": 4096}
        arch = extract_model({
            "hidden_size": 4096,
            "decoder": {"type": "fixed", "num_blocks": 1, "block": {
                "mixer": config, "mlp": {}, "normalization": {},
            }},
        })
        mixer = arch.block_groups[0].block_spec.mixer
        assert isinstance(mixer, MixerSpec)
        assert "SWA" in mixer.label
        assert "4096" in mixer.label

    def test_mamba_label(self):
        config = {"type": "mamba", "d_state": 64, "d_conv": 4}
        arch = extract_model({
            "hidden_size": 4096,
            "decoder": {"type": "fixed", "num_blocks": 1, "block": {
                "mixer": config, "mlp": {}, "normalization": {},
            }},
        })
        mixer = arch.block_groups[0].block_spec.mixer
        assert isinstance(mixer, MixerSpec)
        assert "Mamba" in mixer.label
        assert "64" in mixer.label

    def test_gdn_label(self):
        config = {
            "type": "gdn", "value_heads": 32, "key_heads": 8,
            "key_head_dim": 128, "value_head_dim": 128,
            "convolution_layer": {"kernel_size": 4},
        }
        arch = extract_model({
            "hidden_size": 4096,
            "decoder": {"type": "fixed", "num_blocks": 1, "block": {
                "mixer": config, "mlp": {}, "normalization": {},
            }},
        })
        mixer = arch.block_groups[0].block_spec.mixer
        assert isinstance(mixer, MixerSpec)
        assert "GDN" in mixer.label

    def test_kda_label(self):
        config = {"type": "kda", "heads": 32, "head_dim": 128, "convolution_layer": {"kernel_size": 4}}
        arch = extract_model({
            "hidden_size": 4096,
            "decoder": {"type": "fixed", "num_blocks": 1, "block": {
                "mixer": config, "mlp": {}, "normalization": {},
            }},
        })
        mixer = arch.block_groups[0].block_spec.mixer
        assert isinstance(mixer, MixerSpec)
        assert "KDA" in mixer.label
        assert "k=4" in mixer.label


class TestMlpLabel:
    def test_gated(self):
        cfg = MLPDisplayConfig(gated=True, activation="silu", intermediate_size=11008)
        assert "Gated SILU" in mlp_label(cfg)
        assert "11008" in mlp_label(cfg)

    def test_ungated(self):
        cfg = MLPDisplayConfig(gated=False, activation="gelu")
        assert "GELU" in mlp_label(cfg)
        assert "Gated" not in mlp_label(cfg)

    def test_empty(self):
        cfg = MLPDisplayConfig()
        assert mlp_label(cfg) == "MLP"


# ═══════════════════════════════════════════════════════════════════════
# Layout vocabulary tests
# ═══════════════════════════════════════════════════════════════════════


class _MockRenderable:
    """A fixed-size renderable for testing layout containers."""

    def __init__(self, w: float, h: float):
        self._size = Size(w, h)
        self.rendered_at: BBox | None = None

    def measure(self, th: Theme) -> Size:  # noqa: ARG002
        return self._size

    def render(self, bb: BBox, th: Theme) -> Iterator[S.Element]:  # noqa: ARG002
        self.rendered_at = bb
        yield from ()


class TestVStack:
    def test_measure(self):
        children = [_MockRenderable(100, 30), _MockRenderable(80, 40), _MockRenderable(120, 20)]
        stack = VStack(children, gap=10)
        size = stack.measure(TH)
        assert size.w == 120
        assert size.h == 30 + 40 + 20 + 10 * 2

    def test_empty(self):
        stack = VStack([], gap=10)
        assert stack.measure(TH) == Size(0, 0)

    def test_render_positions(self):
        a = _MockRenderable(100, 30)
        b = _MockRenderable(100, 40)
        stack = VStack([a, b], gap=5, align=Align.START)
        bb = BBox(0, 0, 200, 200)
        list(stack.render(bb, TH))
        assert a.rendered_at is not None
        assert a.rendered_at.y == 0
        assert b.rendered_at is not None
        assert b.rendered_at.y == 35

    def test_alignment_center(self):
        a = _MockRenderable(50, 30)
        stack = VStack([a], gap=0, align=Align.CENTER)
        bb = BBox(0, 0, 200, 100)
        list(stack.render(bb, TH))
        assert a.rendered_at is not None
        assert a.rendered_at.x == 75

    def test_alignment_stretch(self):
        a = _MockRenderable(50, 30)
        stack = VStack([a], gap=0, align=Align.STRETCH)
        bb = BBox(0, 0, 200, 100)
        list(stack.render(bb, TH))
        assert a.rendered_at is not None
        assert a.rendered_at.w == 200

    def test_nested(self):
        inner = VStack([_MockRenderable(50, 20)], gap=0)
        outer = VStack([inner, _MockRenderable(50, 30)], gap=5)
        size = outer.measure(TH)
        assert size.h == 20 + 30 + 5

    def test_gap_none_uses_theme(self):
        children = [_MockRenderable(100, 30), _MockRenderable(100, 30)]
        stack = VStack(children, gap=None)
        size = stack.measure(TH)
        assert size.h == 30 + 30 + TH.geo.gap


class TestHStack:
    def test_measure(self):
        children = [_MockRenderable(100, 30), _MockRenderable(80, 40)]
        stack = HStack(children, gap=10)
        size = stack.measure(TH)
        assert size.w == 100 + 80 + 10
        assert size.h == 40

    def test_empty(self):
        stack = HStack([], gap=10)
        assert stack.measure(TH) == Size(0, 0)

    def test_alignment_stretch(self):
        a = _MockRenderable(50, 30)
        stack = HStack([a], gap=0, align=Align.STRETCH)
        bb = BBox(0, 0, 200, 100)
        list(stack.render(bb, TH))
        assert a.rendered_at is not None
        assert a.rendered_at.h == 100

    def test_nested(self):
        inner = HStack([_MockRenderable(30, 20)], gap=0)
        outer = HStack([inner, _MockRenderable(40, 20)], gap=5)
        size = outer.measure(TH)
        assert size.w == 30 + 40 + 5


class TestZStack:
    def test_measure_union(self):
        children = [_MockRenderable(100, 50), _MockRenderable(80, 60)]
        stack = ZStack(children)
        size = stack.measure(TH)
        assert size.w == 100
        assert size.h == 60

    def test_empty(self):
        stack = ZStack([])
        assert stack.measure(TH) == Size(0, 0)

    def test_overlay_ordering(self):
        a = _MockRenderable(100, 50)
        b = _MockRenderable(80, 40)
        stack = ZStack([a, b], align_x=Align.CENTER, align_y=Align.CENTER)
        bb = BBox(0, 0, 100, 60)
        list(stack.render(bb, TH))
        assert a.rendered_at is not None
        assert b.rendered_at is not None
        # b should be centered within the container
        assert b.rendered_at.x == 10  # (100 - 80) / 2

    def test_start_alignment(self):
        a = _MockRenderable(50, 30)
        stack = ZStack([a], align_x=Align.START, align_y=Align.START)
        bb = BBox(10, 20, 200, 100)
        list(stack.render(bb, TH))
        assert a.rendered_at is not None
        assert a.rendered_at.x == 10
        assert a.rendered_at.y == 20


class TestPadded:
    def test_measure(self):
        child = _MockRenderable(100, 50)
        padded = Padded(child, top=10, right=20, bottom=30, left=40)
        size = padded.measure(TH)
        assert size.w == 100 + 20 + 40
        assert size.h == 50 + 10 + 30

    def test_uniform(self):
        child = _MockRenderable(100, 50)
        padded = Padded.uniform(child, 15)
        size = padded.measure(TH)
        assert size.w == 130
        assert size.h == 80

    def test_render_inset(self):
        child = _MockRenderable(100, 50)
        padded = Padded(child, top=10, right=5, bottom=10, left=5)
        bb = BBox(0, 0, 110, 70)
        list(padded.render(bb, TH))
        assert child.rendered_at is not None
        assert child.rendered_at.x == 5
        assert child.rendered_at.y == 10


class TestSpacer:
    def test_measure(self):
        spacer = Spacer(w=50, h=30)
        size = spacer.measure(TH)
        assert size.w == 50
        assert size.h == 30

    def test_render_no_elements(self):
        spacer = Spacer(w=50, h=30)
        elements = list(spacer.render(BBox(0, 0, 50, 30), TH))
        assert len(elements) == 0


class TestFixedSize:
    def test_override_width(self):
        child = _MockRenderable(100, 50)
        fixed = FixedSize(child, w=200)
        size = fixed.measure(TH)
        assert size.w == 200
        assert size.h == 50

    def test_override_height(self):
        child = _MockRenderable(100, 50)
        fixed = FixedSize(child, h=80)
        size = fixed.measure(TH)
        assert size.w == 100
        assert size.h == 80

    def test_override_both(self):
        child = _MockRenderable(100, 50)
        fixed = FixedSize(child, w=200, h=80)
        size = fixed.measure(TH)
        assert size.w == 200
        assert size.h == 80

    def test_override_neither(self):
        child = _MockRenderable(100, 50)
        fixed = FixedSize(child)
        size = fixed.measure(TH)
        assert size.w == 100
        assert size.h == 50


class TestMinSize:
    def test_floor_applied(self):
        child = _MockRenderable(50, 30)
        ms = MinSize(child, min_w=100, min_h=60)
        size = ms.measure(TH)
        assert size.w == 100
        assert size.h == 60

    def test_noop_when_larger(self):
        child = _MockRenderable(200, 100)
        ms = MinSize(child, min_w=50, min_h=30)
        size = ms.measure(TH)
        assert size.w == 200
        assert size.h == 100


class TestClamped:
    def test_min_max(self):
        child = _MockRenderable(50, 200)
        c = Clamped(child, min_w=100, max_w=300, min_h=50, max_h=150)
        size = c.measure(TH)
        assert size.w == 100  # clamped up
        assert size.h == 150  # clamped down

    def test_within_range(self):
        child = _MockRenderable(150, 100)
        c = Clamped(child, min_w=50, max_w=200, min_h=50, max_h=200)
        size = c.measure(TH)
        assert size.w == 150
        assert size.h == 100


class TestOffset:
    def test_translation(self):
        child = _MockRenderable(100, 50)
        offset = Offset(child, dx=10, dy=20)
        bb = BBox(0, 0, 100, 50)
        list(offset.render(bb, TH))
        assert child.rendered_at is not None
        assert child.rendered_at.x == 10
        assert child.rendered_at.y == 20

    def test_measure_unchanged(self):
        child = _MockRenderable(100, 50)
        offset = Offset(child, dx=10, dy=20)
        size = offset.measure(TH)
        assert size.w == 100
        assert size.h == 50


class TestAligned:
    def test_center(self):
        child = _MockRenderable(50, 30)
        aligned = Aligned(child, align_x=Align.CENTER, align_y=Align.CENTER)
        bb = BBox(0, 0, 200, 100)
        list(aligned.render(bb, TH))
        assert child.rendered_at is not None
        assert child.rendered_at.x == 75
        assert child.rendered_at.y == 35

    def test_end(self):
        child = _MockRenderable(50, 30)
        aligned = Aligned(child, align_x=Align.END, align_y=Align.END)
        bb = BBox(0, 0, 200, 100)
        list(aligned.render(bb, TH))
        assert child.rendered_at is not None
        assert child.rendered_at.x == 150
        assert child.rendered_at.y == 70

    def test_start(self):
        child = _MockRenderable(50, 30)
        aligned = Aligned(child, align_x=Align.START, align_y=Align.START)
        bb = BBox(0, 0, 200, 100)
        list(aligned.render(bb, TH))
        assert child.rendered_at is not None
        assert child.rendered_at.x == 0
        assert child.rendered_at.y == 0

    def test_stretch(self):
        child = _MockRenderable(50, 30)
        aligned = Aligned(child, align_x=Align.STRETCH, align_y=Align.STRETCH)
        bb = BBox(0, 0, 200, 100)
        list(aligned.render(bb, TH))
        assert child.rendered_at is not None
        assert child.rendered_at.w == 200
        assert child.rendered_at.h == 100


class TestBackground:
    def test_bg_rect_first(self):
        child = _MockRenderable(100, 50)
        bg = Background(child, css_class="block-bg", rx=6, padding=10)
        bb = BBox(0, 0, 120, 70)
        elements = list(bg.render(bb, TH))
        assert len(elements) >= 1
        assert isinstance(elements[0], S.Rect)

    def test_measure_includes_padding(self):
        child = _MockRenderable(100, 50)
        bg = Background(child, padding=10)
        size = bg.measure(TH)
        assert size.w == 120
        assert size.h == 70

    def test_child_inset(self):
        child = _MockRenderable(100, 50)
        bg = Background(child, padding=10)
        bb = BBox(0, 0, 120, 70)
        list(bg.render(bb, TH))
        assert child.rendered_at is not None
        assert child.rendered_at.x == 10
        assert child.rendered_at.y == 10


class TestResponsive:
    def test_breakpoint_selection(self):
        small = _MockRenderable(100, 50)
        large = _MockRenderable(200, 50)
        resp = Responsive(breakpoints=[(300, large)], fallback=small)
        # Wide container → large
        bb = BBox(0, 0, 400, 100)
        list(resp.render(bb, TH))
        assert large.rendered_at is not None

    def test_fallback(self):
        small = _MockRenderable(100, 50)
        large = _MockRenderable(200, 50)
        resp = Responsive(breakpoints=[(300, large)], fallback=small)
        bb = BBox(0, 0, 200, 100)
        list(resp.render(bb, TH))
        assert small.rendered_at is not None

    def test_measure_uses_largest(self):
        small = _MockRenderable(100, 50)
        large = _MockRenderable(200, 80)
        resp = Responsive(breakpoints=[(300, large)], fallback=small)
        size = resp.measure(TH)
        assert size.w == 200
        assert size.h == 80


class TestAspectFit:
    def test_ratio_preservation(self):
        child = _MockRenderable(100, 100)
        af = AspectFit(child, ratio=2.0)
        bb = BBox(0, 0, 400, 200)
        list(af.render(bb, TH))
        assert child.rendered_at is not None
        # ratio=2.0 → w should be 2× h
        assert abs(child.rendered_at.w / child.rendered_at.h - 2.0) < 0.01

    def test_measure_adjusts(self):
        child = _MockRenderable(100, 100)
        af = AspectFit(child, ratio=2.0)
        size = af.measure(TH)
        assert abs(size.w / size.h - 2.0) < 0.01


class TestBBox:
    def test_cx_cy_aliases(self):
        bb = BBox(10, 20, 100, 50)
        assert bb.cx == bb.center_x == 60.0
        assert bb.cy == bb.center_y == 45.0

    def test_inset(self):
        bb = BBox(10, 20, 100, 50)
        inner = bb.inset(5)
        assert inner.x == 15
        assert inner.y == 25
        assert inner.w == 90
        assert inner.h == 40

    def test_inset_asymmetric(self):
        bb = BBox(0, 0, 200, 100)
        inner = bb.inset(10, 20)
        assert inner.x == 10
        assert inner.y == 20
        assert inner.w == 180
        assert inner.h == 60


class TestFixedWidth:
    def test_overrides_width(self):
        child = _MockRenderable(100, 50)
        fixed = FixedWidth(child, width=200)
        size = fixed.measure(TH)
        assert size.w == 200
        assert size.h == 50

    def test_render_passes_through(self):
        child = _MockRenderable(100, 50)
        fixed = FixedWidth(child, width=200)
        bb = BBox(0, 0, 200, 50)
        list(fixed.render(bb, TH))
        assert child.rendered_at == bb


# ═══════════════════════════════════════════════════════════════════════
# Overlap detection/resolution tests
# ═══════════════════════════════════════════════════════════════════════


class TestOverlapDetection:
    def test_overlapping(self):
        bboxes = [BBox(0, 0, 100, 100), BBox(50, 50, 100, 100)]
        overlaps = detect_overlaps(bboxes)
        assert overlaps == [(0, 1)]

    def test_non_overlapping(self):
        bboxes = [BBox(0, 0, 50, 50), BBox(100, 100, 50, 50)]
        overlaps = detect_overlaps(bboxes)
        assert overlaps == []

    def test_min_distance(self):
        bboxes = [BBox(0, 0, 50, 50), BBox(52, 0, 50, 50)]
        # Without min_distance they don't overlap
        assert detect_overlaps(bboxes, min_distance=0) == []
        # With min_distance=5 they do
        assert detect_overlaps(bboxes, min_distance=5) == [(0, 1)]


class TestOverlapResolution:
    def test_nudge_y(self):
        bboxes = [BBox(0, 0, 50, 50), BBox(0, 40, 50, 50)]
        result = resolve_overlaps(bboxes, axis="y", min_distance=0)
        assert result[1].y >= result[0].bottom

    def test_nudge_x(self):
        bboxes = [BBox(0, 0, 50, 50), BBox(40, 0, 50, 50)]
        result = resolve_overlaps(bboxes, axis="x", min_distance=0)
        assert result[1].x >= result[0].right

    def test_empty(self):
        assert resolve_overlaps([]) == []

    def test_min_distance(self):
        bboxes = [BBox(0, 0, 50, 50), BBox(0, 50, 50, 50)]
        result = resolve_overlaps(bboxes, axis="y", min_distance=10)
        assert result[1].y >= result[0].bottom + 10


class TestLayoutRoot:
    def test_viewport_clamping(self):
        child = _MockRenderable(500, 400)
        root = LayoutRoot(child, viewport=Size(300, 200))
        size = root.measure(TH)
        assert size.w == 300
        assert size.h == 200

    def test_no_viewport(self):
        child = _MockRenderable(500, 400)
        root = LayoutRoot(child)
        size = root.measure(TH)
        assert size.w == 500
        assert size.h == 400


class TestAnchorPoint:
    def test_all_sides(self):
        bb = BBox(10, 20, 100, 50)
        assert anchor_point(bb, AnchorSide.LEFT) == (10, 45.0)
        assert anchor_point(bb, AnchorSide.RIGHT) == (110, 45.0)
        assert anchor_point(bb, AnchorSide.TOP) == (60.0, 20)
        assert anchor_point(bb, AnchorSide.BOTTOM) == (60.0, 70)


class TestRenderConnector:
    def test_bezier(self):
        from_bb = BBox(0, 0, 50, 50)
        to_bb = BBox(100, 0, 50, 50)
        elements = list(render_connector(from_bb, to_bb, style="bezier"))
        assert len(elements) == 1
        assert isinstance(elements[0], S.Path)

    def test_straight(self):
        from_bb = BBox(0, 0, 50, 50)
        to_bb = BBox(100, 0, 50, 50)
        elements = list(render_connector(from_bb, to_bb, style="straight"))
        assert len(elements) == 1
        assert isinstance(elements[0], S.Line)


class TestRenderBrace:
    def test_left_brace(self):
        bb = BBox(50, 10, 100, 200)
        elements = list(render_brace(bb, side=AnchorSide.LEFT))
        assert len(elements) == 1
        assert isinstance(elements[0], S.Path)

    def test_right_brace(self):
        bb = BBox(50, 10, 100, 200)
        elements = list(render_brace(bb, side=AnchorSide.RIGHT))
        assert len(elements) == 1


# ═══════════════════════════════════════════════════════════════════════
# Element tests
# ═══════════════════════════════════════════════════════════════════════


class TestBox:
    def test_measure_default(self):
        box = Box("Test", "box-norm")
        sz = box.measure(TH)
        assert sz.w == TH.geo.inner_w
        assert sz.h == TH.geo.box_h

    def test_measure_custom(self):
        box = Box("Test", "box-norm", w=100, h=40)
        sz = box.measure(TH)
        assert sz.w == 100
        assert sz.h == 40

    def test_render_yields_group(self):
        box = Box("Test", "box-norm")
        sz = box.measure(TH)
        elements = list(box.render(BBox(10, 20, sz.w, sz.h), TH))
        assert len(elements) == 1
        assert isinstance(elements[0], S.G)

    def test_bold_label(self):
        box = Box("Bold", "box-attention", bold=True)
        elements = list(box.render(BBox(0, 0, 100, 32), TH))
        svg_str = str(elements[0])
        assert "t-label-bold" in svg_str


class TestSymbol:
    def test_measure(self):
        s = Symbol("plus")
        sz = s.measure(TH)
        assert sz.w == TH.geo.symbol_r * 2
        assert sz.h == TH.geo.symbol_r * 2

    def test_render_plus(self):
        s = Symbol("plus")
        elements = list(s.render(BBox(0, 0, 18, 18), TH))
        assert len(elements) == 1
        assert isinstance(elements[0], S.G)

    def test_render_cross(self):
        s = Symbol("cross")
        elements = list(s.render(BBox(0, 0, 18, 18), TH))
        assert len(elements) == 1


class TestArrow:
    def test_measure(self):
        a = Arrow("down", length=30)
        sz = a.measure(TH)
        assert sz.w == 0
        assert sz.h == 30

    def test_render_down(self):
        a = Arrow("down")
        elements = list(a.render(BBox(50, 0, 0, 20), TH))
        assert len(elements) == 2  # line + chevron path
        assert isinstance(elements[0], S.Line)
        assert isinstance(elements[1], S.Path)

    def test_render_up(self):
        a = Arrow("up")
        elements = list(a.render(BBox(50, 0, 0, 20), TH))
        assert len(elements) == 2  # line + chevron path
        assert isinstance(elements[0], S.Line)
        assert isinstance(elements[1], S.Path)


class TestLabel:
    def test_measure(self):
        lbl = Label("Hello", "t-ann")
        sz = lbl.measure(TH)
        assert sz.w > 0
        assert sz.h > 0

    def test_render(self):
        lbl = Label("Hello", "t-ann")
        elements = list(lbl.render(BBox(10, 20, 100, 20), TH))
        assert len(elements) == 1
        assert isinstance(elements[0], S.Text)


class TestValueLabel:
    def test_measure_auto_width(self):
        vl = ValueLabel("hidden states")
        sz = vl.measure(TH)
        expected_w = len("hidden states") * TH.typo.sz_ann * 0.6 + 16
        assert sz.w == pytest.approx(expected_w)

    def test_measure_explicit_width(self):
        vl = ValueLabel("q", w=55)
        sz = vl.measure(TH)
        assert sz.w == 55

    def test_unified_height(self):
        for text in ("q", "k", "v", "hidden states", "text tokens"):
            vl = ValueLabel(text)
            sz = vl.measure(TH)
            assert sz.h == TH.geo.value_label_h, f"{text}: expected {TH.geo.value_label_h}, got {sz.h}"

    def test_render_produces_group(self):
        vl = ValueLabel("q")
        sz = vl.measure(TH)
        elements = list(vl.render(BBox(0, 0, sz.w, sz.h), TH))
        assert len(elements) == 1
        g = elements[0]
        assert isinstance(g, S.G)
        assert "box-transparent" in g.class_

    def test_centers_in_larger_bbox(self):
        vl = ValueLabel("q")
        sz = vl.measure(TH)
        big_bb = BBox(0, 0, sz.w + 40, sz.h + 20)
        elements = list(vl.render(big_bb, TH))
        g = elements[0]
        # Second rect is the main rect (first is shadow)
        rects = [e for e in g.elements if isinstance(e, S.Rect)]
        rect = rects[1]
        # Rect should be centered, not stretched to fill
        assert rect.width == pytest.approx(sz.w)
        assert rect.height == pytest.approx(sz.h)
        assert rect.x == pytest.approx(20)  # (40 extra) / 2
        assert rect.y == pytest.approx(10)  # (20 extra) / 2

    def test_has_shadow_and_sheen(self):
        vl = ValueLabel("hidden states")
        sz = vl.measure(TH)
        elements = list(vl.render(BBox(0, 0, sz.w, sz.h), TH))
        svg_str = "".join(str(e) for e in elements)
        assert "box-shadow-muted" in svg_str
        assert "box-sheen" in svg_str


class TestDetailEnvelope:
    def test_measure_no_labels(self):
        env = DetailEnvelope("Title", "css")
        sz = env.measure_envelope(100, 50, TH)
        g = TH.geo.gap
        title_h = TH.geo.title_h
        assert sz.w == 100 + 2 * g
        assert sz.h == 50 + 2 * g + title_h

    def test_measure_with_labels(self):
        env = DetailEnvelope("Title", "css",
                             output_labels=[ExternalLabel("out")],
                             input_labels=[ExternalLabel("in")])
        sz = env.measure_envelope(100, 50, TH)
        label_h = TH.geo.value_label_h + TH.geo.gap
        sz_no_labels = DetailEnvelope("Title", "css").measure_envelope(100, 50, TH)
        assert sz.h == sz_no_labels.h + 2 * label_h

    def test_label_area_h(self):
        env = DetailEnvelope("Title", "css")
        expected = TH.geo.value_label_h + TH.geo.gap
        assert env.label_area_h(TH) == expected

    def test_render_envelope_returns_result(self):
        env = DetailEnvelope("Title", "css",
                             output_labels=[ExternalLabel("out")])
        sz = env.measure_envelope(100, 50, TH)
        bb = BBox(0, 0, sz.w, sz.h)
        result = env.render_envelope(bb, 100, 50, TH)
        assert isinstance(result, EnvelopeResult)
        assert result.frame_bb.w > 0
        assert result.content_bb.w > 0

    def test_phase2_yields_frame(self):
        env = DetailEnvelope("Title", "css")
        sz = env.measure_envelope(100, 50, TH)
        bb = BBox(0, 0, sz.w, sz.h)
        result = env.render_envelope(bb, 100, 50, TH)
        elements = list(result.phase2_frame())
        assert len(elements) == 1
        assert isinstance(elements[0], S.G)

    def test_phase3_with_output_label(self):
        env = DetailEnvelope("Title", "css",
                             output_labels=[ExternalLabel("hidden states")])
        sz = env.measure_envelope(100, 50, TH)
        bb = BBox(0, 0, sz.w, sz.h)
        result = env.render_envelope(bb, 100, 50, TH)
        elements = list(result.phase3_exit_arrow_and_output_labels())
        svg_str = "".join(str(e) for e in elements)
        assert "hidden states" in svg_str

    def test_phase4_with_single_input_label(self):
        env = DetailEnvelope("Title", "css",
                             input_labels=[ExternalLabel("hidden states")])
        sz = env.measure_envelope(100, 50, TH)
        bb = BBox(0, 0, sz.w, sz.h)
        result = env.render_envelope(bb, 100, 50, TH)
        elements = list(result.phase4_entry_arrow_and_input_labels())
        svg_str = "".join(str(e) for e in elements)
        assert "hidden states" in svg_str

    def test_phase4_with_multiple_input_labels(self):
        env = DetailEnvelope("Title", "css",
                             input_labels=[ExternalLabel("q"), ExternalLabel("k"), ExternalLabel("v")])
        sz = env.measure_envelope(100, 50, TH)
        bb = BBox(0, 0, sz.w, sz.h)
        result = env.render_envelope(bb, 100, 50, TH)
        elements = list(result.phase4_entry_arrow_and_input_labels())
        svg_str = "".join(str(e) for e in elements)
        for lbl in ("q", "k", "v"):
            assert lbl in svg_str

    def test_no_labels_empty_phases(self):
        env = DetailEnvelope("Title", "css")
        sz = env.measure_envelope(100, 50, TH)
        bb = BBox(0, 0, sz.w, sz.h)
        result = env.render_envelope(bb, 100, 50, TH)
        assert list(result.phase3_exit_arrow_and_output_labels()) == []
        assert list(result.phase4_entry_arrow_and_input_labels()) == []

    def test_spine_cx_default(self):
        """Default spine_cx equals content_bb.cx."""
        env = DetailEnvelope("Title", "css")
        sz = env.measure_envelope(100, 50, TH)
        bb = BBox(0, 0, sz.w, sz.h)
        result = env.render_envelope(bb, 100, 50, TH)
        assert result.spine_cx == pytest.approx(result.content_bb.cx)

    def test_spine_cx_override(self):
        """Overriding spine_cx shifts phase arrows to the new x."""
        env = DetailEnvelope("Title", "css",
                             output_labels=[ExternalLabel("out")],
                             input_labels=[ExternalLabel("in")])
        sz = env.measure_envelope(100, 50, TH)
        bb = BBox(0, 0, sz.w, sz.h)
        result = env.render_envelope(bb, 100, 50, TH)
        # Override spine
        new_cx = result.content_bb.x + 10
        result.spine_cx = new_cx
        # Phase 1 arrow should use new_cx
        phase1 = list(result.phase1_behind_title())
        assert len(phase1) >= 1
        line = phase1[0]
        assert line.x1 == pytest.approx(new_cx)
        assert line.x2 == pytest.approx(new_cx)
        # Phase 3 arrow should use new_cx
        phase3 = list(result.phase3_exit_arrow_and_output_labels())
        lines = [e for e in phase3 if isinstance(e, S.Line)]
        assert len(lines) >= 1
        assert lines[0].x1 == pytest.approx(new_cx)

    def test_frame_bb_offset_by_output_area(self):
        env = DetailEnvelope("Title", "css",
                             output_labels=[ExternalLabel("out")])
        sz = env.measure_envelope(100, 50, TH)
        bb = BBox(10, 20, sz.w, sz.h)
        result = env.render_envelope(bb, 100, 50, TH)
        label_h = TH.geo.value_label_h + TH.geo.gap
        assert result.frame_bb.y == pytest.approx(20 + label_h)

    def test_containers_have_hidden_states_labels(self):
        """All refactored containers render 'hidden states' labels."""
        containers = [
            GDNDetail(GDNDisplayConfig(value_heads=8)),
            MambaDetail(MambaDisplayConfig(d_state=64)),
            AttentionDetail(AttentionDisplayConfig(heads=8)),
            MLPDetail(MLPDisplayConfig(gated=True, activation="silu", intermediate_size=512)),
        ]
        for container in containers:
            sz = container.measure(TH)
            elements = list(container.render(BBox(0, 0, sz.w, sz.h), TH))
            svg_str = "".join(str(e) for e in elements)
            assert "hidden states" in svg_str, f"{type(container).__name__} missing 'hidden states'"

    def test_kda_has_qkv_input_labels(self):
        """KDADetail renders q, k, v input labels."""
        detail = KDADetail(KDADisplayConfig(heads=16, head_dim=128))
        sz = detail.measure(TH)
        elements = list(detail.render(BBox(0, 0, sz.w, sz.h), TH))
        svg_str = "".join(str(e) for e in elements)
        assert "hidden states" in svg_str  # output label
        # q, k, v as separate label boxes
        assert svg_str.count("box-transparent") >= 4  # 1 output + 3 input

    def test_decoder_block_has_labels(self):
        """DecoderBlock renders 'hidden states' labels."""
        mixer = MixerSpec(mixer_type="attention", label="Attn",
                          display=AttentionDisplayConfig(heads=8))
        block = DecoderBlock(mixer)
        sz = block.measure(TH)
        elements = list(block.render(BBox(0, 0, sz.w, sz.h), TH))
        svg_str = "".join(str(e) for e in elements)
        assert "hidden states" in svg_str

    def test_stochastic_panel_has_labels(self):
        """StochasticMixerPanel renders 'hidden states' labels."""
        spec = StochasticMixerSpec(
            main_mixer_name="attention",
            sub_mixers=(
                ("attention", MixerSpec(mixer_type="attention", label="Attn")),
            ),
        )
        panel = StochasticMixerPanel(spec)
        sz = panel.measure(TH)
        elements = list(panel.render(BBox(0, 0, sz.w, sz.h), TH))
        svg_str = "".join(str(e) for e in elements)
        assert "hidden states" in svg_str


class TestDecoderBlock:
    def test_measure(self):
        mixer = MixerSpec(mixer_type="attention", label="Attention",
                          display=AttentionDisplayConfig(heads=32))
        block = DecoderBlock(mixer)
        sz = block.measure(TH)
        assert sz.w > 0
        assert sz.h > 0

    def test_render_yields_elements(self):
        mixer = MixerSpec(mixer_type="attention", label="Attention",
                          display=AttentionDisplayConfig(heads=32))
        block = DecoderBlock(mixer)
        sz = block.measure(TH)
        elements = list(block.render(BBox(0, 0, sz.w, sz.h), TH))
        assert len(elements) > 0
        assert isinstance(elements[0], S.G)

    def test_different_mixer_types(self):
        for mtype in ["attention", "gdn", "kda", "mamba"]:
            mixer = MixerSpec(mixer_type=mtype, label=mtype)
            block = DecoderBlock(mixer)
            sz = block.measure(TH)
            elements = list(block.render(BBox(0, 0, sz.w, sz.h), TH))
            assert len(elements) > 0

    def test_stochastic_mixer(self):
        mixer = StochasticMixerSpec(
            main_mixer_name="attention",
            sub_mixers=(("attention", MixerSpec(mixer_type="attention", label="Attention")),),
        )
        block = DecoderBlock(mixer)
        sz = block.measure(TH)
        elements = list(block.render(BBox(0, 0, sz.w, sz.h), TH))
        assert len(elements) > 0

    def test_sliding_window(self):
        mixer = MixerSpec(mixer_type="sliding_window", label="SWA",
                          display=AttentionDisplayConfig(heads=32, window_size=4096))
        block = DecoderBlock(mixer)
        sz = block.measure(TH)
        elements = list(block.render(BBox(0, 0, sz.w, sz.h), TH))
        assert len(elements) > 0


# ═══════════════════════════════════════════════════════════════════════
# Detail panel tests
# ═══════════════════════════════════════════════════════════════════════


class TestAttentionDetail:
    def test_measure(self):
        detail = AttentionDetail(AttentionDisplayConfig(heads=32, kv_heads=8))
        sz = detail.measure(TH)
        assert sz.w > 0
        assert sz.h > 0

    def test_render(self):
        detail = AttentionDetail(AttentionDisplayConfig(heads=32, kv_heads=8))
        sz = detail.measure(TH)
        elements = list(detail.render(BBox(0, 0, sz.w, sz.h), TH))
        assert len(elements) > 0

    def test_has_separate_projections(self):
        """Attention detail shows separate q_proj, k_proj, v_proj (not fused qkv_proj)."""
        detail = AttentionDetail(AttentionDisplayConfig(heads=32, kv_heads=8))
        sz = detail.measure(TH)
        elements = list(detail.render(BBox(0, 0, sz.w, sz.h), TH))
        svg_str = "".join(str(e) for e in elements)
        assert "q_proj" in svg_str
        assert "k_proj" in svg_str
        assert "v_proj" in svg_str
        assert "qkv_proj" not in svg_str

    def test_no_cross_symbol(self):
        """No ×cross symbol in attention detail panel."""
        detail = AttentionDetail(AttentionDisplayConfig(heads=32, kv_heads=8))
        items = detail._build(TH)
        for _, child in items:
            assert not isinstance(child, Symbol)

    def test_sliding_window_variant(self):
        """Sliding window variant uses SWA CSS and title."""
        detail = AttentionDetail(AttentionDisplayConfig(heads=32, kv_heads=8, window_size=4096))
        sz = detail.measure(TH)
        elements = list(detail.render(BBox(0, 0, sz.w, sz.h), TH))
        svg_str = "".join(str(e) for e in elements)
        assert "Sliding-window" in svg_str

    def test_has_rope_boxes(self):
        """Attention detail shows two RoPE boxes (for q and k columns)."""
        detail = AttentionDetail(AttentionDisplayConfig(heads=32, kv_heads=8))
        sz = detail.measure(TH)
        elements = list(detail.render(BBox(0, 0, sz.w, sz.h), TH))
        svg_str = "".join(str(e) for e in elements)
        assert svg_str.count("RoPE") == 2

    def test_has_sdpa_label(self):
        """Attention detail shows 'Scaled dot-product attention' label."""
        detail = AttentionDetail(AttentionDisplayConfig(heads=32, kv_heads=8))
        sz = detail.measure(TH)
        elements = list(detail.render(BBox(0, 0, sz.w, sz.h), TH))
        svg_str = "".join(str(e) for e in elements)
        assert "Scaled dot-product attention" in svg_str

    def test_build_item_count(self):
        """Attention detail _build() returns 5 items."""
        detail = AttentionDetail(AttentionDisplayConfig(heads=32, kv_heads=8))
        items = detail._build(TH)
        assert len(items) == 5


class TestGDNDetail:
    def test_measure(self):
        detail = GDNDetail(GDNDisplayConfig(
            value_heads=32, key_heads=8, key_head_dim=64, value_head_dim=64, conv_kernel=4,
        ))
        sz = detail.measure(TH)
        assert sz.w > 0
        assert sz.h > 0

    def test_render(self):
        detail = GDNDetail(GDNDisplayConfig(
            value_heads=32, key_heads=8, key_head_dim=64, value_head_dim=64, conv_kernel=4,
        ))
        sz = detail.measure(TH)
        elements = list(detail.render(BBox(0, 0, sz.w, sz.h), TH))
        assert len(elements) > 0

    def test_no_cross_symbol(self):
        """No ×cross symbol in GDN detail panel."""
        detail = GDNDetail(GDNDisplayConfig(
            value_heads=32, key_heads=8, key_head_dim=64, value_head_dim=64, conv_kernel=4,
        ))
        items = detail._build(TH)
        for _, child in items:
            assert not isinstance(child, Symbol)

    def test_hstack_equal_width(self):
        """in_proj_qkvz and in_proj_βα should have equal width."""
        detail = GDNDetail(GDNDisplayConfig(
            value_heads=32, key_heads=8, key_head_dim=64, value_head_dim=64, conv_kernel=4,
        ))
        items = detail._build(TH)
        hstack = items[-1][1]
        assert isinstance(hstack, HStack)
        # Both boxes should have the same width
        assert hstack.children[0].w == hstack.children[1].w


class TestKDADetail:
    def test_measure(self):
        detail = KDADetail(KDADisplayConfig(heads=16, head_dim=128))
        sz = detail.measure(TH)
        assert sz.w > 0
        assert sz.h > 0

    def test_render(self):
        detail = KDADetail(KDADisplayConfig(heads=16, head_dim=128))
        sz = detail.measure(TH)
        elements = list(detail.render(BBox(0, 0, sz.w, sz.h), TH))
        assert len(elements) > 0

    def test_no_cross_symbol(self):
        """No ×cross symbol in KDA detail panel."""
        detail = KDADetail(KDADisplayConfig(heads=16, head_dim=128))
        items = detail._build(TH)
        for _, child in items:
            assert not isinstance(child, Symbol)



class TestMambaDetail:
    def test_measure(self):
        detail = MambaDetail(MambaDisplayConfig(d_state=64, d_conv=4, d_inner=256))
        sz = detail.measure(TH)
        assert sz.w > 0
        assert sz.h > 0

    def test_render(self):
        detail = MambaDetail(MambaDisplayConfig(d_state=64, d_conv=4, d_inner=256))
        sz = detail.measure(TH)
        elements = list(detail.render(BBox(0, 0, sz.w, sz.h), TH))
        assert len(elements) > 0
        svg_str = "".join(str(e) for e in elements)
        assert "Mamba SSM" in svg_str
        assert "in_proj" in svg_str
        assert "out_proj" in svg_str
        assert "CausalConv1d" in svg_str

    def test_five_rows(self):
        """Mamba detail has 5 rows: out_proj, norm, ssm, conv, in_proj."""
        detail = MambaDetail(MambaDisplayConfig(d_state=64))
        items = detail._build(TH)
        assert len(items) == 5


class TestMLPDetail:
    def test_measure_gated(self):
        detail = MLPDetail(MLPDisplayConfig(gated=True, activation="silu", intermediate_size=11008))
        sz = detail.measure(TH)
        assert sz.w > 0
        assert sz.h > 0

    def test_measure_ungated(self):
        detail = MLPDetail(MLPDisplayConfig(gated=False, activation="silu", intermediate_size=5504))
        sz = detail.measure(TH)
        assert sz.w > 0
        assert sz.h > 0

    def test_render_gated(self):
        detail = MLPDetail(MLPDisplayConfig(gated=True, activation="silu", intermediate_size=11008))
        sz = detail.measure(TH)
        elements = list(detail.render(BBox(0, 0, sz.w, sz.h), TH))
        assert len(elements) > 0

    def test_labels_gated(self):
        """Gated MLP shows down_proj, gate_proj, up_proj."""
        detail = MLPDetail(MLPDisplayConfig(gated=True, activation="silu", intermediate_size=11008))
        sz = detail.measure(TH)
        elements = list(detail.render(BBox(0, 0, sz.w, sz.h), TH))
        svg_str = "".join(str(e) for e in elements)
        assert "down_proj" in svg_str
        assert "gate_proj" in svg_str
        assert "up_proj" in svg_str

    def test_labels_ungated(self):
        """Ungated MLP shows down_proj and up_proj."""
        detail = MLPDetail(MLPDisplayConfig(gated=False, activation="silu", intermediate_size=5504))
        sz = detail.measure(TH)
        elements = list(detail.render(BBox(0, 0, sz.w, sz.h), TH))
        svg_str = "".join(str(e) for e in elements)
        assert "down_proj" in svg_str
        assert "up_proj" in svg_str

    def test_silu_capitalization(self):
        """Activation label reads 'SiLU' not 'SILU'."""
        detail = MLPDetail(MLPDisplayConfig(gated=True, activation="silu", intermediate_size=11008))
        sz = detail.measure(TH)
        elements = list(detail.render(BBox(0, 0, sz.w, sz.h), TH))
        svg_str = "".join(str(e) for e in elements)
        assert "SiLU" in svg_str
        assert "SILU" not in svg_str

    def test_gelu_capitalization(self):
        """Activation label reads 'GELU' for gelu."""
        detail = MLPDetail(MLPDisplayConfig(gated=False, activation="gelu", intermediate_size=5504))
        sz = detail.measure(TH)
        elements = list(detail.render(BBox(0, 0, sz.w, sz.h), TH))
        svg_str = "".join(str(e) for e in elements)
        assert "GELU" in svg_str

    def test_background_box(self):
        """MLPDetail renders a detail-mlp titled frame."""
        detail = MLPDetail(MLPDisplayConfig(gated=True, activation="silu", intermediate_size=11008))
        sz = detail.measure(TH)
        elements = list(detail.render(BBox(0, 0, sz.w, sz.h), TH))
        svg_str = "".join(str(e) for e in elements)
        assert "detail-mlp" in svg_str

    def test_gated_fork_merge_arrows(self):
        """Gated MLP renders fork/merge Path arrows with manual arrowheads."""
        detail = MLPDetail(MLPDisplayConfig(gated=True, activation="silu", intermediate_size=11008))
        sz = detail.measure(TH)
        elements = list(detail.render(BBox(0, 0, sz.w, sz.h), TH))
        # Fork/merge produces Path elements for the routing lines (≥4 draw commands)
        fork_merge_paths = [e for e in elements if isinstance(e, S.Path)
                            and hasattr(e, 'd') and e.d and len(e.d) >= 4]
        assert len(fork_merge_paths) >= 2, f"Expected ≥2 fork/merge path lines, got {len(fork_merge_paths)}"
        # Arrowheads (4-command: MoveTo, LineTo, LineTo, ClosePath) + fork L-paths (3-command)
        short_paths = [e for e in elements if isinstance(e, S.Path)
                       and hasattr(e, 'd') and e.d and len(e.d) in (3, 4)]
        assert len(short_paths) >= 6, f"Expected ≥6 short paths (arrowheads + L-paths), got {len(short_paths)}"
        # No junction dot (clean fork split)
        circles = [e for e in elements if isinstance(e, S.Circle)]
        assert len(circles) == 0, f"Expected 0 circles (no junction dot), got {len(circles)}"
        # "hidden states" labels as transparent boxes
        svg_str = "".join(str(e) for e in elements)
        assert svg_str.count("hidden states") == 2, "Expected 2 'hidden states' labels"
        assert "box-transparent" in svg_str, "Expected box-transparent class for labels"
        # No marker_end on any element
        for e in elements:
            if hasattr(e, 'marker_end'):
                assert not e.marker_end, "Expected no marker_end (manual arrowheads instead)"

    def test_ungated_simple_arrows(self):
        """Ungated MLP renders simple Line arrows with manual arrowheads."""
        detail = MLPDetail(MLPDisplayConfig(gated=False, activation="silu", intermediate_size=5504))
        sz = detail.measure(TH)
        elements = list(detail.render(BBox(0, 0, sz.w, sz.h), TH))
        # Ungated case: internal arrows + exit/entry arrows produce Lines
        line_arrows = [e for e in elements if isinstance(e, S.Line)
                       and getattr(e, 'class_', None) == ["arrow"]]
        assert len(line_arrows) >= 4, f"Expected ≥4 arrow lines, got {len(line_arrows)}"
        # Manual arrowhead Paths (open chevrons: 3 commands, no ClosePath)
        arrowheads = [e for e in elements if isinstance(e, S.Path)
                      and hasattr(e, 'd') and e.d and len(e.d) == 3]
        assert len(arrowheads) >= 3, f"Expected ≥3 manual arrowheads, got {len(arrowheads)}"
        # "hidden states" labels
        svg_str = "".join(str(e) for e in elements)
        assert svg_str.count("hidden states") == 2, "Expected 2 'hidden states' labels"

    def test_hidden_states_labels(self):
        """Both gated and ungated MLPs render 2 'hidden states' labels as transparent boxes."""
        for gated in (True, False):
            detail = MLPDetail(MLPDisplayConfig(gated=gated, activation="silu",
                                                intermediate_size=11008 if gated else 5504))
            sz = detail.measure(TH)
            elements = list(detail.render(BBox(0, 0, sz.w, sz.h), TH))
            svg_str = "".join(str(e) for e in elements)
            assert svg_str.count("hidden states") == 2, f"gated={gated}: expected 2 'hidden states'"
            assert "box-transparent" in svg_str, f"gated={gated}: expected box-transparent class"

    def test_gated_no_junction_dot(self):
        """Gated MLP does not render a junction dot — just a clean fork split."""
        detail = MLPDetail(MLPDisplayConfig(gated=True, activation="silu", intermediate_size=11008))
        sz = detail.measure(TH)
        elements = list(detail.render(BBox(0, 0, sz.w, sz.h), TH))
        circles = [e for e in elements if isinstance(e, S.Circle)]
        assert len(circles) == 0, f"Expected 0 circles (no junction dot), got {len(circles)}"

    def test_ungated_no_junction_dot(self):
        """Ungated MLP does not render a junction dot."""
        detail = MLPDetail(MLPDisplayConfig(gated=False, activation="silu", intermediate_size=5504))
        sz = detail.measure(TH)
        elements = list(detail.render(BBox(0, 0, sz.w, sz.h), TH))
        circles = [e for e in elements if isinstance(e, S.Circle)]
        assert len(circles) == 0, f"Expected 0 circles in ungated MLP, got {len(circles)}"

    def test_exit_arrow_skips_title_bar(self):
        """The exit arrow from down_proj has a gap where the title bar sits."""
        detail = MLPDetail(MLPDisplayConfig(gated=True, activation="silu", intermediate_size=11008))
        sz = detail.measure(TH)
        elements = list(detail.render(BBox(0, 0, sz.w, sz.h), TH))
        # Collect all arrow Lines that are vertical (x1 == x2)
        vertical_arrows = [e for e in elements if isinstance(e, S.Line)
                           and getattr(e, 'class_', None) == ["arrow"]
                           and e.x1 == e.x2]
        # There should be at least 3 vertical segments for the exit arrow
        # (title bar behind + content area + above frame), verifying no single line spans the full gap
        assert len(vertical_arrows) >= 3, f"Expected ≥3 vertical arrow segments, got {len(vertical_arrows)}"


class TestDecoderBlockMlpBbox:
    def test_mlp_bbox_within_block(self):
        """mlp_bbox returns a BBox inside the block bounds."""
        mixer = MixerSpec(mixer_type="attention", label="Attn",
                          display=AttentionDisplayConfig(heads=8))
        block = DecoderBlock(mixer)
        block_sz = block.measure(TH)
        block_bb = BBox(100, 200, block_sz.w, block_sz.h)
        mlp_bb = block.mlp_bbox(block_bb, TH)
        assert mlp_bb.x >= block_bb.x
        assert mlp_bb.y >= block_bb.y
        assert mlp_bb.right <= block_bb.right
        assert mlp_bb.bottom <= block_bb.bottom
        assert mlp_bb.w > 0
        assert mlp_bb.h > 0

    def test_mlp_bbox_height_is_box_h(self):
        """mlp_bbox height matches the standard box height."""
        mixer = MixerSpec(mixer_type="gdn", label="GDN",
                          display=GDNDisplayConfig(value_heads=8))
        block = DecoderBlock(mixer)
        block_sz = block.measure(TH)
        block_bb = BBox(0, 0, block_sz.w, block_sz.h)
        mlp_bb = block.mlp_bbox(block_bb, TH)
        assert mlp_bb.h == TH.geo.box_h


class TestDetailForMixer:
    def test_attention(self):
        mixer = MixerSpec(mixer_type="attention", label="Attention",
                          display=AttentionDisplayConfig(heads=32))
        detail = detail_for_mixer(mixer)
        assert isinstance(detail, AttentionDetail)

    def test_gdn(self):
        mixer = MixerSpec(mixer_type="gdn", label="GDN",
                          display=GDNDisplayConfig(value_heads=32))
        detail = detail_for_mixer(mixer)
        assert isinstance(detail, GDNDetail)

    def test_kda(self):
        mixer = MixerSpec(mixer_type="kda", label="KDA",
                          display=KDADisplayConfig(heads=16))
        detail = detail_for_mixer(mixer)
        assert isinstance(detail, KDADetail)

    def test_mamba(self):
        mixer = MixerSpec(mixer_type="mamba", label="Mamba",
                          display=MambaDisplayConfig(d_state=64))
        detail = detail_for_mixer(mixer)
        assert isinstance(detail, MambaDetail)


class TestStochasticMixerPanel:
    def _make_spec(self) -> StochasticMixerSpec:
        return StochasticMixerSpec(
            main_mixer_name="attention",
            sub_mixers=(
                ("attention", MixerSpec(mixer_type="attention", label="Attn",
                                        display=AttentionDisplayConfig(heads=8))),
                ("gdn", MixerSpec(mixer_type="gdn", label="GDN",
                                   display=GDNDisplayConfig(value_heads=8))),
                ("kda", MixerSpec(mixer_type="kda", label="KDA",
                                   display=KDADisplayConfig(heads=8))),
            ),
        )

    def test_measure_height(self):
        """Height = n sub-mixers × (box_h + gap) + padding + title."""
        spec = self._make_spec()
        panel = StochasticMixerPanel(spec)
        sz = panel.measure(TH)
        assert sz.h > 0
        assert sz.w > 0

    def test_render_has_title(self):
        """'Stochastic' appears in rendered SVG."""
        spec = self._make_spec()
        panel = StochasticMixerPanel(spec)
        sz = panel.measure(TH)
        elements = list(panel.render(BBox(0, 0, sz.w, sz.h), TH))
        svg_str = "".join(str(e) for e in elements)
        assert "Stochastic" in svg_str

    def test_render_sub_mixer_boxes(self):
        """Each sub-mixer name appears in rendered SVG."""
        spec = self._make_spec()
        panel = StochasticMixerPanel(spec)
        sz = panel.measure(TH)
        elements = list(panel.render(BBox(0, 0, sz.w, sz.h), TH))
        svg_str = "".join(str(e) for e in elements)
        assert "attention" in svg_str
        assert "gdn" in svg_str
        assert "kda" in svg_str

    def test_main_mixer_marked(self):
        """Main mixer has ★ indicator."""
        spec = self._make_spec()
        panel = StochasticMixerPanel(spec)
        sz = panel.measure(TH)
        elements = list(panel.render(BBox(0, 0, sz.w, sz.h), TH))
        svg_str = "".join(str(e) for e in elements)
        assert "\u2605" in svg_str

    def test_sub_mixer_bboxes_count(self):
        """Returns correct number of (bbox, spec) pairs."""
        spec = self._make_spec()
        panel = StochasticMixerPanel(spec)
        sz = panel.measure(TH)
        bboxes = panel.sub_mixer_bboxes(BBox(0, 0, sz.w, sz.h), TH)
        assert len(bboxes) == 3

    def test_sub_mixer_css_classes(self):
        """Each sub-mixer uses correct CSS class."""
        spec = self._make_spec()
        panel = StochasticMixerPanel(spec)
        sz = panel.measure(TH)
        elements = list(panel.render(BBox(0, 0, sz.w, sz.h), TH))
        svg_str = "".join(str(e) for e in elements)
        assert "box-attention" in svg_str
        assert "box-gdn" in svg_str


# ═══════════════════════════════════════════════════════════════════════
# Utility tests
# ═══════════════════════════════════════════════════════════════════════


class TestConnectorBezier:
    def test_returns_path(self):
        path = connector_bezier(0, 50, 200, 50)
        assert isinstance(path, S.Path)

    def test_has_connector_class(self):
        path = connector_bezier(0, 0, 100, 100)
        svg_str = str(path)
        assert "connector" in svg_str


class TestDefs:
    def test_returns_defs(self):
        d = defs(TH)
        assert isinstance(d, S.Defs)

    def test_contains_dotgrid(self):
        d = defs(TH)
        svg_str = str(d)
        assert "dotgrid" in svg_str

    def test_no_markers(self):
        """Markers were removed — only the dotgrid pattern should remain."""
        d = defs(TH)
        svg_str = str(d)
        assert "arr-d" not in svg_str
        assert "arr-u" not in svg_str


# ═══════════════════════════════════════════════════════════════════════
# Theme tests
# ═══════════════════════════════════════════════════════════════════════


class TestDetailedTheme:
    def test_stylesheet_contains_box_classes(self):
        css = TH.stylesheet()
        for name in ["box-attention", "box-swa", "box-gdn", "box-kda",
                      "box-mamba", "box-conv", "box-gate",
                      "box-stochastic", "box-norm", "box-linear",
                      "box-activation", "box-mlp", "box-embedding",
                      "box-transparent"]:
            assert name in css, f"Missing {name} in stylesheet"

    def test_stylesheet_contains_text_classes(self):
        css = TH.stylesheet()
        for name in ["t-label", "t-label-bold",
                      "t-ann", "t-dim", "t-count", "t-note", "t-small"]:
            assert f".{name}" in css, f"Missing .{name} in stylesheet"

    def test_stylesheet_contains_wires(self):
        css = TH.stylesheet()
        for name in [".arrow", ".connector", ".symbol"]:
            assert name in css, f"Missing {name} in stylesheet"

    def test_stylesheet_contains_background(self):
        css = TH.stylesheet()
        assert ".background" in css

    def test_stylesheet_contains_detail_content(self):
        css = TH.stylesheet()
        assert ".detail-content" in css, "Missing .detail-content in stylesheet"

    def test_palette_mixer(self):
        pal = Palette()
        fill, text = pal.mixer("attention")
        assert fill == pal.attention
        assert text == pal.attn_text

    def test_palette_mixer_unknown(self):
        pal = Palette()
        fill, text = pal.mixer("unknown")
        assert fill == pal.linear
        assert text == pal.linear_text


# ═══════════════════════════════════════════════════════════════════════
# Integration tests
# ═══════════════════════════════════════════════════════════════════════


class TestGenerateDiagram:
    def test_fixed_attention(self):
        config = _fixed_attention_config(24)
        svg = generate_diagram(config)
        assert svg.startswith("<svg")
        assert "viewBox" in svg

    def test_hybrid_dil(self):
        config = _hybrid_dil_config()
        svg = generate_diagram(config)
        assert svg.startswith("<svg")
        assert "attn" in svg
        assert "hybrid" in svg

    def test_stochastic_svg(self):
        config = _stochastic_supernet_config()
        svg = generate_diagram(config)
        assert svg.startswith("<svg")

    def test_comprehensive_svg(self):
        config = _comprehensive_config()
        svg = generate_diagram(config)
        assert svg.startswith("<svg")
        for name in ["attn", "swa", "mamba", "gdn", "kda", "stoch_am", "stoch_sg", "stoch_ak"]:
            assert name in svg, f"Expected '{name}' in SVG"

    def test_gdn_model(self):
        config = _gdn_config()
        svg = generate_diagram(config)
        assert svg.startswith("<svg")
        assert "Gated Delta" in svg or "GDN" in svg

    def test_kda_model(self):
        config = _kda_config()
        svg = generate_diagram(config)
        assert svg.startswith("<svg")
        assert "Kimi" in svg or "KDA" in svg

    def test_output_file(self):
        config = _fixed_attention_config(4)
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
            path = f.name
        generate_diagram(config, output_path=path)
        assert Path(path).exists()
        content = Path(path).read_text()
        assert content.startswith("<svg")
        Path(path).unlink()

    def test_custom_theme(self):
        th = Theme(pal=Palette(attention="#FF0000", attn_text="#CC0000"))
        config = _fixed_attention_config(4)
        svg = generate_diagram(config, theme=th)
        assert "#FF0000" in svg

    def test_style_element_present(self):
        config = _fixed_attention_config(4)
        svg = generate_diagram(config)
        assert "<style>" in svg
        assert ".box-attention" in svg

    def test_has_defs(self):
        config = _fixed_attention_config(4)
        svg = generate_diagram(config)
        assert "<defs>" in svg
        assert "dotgrid" in svg


# ═══════════════════════════════════════════════════════════════════════
# mixer_css_class
# ═══════════════════════════════════════════════════════════════════════


class TestMixerCssClass:
    def test_attention(self):
        spec = BlockSpec(
            mixer=MixerSpec(mixer_type="attention", label="Attention"),
            mlp=MLPDisplayConfig(), norm_type="RMSNorm",
        )
        assert mixer_css_class(spec) == "box-attention"

    def test_sliding_window(self):
        spec = BlockSpec(
            mixer=MixerSpec(mixer_type="sliding_window", label="SWA"),
            mlp=MLPDisplayConfig(), norm_type="RMSNorm",
        )
        assert mixer_css_class(spec) == "box-swa"

    def test_gdn(self):
        spec = BlockSpec(
            mixer=MixerSpec(mixer_type="gdn", label="GDN"),
            mlp=MLPDisplayConfig(), norm_type="RMSNorm",
        )
        assert mixer_css_class(spec) == "box-gdn"

    def test_kda(self):
        spec = BlockSpec(
            mixer=MixerSpec(mixer_type="kda", label="KDA"),
            mlp=MLPDisplayConfig(), norm_type="RMSNorm",
        )
        assert mixer_css_class(spec) == "box-kda"

    def test_mamba(self):
        spec = BlockSpec(
            mixer=MixerSpec(mixer_type="mamba", label="Mamba"),
            mlp=MLPDisplayConfig(), norm_type="RMSNorm",
        )
        assert mixer_css_class(spec) == "box-mamba"

    def test_stochastic(self):
        spec = BlockSpec(
            mixer=StochasticMixerSpec(
                main_mixer_name="attention",
                sub_mixers=(
                    ("attention", MixerSpec(mixer_type="attention", label="Attn")),
                    ("gdn", MixerSpec(mixer_type="gdn", label="GDN")),
                ),
            ),
            mlp=MLPDisplayConfig(), norm_type="RMSNorm",
        )
        assert mixer_css_class(spec) == "box-stochastic"

    def test_unknown_type(self):
        spec = BlockSpec(
            mixer=MixerSpec(mixer_type="unknown", label="Unknown"),
            mlp=MLPDisplayConfig(), norm_type="RMSNorm",
        )
        assert mixer_css_class(spec) == "box-linear"


# ═══════════════════════════════════════════════════════════════════════
# ArchitectureOverview
# ═══════════════════════════════════════════════════════════════════════


class TestArchitectureOverview:
    def _fixed_arch(self, num_blocks: int = 24) -> ArchitectureModel:
        return extract_model(_fixed_attention_config(num_blocks))

    def _hybrid_arch(self) -> ArchitectureModel:
        return extract_model(_hybrid_dil_config())

    def _comprehensive_arch(self) -> ArchitectureModel:
        return extract_model(_comprehensive_config())

    def test_text_only_cells(self):
        """Fixed attention → 5 cells: lm_head + norm + 1 group + embed + 1 pre-decoder."""
        arch = self._fixed_arch()
        overview = ArchitectureOverview(arch)
        sz = overview.measure(TH)
        bb = BBox(0, 0, sz.w, sz.h)
        cells = overview.cell_bboxes(bb, TH)
        # 1 lm_head + 1 norm + 1 decoder group + 1 embed + 1 pre-decoder = 5
        assert len(cells) == 5

    def test_hybrid_cells(self):
        """Hybrid 8+32+8 → 7 cells: lm_head + norm + 3 groups + embed + 1 pre-decoder."""
        arch = self._hybrid_arch()
        overview = ArchitectureOverview(arch)
        sz = overview.measure(TH)
        bb = BBox(0, 0, sz.w, sz.h)
        cells = overview.cell_bboxes(bb, TH)
        # 1 lm_head + 1 norm + 3 decoder groups + 1 embed + 1 pre-decoder = 7
        assert len(cells) == 7

    def test_comprehensive_cells(self):
        """Comprehensive 48 groups → 52 cells: lm_head + norm + 48 groups + embed + 1 pre-decoder."""
        arch = self._comprehensive_arch()
        overview = ArchitectureOverview(arch)
        sz = overview.measure(TH)
        bb = BBox(0, 0, sz.w, sz.h)
        cells = overview.cell_bboxes(bb, TH)
        # 1 lm_head + 1 norm + 48 decoder groups + 1 embed + 1 pre-decoder = 52
        assert len(cells) == 52

    def test_cell_height_uniform(self):
        """All cells have stack_cell_h height."""
        arch = self._fixed_arch()
        overview = ArchitectureOverview(arch)
        sz = overview.measure(TH)
        bb = BBox(0, 0, sz.w, sz.h)
        cells = overview.cell_bboxes(bb, TH)
        for cell_bb, _ in cells:
            assert cell_bb.h == TH.geo.stack_cell_h

    def test_cells_non_overlapping(self):
        """Sequential bboxes should not overlap."""
        arch = self._hybrid_arch()
        overview = ArchitectureOverview(arch)
        sz = overview.measure(TH)
        bb = BBox(0, 0, sz.w, sz.h)
        cells = overview.cell_bboxes(bb, TH)
        for i in range(len(cells) - 1):
            assert cells[i][0].bottom <= cells[i + 1][0].y

    def test_contains_embedding_label(self):
        """'Embedding' should appear in rendered SVG."""
        arch = self._fixed_arch()
        overview = ArchitectureOverview(arch)
        sz = overview.measure(TH)
        bb = BBox(0, 0, sz.w, sz.h)
        elements = list(overview.render(bb, TH))
        svg_str = "".join(str(e) for e in elements)
        assert "Embedding" in svg_str

    def test_contains_lm_head_label(self):
        """'LM Head' should appear in rendered SVG."""
        arch = self._fixed_arch()
        overview = ArchitectureOverview(arch)
        sz = overview.measure(TH)
        bb = BBox(0, 0, sz.w, sz.h)
        elements = list(overview.render(bb, TH))
        svg_str = "".join(str(e) for e in elements)
        assert "LM Head" in svg_str

    def test_multiplier_in_label(self):
        """Groups with count>1 show '×N' in the label."""
        arch = self._fixed_arch(24)
        overview = ArchitectureOverview(arch)
        sz = overview.measure(TH)
        bb = BBox(0, 0, sz.w, sz.h)
        elements = list(overview.render(bb, TH))
        svg_str = "".join(str(e) for e in elements)
        assert "\u00d724" in svg_str

    def test_tied_weights(self):
        """When tie_word_embeddings=True, 'tied' appears in SVG."""
        arch = self._hybrid_arch()
        assert arch.tie_word_embeddings is True
        overview = ArchitectureOverview(arch)
        sz = overview.measure(TH)
        bb = BBox(0, 0, sz.w, sz.h)
        elements = list(overview.render(bb, TH))
        svg_str = "".join(str(e) for e in elements)
        assert "tied" in svg_str

    def test_decoder_cells_have_spec(self):
        """Decoder cell_bboxes have non-None spec; non-decoder cells have None."""
        arch = self._fixed_arch()
        overview = ArchitectureOverview(arch)
        sz = overview.measure(TH)
        bb = BBox(0, 0, sz.w, sz.h)
        cells = overview.cell_bboxes(bb, TH)
        # cells[0] = LM Head (None), cells[1] = Norm (None)
        assert cells[0][1] is None  # LM Head
        assert cells[1][1] is None  # Norm
        # cells[-2] = Embedding (None), cells[-1] = text tokens (None)
        assert cells[-2][1] is None  # Embedding
        assert cells[-1][1] is None  # text tokens
        # Decoder block cells should have non-None spec
        for _, spec in cells[2:-2]:
            assert spec is not None

    def test_decoder_frame_label(self):
        """'Decoder' frame title should appear in rendered SVG."""
        arch = self._fixed_arch()
        overview = ArchitectureOverview(arch)
        sz = overview.measure(TH)
        bb = BBox(0, 0, sz.w, sz.h)
        elements = list(overview.render(bb, TH))
        svg_str = "".join(str(e) for e in elements)
        assert "Decoder" in svg_str

    def test_data_labels(self):
        """'text tokens' and 'token probabilities' should appear in rendered SVG."""
        arch = self._fixed_arch()
        overview = ArchitectureOverview(arch)
        sz = overview.measure(TH)
        bb = BBox(0, 0, sz.w, sz.h)
        elements = list(overview.render(bb, TH))
        svg_str = "".join(str(e) for e in elements)
        assert "text tokens" in svg_str
        assert "token probabilities" in svg_str
        assert "box-transparent" in svg_str
        assert "Sample input text" not in svg_str

    def test_no_marker_end_in_overview(self):
        """All overview arrows use manual chevrons, not SVG marker-end."""
        arch = self._fixed_arch()
        overview = ArchitectureOverview(arch)
        sz = overview.measure(TH)
        bb = BBox(0, 0, sz.w, sz.h)
        elements = list(overview.render(bb, TH))
        svg_str = "".join(str(e) for e in elements)
        assert "marker-end" not in svg_str
        assert "marker_end" not in svg_str


class TestVisionEncoderColumn:
    def _vision_spec(self) -> VisionEncoderSpec:
        arch = extract_model(_vision_config())
        assert arch.vision_encoder is not None
        return arch.vision_encoder

    def test_three_cells(self):
        """Measure height = 3 cells + 2 gaps."""
        vision = self._vision_spec()
        col = VisionEncoderColumn(vision)
        sz = col.measure(TH)
        expected_h = 3 * TH.geo.stack_cell_h + 2 * TH.geo.stack_cell_gap
        assert sz.h == expected_h

    def test_render_patch_embed_label(self):
        """'Patch Embed' should appear in rendered SVG."""
        vision = self._vision_spec()
        col = VisionEncoderColumn(vision)
        sz = col.measure(TH)
        bb = BBox(0, 0, sz.w, sz.h)
        elements = list(col.render(bb, TH))
        svg_str = "".join(str(e) for e in elements)
        assert "Patch Embed" in svg_str

    def test_render_adapter_label(self):
        """'Adapter' should appear in rendered SVG."""
        vision = self._vision_spec()
        col = VisionEncoderColumn(vision)
        sz = col.measure(TH)
        bb = BBox(0, 0, sz.w, sz.h)
        elements = list(col.render(bb, TH))
        svg_str = "".join(str(e) for e in elements)
        assert "Adapter" in svg_str

    def test_cell_bboxes(self):
        """Only the encoder cell (middle) has a non-None spec."""
        vision = self._vision_spec()
        col = VisionEncoderColumn(vision)
        sz = col.measure(TH)
        bb = BBox(0, 0, sz.w, sz.h)
        cells = col.cell_bboxes(bb, TH)
        assert len(cells) == 3
        assert cells[0][1] is None  # Patch Embed
        assert cells[1][1] is not None  # Vision Encoder
        assert cells[2][1] is None  # Adapter


class TestLayoutIncludesStack:
    def test_fixed_has_overview(self):
        """End-to-end: fixed decoder SVG contains overview elements."""
        config = _fixed_attention_config(24)
        svg = generate_diagram(config)
        assert "Embedding" in svg
        assert "LM Head" in svg
        assert "RMSNorm" in svg

    def test_hybrid_has_overview_and_tied(self):
        """End-to-end: hybrid decoder SVG has overview + tied weights."""
        config = _hybrid_dil_config()
        svg = generate_diagram(config)
        assert "Embedding" in svg
        assert "tied" in svg
        # Should have connector paths
        assert "connector" in svg

    def test_vision_has_vision_column(self):
        """End-to-end: vision config SVG has vision column elements."""
        config = _vision_config()
        svg = generate_diagram(config)
        assert "Patch Embed" in svg
        assert "Vision Enc" in svg
        assert "Adapter" in svg
        assert "replace" in svg

    def test_comprehensive_has_overview(self):
        """End-to-end: comprehensive config SVG has overview."""
        config = _comprehensive_config()
        svg = generate_diagram(config)
        assert "Embedding" in svg
        assert "LM Head" in svg

    def test_stochastic_has_dispatch_column(self):
        """When stochastic blocks exist, 'Stochastic' appears in SVG."""
        config = _stochastic_supernet_config()
        svg = generate_diagram(config)
        assert "Stochastic" in svg

    def test_no_stochastic_no_column(self):
        """When no stochastic blocks, 'Stochastic Dispatch' absent."""
        config = _fixed_attention_config(4)
        svg = generate_diagram(config)
        assert "Stochastic Dispatch" not in svg

    def test_comprehensive_has_dispatch(self):
        """Comprehensive config shows stochastic panel."""
        config = _comprehensive_config()
        svg = generate_diagram(config)
        assert "Stochastic" in svg


class TestBackgroundClearance:
    """Background rect has rounded corners and 2*gap clearance to content."""

    def _parse_svg(self, svg_str: str):
        import xml.etree.ElementTree as ET
        return ET.fromstring(svg_str)

    def _get_bg_rects(self, root):
        ns = {"svg": "http://www.w3.org/2000/svg"}
        return root.findall("svg:rect", ns)

    def test_background_has_rounded_corners(self):
        config = _fixed_attention_config(4)
        svg = generate_diagram(config)
        root = self._parse_svg(svg)
        bg_rects = self._get_bg_rects(root)
        assert len(bg_rects) >= 2
        for rect in bg_rects[:2]:
            rx = rect.get("rx")
            assert rx is not None, "Background rect missing rx attribute"
            assert float(rx) == TH.geo.rx

    def test_content_wrapped_in_translated_group(self):
        config = _fixed_attention_config(4)
        svg = generate_diagram(config)
        root = self._parse_svg(svg)
        ns = {"svg": "http://www.w3.org/2000/svg"}
        groups = root.findall("svg:g", ns)
        # At least one <g> should have a translate transform
        translated = [g for g in groups if g.get("transform") and "translate" in g.get("transform", "")]
        assert len(translated) >= 1, "Expected a <g> with translate transform"

    def test_clearance_at_least_2gap(self):
        """Content + translation leaves >= 2*gap clearance on all sides."""
        import re as _re
        config = _fixed_attention_config(4)
        th = TH
        svg = generate_diagram(config, theme=th)
        root = self._parse_svg(svg)
        ns = {"svg": "http://www.w3.org/2000/svg"}

        # Parse viewBox dimensions
        vb = root.get("viewBox", "")
        vb_parts = vb.split()
        assert len(vb_parts) == 4
        total_w, total_h = float(vb_parts[2]), float(vb_parts[3])

        # Parse translate transform
        groups = root.findall("svg:g", ns)
        translated = [g for g in groups if g.get("transform") and "translate" in g.get("transform", "")]
        assert translated, "No translated group found"
        transform = translated[0].get("transform", "")
        m = _re.search(r"translate\(([\d.e+-]+)[, ]+([\d.e+-]+)\)", transform)
        assert m, f"Could not parse translate from {transform}"
        dx, dy = float(m.group(1)), float(m.group(2))

        # Get the content bbox from _layout
        from fast_llm_external_models.apriel2.conversion.diagram import _layout
        from fast_llm_external_models.apriel2.conversion.diagram.model import extract_model
        arch = extract_model(config)
        _, content_bb = _layout(arch, th)

        # After translation, content starts at (content_bb.x + dx, content_bb.y + dy)
        # and ends at (content_bb.x + dx + content_bb.w, content_bb.y + dy + content_bb.h)
        min_clearance = 2 * th.geo.gap
        left = content_bb.x + dx
        top = content_bb.y + dy
        right = total_w - (content_bb.x + dx + content_bb.w)
        bottom = total_h - (content_bb.y + dy + content_bb.h)
        assert left >= min_clearance - 0.1, f"left={left} < {min_clearance}"
        assert top >= min_clearance - 0.1, f"top={top} < {min_clearance}"
        assert right >= min_clearance - 0.1, f"right={right} < {min_clearance}"
        assert bottom >= min_clearance - 0.1, f"bottom={bottom} < {min_clearance}"

    def test_all_configs_have_rounded_bg(self):
        """Verify rounded corners for multiple config types."""
        configs = [
            _fixed_attention_config(4),
            _hybrid_dil_config(),
            _stochastic_supernet_config(),
        ]
        for config in configs:
            svg = generate_diagram(config)
            root = self._parse_svg(svg)
            bg_rects = self._get_bg_rects(root)
            assert len(bg_rects) >= 2
            assert bg_rects[0].get("rx") is not None
