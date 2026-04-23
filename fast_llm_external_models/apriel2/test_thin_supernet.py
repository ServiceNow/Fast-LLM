#!/usr/bin/env python3
"""Test thin_supernet conversion.

Runs unit tests and an integration test. Output goes to /tmp (cleared on reboot).
Supports both pytest and direct execution.

Usage:
    pytest fast_llm_external_models/apriel2/test_thin_supernet.py -v
    python fast_llm_external_models/apriel2/test_thin_supernet.py

Requires checkpoint at ~/.cache/huggingface/apriel2-0.5b-dev for integration test.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import yaml

from fast_llm_external_models.apriel2.thin_supernet import (
    build_thin_surgery_config,
    compute_required_mixers_per_layer,
    load_placement_config,
    thin_supernet,
)

# Add project root for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Try pytest for fixture support
try:
    import pytest

    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False


# =============================================================================
# Constants
# =============================================================================

APRIEL2_05B_DEV = Path.home() / ".cache" / "huggingface" / "apriel2-0.5b-dev"
OUTPUT_DIR_TMP = Path("/tmp") / "apriel2-thin-supernet-test"


# =============================================================================
# Unit Tests
# =============================================================================


def test_load_placement_config_placements_format(tmp_path=None):
    """Test loading placements list format."""
    tmp_path = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    config = {
        "placements": [
            ["attention", "gdn", "attention"],
            ["sliding_window", "sliding_window", "attention"],
        ]
    }
    path = tmp_path / "placements.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)

    result = load_placement_config(path)
    assert len(result) == 2
    assert result[0] == ["attention", "gdn", "attention"]
    assert result[1] == ["sliding_window", "sliding_window", "attention"]


def test_load_placement_config_layers_format(tmp_path=None):
    """Test loading single placement via layers format."""
    tmp_path = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    config = {"layers": ["attention", "gdn", "attention", "kda"]}
    path = tmp_path / "layers.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)

    result = load_placement_config(path)
    assert len(result) == 1
    assert result[0] == ["attention", "gdn", "attention", "kda"]


def test_compute_required_mixers_per_layer():
    """Test union of mixers per layer across placements."""
    placements = [
        ["attention", "gdn", "attention"],
        ["sliding_window", "gdn", "kda"],
        ["attention", "attention", "attention"],
    ]
    result = compute_required_mixers_per_layer(placements, num_layers=3)
    assert result[0] == {"attention", "sliding_window"}
    assert result[1] == {"gdn", "attention"}
    assert result[2] == {"attention", "kda"}


def _get_supernet_config():
    """Minimal supernet config for unit tests."""
    return {
        "model_type": "apriel2_text",
        "hidden_size": 896,
        "decoder": {
            "type": "fixed",
            "num_blocks": 24,
            "block": {
                "mixer": {
                    "type": "stochastic",
                    "main_mixer_name": "attention",
                    "mixers": {
                        "attention": {"type": "attention", "heads": 14},
                        "sliding_window": {"type": "attention", "window_size": 4096},
                        "gdn": {"type": "gdn", "convolution_layer": {"kernel_size": 4}},
                        "kda": {"type": "kda", "convolution_layer": {"kernel_size": 4}},
                    },
                },
                "mlp": {"type": "mlp"},
                "normalization": {"type": "rms_norm"},
            },
        },
    }


def test_build_thin_surgery_config_all_same():
    """Test surgery config when all layers need same mixers."""
    supernet_config = _get_supernet_config()
    required = [{"attention", "gdn"}] * 24
    surgery = build_thin_surgery_config(supernet_config, required)
    assert surgery["decoder"]["type"] == "fixed"
    mixers = surgery["decoder"]["block"]["mixer"]["mixers"]
    assert set(mixers.keys()) == {"attention", "gdn"}
    assert mixers["attention"]["init"] == "transfer"
    assert mixers["gdn"]["init"] == "transfer"


def test_build_thin_surgery_config_per_layer():
    """Test surgery config when layers need different mixers."""
    supernet_config = _get_supernet_config()
    required = [{"attention"}] * 8 + [{"attention", "gdn"}] * 8 + [{"gdn"}] * 8
    surgery = build_thin_surgery_config(supernet_config, required)
    assert surgery["decoder"]["type"] == "pattern"
    assert len(surgery["decoder"]["pattern"]) == 24
    assert set(surgery["decoder"]["blocks"]["layer_0"]["mixer"]["mixers"].keys()) == {"attention"}
    assert set(surgery["decoder"]["blocks"]["layer_8"]["mixer"]["mixers"].keys()) == {"attention", "gdn"}
    assert set(surgery["decoder"]["blocks"]["layer_16"]["mixer"]["mixers"].keys()) == {"gdn"}


# =============================================================================
# Integration Test
# =============================================================================


def test_thin_supernet_integration(tmp_path):
    """Integration test: full thin_supernet conversion (requires checkpoint)."""
    if not APRIEL2_05B_DEV.exists():
        pytest.skip(f"Checkpoint not found: {APRIEL2_05B_DEV}")
    output_dir = tmp_path / "thinned"
    run_integration_test(output_dir)


def run_integration_test(output_dir: Path) -> bool:
    """Run thin_supernet full conversion. Returns True on success."""
    if not APRIEL2_05B_DEV.exists():
        print(f"SKIP: Checkpoint not found: {APRIEL2_05B_DEV}")
        return False

    placements_dir = Path(__file__).parent / "examples" / "placements"
    placement_config = placements_dir / "budget_attention_heavy.yaml"
    if not placement_config.exists():
        print(f"SKIP: Placement config not found: {placement_config}")
        return False

    print("=" * 60)
    print("Thin Supernet Integration Test")
    print("=" * 60)
    print(f"Input:  {APRIEL2_05B_DEV}")
    print(f"Output: {output_dir}")
    print(f"Placements: {placement_config}")
    print("=" * 60)

    # Dry run
    print("\n--- Dry run ---")
    thin_supernet(
        input_dir=APRIEL2_05B_DEV,
        output_dir=output_dir,
        placement_configs=[placement_config],
        dry_run=True,
        verbose=True,
    )

    # Full run
    print("\n--- Full conversion ---")
    thin_supernet(
        input_dir=APRIEL2_05B_DEV,
        output_dir=output_dir,
        placement_configs=[placement_config],
        dry_run=False,
        verbose=True,
    )

    # Verify
    print("\n--- Verification ---")
    config_path = output_dir / "config.json"
    assert config_path.exists(), "config.json not created"
    with open(config_path) as f:
        saved = json.load(f)

    decoder = saved.get("decoder", {})
    mixer = decoder.get("block", {}).get("mixer") or decoder.get("blocks", {}).get("layer_0", {}).get("mixer", {})
    mixers = mixer.get("mixers", {})
    print(f"Decoder type: {decoder.get('type')}")
    print(f"Mixers retained: {list(mixers.keys())}")

    safetensors = list(output_dir.glob("*.safetensors"))
    print(f"Safetensor files: {len(safetensors)}")

    assert (output_dir / "tokenizer.json").exists() or (output_dir / "tokenizer_config.json").exists()
    print("\n" + "=" * 60)
    print("Test PASSED")
    print("=" * 60)
    return True


# =============================================================================
# Main
# =============================================================================


def main():
    """Run unit tests and integration test."""
    # Unit tests (no checkpoint needed)
    print("Running unit tests...")
    test_load_placement_config_placements_format()
    test_load_placement_config_layers_format()
    test_compute_required_mixers_per_layer()
    test_build_thin_surgery_config_all_same()
    test_build_thin_surgery_config_per_layer()
    print("Unit tests OK\n")

    # Integration test (uses /tmp, requires checkpoint)
    success = run_integration_test(OUTPUT_DIR_TMP)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
