#!/usr/bin/env python3
"""Generate architecture diagrams from Apriel2 source checkpoints.

Apriel-1.6 (multimodal, 48 layers):
  apriel2_comprehensive.svg  – comprehensive surgery (all mixer types)
  apriel2_fixed.svg          – base text-only model (attention-only)
  diagram_fixed.svg          – same as apriel2_fixed
  diagram_hybrid.svg         – hybrid DIL surgery (attn + stochastic GDN)
  diagram_vision.svg         – base multimodal model (with vision encoder)

Qwen2 (text-only, 24 layers):
  qwen2_fixed.svg            – base model (attention-only)
  qwen2_supernet.svg         – stochastic supernet (all 4 mixer types)
  qwen2_hybrid.svg           – hybrid DIL surgery (attn + stochastic GDN)

Usage:
    python generate_diagram.py
    python generate_diagram.py --apriel /path/to/apriel --qwen2 /path/to/qwen2
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

from fast_llm_external_models.apriel2.convert import build_plan
from fast_llm_external_models.apriel2.conversion.diagram import generate_diagram

DEFAULT_APRIEL = "/tmp/ServiceNow-AI/Apriel-1.6-15b-Thinker"
DEFAULT_QWEN2 = "/tmp/qwen2/teacher"
EXAMPLES_DIR = Path("fast_llm_external_models/apriel2/examples")

# (output_filename, surgery_yaml_or_None, keep_vision)
DiagramSpec = tuple[str, str | None, bool]

APRIEL_DIAGRAMS: list[DiagramSpec] = [
    ("apriel2_comprehensive.svg", "comprehensive.yaml", False),
    ("apriel2_fixed.svg", None, False),
    ("diagram_fixed.svg", None, False),
    ("diagram_hybrid.svg", "hybrid_dil.yaml", False),
    ("diagram_vision.svg", None, True),
]

QWEN2_DIAGRAMS: list[DiagramSpec] = [
    ("qwen2_fixed.svg", None, False),
    ("qwen2_supernet.svg", "stochastic_supernet.yaml", False),
    ("qwen2_hybrid.svg", "hybrid_dil.yaml", False),
]


def _load_source(source_dir: str) -> dict:
    config_path = Path(source_dir) / "config.json"
    with open(config_path) as f:
        return json.load(f)


def _build_config(
    source_config: dict,
    source_format: str,
    surgery_path: str | None,
    keep_vision: bool,
) -> dict:
    """Build final Apriel2 config via the conversion pipeline (dry-run)."""
    surgery_configs = None
    if surgery_path is not None:
        with open(EXAMPLES_DIR / surgery_path) as f:
            surgery_configs = [yaml.safe_load(f)]

    _plan, config = build_plan(source_config, surgery_configs, source_format=source_format)

    if not keep_vision:
        config.pop("vision_encoder", None)
        if config.get("architectures") == ["Apriel2ForConditionalGeneration"]:
            config["architectures"] = ["Apriel2ForCausalLM"]

    return config


def _generate_set(
    source_dir: str,
    source_format: str,
    diagrams: list[DiagramSpec],
    outdir: Path,
    label: str,
) -> int:
    """Generate a set of diagrams from one source model. Returns count."""
    source_config = _load_source(source_dir)
    print(f"\n{label}  ({source_dir})")

    for filename, surgery, keep_vision in diagrams:
        config = _build_config(source_config, source_format, surgery, keep_vision)
        out_path = outdir / filename
        svg = generate_diagram(config, output_path=str(out_path))
        ET.fromstring(svg)
        print(f"  {filename:<35s} {len(svg):>7,} chars")

    return len(diagrams)


def main():
    parser = argparse.ArgumentParser(description="Generate Apriel2 architecture diagrams")
    parser.add_argument(
        "--apriel",
        default=DEFAULT_APRIEL,
        help=f"Path to Apriel source checkpoint (default: {DEFAULT_APRIEL})",
    )
    parser.add_argument(
        "--qwen2",
        default=DEFAULT_QWEN2,
        help=f"Path to Qwen2 source checkpoint (default: {DEFAULT_QWEN2})",
    )
    parser.add_argument(
        "--outdir", "-o",
        default=".",
        help="Output directory (default: current directory)",
    )
    args = parser.parse_args()
    outdir = Path(args.outdir)

    total = 0
    total += _generate_set(args.apriel, "llava", APRIEL_DIAGRAMS, outdir, "Apriel-1.6")
    total += _generate_set(args.qwen2, "qwen2", QWEN2_DIAGRAMS, outdir, "Qwen2")

    print(f"\n{total} diagrams written to {outdir}/")


if __name__ == "__main__":
    main()
