"""Thin a stochastic supernet to retain only mixers needed for placement configs.

A supernet contains multiple mixers per layer (e.g., attention, sliding_window,
gdn, kda). Layer placement studies select one mixer per layer to create subnetworks
under budget constraints. This script takes a full supernet and a set of placement
configurations, then produces a slimmed supernet that retains only the mixers
actually used across those placements.

Usage:
    python thin_supernet.py ~/.cache/huggingface/apriel2-0.5b-dev output/ \\
        -p examples/placements/budget_attention_heavy.yaml

The output is a valid Apriel2 supernet checkpoint with fewer mixers per layer,
reducing model size while preserving all placement configurations of interest.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from fast_llm_external_models.apriel2.conversion import (
    DEFAULT_MAX_SHARD_SIZE,
    SafetensorLoader,
    ShardedSafetensorWriter,
    StreamingExecutor,
    compose_configs,
    plan_surgery,
    strip_init_fields,
)
from fast_llm_external_models.apriel2.convert import copy_model_files, copy_tokenizer_files, resolve_input

# Allow running as script or module
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))


logger = logging.getLogger(__name__)


# =============================================================================
# Placement Config Loading
# =============================================================================


def load_placement_config(path: Path) -> list[list[str]]:
    """Load a placement config from YAML.

    Expected format:
        placements:
          - [attention, gdn, attention, kda, ...]  # one mixer per layer
          - [sliding_window, sliding_window, ...]

    Or alternatively:
        layers:
          - attention
          - gdn
          - attention
          ...

    Returns:
        List of placements, each a list of mixer names (one per layer).
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    if "placements" in data:
        return data["placements"]
    if "layers" in data:
        return [data["layers"]]
    raise ValueError(f"Placement config must have 'placements' or 'layers' key: {path}")


def load_all_placements(paths: list[Path]) -> list[list[str]]:
    """Load and merge placements from multiple config files."""
    all_placements: list[list[str]] = []
    for path in paths:
        placements = load_placement_config(path)
        all_placements.extend(placements)
    return all_placements


def compute_required_mixers_per_layer(placements: list[list[str]], num_layers: int) -> list[set[str]]:
    """Compute the union of mixers needed per layer across all placements.

    Args:
        placements: List of placements, each a list of mixer names per layer.
        num_layers: Number of layers in the model.

    Returns:
        List of sets: required_mixers[i] = set of mixer names needed at layer i.
    """
    required: list[set[str]] = [set() for _ in range(num_layers)]
    for placement in placements:
        for layer_idx, mixer_name in enumerate(placement):
            if layer_idx < num_layers:
                required[layer_idx].add(mixer_name)
    return required


# =============================================================================
# Surgery Config Building
# =============================================================================


def _build_mixer_spec(source_mixers: dict[str, dict], mixer_names: set[str], main_mixer_name: str) -> dict[str, Any]:
    """Build a mixer surgery spec with only the specified mixers, each with init: transfer."""
    mixers_spec: dict[str, Any] = {}
    for name in mixer_names:
        if name not in source_mixers:
            logger.warning(f"Mixer '{name}' not in source supernet, skipping")
            continue
        sub_config = copy.deepcopy(source_mixers[name])
        sub_config["init"] = "transfer"
        mixers_spec[name] = sub_config
    return {
        "type": "stochastic",
        "main_mixer_name": main_mixer_name if main_mixer_name in mixers_spec else next(iter(mixers_spec), "attention"),
        "mixers": mixers_spec,
    }


def build_thin_surgery_config(
    source_config: dict,
    required_mixers_per_layer: list[set[str]],
) -> dict:
    """Build a surgery config that thins the supernet to only required mixers.

    Args:
        source_config: Full supernet config.
        required_mixers_per_layer: required_mixers[i] = set of mixer names for layer i.

    Returns:
        Surgery config (partial) to pass to compose_configs.
    """
    decoder = source_config.get("decoder", {})
    num_blocks = decoder.get("num_blocks", 0)
    decoder_type = decoder.get("type", "fixed")
    source_block = decoder.get("block", {})
    source_mixer = source_block.get("mixer", {})

    if decoder_type != "fixed":
        raise ValueError(f"Thin supernet currently supports only fixed decoder, got: {decoder_type}")

    source_mixers = source_mixer.get("mixers", {})
    if not source_mixers:
        raise ValueError("Source config has no stochastic mixers - not a supernet")

    main_mixer_name = source_mixer.get("main_mixer_name", "attention")

    # Check if all layers need the same mixers
    first_set = required_mixers_per_layer[0] if required_mixers_per_layer else set()
    all_same = all(required_mixers_per_layer[i] == first_set for i in range(1, len(required_mixers_per_layer)))

    if all_same:
        # Fixed decoder: single block with thinned mixers (stochastic)
        mixer_spec = _build_mixer_spec(source_mixers, first_set, main_mixer_name)
        return {
            "decoder": {
                "type": "fixed",
                "num_blocks": num_blocks,
                "block": {
                    "mixer": mixer_spec,
                    "mlp": {"init": "transfer"},
                    "normalization": {"init": "transfer"},
                },
            }
        }
    else:
        # Pattern decoder: one block type per layer with its own mixer subset
        pattern = [f"layer_{i}" for i in range(num_blocks)]
        blocks = {}
        for i in range(num_blocks):
            mixer_spec = _build_mixer_spec(source_mixers, required_mixers_per_layer[i], main_mixer_name)
            blocks[f"layer_{i}"] = {
                "mixer": mixer_spec,
                "mlp": {"init": "transfer"},
                "normalization": {"init": "transfer"},
            }
        return {
            "decoder": {
                "type": "pattern",
                "num_blocks": num_blocks,
                "pattern": pattern,
                "blocks": blocks,
            }
        }


# =============================================================================
# Main Conversion
# =============================================================================


def thin_supernet(
    input_dir: Path,
    output_dir: Path,
    placement_configs: list[Path],
    max_shard_size: int = DEFAULT_MAX_SHARD_SIZE,
    device: str = "cpu",
    dry_run: bool = False,
    verbose: bool = False,
) -> dict:
    """Thin a supernet checkpoint to retain only mixers needed for placements.

    Args:
        input_dir: Path to supernet checkpoint directory.
        output_dir: Path to output directory.
        placement_configs: List of paths to placement YAML configs.
        max_shard_size: Max shard size in bytes.
        device: Device for loading tensors.
        dry_run: If True, only build and show plan, don't execute.
        verbose: Enable verbose logging.

    Returns:
        Final config dict (for testing).
    """
    # Load source config
    config_path = input_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        source_config = json.load(f)

    # Validate source is a stochastic supernet
    decoder = source_config.get("decoder", {})
    block = decoder.get("block", {})
    mixer = block.get("mixer", {})
    if mixer.get("type") != "stochastic":
        raise ValueError(
            f"Source is not a stochastic supernet (mixer.type={mixer.get('type')}). "
            "Thin supernet requires a model with stochastic mixers."
        )

    num_layers = decoder.get("num_blocks", 0)
    if num_layers == 0:
        raise ValueError("Source config has num_blocks=0")

    # Load placements and compute required mixers
    all_placements = load_all_placements(placement_configs)
    if not all_placements:
        raise ValueError("No placements loaded from config files")

    required_mixers_per_layer = compute_required_mixers_per_layer(all_placements, num_layers)

    # Log summary
    logger.info(
        f"Placements: {len(all_placements)}, layers: {num_layers}. "
        f"Mixers per layer: {[len(s) for s in required_mixers_per_layer]}"
    )

    # Build surgery config and transition
    surgery_config = build_thin_surgery_config(source_config, required_mixers_per_layer)
    target_config = compose_configs(source_config, surgery_config)

    # Build plan
    plan = plan_surgery(source_config, target_config)
    logger.info(f"Built thin-supernet plan: {plan.summary()['num_targets']} targets")

    if dry_run:
        print("\n" + "=" * 60)
        print("THIN SUPERNET PLAN (dry-run)")
        print("=" * 60)
        print(plan.render_tree(collapse_layers=True))
        print("=" * 60)
        print(f"Summary: {plan.summary()}")
        return strip_init_fields(target_config)

    # Execute
    output_dir.mkdir(parents=True, exist_ok=True)
    safetensor_files = sorted(input_dir.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensor files in {input_dir}")

    with SafetensorLoader(safetensor_files, device) as loader:
        executor = StreamingExecutor(plan, loader)
        with ShardedSafetensorWriter(output_dir, max_shard_size=max_shard_size) as writer:
            for target_key, tensor in tqdm(executor.execute(seed=0), desc="Thinning", total=len(plan)):
                writer.add(target_key, tensor)

    # Save config
    final_config = strip_init_fields(target_config)
    with open(output_dir / "config.json", "w") as f:
        json.dump(final_config, f, indent=2)

    # Copy tokenizer and model files
    copy_tokenizer_files(input_dir, output_dir)
    copy_model_files(output_dir)

    logger.info(f"Thin supernet complete. Output: {output_dir}")
    return final_config


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Thin a stochastic supernet to retain only mixers needed for placement configs"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to supernet checkpoint or HuggingFace model ID",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Path to output directory",
    )
    parser.add_argument(
        "-p",
        "--placement",
        type=Path,
        action="append",
        dest="placements",
        metavar="YAML",
        required=True,
        help="Path to placement config YAML. Can specify multiple times.",
    )
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Build and show plan without executing",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    parser.add_argument(
        "--max-shard-size",
        type=int,
        default=DEFAULT_MAX_SHARD_SIZE,
        help=f"Max shard size in bytes (default: {DEFAULT_MAX_SHARD_SIZE // (1024**3)}GB)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    input_dir = resolve_input(args.input)
    thin_supernet(
        input_dir=input_dir,
        output_dir=args.output_dir,
        placement_configs=args.placements,
        max_shard_size=args.max_shard_size,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
