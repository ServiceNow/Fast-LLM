"""Convert HuggingFace checkpoints to Apriel2 HF format.

This module provides declarative, plan-based conversion from various source formats to Apriel2.

The converter handles:
- Config conversion: Source config -> Apriel2 config
- Weight conversion: Source state_dict -> Apriel2 state_dict via expression plans

For architecture modifications (adding stochastic mixers, hybridization, etc.),
pass one or more surgery configs. Multiple surgeries are chained in order:

    convert input output -s surgery1.yaml -s surgery2.yaml -s surgery3.yaml

This produces: Source -> Apriel2 -> surgery1 -> surgery2 -> surgery3

Supported source formats:
- llava: Llava/Pixtral models
- qwen2: Qwen2/Qwen2.5 models
- apriel2: Apriel2 models (surgery-only mode - no conversion, just apply surgeries)
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Callable

import yaml
from tqdm import tqdm

# Import source-specific converters
from fast_llm_external_models.apriel2.conversion import (
    DEFAULT_MAX_SHARD_SIZE,
    ExprPlan,
    SafetensorLoader,
    ShardedSafetensorWriter,
    StreamingExecutor,
    compose,
    compose_configs,
)
from fast_llm_external_models.apriel2.conversion import llava as llava_converter
from fast_llm_external_models.apriel2.conversion import (
    plan_surgery,
)
from fast_llm_external_models.apriel2.conversion import qwen2 as qwen2_converter
from fast_llm_external_models.apriel2.conversion import (
    strip_init_fields,
)

# Allow running as script or module
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))


logger = logging.getLogger(__name__)


# =============================================================================
# Source Format Registry
# =============================================================================


def _identity_config(config: dict) -> dict:
    """Identity config converter for Apriel2 source."""
    return config


def _identity_plan(config: dict) -> ExprPlan:
    """Identity plan builder for Apriel2 source (surgery-only mode).

    Creates a plan that references all keys as-is, which will be composed
    with surgery plans to perform modifications.
    """
    return plan_surgery(config, config)


# Registry of supported source formats
# Each entry maps format name to (config_converter, plan_builder)
SOURCE_FORMATS: dict[str, tuple[Callable[[dict], dict], Callable[[dict], ExprPlan]]] = {
    "llava": (llava_converter.convert_config, llava_converter.plan_llava_to_apriel2),
    "qwen2": (qwen2_converter.convert_config, qwen2_converter.plan_qwen2_to_apriel2),
    "apriel2": (_identity_config, _identity_plan),
}


def detect_source_format(config: dict) -> str | None:
    """Auto-detect source format from config.

    Returns format name if detected, None otherwise.
    """
    model_type = config.get("model_type", "")

    # Llava/Pixtral detection
    if model_type in ("llava", "pixtral") or "text_config" in config:
        return "llava"

    # Qwen2/Qwen2.5 detection
    if model_type == "qwen2":
        return "qwen2"

    # Apriel2 detection - check for Apriel2-specific structure
    if model_type in ("apriel2", "apriel2_text") or "decoder" in config:
        return "apriel2"

    return None


def get_converter(source_format: str) -> tuple[Callable[[dict], dict], Callable[[dict], ExprPlan]]:
    """Get config converter and plan builder for a source format."""
    if source_format not in SOURCE_FORMATS:
        available = ", ".join(sorted(SOURCE_FORMATS.keys()))
        raise ValueError(f"Unknown source format: {source_format}. Available: {available}")
    return SOURCE_FORMATS[source_format]


# =============================================================================
# Plan-Based Conversion
# =============================================================================


def build_plan(
    source_config: dict,
    surgery_configs: list[dict] | None = None,
    source_format: str | None = None,
) -> tuple[ExprPlan, dict]:
    """Build conversion plan without executing.

    Args:
        source_config: Source model config dict.
        surgery_configs: Optional list of surgery configs to chain. Each surgery is
            applied in order: Source -> Apriel2 -> surgery[0] -> surgery[1] -> ...
        source_format: Source format name (e.g., "llava"). Auto-detected if not specified.

    Returns:
        Tuple of (plan, final_config).
    """
    if source_format is None:
        source_format = detect_source_format(source_config)
    if source_format is None:
        available = ", ".join(sorted(SOURCE_FORMATS.keys()))
        raise ValueError(f"Unknown source format. Available: {available}")

    config_converter, plan_builder = get_converter(source_format)

    # Build conversion plan (Source -> Apriel2)
    current_plan = plan_builder(source_config)
    logger.info(f"Built conversion plan: {current_plan.summary()['num_targets']} targets")

    # Get intermediate Apriel2 config
    current_config = config_converter(source_config)

    # Apply surgery chain if requested
    if surgery_configs:
        for i, surgery_config in enumerate(surgery_configs, 1):
            # S × P → T: compose state with surgery to get transition spec
            target_config = compose_configs(current_config, surgery_config)

            # S × T → Plan: build plan from source state and transition spec
            surgery_plan = plan_surgery(current_config, target_config)
            logger.info(
                f"Built surgery plan [{i}/{len(surgery_configs)}]: {surgery_plan.summary()['num_targets']} targets"
            )

            # Compose plans
            current_plan = compose(current_plan, surgery_plan)
            logger.info(f"Composed plan [{i}/{len(surgery_configs)}]: {current_plan.summary()['num_targets']} targets")

            # T → S: strip init for next iteration (init is consumed by plan_surgery)
            current_config = strip_init_fields(target_config)

    return current_plan, current_config


def print_plan(plan: ExprPlan, title: str = "CONVERSION PLAN", show_summary: bool = False) -> None:
    """Print a conversion plan tree."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(plan.render_tree(collapse_layers=True))
    print("=" * 60)
    if show_summary:
        summary = plan.summary()
        print(f"\nSummary: {summary['num_targets']} targets, {summary['num_source_refs']} source refs")


def convert(
    source_config: dict,
    source_files: list[Path],
    output_dir: Path,
    surgery_configs: list[dict] | None = None,
    source_format: str | None = None,
    device: str = "cpu",
    max_shard_size: int = DEFAULT_MAX_SHARD_SIZE,
    seed: int = 0,
    show_plan: bool = False,
) -> dict:
    """Convert checkpoint to Apriel2 using plan-based streaming.

    This conversion:
    1. Uses declarative plans that can be inspected and composed
    2. Loads weights on-demand and releases them when done (memory efficient)
    3. Writes output in shards to bound memory usage
    4. Supports surgery chains (multiple architecture modifications) via plan composition

    Args:
        source_config: Source model config dict.
        source_files: List of source safetensor files.
        output_dir: Output directory for safetensor files.
        surgery_configs: Optional list of surgery configs to chain.
        source_format: Source format name (e.g., "llava"). Auto-detected if not specified.
        device: Device to load source tensors onto (default: cpu).
        max_shard_size: Maximum shard size in bytes (default: 5GB).
        seed: Random seed for deterministic initialization (default: 0).
        show_plan: If True, print the plan tree before converting.

    Returns:
        Final Apriel2 config dict.
    """
    # Build the plan
    full_plan, final_config = build_plan(source_config, surgery_configs, source_format)

    if show_plan:
        print_plan(full_plan)

    # Execute with streaming I/O
    with SafetensorLoader(source_files, device) as loader:
        executor = StreamingExecutor(full_plan, loader)

        with ShardedSafetensorWriter(output_dir, max_shard_size=max_shard_size) as writer:
            for target_key, tensor in tqdm(executor.execute(seed), desc="Converting", total=len(full_plan)):
                writer.add(target_key, tensor)

    return final_config


# =============================================================================
# File Operations
# =============================================================================


def copy_tokenizer_files(input_dir: Path, output_dir: Path) -> None:
    """Copy tokenizer files from input to output directory."""
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.model",
    ]

    for filename in tokenizer_files:
        src = input_dir / filename
        if src.exists():
            dst = output_dir / filename
            shutil.copy2(src, dst)
            logger.info(f"Copied {filename}")


def copy_model_files(output_dir: Path) -> None:
    """Copy Apriel2 model files to output directory."""
    apriel2_dir = Path(__file__).parent

    files_to_copy = [
        "configuration_apriel2.py",
        "modeling_apriel2.py",
        "cache.py",
    ]

    for filename in files_to_copy:
        src = apriel2_dir / filename
        if src.exists():
            dst = output_dir / filename
            shutil.copy2(src, dst)
            logger.info(f"Copied {filename}")


def resolve_input(input_path: str) -> Path:
    """Resolve input path - either local directory or HuggingFace model ID."""
    from huggingface_hub import snapshot_download

    path = Path(input_path)
    if path.exists():
        return path

    # Try as HuggingFace model ID
    logger.info(f"Input not found locally, downloading from HuggingFace: {input_path}")
    cache_dir = snapshot_download(
        input_path,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
    )
    return Path(cache_dir)


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Convert HuggingFace checkpoint to Apriel2 HF format")
    parser.add_argument(
        "input",
        type=str,
        help="Path to input checkpoint directory or HuggingFace model ID",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Path to output Apriel2 checkpoint directory",
    )
    parser.add_argument(
        "--source-format",
        "-f",
        type=str,
        choices=list(SOURCE_FORMATS.keys()),
        help="Source model format (auto-detected if not specified)",
    )
    parser.add_argument(
        "--surgery",
        "-s",
        type=Path,
        action="append",
        dest="surgeries",
        metavar="YAML",
        help="Path to YAML surgery config. Can be specified multiple times to chain surgeries.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Build and show the conversion plan without executing",
    )
    parser.add_argument(
        "--show-plan",
        action="store_true",
        help="Print the conversion plan tree before executing",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for deterministic initialization (default: 0)",
    )
    parser.add_argument(
        "--max-shard-size",
        type=int,
        default=DEFAULT_MAX_SHARD_SIZE,
        help=f"Maximum shard size in bytes (default: {DEFAULT_MAX_SHARD_SIZE // (1024**3)}GB)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Resolve input (local or HuggingFace)
    input_dir = resolve_input(args.input)

    config_file = input_dir / "config.json"
    if not config_file.exists():
        raise ValueError(f"Config file not found: {config_file}")

    # Load config
    logger.info(f"Loading source config from {config_file}")
    with open(config_file) as f:
        source_config = json.load(f)

    # Load surgery configs if specified
    surgery_configs = None
    if args.surgeries:
        surgery_configs = []
        for surgery_path in args.surgeries:
            logger.info(f"Loading surgery config from {surgery_path}")
            with open(surgery_path) as f:
                surgery_configs.append(yaml.safe_load(f))
        logger.info(f"Loaded {len(surgery_configs)} surgery config(s)")

    # Dry-run mode: just build and show the plan, don't execute
    if args.dry_run:
        plan, _ = build_plan(source_config, surgery_configs, args.source_format)
        print_plan(plan, title="CONVERSION PLAN (dry-run)", show_summary=True)
        print("Dry-run complete. No files written.")
        return

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find model files (safetensors only)
    safetensor_files = sorted(input_dir.glob("*.safetensors"))
    if not safetensor_files:
        raise ValueError(
            f"No safetensor files found in {input_dir}. " "Plan-based conversion requires safetensor files."
        )

    # Convert using plan-based approach with streaming sharded output
    apriel2_config = convert(
        source_config,
        safetensor_files,
        args.output_dir,
        surgery_configs=surgery_configs,
        source_format=args.source_format,
        max_shard_size=args.max_shard_size,
        seed=args.seed,
        show_plan=args.show_plan or args.verbose,
    )

    # Save config (build_plan returns S which has no init, but strip defensively)
    output_config_file = args.output_dir / "config.json"
    logger.info(f"Saving config to {output_config_file}")
    with open(output_config_file, "w") as f:
        json.dump(strip_init_fields(apriel2_config), f, indent=2)

    # Copy tokenizer files
    copy_tokenizer_files(input_dir, args.output_dir)

    # Copy model files
    copy_model_files(args.output_dir)

    logger.info(f"Conversion complete! Output saved to {args.output_dir}")


if __name__ == "__main__":
    main()
