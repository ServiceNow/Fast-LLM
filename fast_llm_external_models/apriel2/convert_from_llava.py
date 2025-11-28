"""Convert Llava HF checkpoint to Apriel2 HF format.

This module provides declarative, plan-based conversion from Llava/Pixtral models to Apriel2.

The converter handles:
- Config conversion: Llava config -> Apriel2 config (1-to-1 mapping)
- Weight conversion: Llava state_dict -> Apriel2 state_dict via expression plans

For architecture modifications (adding stochastic mixers, hybridization, etc.),
pass a surgery config to compose the conversion with a surgery plan.
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import torch
import yaml
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor
from tqdm import tqdm

logger = logging.getLogger(__name__)


# =============================================================================
# Config Conversion
# =============================================================================


def convert_config(llava_config: dict) -> dict:
    """Convert Llava config to Apriel2 format.

    This is a pure 1-to-1 mapping - no architecture modifications.
    The resulting config has attention-only decoder matching the source structure.

    Args:
        llava_config: Source Llava/Pixtral config dict.

    Returns:
        Apriel2 config dict with equivalent architecture.
    """
    text_config = llava_config["text_config"]

    # Get token IDs - prefer top-level, fall back to text_config
    bos_token_id = llava_config.get("bos_token_id") or text_config.get("bos_token_id")
    eos_token_id = llava_config.get("eos_token_id") or text_config.get("eos_token_id")
    pad_token_id = llava_config.get("pad_token_id") or text_config.get("pad_token_id")

    # Build decoder config (attention-only, matching source)
    hidden_size = text_config["hidden_size"]
    num_heads = text_config["num_attention_heads"]
    num_kv_heads = text_config["num_key_value_heads"]
    rope_theta = text_config["rope_theta"]

    decoder_config = {
        "type": "fixed",
        "num_blocks": text_config["num_hidden_layers"],
        "block": {
            "mixer": {
                "type": "attention",
                "heads": num_heads,
                "head_groups": num_kv_heads,
                "head_size": hidden_size // num_heads,
                "add_linear_biases": False,
                "rotary": {"type": "default", "theta": rope_theta},
            },
            "mlp": {
                "type": "mlp",
                "intermediate_size": text_config["intermediate_size"],
                "activation": text_config["hidden_act"],
                "gated": True,
                "add_linear_biases": False,
            },
            "normalization": {
                "type": "rms_norm",
                "epsilon": text_config["rms_norm_eps"],
            },
        },
    }

    apriel2_config = {
        "architectures": ["Apriel2ForConditionalGeneration"],
        "model_type": "apriel2",
        "auto_map": {
            "AutoConfig": "configuration_apriel2.Apriel2Config",
            "AutoModel": "modeling_apriel2.Apriel2Model",
            "AutoModelForCausalLM": "modeling_apriel2.Apriel2ForConditionalGeneration",
        },
        "hidden_size": hidden_size,
        "vocab_size": text_config["vocab_size"],
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
        "tie_word_embeddings": text_config["tie_word_embeddings"],
        "use_cache": text_config.get("use_cache", True),
        "image_token_index": llava_config["image_token_index"],
        "decoder": decoder_config,
        "embeddings": {
            "max_position_embeddings": text_config["max_position_embeddings"],
        },
        "head": {
            "normalization": {
                "type": "rms_norm",
                "epsilon": text_config["rms_norm_eps"],
            },
        },
        "vision_encoder": _convert_vision_config(llava_config),
    }

    return apriel2_config


def _convert_vision_config(llava_config: dict) -> dict:
    """Convert Llava vision_config to Apriel2 vision_encoder format."""
    vision_config = llava_config["vision_config"]
    text_config = llava_config["text_config"]

    hidden_size = vision_config["hidden_size"]
    num_heads = vision_config["num_attention_heads"]
    num_layers = vision_config["num_hidden_layers"]
    intermediate_size = vision_config["intermediate_size"]
    rope_theta = vision_config["rope_theta"]
    patch_size = vision_config["patch_size"]
    num_channels = vision_config["num_channels"]

    return {
        "hidden_size": hidden_size,
        "patch_convolution": {
            "patch_height": patch_size,
            "patch_width": patch_size,
            "input_channels": num_channels,
            "normalization": {"type": "rms_norm", "epsilon": 1e-5},
        },
        "encoder": {
            "type": "fixed",
            "num_blocks": num_layers,
            "block": {
                "mixer": {
                    "type": "attention",
                    "heads": num_heads,
                    "head_groups": num_heads,
                    "head_size": hidden_size // num_heads,
                    "add_linear_biases": False,
                    "causal": False,
                    "rotary": {"type": "default_2d", "theta": rope_theta},
                },
                "mlp": {
                    "type": "mlp",
                    "intermediate_size": intermediate_size,
                    "activation": vision_config["hidden_act"],
                    "gated": True,
                    "add_linear_biases": False,
                },
                "normalization": {"type": "rms_norm", "epsilon": 1e-5},
            },
        },
        "adapter": {
            "type": "mlp",
            "intermediate_size": text_config["hidden_size"],
            "activation": llava_config["projector_hidden_act"],
            "add_linear_biases": True,
        },
    }


# =============================================================================
# Plan-Based Conversion
# =============================================================================


def convert(
    llava_config: dict,
    source_files: list[Path],
    output_file: Path,
    surgery_config: dict | None = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> dict:
    """Convert Llava checkpoint to Apriel2 using plan-based streaming.

    This conversion:
    1. Uses declarative plans that can be inspected and composed
    2. Loads weights on-demand and releases them when done (memory efficient)
    3. Supports surgery (architecture modification) via plan composition

    Args:
        llava_config: Source Llava config dict.
        source_files: List of source safetensor files.
        output_file: Output safetensor file path.
        surgery_config: Optional target config for surgery (architecture modification).
        device: Device for computation (default: cpu).
        dtype: Data type for weights (default: float32).

    Returns:
        Final Apriel2 config dict.
    """
    from .expr_plan import (
        StreamingExecutor,
        compose,
        plan_llava_to_apriel2,
        plan_surgery,
    )

    # Build conversion plan (Llava -> Apriel2)
    conversion_plan = plan_llava_to_apriel2(llava_config)
    logger.info(f"Built conversion plan: {conversion_plan.summary()['num_targets']} targets")

    # Get intermediate Apriel2 config
    intermediate_config = convert_config(llava_config)

    # Apply surgery if requested
    if surgery_config:
        surgery_plan = plan_surgery(intermediate_config, surgery_config)
        logger.info(f"Built surgery plan: {surgery_plan.summary()['num_targets']} targets")

        # Compose: Llava -> Apriel2 -> Modified Apriel2
        full_plan = compose(conversion_plan, surgery_plan)
        logger.info(f"Composed plan: {full_plan.summary()['num_targets']} targets")
        final_config = surgery_config
    else:
        full_plan = conversion_plan
        final_config = intermediate_config

    # Build weight loader that reads from safetensor files
    source_handles: dict[Path, any] = {}

    def load_source(key: str) -> Tensor:
        """Load a source tensor from safetensor files."""
        for source_file in source_files:
            if source_file not in source_handles:
                source_handles[source_file] = safe_open(
                    source_file, framework="pt", device=device
                )
            handle = source_handles[source_file]
            if key in handle.keys():
                return handle.get_tensor(key)
        raise KeyError(f"Source key not found in any file: {key}")

    # Execute with streaming
    executor = StreamingExecutor(full_plan, load_source, device, dtype)

    # Collect results
    result_weights = {}
    for target_key, tensor in tqdm(executor.execute(), desc="Converting", total=len(full_plan)):
        result_weights[target_key] = tensor

    # Save output
    logger.info(f"Saving {len(result_weights)} weights to {output_file}")
    save_file(result_weights, output_file)

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
    parser = argparse.ArgumentParser(
        description="Convert Llava HF checkpoint to Apriel2 HF format"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to input Llava checkpoint directory or HuggingFace model ID",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Path to output Apriel2 checkpoint directory",
    )
    parser.add_argument(
        "--surgery",
        "-s",
        type=Path,
        help="Path to YAML config for post-conversion surgery (optional)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
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

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    logger.info(f"Loading source config from {config_file}")
    with open(config_file) as f:
        llava_config = json.load(f)

    # Find model files (safetensors only)
    safetensor_files = sorted(input_dir.glob("*.safetensors"))
    if not safetensor_files:
        raise ValueError(
            f"No safetensor files found in {input_dir}. "
            "Plan-based conversion requires safetensor files."
        )

    # Load surgery config if specified
    surgery_config = None
    if args.surgery:
        logger.info(f"Loading surgery config from {args.surgery}")
        with open(args.surgery) as f:
            surgery_config = yaml.safe_load(f)

    # Convert using plan-based approach
    output_weights_file = args.output_dir / "model.safetensors"
    apriel2_config = convert(
        llava_config,
        safetensor_files,
        output_weights_file,
        surgery_config=surgery_config,
    )

    # Save config
    output_config_file = args.output_dir / "config.json"
    logger.info(f"Saving config to {output_config_file}")
    with open(output_config_file, "w") as f:
        json.dump(apriel2_config, f, indent=2)

    # Copy tokenizer files
    copy_tokenizer_files(input_dir, args.output_dir)

    # Copy model files
    copy_model_files(args.output_dir)

    logger.info(f"Conversion complete! Output saved to {args.output_dir}")


if __name__ == "__main__":
    main()
