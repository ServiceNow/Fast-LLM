"""Convert Llava HF checkpoint to Apriel2 HF format.

This module provides pure format conversion from Llava/Pixtral models to Apriel2.
It does NOT modify the architecture - use surgery.py for that.

The converter handles:
- Config conversion: Llava config -> Apriel2 config (1-to-1 mapping)
- Weight conversion: Llava state_dict -> Apriel2 state_dict (pure name mapping)

For architecture modifications (adding stochastic mixers, changing patterns, etc.),
use surgery.py after conversion.
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
# Weight Conversion
# =============================================================================

# Weight name mappings (Llava -> Apriel2)
_STATIC_WEIGHT_MAP = {
    # Embeddings
    "language_model.model.embed_tokens.weight": "model.embed_tokens.weight",
    # Final norm and LM head
    "language_model.model.norm.weight": "model.norm.weight",
    "language_model.lm_head.weight": "lm_head.weight",
    # Vision tower
    "vision_tower.patch_conv.weight": "model.vision_encoder.patch_convolution.conv.weight",
    "vision_tower.ln_pre.weight": "model.vision_encoder.patch_convolution.norm.weight",
    # Vision adapter
    "multi_modal_projector.linear_1.weight": "model.vision_encoder.adapter.linear_1.weight",
    "multi_modal_projector.linear_1.bias": "model.vision_encoder.adapter.linear_1.bias",
    "multi_modal_projector.linear_2.weight": "model.vision_encoder.adapter.linear_2.weight",
    "multi_modal_projector.linear_2.bias": "model.vision_encoder.adapter.linear_2.bias",
}

# Decoder layer component mappings
_DECODER_LAYER_MAP = {
    "self_attn.q_proj.weight": "mixer.self_attn.q_proj.weight",
    "self_attn.k_proj.weight": "mixer.self_attn.k_proj.weight",
    "self_attn.v_proj.weight": "mixer.self_attn.v_proj.weight",
    "self_attn.o_proj.weight": "mixer.self_attn.o_proj.weight",
    "mlp.gate_proj.weight": "mlp.gate_proj.weight",
    "mlp.up_proj.weight": "mlp.up_proj.weight",
    "mlp.down_proj.weight": "mlp.down_proj.weight",
    "input_layernorm.weight": "input_layernorm.weight",
    "post_attention_layernorm.weight": "post_attention_layernorm.weight",
}

# Vision encoder layer component mappings
_VISION_LAYER_MAP = {
    "attention.q_proj.weight": "mixer.self_attn.q_proj.weight",
    "attention.k_proj.weight": "mixer.self_attn.k_proj.weight",
    "attention.v_proj.weight": "mixer.self_attn.v_proj.weight",
    "attention.o_proj.weight": "mixer.self_attn.o_proj.weight",
    "feed_forward.gate_proj.weight": "mlp.gate_proj.weight",
    "feed_forward.up_proj.weight": "mlp.up_proj.weight",
    "feed_forward.down_proj.weight": "mlp.down_proj.weight",
    "attention_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
}


def map_weight_name(llava_name: str) -> str | None:
    """Map a single Llava weight name to Apriel2 format.

    Args:
        llava_name: Llava weight name.

    Returns:
        Apriel2 weight name, or None if unmapped.
    """
    # Check static mappings
    if llava_name in _STATIC_WEIGHT_MAP:
        return _STATIC_WEIGHT_MAP[llava_name]

    # Check decoder layer patterns
    if llava_name.startswith("language_model.model.layers."):
        parts = llava_name.split(".")
        layer_idx = int(parts[3])
        rest = ".".join(parts[4:])
        if rest in _DECODER_LAYER_MAP:
            return f"model.decoder.blocks.{layer_idx}.{_DECODER_LAYER_MAP[rest]}"

    # Check vision layer patterns
    if llava_name.startswith("vision_tower.transformer.layers."):
        parts = llava_name.split(".")
        layer_idx = int(parts[3])
        rest = ".".join(parts[4:])
        if rest in _VISION_LAYER_MAP:
            return f"model.vision_encoder.encoder.blocks.{layer_idx}.{_VISION_LAYER_MAP[rest]}"

    return None


def convert_weights(llava_weights: dict[str, Tensor]) -> dict[str, Tensor]:
    """Convert Llava weights to Apriel2 format.

    This is a pure name mapping - no weight transformations.

    Args:
        llava_weights: Source Llava state_dict.

    Returns:
        Apriel2 state_dict.
    """
    apriel2_weights = {}
    unmapped = []

    for llava_name, tensor in llava_weights.items():
        apriel2_name = map_weight_name(llava_name)
        if apriel2_name:
            apriel2_weights[apriel2_name] = tensor
        else:
            unmapped.append(llava_name)

    if unmapped:
        logger.warning(f"Unmapped weights: {unmapped[:5]}{'...' if len(unmapped) > 5 else ''}")

    return apriel2_weights


def convert_weights_from_files(
    input_dir: Path,
    output_dir: Path,
) -> None:
    """Convert weights from files on disk.

    Args:
        input_dir: Directory containing Llava checkpoint.
        output_dir: Directory to write Apriel2 checkpoint.
    """
    # Find model files
    safetensor_files = sorted(input_dir.glob("*.safetensors"))
    if not safetensor_files:
        bin_files = sorted(input_dir.glob("pytorch_model*.bin"))
        if not bin_files:
            raise ValueError(f"No model files found in {input_dir}")
        use_safetensors = False
        model_files = bin_files
    else:
        use_safetensors = True
        model_files = safetensor_files

    # Load and convert all weights
    all_weights = {}
    for model_file in tqdm(model_files, desc="Loading weights"):
        if use_safetensors:
            with safe_open(model_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    all_weights[key] = f.get_tensor(key)
        else:
            state_dict = torch.load(model_file, map_location="cpu", weights_only=True)
            all_weights.update(state_dict)

    # Convert
    apriel2_weights = convert_weights(all_weights)

    # Save
    output_file = output_dir / "model.safetensors"
    logger.info(f"Saving {len(apriel2_weights)} weights to {output_file}")
    save_file(apriel2_weights, output_file)


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

    # Load and convert config
    logger.info(f"Loading source config from {config_file}")
    with open(config_file) as f:
        llava_config = json.load(f)

    apriel2_config = convert_config(llava_config)

    # Convert weights (to in-memory state dict)
    safetensor_files = sorted(input_dir.glob("*.safetensors"))
    bin_files = sorted(input_dir.glob("pytorch_model*.bin"))

    if safetensor_files:
        model_files = safetensor_files
        use_safetensors = True
    elif bin_files:
        model_files = bin_files
        use_safetensors = False
    else:
        raise ValueError(f"No model files found in {input_dir}")

    all_weights = {}
    for model_file in tqdm(model_files, desc="Loading weights"):
        if use_safetensors:
            with safe_open(model_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    all_weights[key] = f.get_tensor(key)
        else:
            state_dict = torch.load(model_file, map_location="cpu", weights_only=True)
            all_weights.update(state_dict)

    apriel2_weights = convert_weights(all_weights)

    # Apply surgery if requested
    if args.surgery:
        from .surgery import surgery

        logger.info(f"Loading surgery config from {args.surgery}")
        with open(args.surgery) as f:
            surgery_config = yaml.safe_load(f)

        # The surgery config specifies the target architecture
        target_config = surgery_config
        apriel2_weights = surgery(apriel2_config, apriel2_weights, target_config)
        apriel2_config = target_config

    # Save config
    output_config_file = args.output_dir / "config.json"
    logger.info(f"Saving config to {output_config_file}")
    with open(output_config_file, "w") as f:
        json.dump(apriel2_config, f, indent=2)

    # Save weights
    output_weights_file = args.output_dir / "model.safetensors"
    logger.info(f"Saving {len(apriel2_weights)} weights to {output_weights_file}")
    save_file(apriel2_weights, output_weights_file)

    # Copy tokenizer files
    copy_tokenizer_files(input_dir, args.output_dir)

    # Copy model files
    copy_model_files(args.output_dir)

    logger.info(f"Conversion complete! Output saved to {args.output_dir}")


if __name__ == "__main__":
    main()
