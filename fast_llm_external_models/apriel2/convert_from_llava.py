"""Convert Llava HF checkpoint to Apriel2 HF format.

Supports conversion with customizable target decoder structure via YAML config.
Each component can specify `init: transfer` (convert from source) or `init: random`.
"""

import argparse
import copy
import json
import logging
import shutil
from pathlib import Path
from typing import Callable

import torch
import yaml
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

logger = logging.getLogger(__name__)


# =============================================================================
# Weight Converter Registry
# =============================================================================

# Registry: (source_type, target_type) -> converter function
# Converter signature: (source_weights: dict, source_config: dict, target_config: dict) -> dict
_WEIGHT_CONVERTERS: dict[tuple[str, str], Callable] = {}


def register_converter(source_type: str, target_type: str):
    """Decorator to register a weight converter for a (source, target) type pair."""

    def decorator(fn: Callable):
        _WEIGHT_CONVERTERS[(source_type, target_type)] = fn
        return fn

    return decorator


def get_converter(source_type: str, target_type: str) -> Callable:
    """Get converter for (source, target) pair. Returns identity if same type."""
    if source_type == target_type:
        return _identity_converter

    key = (source_type, target_type)
    if key not in _WEIGHT_CONVERTERS:
        raise ValueError(
            f"No converter registered for {source_type} -> {target_type}. "
            f"Use 'init: random' or register a converter."
        )
    return _WEIGHT_CONVERTERS[key]


def _identity_converter(
    source_weights: dict, source_config: dict, target_config: dict
) -> dict:
    """Identity converter - just return source weights."""
    return source_weights


# =============================================================================
# Built-in Converters
# =============================================================================


@register_converter("attention", "sliding_window")
def _attention_to_sliding_window(
    source_weights: dict, source_config: dict, target_config: dict
) -> dict:
    """Attention to sliding window - same architecture, just copy weights."""
    return source_weights


@register_converter("attention", "local_attention")
def _attention_to_local(
    source_weights: dict, source_config: dict, target_config: dict
) -> dict:
    """Attention to local attention - same weights work."""
    return source_weights


# Placeholder for future converters
# @register_converter("attention", "gdn")
# def _attention_to_gdn(source_weights, source_config, target_config):
#     """Convert attention to GDN."""
#     # Implementation would go here
#     pass


# =============================================================================
# Config Conversion
# =============================================================================


def extract_source_mixer_config(llava_config: dict) -> dict:
    """Extract the source mixer config from Llava config."""
    text_config = llava_config["text_config"]
    hidden_size = text_config["hidden_size"]
    num_heads = text_config["num_attention_heads"]
    num_kv_heads = text_config["num_key_value_heads"]
    rope_theta = text_config["rope_theta"]

    return {
        "type": "attention",
        "heads": num_heads,
        "head_groups": num_kv_heads,
        "head_size": hidden_size // num_heads,
        "add_linear_biases": False,
        "rotary": {"type": "default", "theta": rope_theta},
    }


def extract_source_mlp_config(llava_config: dict) -> dict:
    """Extract the source MLP config from Llava config."""
    text_config = llava_config["text_config"]
    return {
        "type": "mlp",
        "intermediate_size": text_config["intermediate_size"],
        "activation": text_config["hidden_act"],
        "gated": True,
        "add_linear_biases": False,
    }


def extract_source_norm_config(llava_config: dict) -> dict:
    """Extract the source normalization config from Llava config."""
    text_config = llava_config["text_config"]
    return {
        "type": "rms_norm",
        "epsilon": text_config["rms_norm_eps"],
    }


# Parameters that affect weight shapes - cannot be overridden with init: transfer
SHAPE_AFFECTING_PARAMS = {
    "heads",
    "head_groups",
    "head_size",
    "intermediate_size",
    "hidden_size",
}

# Parameters that affect behavior but not weight shapes - warn if overridden
BEHAVIOR_AFFECTING_PARAMS = {
    "activation",
    "gated",
}


def validate_transfer_overrides(
    overrides: dict, source_config: dict, component_name: str
) -> None:
    """Validate that overrides are compatible with weight transfer.

    Raises ValueError for shape-incompatible overrides.
    Logs warning for behavior-affecting overrides.
    """
    for param in SHAPE_AFFECTING_PARAMS:
        if param in overrides and param in source_config:
            if overrides[param] != source_config[param]:
                raise ValueError(
                    f"Component '{component_name}': Cannot override '{param}' with "
                    f"init: transfer (source={source_config[param]}, target={overrides[param]}). "
                    f"This would cause weight shape mismatch. Use 'init: random' instead."
                )

    for param in BEHAVIOR_AFFECTING_PARAMS:
        if param in overrides and param in source_config:
            if overrides[param] != source_config[param]:
                logger.warning(
                    f"Component '{component_name}': Overriding '{param}' with init: transfer "
                    f"(source={source_config[param]}, target={overrides[param]}). "
                    f"Weights will be transferred but behavior will differ."
                )


def build_component_config(
    component_spec: dict, source_config: dict, component_name: str
) -> dict:
    """Build final component config from spec and source.

    If spec has 'init: transfer' and no explicit type (or same type as source),
    inherit from source config with any overrides applied.

    Raises ValueError if overrides are incompatible with weight transfer.
    """
    init_mode = component_spec.get("init", "transfer")

    # Extract fields that aren't config (init is a control field)
    config_fields = {k: v for k, v in component_spec.items() if k != "init"}

    if init_mode == "transfer":
        # Check if type is specified and different from source
        target_type = config_fields.get("type", source_config.get("type"))
        source_type = source_config.get("type")

        if target_type == source_type or "type" not in config_fields:
            # Validate overrides are compatible with transfer
            validate_transfer_overrides(config_fields, source_config, component_name)

            # Same type or no type specified - inherit from source with overrides
            result = copy.deepcopy(source_config)
            result.update(config_fields)
            return result
        else:
            # Different type - must have full config specified
            if "type" not in config_fields:
                raise ValueError(
                    f"Component '{component_name}' has different type but no config specified"
                )
            return config_fields
    else:  # init: random
        # Must have full config specified
        if "type" not in config_fields:
            raise ValueError(
                f"Component '{component_name}' with 'init: random' must specify full config including 'type'"
            )
        return config_fields


def build_stochastic_mixer_config(
    stochastic_spec: dict, source_mixer_config: dict
) -> dict:
    """Build stochastic mixer config from spec."""
    mixers_spec = stochastic_spec.get("mixers", {})
    main_mixer_name = stochastic_spec.get("main_mixer_name", "attention")
    sampling_strategy = stochastic_spec.get("sampling_strategy", "uniform")

    built_mixers = {}
    for mixer_name, mixer_spec in mixers_spec.items():
        built_mixers[mixer_name] = build_component_config(
            mixer_spec, source_mixer_config, f"mixer.{mixer_name}"
        )

    return {
        "type": "stochastic",
        "main_mixer_name": main_mixer_name,
        "sampling_strategy": sampling_strategy,
        "mixers": built_mixers,
    }


def build_decoder_config(
    target_decoder: dict, llava_config: dict
) -> dict:
    """Build decoder config from target spec and source config."""
    text_config = llava_config["text_config"]
    num_layers = text_config["num_hidden_layers"]

    source_mixer = extract_source_mixer_config(llava_config)
    source_mlp = extract_source_mlp_config(llava_config)
    source_norm = extract_source_norm_config(llava_config)

    decoder_type = target_decoder.get("type", "fixed")

    if decoder_type == "fixed":
        block_spec = target_decoder.get("block", {})
        mixer_spec = block_spec.get("mixer", {"init": "transfer"})
        mlp_spec = block_spec.get("mlp", {"init": "transfer"})
        norm_spec = block_spec.get("normalization", {"init": "transfer"})

        # Handle stochastic mixer
        if mixer_spec.get("type") == "stochastic":
            mixer_config = build_stochastic_mixer_config(mixer_spec, source_mixer)
        else:
            mixer_config = build_component_config(mixer_spec, source_mixer, "mixer")

        mlp_config = build_component_config(mlp_spec, source_mlp, "mlp")
        norm_config = build_component_config(norm_spec, source_norm, "normalization")

        return {
            "type": "fixed",
            "num_blocks": target_decoder.get("num_blocks", num_layers),
            "block": {
                "mixer": mixer_config,
                "mlp": mlp_config,
                "normalization": norm_config,
            },
        }

    elif decoder_type == "pattern":
        pattern = target_decoder.get("pattern", [])
        blocks_spec = target_decoder.get("blocks", {})

        built_blocks = {}
        for block_name, block_spec in blocks_spec.items():
            mixer_spec = block_spec.get("mixer", {"init": "transfer"})
            mlp_spec = block_spec.get("mlp", {"init": "transfer"})
            norm_spec = block_spec.get("normalization", {"init": "transfer"})

            if mixer_spec.get("type") == "stochastic":
                mixer_config = build_stochastic_mixer_config(mixer_spec, source_mixer)
            else:
                mixer_config = build_component_config(
                    mixer_spec, source_mixer, f"blocks.{block_name}.mixer"
                )

            mlp_config = build_component_config(
                mlp_spec, source_mlp, f"blocks.{block_name}.mlp"
            )
            norm_config = build_component_config(
                norm_spec, source_norm, f"blocks.{block_name}.normalization"
            )

            built_blocks[block_name] = {
                "mixer": mixer_config,
                "mlp": mlp_config,
                "normalization": norm_config,
            }

        return {
            "type": "pattern",
            "num_blocks": target_decoder.get("num_blocks", num_layers),
            "pattern": pattern,
            "blocks": built_blocks,
        }

    else:
        raise ValueError(f"Unknown decoder type: {decoder_type}")


def convert_vision_config(llava_config: dict) -> dict:
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


def convert_config(llava_config: dict, target_config: dict | None = None) -> dict:
    """Convert full Llava config to Apriel2 format.

    Args:
        llava_config: Source Llava config
        target_config: Optional target structure config (from YAML).
                      If None, creates a simple attention-only decoder.
    """
    text_config = llava_config["text_config"]

    # Get token IDs - prefer top-level, fall back to text_config (no silent defaults)
    bos_token_id = llava_config.get("bos_token_id") or text_config["bos_token_id"]
    eos_token_id = llava_config.get("eos_token_id") or text_config["eos_token_id"]
    pad_token_id = llava_config.get("pad_token_id") or text_config.get("pad_token_id")

    # Build decoder config
    if target_config and "decoder" in target_config:
        decoder_config = build_decoder_config(target_config["decoder"], llava_config)
    else:
        # Default: simple attention decoder (transfer everything)
        decoder_config = build_decoder_config(
            {
                "type": "fixed",
                "block": {
                    "mixer": {"init": "transfer"},
                    "mlp": {"init": "transfer"},
                    "normalization": {"init": "transfer"},
                },
            },
            llava_config,
        )

    apriel2_config = {
        "architectures": ["Apriel2ForConditionalGeneration"],
        "model_type": "apriel2",
        "auto_map": {
            "AutoConfig": "configuration_apriel2.Apriel2Config",
            "AutoModel": "modeling_apriel2.Apriel2Model",
            "AutoModelForCausalLM": "modeling_apriel2.Apriel2ForConditionalGeneration",
        },
        "hidden_size": text_config["hidden_size"],
        "vocab_size": text_config["vocab_size"],
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
        "tie_word_embeddings": text_config["tie_word_embeddings"],
        "use_cache": text_config.get("use_cache", True),  # use_cache commonly omitted when True
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
        "vision_encoder": convert_vision_config(llava_config),
    }

    return apriel2_config


# =============================================================================
# Weight Conversion
# =============================================================================

# Weight mapping from Llava to Apriel2 naming (for non-layer weights)
WEIGHT_MAP = {
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

# Llava layer component -> Apriel2 component
LLAVA_LAYER_MAP = {
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

# Vision layer component -> Apriel2 component
LLAVA_VISION_LAYER_MAP = {
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


def get_init_mode_for_layer(
    layer_idx: int, component: str, target_decoder: dict
) -> tuple[str, dict, dict]:
    """Get init mode and configs for a component at a specific layer.

    Returns: (init_mode, source_config, target_config)
    """
    decoder_type = target_decoder.get("type", "fixed")

    if decoder_type == "fixed":
        block = target_decoder.get("block", {})
        if component == "mixer":
            spec = block.get("mixer", {})
        elif component == "mlp":
            spec = block.get("mlp", {})
        elif component == "normalization":
            spec = block.get("normalization", {})
        else:
            spec = {}

    elif decoder_type == "pattern":
        pattern = target_decoder.get("pattern", [])
        blocks = target_decoder.get("blocks", {})
        if pattern:
            block_name = pattern[layer_idx % len(pattern)]
            block = blocks.get(block_name, {})
        else:
            block = {}

        if component == "mixer":
            spec = block.get("mixer", {})
        elif component == "mlp":
            spec = block.get("mlp", {})
        elif component == "normalization":
            spec = block.get("normalization", {})
        else:
            spec = {}
    else:
        spec = {}

    init_mode = spec.get("init", "transfer")
    return init_mode, spec


def get_mixer_init_for_stochastic(
    layer_idx: int, mixer_name: str, target_decoder: dict
) -> str:
    """Get init mode for a specific mixer within a stochastic mixer."""
    decoder_type = target_decoder.get("type", "fixed")

    if decoder_type == "fixed":
        mixer_spec = target_decoder.get("block", {}).get("mixer", {})
    elif decoder_type == "pattern":
        pattern = target_decoder.get("pattern", [])
        blocks = target_decoder.get("blocks", {})
        if pattern:
            block_name = pattern[layer_idx % len(pattern)]
            mixer_spec = blocks.get(block_name, {}).get("mixer", {})
        else:
            mixer_spec = {}
    else:
        mixer_spec = {}

    if mixer_spec.get("type") != "stochastic":
        return "transfer"

    mixers = mixer_spec.get("mixers", {})
    sub_mixer = mixers.get(mixer_name, {})
    return sub_mixer.get("init", "transfer")


def convert_weights(
    input_dir: Path,
    output_dir: Path,
    target_config: dict | None = None,
    apriel2_config: dict | None = None,
) -> None:
    """Convert weights from Llava to Apriel2 format.

    Handles init modes (transfer vs random) based on target_config.
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

    # Load all source weights
    all_weights = {}
    for model_file in tqdm(model_files, desc="Loading weights"):
        if use_safetensors:
            with safe_open(model_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    all_weights[key] = f.get_tensor(key)
        else:
            state_dict = torch.load(model_file, map_location="cpu", weights_only=True)
            all_weights.update(state_dict)

    # Organize source weights by layer
    source_layer_weights = {}  # layer_idx -> {component -> {weight_name -> tensor}}
    other_weights = {}

    for llava_name, tensor in all_weights.items():
        if llava_name in WEIGHT_MAP:
            other_weights[WEIGHT_MAP[llava_name]] = tensor
        elif llava_name.startswith("language_model.model.layers."):
            parts = llava_name.split(".")
            layer_idx = int(parts[3])
            rest = ".".join(parts[4:])
            if layer_idx not in source_layer_weights:
                source_layer_weights[layer_idx] = {}
            source_layer_weights[layer_idx][rest] = tensor
        elif llava_name.startswith("vision_tower.transformer.layers."):
            parts = llava_name.split(".")
            layer_idx = int(parts[3])
            rest = ".".join(parts[4:])
            if rest in LLAVA_VISION_LAYER_MAP:
                apriel2_name = f"model.vision_encoder.encoder.blocks.{layer_idx}.{LLAVA_VISION_LAYER_MAP[rest]}"
                other_weights[apriel2_name] = tensor
        else:
            logger.warning(f"Unknown weight: {llava_name}")

    # Get target decoder config
    target_decoder = {}
    if target_config and "decoder" in target_config:
        target_decoder = target_config["decoder"]
    if apriel2_config and "decoder" in apriel2_config:
        built_decoder = apriel2_config["decoder"]
    else:
        built_decoder = {"type": "fixed", "block": {"mixer": {"type": "attention"}}}

    # Convert layer weights
    converted_weights = dict(other_weights)

    for layer_idx in tqdm(sorted(source_layer_weights.keys()), desc="Converting layers"):
        layer_weights = source_layer_weights[layer_idx]

        # Get block config for this layer
        if built_decoder.get("type") == "fixed":
            block_config = built_decoder.get("block", {})
        elif built_decoder.get("type") == "pattern":
            pattern = built_decoder.get("pattern", [])
            blocks = built_decoder.get("blocks", {})
            if pattern:
                block_name = pattern[layer_idx % len(pattern)]
                block_config = blocks.get(block_name, {})
            else:
                block_config = {}
        else:
            block_config = {}

        mixer_config = block_config.get("mixer", {})
        is_stochastic = mixer_config.get("type") == "stochastic"

        # Process mixer weights
        mixer_init, _ = get_init_mode_for_layer(layer_idx, "mixer", target_decoder)

        for src_name, tensor in layer_weights.items():
            if src_name not in LLAVA_LAYER_MAP:
                logger.warning(f"Unknown layer weight: {src_name}")
                continue

            apriel2_suffix = LLAVA_LAYER_MAP[src_name]

            # Determine if this is a mixer weight
            is_mixer_weight = apriel2_suffix.startswith("mixer.")

            if is_mixer_weight and is_stochastic:
                # For stochastic mixer, we need to handle each sub-mixer
                mixers = mixer_config.get("mixers", {})
                for mixer_name, sub_mixer_config in mixers.items():
                    # Get init mode for this specific sub-mixer
                    sub_init = get_mixer_init_for_stochastic(
                        layer_idx, mixer_name, target_decoder
                    )

                    if sub_init == "random":
                        # Skip - will be randomly initialized
                        logger.debug(
                            f"Skipping {mixer_name} weights at layer {layer_idx} (init: random)"
                        )
                        continue

                    # Transfer weights
                    # For stochastic, path is: mixer.mixers.<name>.self_attn.xxx
                    stochastic_suffix = apriel2_suffix.replace(
                        "mixer.", f"mixer.mixers.{mixer_name}."
                    )
                    full_name = f"model.decoder.blocks.{layer_idx}.{stochastic_suffix}"
                    # Clone tensor to avoid shared memory issues with safetensors
                    converted_weights[full_name] = tensor.clone()

            elif is_mixer_weight:
                # Non-stochastic mixer
                if mixer_init == "random":
                    logger.debug(
                        f"Skipping mixer weights at layer {layer_idx} (init: random)"
                    )
                    continue
                full_name = f"model.decoder.blocks.{layer_idx}.{apriel2_suffix}"
                converted_weights[full_name] = tensor

            else:
                # MLP or norm weights
                if apriel2_suffix.startswith("mlp."):
                    component_init, _ = get_init_mode_for_layer(
                        layer_idx, "mlp", target_decoder
                    )
                else:
                    component_init, _ = get_init_mode_for_layer(
                        layer_idx, "normalization", target_decoder
                    )

                if component_init == "random":
                    logger.debug(
                        f"Skipping {apriel2_suffix} at layer {layer_idx} (init: random)"
                    )
                    continue

                full_name = f"model.decoder.blocks.{layer_idx}.{apriel2_suffix}"
                converted_weights[full_name] = tensor

    # Save converted weights
    output_file = output_dir / "model.safetensors"
    logger.info(f"Saving {len(converted_weights)} weights to {output_file}")
    save_file(converted_weights, output_file)


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
        "--config",
        "-c",
        type=Path,
        help="Path to YAML config specifying target decoder structure",
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

    # Load target config if provided
    target_config = None
    if args.config:
        logger.info(f"Loading target config from {args.config}")
        with open(args.config) as f:
            target_config = yaml.safe_load(f)

    # Resolve input (local or HuggingFace)
    input_dir = resolve_input(args.input)

    config_file = input_dir / "config.json"
    if not config_file.exists():
        raise ValueError(f"Config file not found: {config_file}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load source config
    logger.info(f"Loading source config from {config_file}")
    with open(config_file) as f:
        llava_config = json.load(f)

    # Convert config
    apriel2_config = convert_config(llava_config, target_config)

    # Save converted config
    output_config_file = args.output_dir / "config.json"
    logger.info(f"Saving converted config to {output_config_file}")
    with open(output_config_file, "w") as f:
        json.dump(apriel2_config, f, indent=2)

    # Convert weights
    convert_weights(input_dir, args.output_dir, target_config, apriel2_config)

    # Copy tokenizer files
    copy_tokenizer_files(input_dir, args.output_dir)

    # Copy model files
    copy_model_files(args.output_dir)

    logger.info(f"Conversion complete! Output saved to {args.output_dir}")


if __name__ == "__main__":
    main()
