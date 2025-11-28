"""Generic Apriel2 -> Apriel2 model surgery.

This module provides a generic surgery function that transforms any Apriel2 model
(config + weights) to a different Apriel2 architecture. It uses the converter
registry to transform components layer by layer.

Key concepts:
- Source: Any valid Apriel2 config + state_dict
- Target: Any valid Apriel2 config (weights will be generated)
- For stochastic mixers, the source is always the main mixer
- Converters handle type transformations (attention -> swa, etc.)
- Missing converters trigger random initialization
"""

import copy
import logging
import re
from typing import Callable

import torch
from torch import Tensor

from .converters import (
    get_converter,
    has_converter,
    random_init_mixer,
    random_init_mlp,
    random_init_norm,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Surgery Function
# =============================================================================


def surgery(
    source_config: dict,
    source_weights: dict[str, Tensor],
    target_config: dict,
    device: str = "cpu",
    dtype: torch.dtype | None = None,
) -> dict[str, Tensor]:
    """Transform Apriel2 model to a different architecture.

    This is the main entry point for model surgery. It takes a source model
    (config + weights) and a target config, and produces weights for the target.

    Args:
        source_config: Source Apriel2 config dict.
        source_weights: Source model state_dict.
        target_config: Target Apriel2 config dict.
        device: Device for new tensors.
        dtype: Data type for new tensors. If None, infers from source weights.

    Returns:
        Target model state_dict.
    """
    if dtype is None:
        # Infer dtype from source weights
        for v in source_weights.values():
            if isinstance(v, Tensor):
                dtype = v.dtype
                break
        if dtype is None:
            dtype = torch.float32

    hidden_size = target_config.get("hidden_size", source_config.get("hidden_size"))

    target_weights = {}

    # Copy non-decoder weights (embeddings, vision encoder, head)
    _copy_non_decoder_weights(source_weights, target_weights)

    # Process decoder layers
    source_decoder = source_config.get("decoder", {})
    target_decoder = target_config.get("decoder", {})

    num_source_layers = source_decoder.get("num_blocks", 0)
    num_target_layers = target_decoder.get("num_blocks", 0)

    if num_target_layers > num_source_layers:
        logger.warning(
            f"Target has more layers ({num_target_layers}) than source ({num_source_layers}). "
            f"Extra layers will use source layer (idx % num_source_layers) as source."
        )

    for layer_idx in range(num_target_layers):
        # Get source layer index (wrap around if target has more layers)
        source_layer_idx = layer_idx % num_source_layers if num_source_layers > 0 else 0

        source_block = _get_block_config(source_decoder, source_layer_idx)
        target_block = _get_block_config(target_decoder, layer_idx)

        # Convert mixer
        _convert_mixer(
            layer_idx,
            source_layer_idx,
            source_block.get("mixer", {}),
            target_block.get("mixer", {}),
            source_weights,
            target_weights,
            hidden_size,
            device,
            dtype,
        )

        # Convert MLP
        _convert_mlp(
            layer_idx,
            source_layer_idx,
            source_block.get("mlp", {}),
            target_block.get("mlp", {}),
            source_weights,
            target_weights,
            hidden_size,
            device,
            dtype,
        )

        # Convert normalizations
        _convert_norms(
            layer_idx,
            source_layer_idx,
            source_block,
            target_block,
            source_weights,
            target_weights,
            hidden_size,
            device,
            dtype,
        )

    return target_weights


# =============================================================================
# Block Config Utilities
# =============================================================================


def _get_block_config(decoder_config: dict, layer_idx: int) -> dict:
    """Get block config for a specific layer index."""
    decoder_type = decoder_config.get("type", "fixed")

    if decoder_type == "fixed":
        return decoder_config.get("block", {})
    elif decoder_type == "pattern":
        pattern = decoder_config.get("pattern", [])
        blocks = decoder_config.get("blocks", {})
        if pattern:
            block_name = pattern[layer_idx % len(pattern)]
            return blocks.get(block_name, {})
        return {}
    else:
        return {}


# =============================================================================
# Weight Extraction Utilities
# =============================================================================


def _copy_non_decoder_weights(
    source_weights: dict[str, Tensor],
    target_weights: dict[str, Tensor],
) -> None:
    """Copy non-decoder weights (embeddings, vision encoder, head, etc.)."""
    decoder_pattern = re.compile(r"model\.decoder\.blocks\.\d+\.")

    for key, tensor in source_weights.items():
        if not decoder_pattern.search(key):
            target_weights[key] = tensor.clone()


def _extract_component_weights(
    state_dict: dict[str, Tensor],
    prefix: str,
) -> dict[str, Tensor]:
    """Extract weights for a component with the given prefix.

    Returns weights with the prefix stripped from keys.
    """
    result = {}
    for key, tensor in state_dict.items():
        if key.startswith(prefix):
            relative_key = key[len(prefix):]
            result[relative_key] = tensor
    return result


def _add_prefix(weights: dict[str, Tensor], prefix: str) -> dict[str, Tensor]:
    """Add prefix to all weight keys."""
    return {prefix + key: tensor for key, tensor in weights.items()}


# =============================================================================
# Mixer Conversion
# =============================================================================


def _convert_mixer(
    target_layer_idx: int,
    source_layer_idx: int,
    source_mixer: dict,
    target_mixer: dict,
    source_weights: dict[str, Tensor],
    target_weights: dict[str, Tensor],
    hidden_size: int,
    device: str,
    dtype: torch.dtype,
) -> None:
    """Convert mixer weights from source to target config."""
    source_type = source_mixer.get("type", "attention")
    target_type = target_mixer.get("type", "attention")

    # Determine actual source (unwrap stochastic to main mixer)
    if source_type == "stochastic":
        main_name = source_mixer.get("main_mixer_name", "attention")
        actual_source_config = source_mixer.get("mixers", {}).get(main_name, {})
        actual_source_type = actual_source_config.get("type", "attention")
        source_prefix = f"model.decoder.blocks.{source_layer_idx}.mixer.mixers.{main_name}."
    else:
        actual_source_config = source_mixer
        actual_source_type = source_type
        source_prefix = f"model.decoder.blocks.{source_layer_idx}.mixer."

    source_component_weights = _extract_component_weights(source_weights, source_prefix)

    # Handle target
    if target_type == "stochastic":
        # Target is stochastic - convert to each sub-mixer
        for sub_name, sub_config in target_mixer.get("mixers", {}).items():
            sub_type = sub_config.get("type", "attention")
            target_prefix = f"model.decoder.blocks.{target_layer_idx}.mixer.mixers.{sub_name}."

            converter = get_converter(actual_source_type, sub_type)
            if converter:
                converted = converter(
                    source_component_weights,
                    actual_source_config,
                    sub_config,
                    hidden_size,
                )
                logger.debug(
                    f"Layer {target_layer_idx}: {actual_source_type} -> {sub_name}:{sub_type} (converted)"
                )
            else:
                # No converter - random init
                converted = random_init_mixer(sub_config, hidden_size, device, dtype)
                logger.info(
                    f"Layer {target_layer_idx}: {actual_source_type} -> {sub_name}:{sub_type} (random init)"
                )

            target_weights.update(_add_prefix(converted, target_prefix))
    else:
        # Target is not stochastic
        target_prefix = f"model.decoder.blocks.{target_layer_idx}.mixer."

        converter = get_converter(actual_source_type, target_type)
        if converter:
            converted = converter(
                source_component_weights,
                actual_source_config,
                target_mixer,
                hidden_size,
            )
            logger.debug(
                f"Layer {target_layer_idx}: {actual_source_type} -> {target_type} (converted)"
            )
        else:
            # No converter - random init
            converted = random_init_mixer(target_mixer, hidden_size, device, dtype)
            logger.info(
                f"Layer {target_layer_idx}: {actual_source_type} -> {target_type} (random init)"
            )

        target_weights.update(_add_prefix(converted, target_prefix))


# =============================================================================
# MLP Conversion
# =============================================================================


def _convert_mlp(
    target_layer_idx: int,
    source_layer_idx: int,
    source_mlp: dict,
    target_mlp: dict,
    source_weights: dict[str, Tensor],
    target_weights: dict[str, Tensor],
    hidden_size: int,
    device: str,
    dtype: torch.dtype,
) -> None:
    """Convert MLP weights from source to target config."""
    source_prefix = f"model.decoder.blocks.{source_layer_idx}.mlp."
    target_prefix = f"model.decoder.blocks.{target_layer_idx}.mlp."

    source_component_weights = _extract_component_weights(source_weights, source_prefix)

    source_type = source_mlp.get("type", "mlp")
    target_type = target_mlp.get("type", "mlp")

    converter = get_converter(source_type, target_type)
    if converter:
        converted = converter(
            source_component_weights,
            source_mlp,
            target_mlp,
            hidden_size,
        )
    else:
        # No converter - random init
        converted = random_init_mlp(target_mlp, hidden_size, device, dtype)
        logger.info(f"Layer {target_layer_idx}: MLP {source_type} -> {target_type} (random init)")

    target_weights.update(_add_prefix(converted, target_prefix))


# =============================================================================
# Normalization Conversion
# =============================================================================


def _convert_norms(
    target_layer_idx: int,
    source_layer_idx: int,
    source_block: dict,
    target_block: dict,
    source_weights: dict[str, Tensor],
    target_weights: dict[str, Tensor],
    hidden_size: int,
    device: str,
    dtype: torch.dtype,
) -> None:
    """Convert normalization weights from source to target config."""
    # Input layernorm
    _convert_single_norm(
        target_layer_idx,
        source_layer_idx,
        "input_layernorm",
        source_block.get("normalization", {}),
        target_block.get("normalization", {}),
        source_weights,
        target_weights,
        hidden_size,
        device,
        dtype,
    )

    # Post-attention layernorm
    _convert_single_norm(
        target_layer_idx,
        source_layer_idx,
        "post_attention_layernorm",
        source_block.get("normalization", {}),
        target_block.get("normalization", {}),
        source_weights,
        target_weights,
        hidden_size,
        device,
        dtype,
    )


def _convert_single_norm(
    target_layer_idx: int,
    source_layer_idx: int,
    norm_name: str,
    source_norm: dict,
    target_norm: dict,
    source_weights: dict[str, Tensor],
    target_weights: dict[str, Tensor],
    hidden_size: int,
    device: str,
    dtype: torch.dtype,
) -> None:
    """Convert a single normalization layer."""
    source_prefix = f"model.decoder.blocks.{source_layer_idx}.{norm_name}."
    target_prefix = f"model.decoder.blocks.{target_layer_idx}.{norm_name}."

    source_component_weights = _extract_component_weights(source_weights, source_prefix)

    source_type = source_norm.get("type", "rms_norm")
    target_type = target_norm.get("type", "rms_norm")

    converter = get_converter(source_type, target_type)
    if converter:
        converted = converter(
            source_component_weights,
            source_norm,
            target_norm,
            hidden_size,
        )
    else:
        # No converter - random init
        converted = random_init_norm(target_norm, hidden_size, device, dtype)
        logger.info(
            f"Layer {target_layer_idx}: {norm_name} {source_type} -> {target_type} (random init)"
        )

    target_weights.update(_add_prefix(converted, target_prefix))


# =============================================================================
# Config Surgery (Convenience Functions)
# =============================================================================


def build_target_config(
    source_config: dict,
    modifications: dict,
) -> dict:
    """Build target config by applying modifications to source config.

    This is a convenience function for creating target configs from source configs
    with specific modifications.

    Args:
        source_config: Source Apriel2 config.
        modifications: Dict of modifications to apply. Supports nested paths
                      like "decoder.block.mixer.type".

    Returns:
        New config dict with modifications applied.
    """
    target = copy.deepcopy(source_config)

    for path, value in modifications.items():
        parts = path.split(".")
        obj = target
        for part in parts[:-1]:
            if part not in obj:
                obj[part] = {}
            obj = obj[part]
        obj[parts[-1]] = value

    return target


def wrap_with_stochastic(
    source_config: dict,
    mixers: dict[str, dict],
    main_mixer_name: str = "attention",
    layer_selector: Callable[[int], bool] | None = None,
) -> dict:
    """Create target config that wraps attention with stochastic mixer.

    Args:
        source_config: Source Apriel2 config with attention mixers.
        mixers: Dict of mixer configs to include in stochastic wrapper.
                The main mixer should be included.
        main_mixer_name: Name of the main mixer in the mixers dict.
        layer_selector: Optional function to select which layers to wrap.
                       If None, all layers are wrapped.

    Returns:
        New config with stochastic mixer wrapper.
    """
    target = copy.deepcopy(source_config)

    # Get the source mixer config to use as base for main mixer
    source_decoder = source_config.get("decoder", {})
    source_block = _get_block_config(source_decoder, 0)
    source_mixer = source_block.get("mixer", {})

    # Build stochastic mixer config
    stochastic_mixer = {
        "type": "stochastic",
        "main_mixer_name": main_mixer_name,
        "mixers": mixers,
    }

    # Apply to decoder
    decoder = target.get("decoder", {})
    decoder_type = decoder.get("type", "fixed")

    if decoder_type == "fixed":
        decoder.setdefault("block", {})["mixer"] = stochastic_mixer
    elif decoder_type == "pattern":
        # Apply to all blocks (or could be selective with layer_selector)
        for block_name in decoder.get("blocks", {}):
            decoder["blocks"][block_name]["mixer"] = stochastic_mixer

    return target
