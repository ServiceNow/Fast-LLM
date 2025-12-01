"""Config composition for Apriel2 architecture transformations.

This module handles STRUCTURAL composition of configs, independent of weight handling.
The `init` field in surgery specs is preserved as metadata for the plan builder but
does not affect how configs are composed.

Composition Cases
=================

compose_configs(base, overlay) handles four cases based on completeness:

1. **Complete + Partial** → Apply surgery semantics (inheritance, cross-type derivation)
2. **Partial + Partial** → Deep merge (monoid operation on surgery specs)
3. **Partial + Complete** → Overlay wins (complete config replaces partial)
4. **Complete + Complete** → Deep merge, then strip `init` fields

A config is "complete" if it has `hidden_size` and `decoder` (i.e., it's a full model
config, not a surgery spec).

Surgery Semantics
=================

When applying a surgery spec to a complete config:

**Inheritance**
    Unspecified parameters inherit from the source config. New blocks inherit
    from the "default" block (first block in pattern, or the single fixed block).

**Cross-Type Derivation**
    When changing mixer types, geometric parameters are derived where possible:
    - attention → sliding_window: preserve heads, head_groups, head_size
    - attention → gated_delta_net: heads → num_value_heads, head_groups → num_key_heads
    - attention → mamba: derive d_inner, d_xb, dt_rank from hidden_size

**Stochastic Mixer Composition**
    Two semantics based on whether surgery declares `type: stochastic`:
    - Replacement: surgery declares type → only surgery's sub-mixers included
    - Additive: surgery omits type → source sub-mixers preserved, surgery adds/modifies

    This distinction means the monoid action law holds for additive surgeries but
    intentionally fails for replacement surgeries (they have "last-write-wins" semantics).

The `init` Field
================

The `init` field is metadata for the plan builder, NOT for config composition:
- `init: transfer` → plan builder creates weight transfer mappings
- `init: random` → plan builder creates random initialization

After surgery is applied to produce a complete config, ALL `init` fields are stripped.
This ensures configs are purely structural and plan creation is Markovian (depends only
on current config + surgery, not on history).
"""

from __future__ import annotations

import copy
from typing import Any


def is_complete(config: dict) -> bool:
    """Check if a config is complete (has required top-level fields)."""
    return "hidden_size" in config and "decoder" in config


def compose_configs(base: dict, overlay: dict | None) -> dict:
    """Compose two configs.

    Args:
        base: Base config (complete or partial surgery spec).
        overlay: Overlay config (complete or partial surgery spec).

    Returns:
        Composed config.
    """
    if not overlay:
        return copy.deepcopy(base)
    if not base:
        return copy.deepcopy(overlay)

    base_complete = is_complete(base)
    overlay_complete = is_complete(overlay)

    # Case 1: Complete + partial surgery -> apply full surgery semantics
    if base_complete and not overlay_complete:
        return apply_surgery(base, overlay)

    # Case 2: Both partial -> deep merge (monoid operation on surgery specs)
    if not base_complete and not overlay_complete:
        return _deep_merge(base, overlay)

    # Case 3: Partial + complete -> overlay wins
    if not base_complete and overlay_complete:
        return copy.deepcopy(overlay)

    # Case 4: Both complete -> deep merge
    result = _deep_merge(base, overlay)
    _strip_keys(result, {"init"})
    return result


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Deep merge overlay into base. Overlay wins on conflicts."""
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if value is None:
            # Null deletion
            result.pop(key, None)
        elif key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _strip_keys(config: Any, keys_to_strip: set[str]) -> None:
    """Recursively strip specified keys from config."""
    if not isinstance(config, dict):
        return
    for key in list(config.keys()):
        if key in keys_to_strip:
            del config[key]
        elif isinstance(config[key], dict):
            _strip_keys(config[key], keys_to_strip)
        elif isinstance(config[key], list):
            for item in config[key]:
                _strip_keys(item, keys_to_strip)


# =============================================================================
# Surgery application with full semantics
# =============================================================================


def apply_surgery(source_config: dict, surgery_config: dict | None) -> dict:
    """Apply surgery specification to a complete source config.

    This handles:
    - Top-level scalar overrides
    - Decoder composition (fixed vs pattern)
    - Stochastic mixer sub-mixer inheritance
    - Cross-type derivation (attention → gdn, attention → mamba)

    Args:
        source_config: Complete Apriel2 config.
        surgery_config: Partial surgery specification.

    Returns:
        Complete Apriel2 config with surgery applied.
    """
    if not surgery_config:
        return copy.deepcopy(source_config)

    result = copy.deepcopy(source_config)
    hidden_size = result.get("hidden_size", 0)

    # Top-level scalar overrides
    for key in [
        "model_type",
        "architectures",
        "hidden_size",
        "vocab_size",
        "bos_token_id",
        "eos_token_id",
        "tie_word_embeddings",
        "image_token_index",
    ]:
        if key in surgery_config:
            result[key] = surgery_config[key]
            if key == "hidden_size":
                hidden_size = surgery_config[key]

    # Compose decoder
    if "decoder" in surgery_config:
        result["decoder"] = _compose_decoder(
            result.get("decoder", {}),
            surgery_config["decoder"],
            hidden_size,
        )

    # Vision encoder: deep merge
    if "vision_encoder" in surgery_config:
        if surgery_config["vision_encoder"] is None:
            result.pop("vision_encoder", None)
        else:
            result["vision_encoder"] = _deep_merge(
                result.get("vision_encoder", {}),
                surgery_config["vision_encoder"],
            )

    # Strip init keys from final result
    _strip_keys(result, {"init"})

    return result


def _compose_decoder(source: dict, surgery: dict, hidden_size: int) -> dict:
    """Compose decoder config with full surgery semantics."""
    result: dict[str, Any] = {}

    result["type"] = surgery.get("type", source.get("type", "fixed"))
    result["num_blocks"] = surgery.get("num_blocks", source.get("num_blocks"))

    source_type = source.get("type", "fixed")

    # Get the "default" block for inheritance when surgery introduces new blocks
    # - For fixed decoder: the single block
    # - For pattern decoder: the first block in the pattern
    if source_type == "fixed":
        default_block = source.get("block", {})
    else:  # pattern
        source_blocks = source.get("blocks", {})
        source_pattern = source.get("pattern", [])
        if source_pattern and source_pattern[0] in source_blocks:
            default_block = source_blocks[source_pattern[0]]
        elif source_blocks:
            default_block = next(iter(source_blocks.values()))
        else:
            default_block = {}

    if result["type"] == "fixed":
        surgery_block = surgery.get("block", {})
        result["block"] = _compose_block(default_block, surgery_block, hidden_size)

    elif result["type"] == "pattern":
        result["pattern"] = surgery.get("pattern", source.get("pattern", []))
        source_blocks = source.get("blocks", {})
        surgery_blocks = surgery.get("blocks", {})
        result["blocks"] = {}

        # For each block in surgery, compose with appropriate base
        for name, surgery_block in surgery_blocks.items():
            # If source has this named block, use it; otherwise use default
            base_block = source_blocks.get(name, default_block)
            result["blocks"][name] = _compose_block(base_block, surgery_block, hidden_size)

        # Preserve blocks from source that aren't in surgery
        for name, block in source_blocks.items():
            if name not in result["blocks"]:
                result["blocks"][name] = copy.deepcopy(block)

    return result


def _compose_block(source: dict, surgery: dict, hidden_size: int) -> dict:
    """Compose a single block config."""
    result: dict[str, Any] = {}

    source_mixer = source.get("mixer", {})
    surgery_mixer = surgery.get("mixer", {})
    result["mixer"] = _compose_mixer(source_mixer, surgery_mixer, hidden_size)

    source_mlp = source.get("mlp", {})
    surgery_mlp = surgery.get("mlp", {})
    result["mlp"] = _compose_simple(source_mlp, surgery_mlp)

    source_norm = source.get("normalization", {})
    surgery_norm = surgery.get("normalization", {})
    result["normalization"] = _compose_simple(source_norm, surgery_norm)

    return result


def _compose_mixer(source: dict, surgery: dict, hidden_size: int) -> dict:
    """Compose mixer config, handling stochastic wrappers.

    Key rules:
    - When wrapping non-stochastic in stochastic, sub-mixers inherit from source
    - When source is stochastic, new sub-mixers inherit from main mixer
    - Cross-type derivation always applies (attention → gdn geometry mapping)
    """
    source_type = source.get("type", "attention")
    source_is_stochastic = source_type == "stochastic"

    # Get the "base mixer" for inheritance
    # - If source is stochastic: use the main mixer
    # - If source is non-stochastic: use source directly
    if source_is_stochastic:
        main_name = source.get("main_mixer_name", "attention")
        source_base = source.get("mixers", {}).get(main_name, {})
        source_mixers = source.get("mixers", {})
    else:
        source_base = source
        source_mixers = {}

    surgery_type = surgery.get("type", source_type)

    if surgery_type == "stochastic":
        result: dict[str, Any] = {
            "type": "stochastic",
            "main_mixer_name": surgery.get(
                "main_mixer_name",
                source.get("main_mixer_name", "attention") if source_is_stochastic else "attention",
            ),
        }

        # Copy other stochastic-level fields
        for key in ["sampling_strategy"]:
            if key in surgery:
                result[key] = surgery[key]
            elif source_is_stochastic and key in source:
                result[key] = source[key]

        # Compose mixers
        result["mixers"] = {}

        surgery_mixers = surgery.get("mixers", {})

        # Determine semantics: replacement vs additive
        # - If surgery explicitly declares type: stochastic, use replacement semantics
        #   (only mixers in surgery.mixers are included)
        # - Otherwise, use additive semantics (source mixers are preserved unless
        #   explicitly null-deleted)
        surgery_declares_stochastic = surgery.get("type") == "stochastic"

        if surgery_declares_stochastic:
            # Replacement semantics: only include mixers explicitly in surgery
            for name, sub_surgery in surgery_mixers.items():
                if sub_surgery is None:
                    # Null deletion - explicitly exclude this mixer
                    continue
                # Get base for this sub-mixer
                if name in source_mixers:
                    # Existing sub-mixer: inherit from it
                    sub_base = source_mixers[name]
                else:
                    # New sub-mixer: inherit from base mixer
                    sub_base = source_base
                result["mixers"][name] = _compose_single_mixer(sub_base, sub_surgery, hidden_size)
        else:
            # Additive semantics: preserve source mixers, then apply surgery modifications
            # First, copy all source mixers
            for name, existing_mixer in source_mixers.items():
                result["mixers"][name] = copy.deepcopy(existing_mixer)

            # Then, compose surgery mixers (overwrite or null-delete)
            for name, sub_surgery in surgery_mixers.items():
                if sub_surgery is None:
                    # Null deletion
                    result["mixers"].pop(name, None)
                else:
                    # Get base for this sub-mixer
                    if name in source_mixers:
                        # Existing sub-mixer: inherit from it
                        sub_base = source_mixers[name]
                    else:
                        # New sub-mixer: inherit from base mixer
                        sub_base = source_base
                    result["mixers"][name] = _compose_single_mixer(sub_base, sub_surgery, hidden_size)

        return result
    else:
        # Non-stochastic result
        return _compose_single_mixer(source_base, surgery, hidden_size)


def _compose_single_mixer(source: dict, surgery: dict, hidden_size: int) -> dict:
    """Compose a single mixer with cross-type derivation.

    Config inheritance is based on STRUCTURE, not `init`.
    `init` is preserved as data for the plan builder.
    """
    source_type = source.get("type", "attention")
    target_type = surgery.get("type", source_type)

    # Start with cross-type derivation or same-type inheritance
    if source_type == target_type:
        # Same type: deep merge
        result = _deep_merge(source, surgery)
        result["type"] = target_type
        return result

    # Cross-type: derive what we can, then apply surgery overrides
    if source_type in ("attention", "sliding_window"):
        # Extract source attention geometry
        heads = source.get("heads", 32)
        head_groups = source.get("head_groups", heads)
        head_size = source.get("head_size", hidden_size // heads if heads else 128)

        if target_type in ("attention", "sliding_window"):
            # Attention → Attention variant: preserve geometry
            result = {
                "type": target_type,
                "heads": surgery.get("heads", heads),
                "head_groups": surgery.get("head_groups", head_groups),
                "head_size": surgery.get("head_size", head_size),
            }
            # Copy other attention fields
            for key in ["sliding_window", "window_size", "rope_theta", "rope_scaling"]:
                if key in surgery:
                    result[key] = surgery[key]
                elif key in source:
                    result[key] = source[key]
            # Preserve init
            if "init" in surgery:
                result["init"] = surgery["init"]
            return result

        elif target_type == "gated_delta_net":
            # Attention → GDN: derive GDN dims from attention geometry
            result = {
                "type": "gated_delta_net",
                "num_value_heads": surgery.get("num_value_heads", heads),
                "num_key_heads": surgery.get("num_key_heads", head_groups),
                "key_head_dim": surgery.get("key_head_dim", head_size),
                "value_head_dim": surgery.get("value_head_dim", head_size),
                "conv_kernel_size": surgery.get("conv_kernel_size", 4),
            }
            # Preserve init
            if "init" in surgery:
                result["init"] = surgery["init"]
            return result

        elif target_type == "mamba":
            # Attention → Mamba: derive what we can
            result = {
                "type": "mamba",
                "d_inner": surgery.get("d_inner", 2 * hidden_size),
                "d_xb": surgery.get("d_xb", hidden_size // 4),
                "dt_rank": surgery.get("dt_rank", hidden_size // 16),
            }
            # Copy mamba-specific fields from surgery
            for key in [
                "d_state", "d_conv", "repeat_kv_before_conv", "conv_bias",
                "dt_proj_bias", "dt_min", "dt_max", "dt_init_floor",
            ]:
                if key in surgery:
                    result[key] = surgery[key]
            # Preserve init
            if "init" in surgery:
                result["init"] = surgery["init"]
            return result

    # Fallback: start fresh with surgery, no inheritance
    result = copy.deepcopy(surgery)
    result["type"] = target_type
    return result


def _compose_simple(source: dict, surgery: dict) -> dict:
    """Compose a simple component (mlp, normalization).

    Always inherits from source, surgery overrides.
    """
    if not surgery:
        return copy.deepcopy(source)

    # Deep merge: inherit from source, surgery wins on conflicts
    return _deep_merge(source, surgery)
