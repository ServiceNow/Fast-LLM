"""Config composition for Apriel2 architecture transformations.

Conceptual Types
================

The system operates on three conceptual types, all represented as ``dict``:

**State (S)**
    A complete structural description of a model. Has ``hidden_size`` and ``decoder``.
    Does NOT contain ``init`` fields. Represents WHAT a model looks like.

    Example: A saved config.json, or a model you're about to transform.

**Partial Surgery (P)**
    An incomplete config specifying fields to change. Missing ``hidden_size`` or
    ``decoder``. May contain ``init`` fields specifying weight initialization mode.

    Example: ``{"decoder": {"block": {"mixer": {"type": "gdn", "init": "random"}}}}``

**Transition Spec (T)**
    A complete config WITH ``init`` fields. Describes both the target structure
    AND how to initialize weights. This is the output of applying a surgery to
    a state - it's a complete specification of the transformation.

    Example: The result of ``compose_configs(state, surgery)`` before stripping.

The distinction between S and T is semantic (presence of ``init``), not structural.
Both are "complete" in the sense of having ``hidden_size`` and ``decoder``.

Algebraic Structure
===================

**Partial Surgeries form a Monoid (P, ∘, {})**::

    compose_configs : P × P → P     (deep merge, overlay wins)

    Identity:      compose_configs(p, {}) = compose_configs({}, p) = p
    Associativity: compose_configs(compose_configs(a, b), c)
                 = compose_configs(a, compose_configs(b, c))

**Surgeries act on States to produce Transition Specs**::

    compose_configs : S × P → T     (apply surgery with inheritance)
    compose_configs : T × P → T     (extend transition with more surgery)

**Action Law (for additive surgeries)**::

    compose_configs(compose_configs(s, p₁), p₂) = compose_configs(s, compose_configs(p₁, p₂))

This law holds when surgeries are "additive" (modifying existing structure without
declaring new types). For "replacement" surgeries (explicitly declaring ``type:``),
the action law intentionally fails - this is last-write-wins semantics.

**State Extraction**::

    strip_init_fields : T → S       (remove init metadata for saving)

Operations Summary
==================

``compose_configs(base, overlay)`` dispatches based on completeness:

1. **S × P → T** : Apply surgery to state (inheritance, cross-type derivation)
2. **T × P → T** : Extend transition spec with more surgery
3. **P × P → P** : Merge partial surgeries (monoid operation)
4. **S × S → S** : Merge states (deep merge, rare)
5. **P × S → S** : Overlay wins (complete replaces partial)

``strip_init_fields(config)`` removes all ``init`` fields, converting T → S.

Inheritance Semantics
=====================

When applying a surgery (S × P → T):

- Unspecified fields inherit from source state
- New decoder blocks inherit from the "default" block
- Cross-type derivation maps geometry (attention.heads → gdn.value_heads, etc.)
- Stochastic mixers: additive surgery preserves source mixers, replacement replaces

The ``init`` Field
==================

The ``init`` field specifies weight initialization mode for ``plan_surgery()``:

- ``init: transfer`` → transfer weights from source (possibly with conversion)
- ``init: random`` → randomly initialize weights

**Key invariant**: ``init`` is preserved through composition so ``plan_surgery()``
can read it. Use ``strip_init_fields()`` to obtain a pure state for:

- Saving to disk (config.json should not contain ``init``)
- Starting the next surgery iteration (current_state should be S, not T)

Typical Usage Pattern
=====================

::

    current_state: S = load_config(...)

    for surgery: P in surgery_chain:
        transition: T = compose_configs(current_state, surgery)  # S × P → T
        plan = plan_surgery(current_state, transition)           # plan reads init from T
        current_state: S = strip_init_fields(transition)         # T → S for next iteration

    save_config(current_state)  # S has no init fields

Sequential vs Merged Surgery Application
========================================

**IMPORTANT**: Applying surgeries sequentially (with stripping) differs from merging
surgeries first then applying once. This affects ``init`` semantics:

**Sequential** (recommended)::

    t1 = compose_configs(s, p1)      # GDN gets init: random
    s1 = strip_init_fields(t1)       # GDN loses init
    t2 = compose_configs(s1, p2)     # GDN has init: None → transfer mode

**Merged**::

    merged = compose_configs(p1, p2) # GDN keeps init: random from p1
    t = compose_configs(s, merged)   # GDN has init: random → random mode

The sequential approach means ``init: random`` applies **only to the surgery that
introduces a component**. Subsequent surgeries transfer existing weights by default.

This is the intended behavior: if surgery 1 adds GDN with random init, and surgery 2
adds sliding window (not mentioning GDN), GDN keeps its weights from surgery 1.

The merged approach would re-randomize GDN in every execution, which is rarely desired.
Always use the sequential pattern shown in "Typical Usage Pattern" above.
"""

from __future__ import annotations

import copy
from typing import Any


def is_complete(config: dict) -> bool:
    """Check if a config is complete (has required top-level fields)."""
    return "hidden_size" in config and "decoder" in config


def compose_configs(base: dict, overlay: dict | None) -> dict:
    """Compose configs. Dispatches based on completeness of arguments.

    Type Signatures (see module docstring for S, P, T definitions)::

        S × P → T    Apply surgery to state, get transition spec
        T × P → T    Extend transition spec with more surgery
        P × P → P    Merge partial surgeries (monoid operation)
        S × S → S    Merge states (deep merge)
        P × S → S    Overlay wins

    The ``init`` field is preserved in all cases. Use ``strip_init_fields()``
    to convert T → S for saving or iteration.

    Args:
        base: State (S), transition spec (T), or partial surgery (P).
        overlay: Partial surgery (P) or state (S).

    Returns:
        Composed config. Type depends on inputs (see signatures above).

    Algebraic Properties:
        Monoid: ``compose(compose(p1, p2), p3) == compose(p1, compose(p2, p3))``

        Action law (additive surgeries):
            ``compose(compose(s, p1), p2) == compose(s, compose(p1, p2))``

    Example::

        # S × P → T (apply surgery to state)
        state = {"hidden_size": 256, "decoder": {...}}
        surgery = {"decoder": {"block": {"mixer": {"init": "random"}}}}
        transition = compose_configs(state, surgery)  # T, has init

        # Build plan, then extract state
        plan = plan_surgery(state, transition)
        new_state = strip_init_fields(transition)  # S, no init
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

    # Case 4: Both complete -> deep merge (init preserved for plan_surgery)
    result = _deep_merge(base, overlay)
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


def strip_init_fields(config: dict) -> dict:
    """Return a copy of config with all ``init`` fields stripped (T → S).

    Converts a transition spec (T) to a state (S) by removing ``init`` metadata.
    Use this:

    1. Before saving configs to disk (config.json should be purely structural)
    2. Between surgery iterations (so subsequent surgeries don't re-randomize)

    See module docstring section "Sequential vs Merged Surgery Application" for
    why stripping between iterations is critical.

    Args:
        config: Config dict (not modified). Typically a transition spec (T).

    Returns:
        A deep copy with all ``init`` fields recursively removed (a state S).
    """
    result = copy.deepcopy(config)
    _strip_keys(result, {"init"})
    return result


# =============================================================================
# Surgery application with full semantics
# =============================================================================


def apply_surgery(source_config: dict, surgery_config: dict | None) -> dict:
    """Apply surgery spec to complete config (the monoid action).

    This is the internal implementation of the monoid action: surgery specs
    acting on complete configs. Called by compose_configs when base is complete
    and overlay is partial.

    Implements inheritance semantics:
    - Unspecified fields inherit from source
    - Cross-type derivation maps geometry (attention → gdn, etc.)
    - Stochastic sub-mixers inherit from source's main mixer
    - `init` fields are PRESERVED for plan_surgery() to see

    Args:
        source_config: Complete Apriel2 config (the state being acted on).
        surgery_config: Partial surgery spec (the monoid element acting).

    Returns:
        Complete config with surgery applied. `init` fields preserved.
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

    # NOTE: We do NOT strip init keys here. The `init` field is preserved through
    # composition so that plan_surgery() can see it and decide between transfer
    # vs random initialization. The caller (convert.py) strips init before saving.

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
        heads = source.get("heads")
        head_groups = source.get("head_groups", heads)
        head_size = source.get("head_size", hidden_size // heads if heads else None)

        if target_type in ("attention", "sliding_window"):
            # Attention → Attention variant: preserve geometry
            result = {
                "type": target_type,
                "heads": surgery.get("heads", heads),
                "head_groups": surgery.get("head_groups", head_groups),
                "head_size": surgery.get("head_size", head_size),
            }
            # Copy other attention fields (rotary is critical for position embeddings)
            for key in ["window_size", "rope_theta", "rope_scaling", "rotary"]:
                if key in surgery:
                    result[key] = surgery[key]
                elif key in source:
                    result[key] = source[key]
            # Copy per-layer bias settings (query_layer, key_layer, value_layer, dense_layer)
            for key in ["query_layer", "key_layer", "value_layer", "dense_layer", "add_linear_biases"]:
                if key in surgery:
                    result[key] = surgery[key]
                elif key in source:
                    result[key] = copy.deepcopy(source[key])
            # Preserve init
            if "init" in surgery:
                result["init"] = surgery["init"]
            return result

        elif target_type == "gdn":
            # Attention → GDN: derive GDN dims from attention geometry
            result = {
                "type": "gdn",
                "value_heads": surgery.get("value_heads", heads),
                "key_heads": surgery.get("key_heads", head_groups),
                "key_head_dim": surgery.get("key_head_dim", head_size),
                "value_head_dim": surgery.get("value_head_dim", head_size),
            }
            # Pass through convolution_layer if provided (required at conversion time)
            if "convolution_layer" in surgery:
                result["convolution_layer"] = surgery["convolution_layer"]
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
                "d_state",
                "d_conv",
                "repeat_kv_before_conv",
                "conv_bias",
                "dt_proj_bias",
                "dt_min",
                "dt_max",
                "dt_init_floor",
            ]:
                if key in surgery:
                    result[key] = surgery[key]
            # Preserve init
            if "init" in surgery:
                result["init"] = surgery["init"]
            return result

        elif target_type == "kda":
            # Attention → KDA: derive heads/head_dim from attention geometry
            result = {
                "type": "kda",
                "heads": surgery.get("heads", heads),
                "head_dim": surgery.get("head_dim", head_size),
            }
            # Copy KDA-specific fields from surgery
            for key in ["convolution_layer", "normalization"]:
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
