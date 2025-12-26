"""Weight conversion system for Apriel2 models.

Overview
========

This package implements a declarative weight transformation system. The core
abstraction separates config composition (structural) from plan execution (weights).

Conceptual Types
================

All configs are ``dict``, but we distinguish three conceptual types:

**State (S)** - A complete model config without ``init`` fields.
    What you load from disk or save after conversion.

**Partial Surgery (P)** - An incomplete config specifying changes.
    May contain ``init`` fields (``transfer`` or ``random``).

**Transition Spec (T)** - A complete config WITH ``init`` fields.
    The result of applying surgery to a state. Describes both target
    structure and weight initialization mode.

Algebraic Structure
===================

**Monoid**: Partial surgeries compose via deep merge::

    compose_configs : P × P → P

**Action**: Surgeries act on states to produce transition specs::

    compose_configs : S × P → T
    compose_configs : T × P → T

**Extraction**: Strip init to get a state::

    strip_init_fields : T → S

**Planning**: Build weight transformation from source state + transition spec::

    plan_surgery : S × T → Plan

The ``init`` Field
==================

The ``init`` field in surgeries specifies weight initialization:

- ``init: transfer`` → transfer/convert weights from source
- ``init: random`` → randomly initialize weights

This field is preserved through ``compose_configs`` so ``plan_surgery`` can read it.
Use ``strip_init_fields`` before saving configs to disk.

Typical Usage
=============

::

    from fast_llm_external_models.apriel2.conversion import (
        compose_configs,
        plan_surgery,
        strip_init_fields,
        execute,
    )

    # Load source state
    source_state = load_config(...)  # S

    # Apply surgery
    surgery = {"decoder": {"block": {"mixer": {"type": "gdn", "init": "random"}}}}  # P
    transition = compose_configs(source_state, surgery)  # T

    # Build and execute plan
    plan = plan_surgery(source_state, transition)
    weights = execute(plan, source_weights, seed=42)

    # Save (strip init first)
    target_state = strip_init_fields(transition)  # S
    save_config(target_state)

For chained surgeries::

    current_state = source_state  # S
    current_plan = identity_plan

    for surgery in surgery_chain:  # each P
        transition = compose_configs(current_state, surgery)  # T
        plan = plan_surgery(current_state, transition)
        current_plan = compose(current_plan, plan)
        current_state = strip_init_fields(transition)  # S  <- IMPORTANT!

**Note**: The ``strip_init_fields`` call is critical. It ensures that ``init: random``
applies only to the surgery that introduces a component. Without stripping, subsequent
surgeries would re-randomize existing components. See ``config.py`` docstring for details.

Key Design Decisions
====================

**Declarative Plans**
    Plans are data (expressions), not functions. Enables inspection,
    serialization, and composition via substitution.

**Inheritance Semantics**
    When S × P → T, unspecified fields inherit from source.
    Cross-type derivation maps geometry (attention.heads → gdn.value_heads).

**Additive vs Replacement Surgeries**
    Additive surgeries (no ``type:`` declaration) satisfy the action law.
    Replacement surgeries (explicit ``type:``) use last-write-wins.

Module Structure
================

- ``config.py`` - Config composition (compose_configs, strip_init_fields)
- ``converters.py`` - Plan builders (plan_surgery, plan_mil_attention_to_mamba)
- ``expr.py`` - Expression types (Ref, Slice, Concat, Init, ExprPlan)
- ``executor.py`` - Plan execution (StreamingExecutor, execute)
- ``io.py`` - Streaming I/O (SafetensorLoader, ShardedSafetensorWriter)
"""

# Core types and plan operations
from fast_llm_external_models.apriel2.conversion.expr import (
    Concat,
    EvalKwargs,
    Expr,
    ExprAdapter,
    ExprPlan,
    Init,
    Ref,
    Reshape,
    Slice,
    W,
    compose,
    full_slice,
    fuse,
    make_slice,
    merge,
    slice_spec,
    substitute,
)

# Execution
from fast_llm_external_models.apriel2.conversion.executor import (
    MAX_SEED,
    StreamingExecutor,
    execute,
)

# I/O utilities
from fast_llm_external_models.apriel2.conversion.io import (
    DEFAULT_MAX_SHARD_SIZE,
    SafetensorLoader,
    ShardedSafetensorWriter,
)

# Plan builders (generic)
from fast_llm_external_models.apriel2.conversion.converters import (
    plan_mil_attention_to_mamba,
    plan_dil_attention_to_gdn,
    plan_kil_attention_to_kda,
    plan_surgery,
)

# Config composition
from fast_llm_external_models.apriel2.conversion.config import compose_configs, strip_init_fields

# Source-specific converters
from fast_llm_external_models.apriel2.conversion.llava import (
    convert_config as convert_llava_config,
    plan_llava_to_apriel2,
)

# Rendering (optional, imported lazily by ExprPlan.render_tree)
# from fast_llm_external_models.apriel2.conversion.render import render_tree

__all__ = [
    # Core types
    "W",
    "EvalKwargs",
    "Ref",
    "Slice",
    "Concat",
    "Init",
    "Reshape",
    "Expr",
    "ExprAdapter",
    "ExprPlan",
    # Slice helpers
    "slice_spec",
    "full_slice",
    "make_slice",
    # Expression utilities
    "substitute",
    "fuse",
    # Plan operations
    "compose",
    "merge",
    # Execution
    "MAX_SEED",
    "StreamingExecutor",
    "execute",
    # I/O
    "DEFAULT_MAX_SHARD_SIZE",
    "SafetensorLoader",
    "ShardedSafetensorWriter",
    # Plan builders (generic)
    "plan_surgery",
    "plan_mil_attention_to_mamba",
    "plan_dil_attention_to_gdn",
    "plan_kil_attention_to_kda",
    # Config composition
    "compose_configs",
    "strip_init_fields",
    # Source-specific converters
    "convert_llava_config",
    "plan_llava_to_apriel2",
]
