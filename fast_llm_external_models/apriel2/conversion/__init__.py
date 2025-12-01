"""Weight conversion system for Apriel2 models.

Architecture Overview
=====================

This package implements a declarative weight transformation system with two
orthogonal concerns:

1. **Config Composition** - Structural transformations of model configs
2. **Plan Building & Execution** - Weight transformations between configs

These concerns are intentionally separated:
- Config composition determines WHAT the target architecture looks like
- Plan building determines HOW weights are transformed to match
- The `init` field bridges them: it's config metadata consumed by the plan builder

Key Design Decisions
====================

**Declarative Plans**
    Plans are DATA (JSON-serializable expressions), not functions. This enables:
    - Inspection and debugging of transformations
    - Serialization for distributed execution
    - Composition via substitution rather than function composition

**Separation of Config and Weights**
    The `init` field in surgery specs controls weight handling (transfer vs random)
    but does NOT affect config composition. Config composition is purely structural.
    After composition, `init` fields are stripped from complete configs.

**Composition Semantics**
    Surgery specs use declarative (merge) composition, not operational (function)
    composition. For "additive" surgeries (modifying existing structure), the
    monoid action law holds. For "replacement" surgeries (defining complete new
    structure), sequential application differs from composed application by design.

**Cross-Type Derivation**
    When converting between mixer types (e.g., attention → mamba), geometric
    parameters are derived where possible:
    - attention.heads → mamba dimensions (MIL conversion)
    - attention.heads → gated_delta_net heads (DIL conversion)

Module Structure
================

- `config.py` - Config composition (compose_configs, apply_surgery)
- `converters.py` - Plan builders (plan_surgery, plan_mil_attention_to_mamba, etc.)
- `expr.py` - Expression types and plan class (Ref, Slice, Concat, Init, ExprPlan)
- `executor.py` - Plan execution (StreamingExecutor, execute)
- `io.py` - Streaming I/O (SafetensorLoader, ShardedSafetensorWriter)
- `llava/` - Source-specific converter for Llava → Apriel2

Example Usage
=============

    from fast_llm_external_models.apriel2.conversion import (
        compose_configs,
        plan_surgery,
        execute,
    )

    # 1. Compose configs to get target architecture
    target_config = compose_configs(source_config, surgery_spec)

    # 2. Build plan for weight transformation
    plan = plan_surgery(source_config, surgery_spec)

    # 3. Execute plan to transform weights
    target_weights = execute(plan, source_weights, seed=42)

For streaming I/O with large models:

    from fast_llm_external_models.apriel2.conversion import (
        StreamingExecutor,
        SafetensorLoader,
        ShardedSafetensorWriter,
    )

    with SafetensorLoader(source_files) as loader:
        executor = StreamingExecutor(plan, loader)
        with ShardedSafetensorWriter(output_dir) as writer:
            for key, tensor in executor.execute(seed=42):
                writer.add(key, tensor)
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
    plan_attention_to_gated_delta_net,
    plan_mil_attention_to_mamba,
    plan_surgery,
)

# Config composition
from fast_llm_external_models.apriel2.conversion.config import compose_configs

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
    "plan_attention_to_gated_delta_net",
    # Config composition
    "compose_configs",
    # Source-specific converters
    "convert_llava_config",
    "plan_llava_to_apriel2",
]
