"""Weight conversion DSL for Apriel2 models.

This package provides a declarative approach to weight transformations:
- Expression types define how target tensors are computed from sources
- Plans map target keys to expressions
- Composition via | operator chains plans together
- Streaming execution for memory-efficient conversion

Example usage:
    from fast_llm_external_models.apriel2.conversion import (
        plan_llava_to_apriel2,
        plan_surgery,
        compose,
        StreamingExecutor,
        SafetensorLoader,
        ShardedSafetensorWriter,
    )

    # Build plans
    conversion_plan = plan_llava_to_apriel2(llava_config)
    surgery_plan = plan_surgery(apriel2_config, target_config)
    full_plan = conversion_plan | surgery_plan

    # Execute with streaming I/O
    with SafetensorLoader(source_files) as loader:
        executor = StreamingExecutor(full_plan, loader)
        with ShardedSafetensorWriter(output_dir) as writer:
            for key, tensor in executor.execute(seed=0):
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
    # Source-specific converters
    "convert_llava_config",
    "plan_llava_to_apriel2",
]
