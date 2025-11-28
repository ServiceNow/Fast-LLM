"""Expression-based plan system for weight transformations.

This module implements a declarative approach where each target tensor is defined
as an expression over source tensors. This enables:
- Composition via expression substitution
- Fusion via tree rewriting
- Streaming execution with ref-counting for memory efficiency

Core expression types:
- Ref(key): Reference to a source tensor
- Slice(expr, slices): Slice an expression
- Concat(exprs, dim): Concatenate expressions along a dimension
- Init(shape, init_type): Random/constant initialization
- Reshape(expr, shape): Reshape an expression

Weight path utilities:
- WeightPath: Builder for structured weight key paths
"""

from __future__ import annotations

import hashlib
import json
import math
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import torch
from torch import Tensor


# =============================================================================
# Weight Path Builder
# =============================================================================


class W(str):
    """Weight path that IS a string, composable via /.

    Usage:
        mixer = W("model", "decoder", "blocks", 0, "mixer")
        q = mixer / "self_attn" / "q_proj" / "weight"
        # Result: "model.decoder.blocks.0.mixer.self_attn.q_proj.weight"

        # Use directly - it's already a string!
        plan.define(q, Ref(source_q))
    """

    def __new__(cls, *parts) -> "W":
        # Join parts, stripping any leading/trailing dots from each
        cleaned = []
        for p in parts:
            if p is None:
                continue
            s = str(p).strip(".")
            if s:
                cleaned.append(s)
        return super().__new__(cls, ".".join(cleaned))

    def __truediv__(self, other) -> "W":
        """Join with another path segment via /."""
        if isinstance(other, (list, tuple)):
            return W(self, *other)
        return W(self, other)

    def __rtruediv__(self, other) -> "W":
        """Support other / W."""
        return W(other, self)


# =============================================================================
# Expression Types
# =============================================================================


class Expr(ABC):
    """Base class for all expressions."""

    @abstractmethod
    def find_refs(self) -> set[str]:
        """Find all source references in this expression."""
        pass

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        pass

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Expr:
        """Deserialize from dictionary."""
        expr_type = d.get("type")
        if expr_type == "ref":
            return Ref.from_dict(d)
        elif expr_type == "slice":
            return Slice.from_dict(d)
        elif expr_type == "concat":
            return Concat.from_dict(d)
        elif expr_type == "init":
            return Init.from_dict(d)
        elif expr_type == "reshape":
            return Reshape.from_dict(d)
        else:
            raise ValueError(f"Unknown expression type: {expr_type}")

    @abstractmethod
    def evaluate(
        self,
        sources: dict[str, Tensor],
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        target_key: str | None = None,
    ) -> Tensor:
        """Evaluate this expression given source tensors."""
        pass


@dataclass(frozen=True)
class Ref(Expr):
    """Reference to a source tensor by key."""

    key: str

    def find_refs(self) -> set[str]:
        return {self.key}

    def to_dict(self) -> dict[str, Any]:
        return {"type": "ref", "key": self.key}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Ref:
        return cls(key=d["key"])

    def evaluate(
        self,
        sources: dict[str, Tensor],
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        target_key: str | None = None,
    ) -> Tensor:
        if self.key not in sources:
            raise KeyError(f"Source key not found: {self.key}")
        return sources[self.key].clone().to(device=device, dtype=dtype)

    def __repr__(self) -> str:
        return f"Ref({self.key!r})"


@dataclass(frozen=True)
class Slice(Expr):
    """Slice an expression along dimensions.

    slices is a tuple of (start, stop, step) tuples, one per dimension.
    None values mean "use default" (0, size, 1).
    """

    expr: Expr
    slices: tuple[tuple[int | None, int | None, int | None], ...]

    def find_refs(self) -> set[str]:
        return self.expr.find_refs()

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "slice",
            "expr": self.expr.to_dict(),
            "slices": self.slices,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Slice:
        return cls(
            expr=Expr.from_dict(d["expr"]),
            slices=tuple(tuple(s) for s in d["slices"]),
        )

    def evaluate(
        self,
        sources: dict[str, Tensor],
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        target_key: str | None = None,
    ) -> Tensor:
        tensor = self.expr.evaluate(sources, device, dtype, target_key)
        slice_objs = tuple(
            slice(s[0], s[1], s[2]) for s in self.slices
        )
        return tensor[slice_objs].clone()

    def __repr__(self) -> str:
        slice_strs = []
        for s in self.slices:
            start, stop, step = s
            if start is None and stop is None and step is None:
                slice_strs.append(":")
            elif step is None or step == 1:
                slice_strs.append(f"{start or ''}:{stop or ''}")
            else:
                slice_strs.append(f"{start or ''}:{stop or ''}:{step}")
        return f"{self.expr}[{', '.join(slice_strs)}]"


@dataclass(frozen=True)
class Concat(Expr):
    """Concatenate multiple expressions along a dimension."""

    exprs: tuple[Expr, ...]
    dim: int = 0

    def find_refs(self) -> set[str]:
        refs = set()
        for expr in self.exprs:
            refs.update(expr.find_refs())
        return refs

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "concat",
            "exprs": [e.to_dict() for e in self.exprs],
            "dim": self.dim,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Concat:
        return cls(
            exprs=tuple(Expr.from_dict(e) for e in d["exprs"]),
            dim=d["dim"],
        )

    def evaluate(
        self,
        sources: dict[str, Tensor],
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        target_key: str | None = None,
    ) -> Tensor:
        tensors = [e.evaluate(sources, device, dtype, target_key) for e in self.exprs]
        return torch.cat(tensors, dim=self.dim)

    def __repr__(self) -> str:
        exprs_str = ", ".join(repr(e) for e in self.exprs)
        return f"Concat([{exprs_str}], dim={self.dim})"


@dataclass(frozen=True)
class Init(Expr):
    """Initialize a tensor with random or constant values.

    init_type can be:
    - "zeros": All zeros
    - "ones": All ones
    - "kaiming": Kaiming uniform initialization
    - "normal": Normal distribution with std=0.02
    - "s4d": S4D real initialization for Mamba A_log (log of 1..d_state expanded)
    - "dt_bias": Special dt_proj.bias initialization (log-space from dt_min/dt_max)
    """

    shape: tuple[int, ...]
    init_type: str = "kaiming"
    init_params: dict[str, Any] | None = None  # For special inits

    def find_refs(self) -> set[str]:
        return set()  # Init has no dependencies

    def to_dict(self) -> dict[str, Any]:
        d = {
            "type": "init",
            "shape": list(self.shape),
            "init_type": self.init_type,
        }
        if self.init_params:
            d["init_params"] = self.init_params
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Init:
        return cls(
            shape=tuple(d["shape"]),
            init_type=d.get("init_type", "kaiming"),
            init_params=d.get("init_params"),
        )

    def evaluate(
        self,
        sources: dict[str, Tensor],
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        target_key: str | None = None,
    ) -> Tensor:
        # Deterministic seeding based on target key for reproducibility
        if target_key:
            seed = int(hashlib.md5(target_key.encode()).hexdigest()[:8], 16)
            gen = torch.Generator(device=device).manual_seed(seed)
        else:
            gen = None

        if self.init_type == "zeros":
            return torch.zeros(self.shape, device=device, dtype=dtype)

        elif self.init_type == "ones":
            return torch.ones(self.shape, device=device, dtype=dtype)

        elif self.init_type == "kaiming":
            tensor = torch.empty(self.shape, device=device, dtype=dtype)
            if len(self.shape) >= 2:
                # Kaiming uniform for weight matrices
                fan_in = self.shape[1]
                bound = math.sqrt(1.0 / fan_in)
                tensor.uniform_(-bound, bound, generator=gen)
            else:
                # For 1D, use normal init
                tensor.normal_(0, 0.02, generator=gen)
            return tensor

        elif self.init_type == "normal":
            tensor = torch.empty(self.shape, device=device, dtype=dtype)
            tensor.normal_(0, 0.02, generator=gen)
            return tensor

        elif self.init_type == "s4d":
            # S4D real initialization for Mamba A_log
            # Shape should be (d_inner, d_state)
            if len(self.shape) != 2:
                raise ValueError(f"S4D init requires 2D shape, got {self.shape}")
            d_inner, d_state = self.shape
            A = torch.arange(1, d_state + 1, device=device, dtype=torch.float32)
            A = A.unsqueeze(0).expand(d_inner, -1).contiguous()
            return torch.log(A).to(dtype)

        elif self.init_type == "dt_bias":
            # Special dt_proj.bias initialization
            # Log-space initialization from dt_min/dt_max for good training dynamics
            params = self.init_params or {}
            dt_min = params.get("dt_min", 0.001)
            dt_max = params.get("dt_max", 0.1)
            dt_init_floor = params.get("dt_init_floor", 1e-4)

            if len(self.shape) != 1:
                raise ValueError(f"dt_bias init requires 1D shape, got {self.shape}")
            d_inner = self.shape[0]

            # Random dt values in [dt_min, dt_max] log-space
            tensor = torch.empty(d_inner, device=device, dtype=dtype)
            tensor.uniform_(generator=gen)
            dt = torch.exp(
                tensor * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
            )
            dt = dt.clamp(min=dt_init_floor)
            # Inverse softplus to get the bias that produces these dt values
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            return inv_dt

        else:
            raise ValueError(f"Unknown init type: {self.init_type}")

    def __repr__(self) -> str:
        if self.init_params:
            return f"Init({self.shape}, {self.init_type!r}, {self.init_params!r})"
        return f"Init({self.shape}, {self.init_type!r})"


@dataclass(frozen=True)
class Reshape(Expr):
    """Reshape an expression to a new shape."""

    expr: Expr
    shape: tuple[int, ...]

    def find_refs(self) -> set[str]:
        return self.expr.find_refs()

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "reshape",
            "expr": self.expr.to_dict(),
            "shape": list(self.shape),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Reshape:
        return cls(
            expr=Expr.from_dict(d["expr"]),
            shape=tuple(d["shape"]),
        )

    def evaluate(
        self,
        sources: dict[str, Tensor],
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        target_key: str | None = None,
    ) -> Tensor:
        tensor = self.expr.evaluate(sources, device, dtype, target_key)
        return tensor.reshape(self.shape)

    def __repr__(self) -> str:
        return f"Reshape({self.expr}, {self.shape})"


# =============================================================================
# Slice Helpers
# =============================================================================


def slice_spec(
    start: int | None = None,
    stop: int | None = None,
    step: int | None = None,
) -> tuple[int | None, int | None, int | None]:
    """Create a slice specification tuple."""
    return (start, stop, step)


def full_slice() -> tuple[int | None, int | None, int | None]:
    """Create a full slice (equivalent to :)."""
    return (None, None, None)


def make_slice(expr: Expr, dim_slices: list[tuple[int | None, int | None, int | None]]) -> Slice:
    """Convenience function to create a Slice expression."""
    return Slice(expr, tuple(dim_slices))


# =============================================================================
# Expression Utilities
# =============================================================================


def substitute(expr: Expr, bindings: dict[str, Expr]) -> Expr:
    """Substitute Ref expressions with their bindings.

    This is the core of composition: replace Ref(x) with the expression
    that produces x in the source plan.

    Args:
        expr: Expression to transform.
        bindings: Map from ref keys to their producing expressions.

    Returns:
        New expression with substitutions applied.
    """
    if isinstance(expr, Ref):
        if expr.key in bindings:
            return bindings[expr.key]
        return expr  # Keep as-is (source passthrough)

    elif isinstance(expr, Slice):
        return Slice(substitute(expr.expr, bindings), expr.slices)

    elif isinstance(expr, Concat):
        return Concat(
            tuple(substitute(e, bindings) for e in expr.exprs),
            expr.dim,
        )

    elif isinstance(expr, Init):
        return expr  # Init has no refs

    elif isinstance(expr, Reshape):
        return Reshape(substitute(expr.expr, bindings), expr.shape)

    else:
        raise TypeError(f"Unknown expression type: {type(expr)}")


def fuse(expr: Expr) -> Expr:
    """Apply fusion/optimization rules to an expression.

    Current rules:
    - Flatten nested Concat with same dim
    - (Future: compose nested slices)
    """
    if isinstance(expr, Ref):
        return expr

    elif isinstance(expr, Slice):
        inner = fuse(expr.expr)
        # Future: compose Slice(Slice(x, s1), s2) -> Slice(x, compose(s1, s2))
        return Slice(inner, expr.slices)

    elif isinstance(expr, Concat):
        # Recursively fuse children
        fused_children = [fuse(e) for e in expr.exprs]

        # Flatten nested Concat with same dim
        flattened = []
        for child in fused_children:
            if isinstance(child, Concat) and child.dim == expr.dim:
                flattened.extend(child.exprs)
            else:
                flattened.append(child)

        return Concat(tuple(flattened), expr.dim)

    elif isinstance(expr, Init):
        return expr

    elif isinstance(expr, Reshape):
        inner = fuse(expr.expr)
        # Future: Reshape(Reshape(x, s1), s2) -> Reshape(x, s2)
        if isinstance(inner, Reshape):
            return Reshape(inner.expr, expr.shape)
        return Reshape(inner, expr.shape)

    else:
        raise TypeError(f"Unknown expression type: {type(expr)}")


# =============================================================================
# Plan Class
# =============================================================================


@dataclass
class ExprPlan:
    """A plan mapping target keys to expressions over sources.

    The plan is declarative: each target is defined as an expression.
    Composition is achieved by substituting Ref expressions.
    """

    mappings: dict[str, Expr] = field(default_factory=dict)
    source_format: str = ""
    target_format: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.mappings)

    def __iter__(self) -> Iterator[tuple[str, Expr]]:
        return iter(self.mappings.items())

    def __getitem__(self, key: str) -> Expr:
        return self.mappings[key]

    def __setitem__(self, key: str, expr: Expr) -> None:
        self.mappings[key] = expr

    def __contains__(self, key: str) -> bool:
        return key in self.mappings

    def define(self, target_key: str, expr: Expr) -> None:
        """Define a target key as an expression."""
        self.mappings[target_key] = expr

    def source_keys(self) -> set[str]:
        """Get all source keys referenced by this plan."""
        refs = set()
        for expr in self.mappings.values():
            refs.update(expr.find_refs())
        return refs

    def target_keys(self) -> set[str]:
        """Get all target keys produced by this plan."""
        return set(self.mappings.keys())

    def summary(self) -> dict[str, Any]:
        """Get a summary of this plan."""
        expr_counts: dict[str, int] = defaultdict(int)
        for expr in self.mappings.values():
            expr_counts[type(expr).__name__] += 1

        return {
            "source_format": self.source_format,
            "target_format": self.target_format,
            "num_targets": len(self.mappings),
            "num_source_refs": len(self.source_keys()),
            "expr_counts": dict(expr_counts),
            "metadata": self.metadata,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize plan to dictionary."""
        return {
            "source_format": self.source_format,
            "target_format": self.target_format,
            "mappings": {k: v.to_dict() for k, v in self.mappings.items()},
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExprPlan:
        """Deserialize plan from dictionary."""
        return cls(
            mappings={k: Expr.from_dict(v) for k, v in d.get("mappings", {}).items()},
            source_format=d.get("source_format", ""),
            target_format=d.get("target_format", ""),
            metadata=d.get("metadata", {}),
        )

    def fuse(self) -> ExprPlan:
        """Return a new plan with fusion optimizations applied."""
        return ExprPlan(
            mappings={k: fuse(v) for k, v in self.mappings.items()},
            source_format=self.source_format,
            target_format=self.target_format,
            metadata=self.metadata,
        )


# =============================================================================
# Plan Composition
# =============================================================================


def compose(plan1: ExprPlan, plan2: ExprPlan) -> ExprPlan:
    """Compose two plans: plan1 (A→B) + plan2 (B→C) = composed (A→C).

    For each target in plan2, substitute its Ref expressions with
    the corresponding expressions from plan1.

    Args:
        plan1: First plan (source format → intermediate format).
        plan2: Second plan (intermediate format → target format).

    Returns:
        Composed plan (source format → target format).
    """
    # Build bindings from plan1's mappings
    bindings = plan1.mappings

    # Substitute in plan2
    composed_mappings = {}
    for target_key, expr in plan2.mappings.items():
        composed_mappings[target_key] = substitute(expr, bindings)

    composed = ExprPlan(
        mappings=composed_mappings,
        source_format=plan1.source_format,
        target_format=plan2.target_format,
        metadata={
            "composed_from": [plan1.source_format, plan1.target_format, plan2.target_format],
            "plan1_metadata": plan1.metadata,
            "plan2_metadata": plan2.metadata,
        },
    )

    # Apply fusion optimizations
    return composed.fuse()


# =============================================================================
# Streaming Execution
# =============================================================================


class StreamingExecutor:
    """Execute a plan with streaming and ref-counting for memory efficiency.

    This executor:
    1. Analyzes dependencies to determine evaluation order
    2. Loads source tensors on-demand
    3. Releases source tensors when no longer needed (ref-counting)
    4. Yields (target_key, tensor) pairs as they're computed
    """

    def __init__(
        self,
        plan: ExprPlan,
        source_loader: Callable[[str], Tensor],
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.plan = plan
        self.source_loader = source_loader
        self.device = device
        self.dtype = dtype

        # Analyze dependencies
        self._analyze_dependencies()

    def _analyze_dependencies(self) -> None:
        """Analyze source dependencies and compute ref counts."""
        # Count how many times each source is referenced
        self.ref_counts: dict[str, int] = defaultdict(int)

        for target_key, expr in self.plan.mappings.items():
            for ref_key in expr.find_refs():
                self.ref_counts[ref_key] += 1

        # Track which sources are needed for which targets
        self.target_deps: dict[str, set[str]] = {}
        for target_key, expr in self.plan.mappings.items():
            self.target_deps[target_key] = expr.find_refs()

    def _topological_order(self) -> list[str]:
        """Compute evaluation order for targets.

        For now, use a simple heuristic: evaluate targets that share
        sources together to maximize cache reuse.

        Future: more sophisticated ordering based on source loading order.
        """
        # Group targets by their first source ref (if any)
        by_first_ref: dict[str, list[str]] = defaultdict(list)
        no_refs: list[str] = []

        for target_key in self.plan.mappings:
            deps = self.target_deps[target_key]
            if deps:
                first_ref = min(deps)  # Deterministic ordering
                by_first_ref[first_ref].append(target_key)
            else:
                no_refs.append(target_key)

        # Order: first targets with no refs, then grouped by first ref
        order = sorted(no_refs)
        for ref_key in sorted(by_first_ref.keys()):
            order.extend(sorted(by_first_ref[ref_key]))

        return order

    def execute(self) -> Iterator[tuple[str, Tensor]]:
        """Execute the plan, yielding (target_key, tensor) pairs.

        Sources are loaded on-demand and released when no longer needed.
        """
        # Cache for loaded sources
        cache: dict[str, Tensor] = {}

        # Remaining ref counts (decremented as we use sources)
        remaining_refs = dict(self.ref_counts)

        def get_source(key: str) -> Tensor:
            """Load a source tensor, caching it."""
            if key not in cache:
                cache[key] = self.source_loader(key)
            return cache[key]

        def release_refs(refs: set[str]) -> None:
            """Decrement ref counts and release unused sources."""
            for ref_key in refs:
                remaining_refs[ref_key] -= 1
                if remaining_refs[ref_key] == 0 and ref_key in cache:
                    del cache[ref_key]

        # Process targets in order
        for target_key in self._topological_order():
            expr = self.plan.mappings[target_key]
            deps = self.target_deps[target_key]

            # Load needed sources
            sources = {key: get_source(key) for key in deps}

            # Evaluate expression
            result = expr.evaluate(sources, self.device, self.dtype, target_key)

            # Release refs that are no longer needed
            release_refs(deps)

            yield target_key, result

        # Verify all sources were released
        assert len(cache) == 0, f"Memory leak: {list(cache.keys())} not released"

    def execute_all(self) -> dict[str, Tensor]:
        """Execute the plan and return all results as a dict."""
        return dict(self.execute())


def execute(
    plan: ExprPlan,
    source_weights: dict[str, Tensor],
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> dict[str, Tensor]:
    """Execute a plan with in-memory sources.

    This is a convenience function for when all sources are already loaded.
    For streaming, use StreamingExecutor directly.
    """
    def loader(key: str) -> Tensor:
        if key not in source_weights:
            raise KeyError(f"Source key not found: {key}")
        return source_weights[key]

    executor = StreamingExecutor(plan, loader, device, dtype)
    return executor.execute_all()


# =============================================================================
# Plan Builders
# =============================================================================


def plan_llava_to_apriel2(llava_config: dict) -> ExprPlan:
    """Build an expression plan for Llava to Apriel2 conversion.

    This is a pure mapping (all Ref expressions) since Llava→Apriel2
    is just renaming keys.
    """
    plan = ExprPlan(source_format="llava", target_format="apriel2")

    num_text_layers = llava_config.get("text_config", {}).get("num_hidden_layers", 0)
    num_vision_layers = llava_config.get("vision_config", {}).get("num_hidden_layers", 0)

    # Static mappings (must match convert_from_llava._STATIC_WEIGHT_MAP)
    static_mappings = [
        (W("language_model", "model", "embed_tokens", "weight"),
         W("model", "embed_tokens", "weight")),
        (W("language_model", "lm_head", "weight"),
         W("lm_head", "weight")),
        (W("language_model", "model", "norm", "weight"),
         W("model", "norm", "weight")),
        (W("vision_tower", "patch_conv", "weight"),
         W("model", "vision_encoder", "patch_convolution", "conv", "weight")),
        (W("vision_tower", "ln_pre", "weight"),
         W("model", "vision_encoder", "patch_convolution", "norm", "weight")),
        (W("multi_modal_projector", "linear_1", "weight"),
         W("model", "vision_encoder", "adapter", "linear_1", "weight")),
        (W("multi_modal_projector", "linear_1", "bias"),
         W("model", "vision_encoder", "adapter", "linear_1", "bias")),
        (W("multi_modal_projector", "linear_2", "weight"),
         W("model", "vision_encoder", "adapter", "linear_2", "weight")),
        (W("multi_modal_projector", "linear_2", "bias"),
         W("model", "vision_encoder", "adapter", "linear_2", "bias")),
    ]

    for src, tgt in static_mappings:
        plan.define(tgt, Ref(src))

    # Text decoder layers
    for layer in range(num_text_layers):
        llava_layer = W("language_model", "model", "layers", layer)
        apriel_layer = W("model", "decoder", "blocks", layer)

        # Attention projections
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            src = llava_layer / "self_attn" / proj / "weight"
            tgt = apriel_layer / "mixer" / "self_attn" / proj / "weight"
            plan.define(tgt, Ref(src))

        # MLP projections
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            src = llava_layer / "mlp" / proj / "weight"
            tgt = apriel_layer / "mlp" / proj / "weight"
            plan.define(tgt, Ref(src))

        # Layer norms
        plan.define(
            apriel_layer / "input_layernorm" / "weight",
            Ref(llava_layer / "input_layernorm" / "weight"),
        )
        plan.define(
            apriel_layer / "post_attention_layernorm" / "weight",
            Ref(llava_layer / "post_attention_layernorm" / "weight"),
        )

    # Vision encoder layers
    for layer in range(num_vision_layers):
        llava_layer = W("vision_tower", "transformer", "layers", layer)
        apriel_layer = W("model", "vision_encoder", "encoder", "blocks", layer)

        # Attention projections
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            src = llava_layer / "attention" / proj / "weight"
            tgt = apriel_layer / "mixer" / "self_attn" / proj / "weight"
            plan.define(tgt, Ref(src))

        # MLP projections (llava uses feed_forward, apriel uses mlp)
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            src = llava_layer / "feed_forward" / proj / "weight"
            tgt = apriel_layer / "mlp" / proj / "weight"
            plan.define(tgt, Ref(src))

        # Layer norms (different naming)
        plan.define(
            apriel_layer / "input_layernorm" / "weight",
            Ref(llava_layer / "attention_norm" / "weight"),
        )
        plan.define(
            apriel_layer / "post_attention_layernorm" / "weight",
            Ref(llava_layer / "ffn_norm" / "weight"),
        )

    plan.metadata = {
        "num_text_layers": num_text_layers,
        "num_vision_layers": num_vision_layers,
    }

    return plan


def plan_mil_attention_to_mamba(
    layer_idx: int,
    hidden_size: int,
    d_inner: int,
    d_xb: int,
    dt_rank: int,
    d_state: int,
    d_conv: int = 4,
    repeat_kv_before_conv: bool = True,
    conv_bias: bool = True,
    dt_bias: bool = True,
    dt_min: float = 0.001,
    dt_max: float = 0.1,
    source_prefix: W | str = "",
    target_prefix: W | str = "",
) -> dict[str, Expr]:
    """Build MIL (Mamba Initialization from LLM) expressions for one layer.

    MIL maps attention projections to Mamba's composite in_proj:
    - Q -> C (readout)
    - K -> B (input-dependent state transition)
    - V -> x (input)
    - z stays random
    - O -> out_proj

    Args:
        layer_idx: Layer index.
        hidden_size: Model hidden size.
        d_inner: Mamba inner dimension (usually 2 * hidden_size).
        d_xb: Mamba x/B dimension.
        dt_rank: Mamba dt rank.
        d_state: Mamba state dimension.
        d_conv: Convolution kernel size (default 4).
        repeat_kv_before_conv: If True, conv has d_inner channels; else d_xb.
        conv_bias: Whether conv1d has bias (default True).
        dt_bias: Whether dt_proj has bias (default True).
        dt_min: Minimum dt value for bias init (default 0.001).
        dt_max: Maximum dt value for bias init (default 0.1).
        source_prefix: Prefix for source attention keys (e.g. layer.mixer.self_attn).
        target_prefix: Prefix for target mamba keys (e.g. layer.mixer).

    Returns:
        Dict mapping target keys to expressions.
    """
    # Convert to W for consistent path handling
    if not source_prefix:
        src = W("model", "decoder", "blocks", layer_idx, "mixer", "self_attn")
    else:
        src = W(source_prefix)

    if not target_prefix:
        tgt = W("model", "decoder", "blocks", layer_idx, "mixer")
    else:
        tgt = W(target_prefix)

    # in_proj layout: [z, x, B, C] with sizes [d_inner, d_xb, d_xb, d_inner]
    # Total: 2*d_inner + 2*d_xb
    in_proj_expr = Concat((
        Init((d_inner, hidden_size), "kaiming"),  # z: random
        Slice(Ref(src / "v_proj" / "weight"), ((0, d_xb, None), (None, None, None))),  # x <- V
        Slice(Ref(src / "k_proj" / "weight"), ((0, d_xb, None), (None, None, None))),  # B <- K
        Slice(Ref(src / "q_proj" / "weight"), ((0, d_inner, None), (None, None, None))),  # C <- Q
    ), dim=0)

    # Conv1d channels depend on repeat_kv_before_conv
    conv_channels = d_inner if repeat_kv_before_conv else d_xb

    result = {
        # Core projections
        tgt / "in_proj" / "weight": in_proj_expr,
        tgt / "out_proj" / "weight": Ref(src / "o_proj" / "weight"),
        # dt projections
        tgt / "dt_in_proj" / "weight": Init((dt_rank, hidden_size), "kaiming"),
        tgt / "dt_proj" / "weight": Init((d_inner, dt_rank), "kaiming"),
        # Conv1d
        tgt / "conv1d" / "weight": Init((conv_channels, 1, d_conv), "kaiming"),
        # SSM parameters
        tgt / "A_log": Init((d_inner, d_state), "s4d"),  # S4D initialization
        tgt / "D": Init((d_inner,), "ones"),
    }

    # Optional biases
    if dt_bias:
        result[tgt / "dt_proj" / "bias"] = Init(
            (d_inner,), "dt_bias",
            init_params={"dt_min": dt_min, "dt_max": dt_max}
        )

    if conv_bias:
        result[tgt / "conv1d" / "bias"] = Init((conv_channels,), "zeros")

    return result


def _plan_non_decoder_weights(plan: ExprPlan, config: dict) -> None:
    """Add passthrough mappings for non-decoder weights.

    These weights are typically unchanged during surgery:
    - Embeddings
    - LM head
    - Final norm
    - Vision encoder (if present)
    """
    # Core model weights (passthrough as identity)
    embed = W("model", "embed_tokens", "weight")
    plan.define(embed, Ref(embed))

    head = W("lm_head", "weight")
    plan.define(head, Ref(head))

    norm = W("model", "norm", "weight")
    plan.define(norm, Ref(norm))

    # Vision encoder (if present)
    if "vision_encoder" in config:
        vision_config = config["vision_encoder"]
        vision = W("model", "vision_encoder")

        # Patch convolution
        patch_conv = vision / "patch_convolution" / "conv" / "weight"
        plan.define(patch_conv, Ref(patch_conv))

        patch_norm = vision / "patch_convolution" / "norm" / "weight"
        plan.define(patch_norm, Ref(patch_norm))

        # Vision encoder blocks
        encoder_config = vision_config.get("encoder", {})
        num_vision_layers = encoder_config.get("num_blocks", 0)

        for layer in range(num_vision_layers):
            block = vision / "encoder" / "blocks" / layer

            # Attention projections
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                key = block / "mixer" / "self_attn" / proj / "weight"
                plan.define(key, Ref(key))

            # MLP projections
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                key = block / "mlp" / proj / "weight"
                plan.define(key, Ref(key))

            # Layer norms
            for norm_name in ["input_layernorm", "post_attention_layernorm"]:
                key = block / norm_name / "weight"
                plan.define(key, Ref(key))

        # Adapter
        adapter_config = vision_config.get("adapter", {})
        add_biases = adapter_config.get("add_linear_biases", False)
        adapter = vision / "adapter"

        for proj in ["linear_1", "linear_2"]:
            weight_key = adapter / proj / "weight"
            plan.define(weight_key, Ref(weight_key))
            if add_biases:
                bias_key = adapter / proj / "bias"
                plan.define(bias_key, Ref(bias_key))


def _get_block_config(decoder_config: dict, layer_idx: int) -> dict:
    """Get block config for a specific layer index.

    Supports both 'fixed' (single block config) and 'pattern' (multiple block configs).
    """
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


def plan_surgery(
    source_config: dict,
    target_config: dict,
) -> ExprPlan:
    """Build an expression plan for Apriel2 surgery.

    This handles converting between different Apriel2 architectures,
    including attention → mamba (MIL) and stochastic mixer wrapping.
    """
    plan = ExprPlan(source_format="apriel2", target_format="apriel2")

    hidden_size = target_config.get("hidden_size", source_config.get("hidden_size"))

    source_decoder = source_config.get("decoder", {})
    target_decoder = target_config.get("decoder", {})

    num_source_layers = source_decoder.get("num_blocks", 0)
    num_target_layers = target_decoder.get("num_blocks", 0)

    # Non-decoder weights: passthrough as Ref(key)
    _plan_non_decoder_weights(plan, source_config)

    # Process decoder layers
    for target_layer_idx in range(num_target_layers):
        source_layer_idx = target_layer_idx % num_source_layers if num_source_layers > 0 else 0

        source_block = _get_block_config(source_decoder, source_layer_idx)
        target_block = _get_block_config(target_decoder, target_layer_idx)

        # Mixer conversion
        _plan_mixer(
            plan,
            target_layer_idx,
            source_layer_idx,
            source_block.get("mixer", {}),
            target_block.get("mixer", {}),
            hidden_size,
        )

        # MLP conversion (usually passthrough)
        _plan_mlp(
            plan,
            target_layer_idx,
            source_layer_idx,
            source_block.get("mlp", {}),
            target_block.get("mlp", {}),
            hidden_size,
        )

        # Norm conversion (usually passthrough)
        _plan_norms(
            plan,
            target_layer_idx,
            source_layer_idx,
            source_block,
            target_block,
            hidden_size,
        )

    return plan


def _plan_mixer(
    plan: ExprPlan,
    target_layer_idx: int,
    source_layer_idx: int,
    source_mixer: dict,
    target_mixer: dict,
    hidden_size: int,
) -> None:
    """Add mixer conversion expressions to plan."""
    source_type = source_mixer.get("type", "attention")
    target_type = target_mixer.get("type", "attention")

    source_layer = W("model", "decoder", "blocks", source_layer_idx)
    target_layer = W("model", "decoder", "blocks", target_layer_idx)

    # Unwrap stochastic source
    if source_type == "stochastic":
        main_name = source_mixer.get("main_mixer_name", "attention")
        actual_source = source_mixer.get("mixers", {}).get(main_name, {})
        actual_source_type = actual_source.get("type", "attention")
        source_mixer_base = source_layer / "mixer" / "mixers" / main_name
    else:
        actual_source = source_mixer
        actual_source_type = source_type
        source_mixer_base = source_layer / "mixer"

    # Add self_attn for attention types
    if actual_source_type in ("attention", "sliding_window"):
        source_prefix = source_mixer_base / "self_attn"
    else:
        source_prefix = source_mixer_base

    # Handle target
    if target_type == "stochastic":
        for sub_name, sub_config in target_mixer.get("mixers", {}).items():
            sub_type = sub_config.get("type", "attention")
            target_prefix = target_layer / "mixer" / "mixers" / sub_name

            _plan_mixer_conversion(
                plan, actual_source_type, sub_type,
                actual_source, sub_config,
                source_prefix, target_prefix, hidden_size,
            )
    else:
        target_prefix = target_layer / "mixer"
        _plan_mixer_conversion(
            plan, actual_source_type, target_type,
            actual_source, target_mixer,
            source_prefix, target_prefix, hidden_size,
        )


def _plan_mixer_conversion(
    plan: ExprPlan,
    source_type: str,
    target_type: str,
    source_config: dict,
    target_config: dict,
    source_prefix: W,
    target_prefix: W,
    hidden_size: int,
) -> None:
    """Add expressions for converting between mixer types.

    Note: source_prefix already includes self_attn for attention types.
    """
    if source_type in ("attention", "sliding_window") and target_type in ("attention", "sliding_window"):
        # Attention to attention: direct copy
        # Source prefix already includes self_attn, target needs it added
        target_attn = target_prefix / "self_attn"
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            plan.define(target_attn / proj / "weight", Ref(source_prefix / proj / "weight"))

    elif source_type in ("attention", "sliding_window") and target_type == "mamba":
        # Attention to Mamba: MIL conversion
        d_inner = target_config.get("d_inner", 2 * hidden_size)
        d_state = target_config.get("d_state", 128)
        dt_rank = target_config.get("dt_rank", hidden_size // 16)

        # d_xb should match k/v size from source if possible
        source_head_groups = source_config.get("head_groups", 8)
        source_head_size = source_config.get("head_size", hidden_size // 32)
        d_xb = target_config.get("d_xb", source_head_groups * source_head_size)

        # Extract Mamba config params
        d_conv = target_config.get("d_conv", 4)
        repeat_kv_before_conv = target_config.get("repeat_kv_before_conv", True)
        conv_bias = target_config.get("conv_bias", True)
        dt_bias = target_config.get("dt_proj_bias", True)
        dt_min = target_config.get("dt_min", 0.001)
        dt_max = target_config.get("dt_max", 0.1)

        mil_exprs = plan_mil_attention_to_mamba(
            layer_idx=0,  # Not used, we provide prefixes
            hidden_size=hidden_size,
            d_inner=d_inner,
            d_xb=d_xb,
            dt_rank=dt_rank,
            d_state=d_state,
            d_conv=d_conv,
            repeat_kv_before_conv=repeat_kv_before_conv,
            conv_bias=conv_bias,
            dt_bias=dt_bias,
            dt_min=dt_min,
            dt_max=dt_max,
            source_prefix=source_prefix,
            target_prefix=target_prefix,
        )
        for key, expr in mil_exprs.items():
            plan.define(key, expr)

    elif source_type == "mamba" and target_type == "mamba":
        # Mamba to Mamba: direct copy (including conv1d)
        for name in ["in_proj.weight", "out_proj.weight", "dt_in_proj.weight",
                     "dt_proj.weight", "dt_proj.bias", "conv1d.weight", "conv1d.bias",
                     "A_log", "D"]:
            plan.define(target_prefix / name, Ref(source_prefix / name))

    else:
        # No converter: random init
        _plan_random_mixer(plan, target_prefix, target_type, target_config, hidden_size)


def _plan_random_mixer(
    plan: ExprPlan,
    prefix: W,
    mixer_type: str,
    config: dict,
    hidden_size: int,
) -> None:
    """Add random initialization expressions for a mixer."""
    if mixer_type in ("attention", "sliding_window"):
        heads = config.get("heads", 32)
        head_groups = config.get("head_groups", heads)
        head_size = config.get("head_size", hidden_size // heads)
        q_size = heads * head_size
        kv_size = head_groups * head_size

        attn = prefix / "self_attn"
        plan.define(attn / "q_proj" / "weight", Init((q_size, hidden_size), "kaiming"))
        plan.define(attn / "k_proj" / "weight", Init((kv_size, hidden_size), "kaiming"))
        plan.define(attn / "v_proj" / "weight", Init((kv_size, hidden_size), "kaiming"))
        plan.define(attn / "o_proj" / "weight", Init((hidden_size, q_size), "kaiming"))

    elif mixer_type == "mamba":
        d_inner = config.get("d_inner", 2 * hidden_size)
        d_state = config.get("d_state", 128)
        dt_rank = config.get("dt_rank", hidden_size // 16)
        d_xb = config.get("d_xb", d_inner // 2)
        d_conv = config.get("d_conv", 4)
        repeat_kv_before_conv = config.get("repeat_kv_before_conv", True)
        conv_bias = config.get("conv_bias", True)
        dt_bias = config.get("dt_proj_bias", True)
        dt_min = config.get("dt_min", 0.001)
        dt_max = config.get("dt_max", 0.1)

        # Conv1d channels depend on repeat_kv_before_conv
        conv_channels = d_inner if repeat_kv_before_conv else d_xb

        # Core projections
        plan.define(prefix / "in_proj" / "weight", Init((2 * d_inner + 2 * d_xb, hidden_size), "kaiming"))
        plan.define(prefix / "out_proj" / "weight", Init((hidden_size, d_inner), "kaiming"))

        # dt projections
        plan.define(prefix / "dt_in_proj" / "weight", Init((dt_rank, hidden_size), "kaiming"))
        plan.define(prefix / "dt_proj" / "weight", Init((d_inner, dt_rank), "kaiming"))

        # Conv1d
        plan.define(prefix / "conv1d" / "weight", Init((conv_channels, 1, d_conv), "kaiming"))
        if conv_bias:
            plan.define(prefix / "conv1d" / "bias", Init((conv_channels,), "zeros"))

        # dt_proj bias with proper initialization
        if dt_bias:
            plan.define(prefix / "dt_proj" / "bias", Init(
                (d_inner,), "dt_bias",
                init_params={"dt_min": dt_min, "dt_max": dt_max}
            ))

        # SSM parameters - S4D initialization for A_log
        plan.define(prefix / "A_log", Init((d_inner, d_state), "s4d"))
        plan.define(prefix / "D", Init((d_inner,), "ones"))


def _plan_mlp(
    plan: ExprPlan,
    target_layer_idx: int,
    source_layer_idx: int,
    source_mlp: dict,
    target_mlp: dict,
    hidden_size: int,
) -> None:
    """Add MLP conversion expressions to plan."""
    source_mlp_path = W("model", "decoder", "blocks", source_layer_idx, "mlp")
    target_mlp_path = W("model", "decoder", "blocks", target_layer_idx, "mlp")

    source_type = source_mlp.get("type", "mlp")
    target_type = target_mlp.get("type", "mlp")

    if source_type == target_type:
        # Same type: direct copy
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            plan.define(target_mlp_path / proj / "weight", Ref(source_mlp_path / proj / "weight"))
    else:
        # Different types: random init
        intermediate_size = target_mlp.get("intermediate_size", 4 * hidden_size)
        plan.define(target_mlp_path / "gate_proj" / "weight", Init((intermediate_size, hidden_size), "kaiming"))
        plan.define(target_mlp_path / "up_proj" / "weight", Init((intermediate_size, hidden_size), "kaiming"))
        plan.define(target_mlp_path / "down_proj" / "weight", Init((hidden_size, intermediate_size), "kaiming"))


def _plan_norms(
    plan: ExprPlan,
    target_layer_idx: int,
    source_layer_idx: int,
    source_block: dict,
    target_block: dict,
    hidden_size: int,
) -> None:
    """Add normalization conversion expressions to plan."""
    source_layer = W("model", "decoder", "blocks", source_layer_idx)
    target_layer = W("model", "decoder", "blocks", target_layer_idx)

    for norm_name in ["input_layernorm", "post_attention_layernorm"]:
        source_norm_path = source_layer / norm_name
        target_norm_path = target_layer / norm_name

        source_norm = source_block.get("normalization", {})
        target_norm = target_block.get("normalization", {})

        source_type = source_norm.get("type", "rms_norm")
        target_type = target_norm.get("type", "rms_norm")

        if source_type == target_type:
            plan.define(target_norm_path / "weight", Ref(source_norm_path / "weight"))
        else:
            plan.define(target_norm_path / "weight", Init((hidden_size,), "ones"))
