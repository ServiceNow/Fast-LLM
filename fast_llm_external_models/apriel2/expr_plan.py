"""Expression-based plan system for weight transformations.

This module implements a declarative approach where each target tensor is defined
as an expression over source tensors. This enables:
- Composition via expression substitution
- Fusion via tree rewriting
- Streaming execution with ref-counting for memory efficiency

Core expression types (Pydantic discriminated union):
- Ref(key): Reference to a source tensor
- Slice(expr, slices): Slice an expression
- Concat(exprs, dim): Concatenate expressions along a dimension
- Init(shape=shape, init_type=init_type): Random/constant initialization
- Reshape(expr, shape): Reshape an expression

Weight path utilities:
- W: Builder for structured weight key paths
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Callable, Iterator, Literal, Union

import torch
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor

logger = logging.getLogger(__name__)


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
        mappings[q] = Ref(key=source_q)
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
# Expression Types (Pydantic Discriminated Union)
# =============================================================================


class Ref(BaseModel):
    """Reference to a source tensor by key."""

    model_config = ConfigDict(frozen=True)

    type: Literal["ref"] = "ref"
    key: str

    def find_refs(self) -> set[str]:
        return {self.key}

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
        return f"Ref(key={self.key!r})"


class Slice(BaseModel):
    """Slice an expression along dimensions.

    slices is a tuple of (start, stop, step) tuples, one per dimension.
    None values mean "use default" (0, size, 1).
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["slice"] = "slice"
    expr: "Expr"
    slices: tuple[tuple[int | None, int | None, int | None], ...]

    def find_refs(self) -> set[str]:
        return self.expr.find_refs()

    def evaluate(
        self,
        sources: dict[str, Tensor],
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        target_key: str | None = None,
    ) -> Tensor:
        tensor = self.expr.evaluate(sources, device, dtype, target_key)
        slice_objs = tuple(slice(s[0], s[1], s[2]) for s in self.slices)
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


class Concat(BaseModel):
    """Concatenate multiple expressions along a dimension."""

    model_config = ConfigDict(frozen=True)

    type: Literal["concat"] = "concat"
    exprs: tuple["Expr", ...]
    dim: int = 0

    def find_refs(self) -> set[str]:
        refs = set()
        for expr in self.exprs:
            refs.update(expr.find_refs())
        return refs

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


class Init(BaseModel):
    """Initialize a tensor with random or constant values.

    init_type can be:
    - "zeros": All zeros
    - "ones": All ones
    - "kaiming": Kaiming uniform initialization
    - "normal": Normal distribution with std=0.02
    - "s4d": S4D real initialization for Mamba A_log (log of 1..d_state expanded)
    - "dt_bias": Special dt_proj.bias initialization (log-space from dt_min/dt_max)
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["init"] = "init"
    shape: tuple[int, ...]
    init_type: str = "kaiming"
    init_params: dict[str, Any] | None = None

    def find_refs(self) -> set[str]:
        return set()  # Init has no dependencies

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
            if not self.init_params:
                raise ValueError("dt_bias init requires init_params with dt_min, dt_max, dt_init_floor")
            dt_min = self.init_params["dt_min"]
            dt_max = self.init_params["dt_max"]
            dt_init_floor = self.init_params["dt_init_floor"]

            if len(self.shape) != 1:
                raise ValueError(f"dt_bias init requires 1D shape, got {self.shape}")
            d_inner = self.shape[0]

            # Random dt values in [dt_min, dt_max] log-space
            tensor = torch.empty(d_inner, device=device, dtype=dtype)
            tensor.uniform_(generator=gen)
            dt = torch.exp(tensor * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
            dt = dt.clamp(min=dt_init_floor)
            # Inverse softplus to get the bias that produces these dt values
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            return inv_dt

        elif self.init_type == "identity_conv":
            # Identity kernel for depthwise conv: delta at last position
            # Shape: (channels, 1, kernel_size)
            if len(self.shape) != 3 or self.shape[1] != 1:
                raise ValueError(f"identity_conv requires shape (C, 1, K), got {self.shape}")
            channels, _, kernel_size = self.shape
            tensor = torch.zeros(self.shape, device=device, dtype=dtype)
            tensor[:, 0, -1] = 1.0  # Delta at last position (current timestep)
            return tensor

        elif self.init_type == "slow_decay":
            # Small A_log for slow decay in GatedDeltaNet
            # exp(A_log) ≈ 0.1, giving ~10 step half-life
            # With dt_bias=0: g = -exp(A_log) * softplus(0) ≈ -0.1 * 0.693 ≈ -0.07
            # exp(g) ≈ 0.93 per step
            A = torch.full(self.shape, 0.1, device=device, dtype=torch.float32)
            return torch.log(A).to(dtype)

        else:
            raise ValueError(f"Unknown init type: {self.init_type}")

    def __repr__(self) -> str:
        if self.init_params:
            return f"Init(shape={self.shape}, init_type={self.init_type!r}, {self.init_params!r})"
        return f"Init(shape={self.shape}, init_type={self.init_type!r})"


class Reshape(BaseModel):
    """Reshape an expression to a new shape."""

    model_config = ConfigDict(frozen=True)

    type: Literal["reshape"] = "reshape"
    expr: "Expr"
    shape: tuple[int, ...]

    def find_refs(self) -> set[str]:
        return self.expr.find_refs()

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


# Discriminated union type for all expressions
Expr = Annotated[
    Union[Ref, Slice, Concat, Init, Reshape],
    Field(discriminator="type"),
]

# Rebuild models to resolve forward references
Slice.model_rebuild()
Concat.model_rebuild()
Reshape.model_rebuild()

# TypeAdapter for deserializing Expr from dict/JSON
ExprAdapter: TypeAdapter[Expr] = TypeAdapter(Expr)


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
    return Slice(expr=expr, slices=tuple(dim_slices))


# =============================================================================
# Expression Utilities
# =============================================================================


def substitute(expr: Expr, bindings: dict[str, Expr]) -> Expr:
    """Substitute Ref expressions with their bindings.

    This is the core of composition: replace Ref(key=x) with the expression
    that produces x in the source plan.

    Args:
        expr: Expression to transform.
        bindings: Map from ref keys to their producing expressions.

    Returns:
        New expression with substitutions applied.
    """
    match expr:
        case Ref(key=key):
            return bindings.get(key, expr)
        case Slice(expr=inner, slices=slices):
            return Slice(expr=substitute(inner, bindings), slices=slices)
        case Concat(exprs=exprs, dim=dim):
            return Concat(exprs=tuple(substitute(e, bindings) for e in exprs), dim=dim)
        case Init():
            return expr
        case Reshape(expr=inner, shape=shape):
            return Reshape(expr=substitute(inner, bindings), shape=shape)
        case _:
            raise TypeError(f"Unknown expression type: {type(expr)}")


def fuse(expr: Expr) -> Expr:
    """Apply fusion/optimization rules to an expression.

    Current rules:
    - Flatten nested Concat with same dim
    - Collapse nested Reshape
    """
    match expr:
        case Ref():
            return expr

        case Slice(expr=inner, slices=slices):
            # Future: compose Slice(Slice(x, s1), s2) -> Slice(x, compose(s1, s2))
            return Slice(expr=fuse(inner), slices=slices)

        case Concat(exprs=exprs, dim=dim):
            # Recursively fuse children, then flatten nested Concat with same dim
            flattened: list[Expr] = []
            for child in (fuse(e) for e in exprs):
                match child:
                    case Concat(exprs=inner_exprs, dim=inner_dim) if inner_dim == dim:
                        flattened.extend(inner_exprs)
                    case _:
                        flattened.append(child)
            return Concat(exprs=tuple(flattened), dim=dim)

        case Init():
            return expr

        case Reshape(expr=inner, shape=shape):
            fused_inner = fuse(inner)
            # Reshape(Reshape(x, _), s2) -> Reshape(x, s2)
            match fused_inner:
                case Reshape(expr=innermost):
                    return Reshape(expr=innermost, shape=shape)
                case _:
                    return Reshape(expr=fused_inner, shape=shape)

        case _:
            raise TypeError(f"Unknown expression type: {type(expr)}")


# =============================================================================
# Plan Class
# =============================================================================


class ExprPlan(BaseModel):
    """A plan mapping target keys to expressions over sources.

    The plan is declarative: each target is defined as an expression.
    Composition is achieved via the `|` operator or `compose()` function.

    Example:
        plan = ExprPlan(mappings={
            "out.weight": Ref(key="in.weight"),
            "out.bias": Init(shape=(10,), init_type="zeros"),
        })

        # Compose plans with |
        full_pipeline = plan1 | plan2 | plan3
    """

    model_config = ConfigDict(frozen=True)

    mappings: dict[str, Expr] = Field(default_factory=dict)
    source_format: str = ""
    target_format: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.mappings)

    def __iter__(self) -> Iterator[tuple[str, Expr]]:
        return iter(self.mappings.items())

    def __getitem__(self, key: str) -> Expr:
        return self.mappings[key]

    def __contains__(self, key: str) -> bool:
        return key in self.mappings

    def __or__(self, other: "ExprPlan") -> "ExprPlan":
        """Compose plans: self | other means self (A→B) then other (B→C) = (A→C)."""
        return compose(self, other)

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

    def fuse(self) -> ExprPlan:
        """Return a new plan with fusion optimizations applied."""
        return ExprPlan(
            mappings={k: fuse(v) for k, v in self.mappings.items()},
            source_format=self.source_format,
            target_format=self.target_format,
            metadata=self.metadata,
        )

    def render_tree(self, collapse_layers: bool = True) -> str:
        """Render the plan as a hierarchical tree.

        Args:
            collapse_layers: If True, collapse repeated layer patterns like
                blocks.0, blocks.1, ... into blocks.[0..47].

        Returns:
            Tree-formatted string representation.
        """
        return render_tree(self, collapse_layers=collapse_layers)


# =============================================================================
# Plan Tree: Proper tree structure for collapsing and rendering
# =============================================================================


@dataclass
class PlanTreeNode:
    """A node in the plan tree.

    Either an internal node (has children) or a leaf node (has values).
    After merging, leaf nodes contain aggregated values from multiple siblings.
    """

    children: dict[str, "PlanTreeNode"] = field(default_factory=dict)
    # For leaf nodes: list of (sibling_key, expr) pairs
    # Before merge: single item, after merge: multiple items from merged siblings
    values: list[tuple[str, "Expr"]] = field(default_factory=list)

    def is_leaf(self) -> bool:
        return len(self.children) == 0


def _build_plan_tree(plan: ExprPlan) -> PlanTreeNode:
    """Convert flat plan to proper tree structure."""
    root = PlanTreeNode()

    for target, expr in plan:
        parts = target.split(".")
        node = root

        # Navigate/create path to parent
        for part in parts[:-1]:
            if part not in node.children:
                node.children[part] = PlanTreeNode()
            node = node.children[part]

        # Create leaf
        leaf_name = parts[-1]
        if leaf_name not in node.children:
            node.children[leaf_name] = PlanTreeNode()
        # Store with empty key (will be set during merge)
        node.children[leaf_name].values.append(("", expr))

    return root


def _expr_signature(expr: "Expr") -> tuple:
    """Get a signature for an expression that determines merge compatibility.

    Expressions with different signatures should not be merged together.
    """
    match expr:
        case Ref():
            return ("ref",)
        case Init(shape=shape, init_type=init_type):
            # Init expressions must have same type and shape to be merged
            return ("init", init_type, shape)
        case Concat(dim=dim, exprs=exprs):
            # Concat must have same dim and same number of parts
            return ("concat", dim, len(exprs))
        case Slice(slices=slices):
            return ("slice", slices)
        case Reshape(shape=shape):
            return ("reshape", shape)
        case _:
            return (type(expr).__name__,)


def _tree_structure_signature(node: PlanTreeNode) -> tuple:
    """Get structural signature of a subtree.

    Two subtrees are structurally equivalent if they have the same signature.
    For leaves, includes expression type info to prevent merging incompatible expressions.
    """
    if node.is_leaf():
        # Include expression signature for leaves
        if node.values:
            _, first_expr = node.values[0]
            return ("leaf", _expr_signature(first_expr))
        return ("leaf",)

    # Internal node - structure is the set of children with their signatures
    child_sigs = tuple(
        sorted((name, _tree_structure_signature(child))
               for name, child in node.children.items())
    )
    return ("node", child_sigs)


def _merge_sibling_trees(
    nodes: list[tuple[str, PlanTreeNode]]
) -> PlanTreeNode:
    """Merge structurally identical sibling trees into one with aggregated leaves.

    Args:
        nodes: List of (sibling_key, node) pairs to merge

    Returns:
        Merged node with aggregated leaf values
    """
    if len(nodes) == 1:
        key, node = nodes[0]
        # Tag leaf values with the sibling key
        if node.is_leaf():
            return PlanTreeNode(
                values=[(key, expr) for _, expr in node.values]
            )
        else:
            return PlanTreeNode(
                children={
                    name: _merge_sibling_trees([(key, child)])
                    for name, child in node.children.items()
                }
            )

    # Multiple nodes to merge - they must have identical structure
    first_key, first_node = nodes[0]

    if first_node.is_leaf():
        # Merge leaf values from all siblings
        merged_values = []
        for key, node in nodes:
            for _, expr in node.values:
                merged_values.append((key, expr))
        return PlanTreeNode(values=merged_values)
    else:
        # Merge children recursively
        merged_children = {}
        for child_name in first_node.children:
            child_nodes = [(key, node.children[child_name]) for key, node in nodes]
            merged_children[child_name] = _merge_sibling_trees(child_nodes)
        return PlanTreeNode(children=merged_children)


def _collect_leaf_refs(node: PlanTreeNode) -> list[str]:
    """Collect all Ref keys from leaf nodes in a subtree."""
    refs = []
    if node.is_leaf():
        for _, expr in node.values:
            if isinstance(expr, Ref):
                refs.append(expr.key)
    else:
        for child in node.children.values():
            refs.extend(_collect_leaf_refs(child))
    return refs


def _find_varying_positions_within_group(refs: list[str]) -> set[int] | None:
    """Find positions where refs within a single group vary.

    Returns:
        Set of varying positions, or None if refs have different structures
        (different lengths), meaning they can't be compared position-by-position.
    """
    if len(refs) <= 1:
        return set()

    parts_list = [ref.split(".") for ref in refs]
    lengths = {len(p) for p in parts_list}

    # Different lengths = different structures, can't compare positionally
    if len(lengths) != 1:
        return None

    ref_length = next(iter(lengths))
    varying = set()

    for part_idx in range(ref_length):
        values = {parts[part_idx] for parts in parts_list}
        if len(values) > 1:
            varying.add(part_idx)

    return varying


def _refs_differ_in_one_part(ref_groups: list[list[str]]) -> bool:
    """Check if refs across groups can be merged.

    The key insight: if refs within a group already vary at some position
    (due to a previous merge), we shouldn't allow another merge that would
    introduce variation at a DIFFERENT position.

    Algorithm:
    1. Find positions where refs vary WITHIN each group (P_within)
    2. Find positions where refs vary ACROSS groups (P_across)
    3. Allow merge only if:
       - P_within is undefined (refs have different structures) → check P_across only
       - OR P_within == P_across (variation is at the same position)

    Args:
        ref_groups: List of ref key lists, one per sibling being considered for merge.

    Returns:
        True if merge is allowed.
    """
    if len(ref_groups) < 2:
        return True

    # All groups must have same number of refs
    first_len = len(ref_groups[0])
    if not all(len(g) == first_len for g in ref_groups):
        return False

    if first_len == 0:
        return True

    # Step 1: Find positions varying WITHIN each group
    # If any group has refs with different structures, P_within is "undefined"
    p_within: set[int] | None = set()
    for group in ref_groups:
        group_varying = _find_varying_positions_within_group(group)
        if group_varying is None:
            # Different structures within group - can't determine P_within
            p_within = None
            break
        p_within = p_within | group_varying

    # Step 2: Find positions varying ACROSS groups (using sorted alignment)
    sorted_groups = [sorted(group) for group in ref_groups]
    p_across: set[int] = set()

    for ref_idx in range(first_len):
        refs_at_pos = [group[ref_idx] for group in sorted_groups]
        parts_list = [ref.split(".") for ref in refs_at_pos]

        # All refs at this position must have the same length for cross-comparison
        lengths = {len(p) for p in parts_list}
        if len(lengths) != 1:
            return False

        ref_length = next(iter(lengths))
        for part_idx in range(ref_length):
            values_at_idx = {parts[part_idx] for parts in parts_list}
            if len(values_at_idx) > 1:
                p_across.add(part_idx)

    # Step 3: Check merge conditions
    # Must have exactly one differing position across groups
    if len(p_across) != 1:
        return False

    # If P_within is defined and non-empty, it must match P_across
    if p_within is not None and len(p_within) > 0:
        if p_within != p_across:
            return False

    return True


def _collapse_siblings(node: PlanTreeNode) -> PlanTreeNode:
    """Recursively collapse structurally identical siblings (TOP-DOWN).

    We try to merge siblings at each level FIRST, then recurse into children.
    This ensures we merge at the highest level possible (e.g., layer indices)
    before lower levels (e.g., projection names), using up the "one differing
    part budget" at the right level.
    """
    if node.is_leaf():
        return node

    # Step 1: Try to merge siblings at THIS level first (before recursing)
    groups: dict[tuple, list[tuple[str, PlanTreeNode]]] = {}
    for name, child in node.children.items():
        sig = _tree_structure_signature(child)
        if sig not in groups:
            groups[sig] = []
        groups[sig].append((name, child))

    # Merge groups where refs differ in at most one part
    merged_children: dict[str, PlanTreeNode] = {}
    for members in groups.values():
        if len(members) > 1:
            ref_groups = [sorted(_collect_leaf_refs(child)) for _, child in members]

            if _refs_differ_in_one_part(ref_groups):
                # Merge these siblings - this aggregates refs from all of them
                merged = _merge_sibling_trees(members)
                keys = [name for name, _ in members]
                merged_key = _format_key_group(keys)
                merged_children[merged_key] = merged
            else:
                # Can't merge - keep separate
                for name, child in members:
                    merged_children[name] = _merge_sibling_trees([(name, child)])
        else:
            name, child = members[0]
            merged_children[name] = _merge_sibling_trees([(name, child)])

    # Step 2: NOW recurse into children (after merging at this level)
    # The merged children now have aggregated refs, so lower-level merging
    # will fail the "one part differs" check if this level already merged.
    result_children = {
        name: _collapse_siblings(child)
        for name, child in merged_children.items()
    }

    return PlanTreeNode(children=result_children)


def _format_key_group(keys: list[str]) -> str:
    """Format a group of keys, using range notation for consecutive integers."""
    # Try to parse as integers
    try:
        nums = sorted(int(k) for k in keys)
        ranges = _find_contiguous_ranges(nums)
        range_strs = []
        for start, end in ranges:
            if start == end:
                range_strs.append(str(start))
            else:
                range_strs.append(f"{start}..{end}")
        return "[" + ", ".join(range_strs) + "]"
    except ValueError:
        # Not all integers, just list them
        return "[" + ", ".join(sorted(keys)) + "]"


def _find_contiguous_ranges(indices: list[int]) -> list[tuple[int, int]]:
    """Find contiguous ranges in a sorted list of indices."""
    if not indices:
        return []

    ranges = []
    start = indices[0]
    end = indices[0]

    for idx in indices[1:]:
        if idx == end + 1:
            end = idx
        else:
            ranges.append((start, end))
            start = idx
            end = idx

    ranges.append((start, end))
    return ranges


def _find_string_pattern(strings: list[str]) -> str:
    """Find pattern in list of strings, render varying parts as ranges.

    Examples:
        ["a.0.b", "a.1.b", "a.2.b"] -> "a.[0..2].b"
        ["x.foo.y", "x.bar.y"] -> "x.[bar, foo].y"
    """
    if len(strings) == 1:
        return strings[0]

    # Find common prefix
    prefix = strings[0]
    for s in strings[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                break

    # Find common suffix
    suffix = strings[0]
    for s in strings[1:]:
        while not s.endswith(suffix):
            suffix = suffix[1:]
            if not suffix:
                break

    # Handle overlap between prefix and suffix
    if len(prefix) + len(suffix) > len(strings[0]):
        suffix = suffix[len(prefix) + len(suffix) - len(strings[0]):]

    # Extract varying parts
    varying = []
    for s in strings:
        end_idx = len(s) - len(suffix) if suffix else len(s)
        varying.append(s[len(prefix):end_idx])

    # Format varying part
    varying_str = _format_key_group(varying)

    return f"{prefix}{varying_str}{suffix}"


def render_tree(plan: ExprPlan, collapse_layers: bool = True) -> str:
    """Render a plan as a hierarchical tree.

    Uses principled tree-based collapsing:
    1. Build proper tree structure from flat plan
    2. Recursively merge structurally identical siblings
    3. Render with pattern discovery for aggregated leaves

    Example output:
        model/
        ├── embed_tokens/
        │   └── weight ← language_model.embed_tokens.weight
        ├── decoder/
        │   └── blocks/
        │       └── [0..47]/
        │           ├── mixer/
        │           │   └── self_attn/
        │           │       ├── q_proj/
        │           │       │   └── weight ← ...layers.[0..47]...q_proj.weight
    """
    # Build tree
    tree = _build_plan_tree(plan)

    # Collapse if requested
    if collapse_layers:
        tree = _collapse_siblings(tree)

    # Render
    lines: list[str] = []
    _render_plan_tree(tree, lines, prefix="", is_last=True, is_root=True, name="")
    return "\n".join(lines)


def _render_plan_tree(
    node: PlanTreeNode,
    lines: list[str],
    prefix: str,
    is_last: bool,
    is_root: bool,
    name: str,
) -> None:
    """Recursively render a PlanTreeNode with pattern discovery for aggregated leaves."""
    # Determine connectors
    if is_root:
        connector = ""
        child_prefix = ""
    else:
        connector = "└── " if is_last else "├── "
        child_prefix = prefix + ("    " if is_last else "│   ")

    if node.is_leaf():
        # Leaf node with (possibly aggregated) values
        expr_str = _format_aggregated_leaf(node.values)
        lines.append(f"{prefix}{connector}{name} {expr_str}")
    else:
        # Internal node
        if name:
            lines.append(f"{prefix}{connector}{name}/")

        items = list(node.children.items())
        for i, (child_name, child) in enumerate(items):
            is_last_child = i == len(items) - 1
            _render_plan_tree(
                child,
                lines,
                child_prefix if name else prefix,
                is_last_child,
                is_root=False,
                name=child_name,
            )


def _format_aggregated_leaf(values: list[tuple[str, "Expr"]]) -> str:
    """Format a leaf with aggregated values using pattern discovery.

    Args:
        values: List of (sibling_key, expr) pairs

    Returns:
        Formatted string with patterns discovered in source refs
    """
    if len(values) == 1:
        # Single value - format directly
        _, expr = values[0]
        return _format_single_expr(expr)

    # Multiple values - need pattern discovery
    # First, check if all expressions have the same structure
    first_expr = values[0][1]

    # For simple Ref expressions, use pattern discovery
    if isinstance(first_expr, Ref):
        if all(isinstance(e, Ref) for _, e in values):
            keys = [e.key for _, e in values]
            pattern = _find_string_pattern(keys)
            return f"← {pattern}"

    # For Init expressions, they should all be identical
    if isinstance(first_expr, Init):
        return _format_single_expr(first_expr)

    # For Concat expressions, format with pattern discovery
    if isinstance(first_expr, Concat):
        return _format_aggregated_concat(values)

    # For Slice expressions
    if isinstance(first_expr, Slice):
        return _format_aggregated_slice(values)

    # Fallback
    return _format_single_expr(first_expr)


def _format_single_expr(expr: "Expr") -> str:
    """Format a single expression using ML notation."""
    match expr:
        case Ref(key=key):
            return f"← {key}"
        case Init(shape=shape, init_type=init_type):
            shape_str = "×".join(str(d) for d in shape)
            if init_type == "zeros":
                return f"= 𝟎({shape_str})"
            elif init_type == "ones":
                return f"= 𝟏({shape_str})"
            elif init_type == "identity_conv":
                return f"= I_conv({shape_str})"
            elif init_type == "slow_decay":
                return f"= A_log({shape_str})"
            else:
                return f"= {init_type}({shape_str})"
        case Concat(exprs=exprs, dim=dim):
            parts = [_format_concat_part(e) for e in exprs]
            sep = "; " if dim == 0 else ", "
            return f"= [{sep.join(parts)}]"
        case Slice(expr=inner, slices=slices):
            slice_str = _format_slice_notation(slices)
            inner_str = _format_single_expr(inner)
            # Remove the prefix (← or =) and add slice
            if inner_str.startswith("← "):
                return f"← {inner_str[2:]}{slice_str}"
            elif inner_str.startswith("= "):
                return f"= {inner_str[2:]}{slice_str}"
            return f"{inner_str}{slice_str}"
        case Reshape(shape=shape):
            shape_str = "×".join(str(d) for d in shape)
            return f"= reshape({shape_str})"
        case _:
            return f"= {type(expr).__name__}"


def _format_concat_part(expr: "Expr") -> str:
    """Format a single part of a concat (for short display)."""
    match expr:
        case Ref(key=key):
            # Extract last 2 components
            parts = key.split(".")
            if len(parts) >= 2:
                return ".".join(parts[-2:])
            return parts[-1] if parts else "?"
        case Init(shape=shape, init_type=init_type):
            shape_str = "×".join(str(d) for d in shape)
            if init_type == "zeros":
                return f"𝟎({shape_str})"
            elif init_type == "ones":
                return f"𝟏({shape_str})"
            else:
                return f"{init_type}({shape_str})"
        case Slice(expr=inner, slices=slices):
            inner_str = _format_concat_part(inner)
            slice_str = _format_slice_notation(slices)
            return f"{inner_str}{slice_str}"
        case _:
            return "?"


def _format_slice_notation(slices: tuple) -> str:
    """Format slice notation like [0:10, :]."""
    slice_strs = []
    for s in slices:
        start, stop, step = s
        if start is None and stop is None and step is None:
            slice_strs.append(":")
        elif step is None or step == 1:
            slice_strs.append(f"{start or ''}:{stop or ''}")
        else:
            slice_strs.append(f"{start or ''}:{stop or ''}:{step}")
    return f"[{', '.join(slice_strs)}]"


def _format_aggregated_concat(values: list[tuple[str, "Expr"]]) -> str:
    """Format aggregated Concat expressions with pattern discovery."""
    # Get the first concat to understand structure
    first_concat = values[0][1]
    if not isinstance(first_concat, Concat):
        return _format_single_expr(first_concat)

    # For each position in the concat, aggregate across all values
    num_parts = len(first_concat.exprs)
    dim = first_concat.dim

    formatted_parts = []
    for i in range(num_parts):
        part_exprs = [(key, expr.exprs[i]) for key, expr in values
                      if isinstance(expr, Concat) and len(expr.exprs) > i]
        formatted_parts.append(_format_aggregated_concat_part(part_exprs))

    sep = "; " if dim == 0 else ", "
    return f"= [{sep.join(formatted_parts)}]"


def _format_aggregated_concat_part(values: list[tuple[str, "Expr"]]) -> str:
    """Format a single part of an aggregated concat."""
    if len(values) == 1:
        return _format_concat_part(values[0][1])

    first_expr = values[0][1]

    # For Refs, use pattern discovery
    if isinstance(first_expr, Ref):
        if all(isinstance(e, Ref) for _, e in values):
            keys = [e.key for _, e in values]
            pattern = _find_string_pattern(keys)
            return pattern

    # For Slice(Ref), extract refs and find pattern, then add slice
    if isinstance(first_expr, Slice) and isinstance(first_expr.expr, Ref):
        if all(isinstance(e, Slice) and isinstance(e.expr, Ref) for _, e in values):
            keys = [e.expr.key for _, e in values]
            pattern = _find_string_pattern(keys)
            slice_str = _format_slice_notation(first_expr.slices)
            return f"{pattern}{slice_str}"

    # For Init, they should all be identical
    if isinstance(first_expr, Init):
        return _format_concat_part(first_expr)

    return _format_concat_part(first_expr)


def _format_aggregated_slice(values: list[tuple[str, "Expr"]]) -> str:
    """Format aggregated Slice expressions with pattern discovery."""
    first_slice = values[0][1]
    if not isinstance(first_slice, Slice):
        return _format_single_expr(first_slice)

    # Get inner expressions and find pattern
    inner_values = [(key, expr.expr) for key, expr in values if isinstance(expr, Slice)]
    inner_str = _format_aggregated_leaf(inner_values)

    # Add slice notation
    slice_str = _format_slice_notation(first_slice.slices)

    # Combine
    if inner_str.startswith("← "):
        return f"← {inner_str[2:]}{slice_str}"
    elif inner_str.startswith("= "):
        return f"= {inner_str[2:]}{slice_str}"
    return f"{inner_str}{slice_str}"


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


# Default shard size: 5GB (HuggingFace default)
DEFAULT_MAX_SHARD_SIZE = 5 * 1024 * 1024 * 1024


class SafetensorLoader:
    """Context manager for streaming reads from sharded safetensors.

    Pre-builds a key index for O(1) lookups and manages file handle lifecycle.

    Usage:
        with SafetensorLoader(source_files) as loader:
            executor = StreamingExecutor(plan, loader, device, dtype)
            for key, tensor in executor.execute():
                ...
    """

    def __init__(self, files: list[Path], device: str = "cpu"):
        self.files = [Path(f) for f in files]
        self.device = device
        self._handles: dict[Path, Any] = {}
        self._key_index: dict[str, Path] = {}

    def __enter__(self) -> "SafetensorLoader":
        # Pre-build index: key -> file (one-time O(n×m), then O(1) lookups)
        for f in self.files:
            handle = safe_open(f, framework="pt", device=self.device)
            self._handles[f] = handle
            for key in handle.keys():
                self._key_index[key] = f
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._handles.clear()
        self._key_index.clear()

    def __call__(self, key: str) -> Tensor:
        """Load a tensor by key. Raises KeyError if not found."""
        if key not in self._key_index:
            raise KeyError(f"Source key not found in any file: {key}")
        return self._handles[self._key_index[key]].get_tensor(key)

    def keys(self) -> set[str]:
        """Return all available keys across all files."""
        return set(self._key_index.keys())


class ShardedSafetensorWriter:
    """Context manager for streaming writes to sharded safetensors.

    Accumulates tensors until a size threshold is reached, then flushes
    to a shard file. This bounds peak memory to ~max_shard_size instead
    of accumulating all tensors before writing.

    Output follows HuggingFace conventions:
    - model-00001-of-00003.safetensors, model-00002-of-00003.safetensors, etc.
    - model.safetensors.index.json with weight_map and metadata

    Usage:
        with ShardedSafetensorWriter(output_dir) as writer:
            for key, tensor in executor.execute():
                writer.add(key, tensor)
        # Automatically finalizes on exit, cleans up temp files on error
    """

    def __init__(
        self,
        output_dir: Path,
        max_shard_size: int = DEFAULT_MAX_SHARD_SIZE,
        base_name: str = "model",
    ):
        self.output_dir = Path(output_dir)
        self.max_shard_size = max_shard_size
        self.base_name = base_name

        # Accumulator state
        self._buffer: dict[str, Tensor] = {}
        self._buffer_bytes: int = 0
        self._shard_index: int = 0
        self._shard_files: list[Path] = []

        # For building the index
        self._weight_map: dict[str, str] = {}
        self._total_bytes: int = 0

        # Context manager state
        self._finalized: bool = False

    def __enter__(self) -> "ShardedWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            # Error occurred - clean up temp files
            self._cleanup_temp_files()
        else:
            # Success - finalize
            self._finalize()
        return False  # Don't suppress exceptions

    def _cleanup_temp_files(self) -> None:
        """Remove any temporary shard files on error."""
        for tmp_file in self._shard_files:
            if tmp_file.exists():
                tmp_file.unlink()
                logger.debug(f"Cleaned up temp file: {tmp_file}")

    def _tensor_bytes(self, tensor: Tensor) -> int:
        """Calculate tensor size in bytes."""
        return tensor.numel() * tensor.element_size()

    def add(self, key: str, tensor: Tensor) -> None:
        """Add a tensor to the current shard buffer.

        If adding this tensor would exceed max_shard_size, the current
        buffer is flushed first.
        """
        if self._finalized:
            raise RuntimeError("Cannot add tensors after finalization")

        tensor_size = self._tensor_bytes(tensor)

        # Flush if this would exceed the threshold (but always allow at least one tensor)
        if self._buffer and self._buffer_bytes + tensor_size > self.max_shard_size:
            self._flush()

        self._buffer[key] = tensor
        self._buffer_bytes += tensor_size
        self._total_bytes += tensor_size

    def _flush(self) -> None:
        """Write the current buffer to a shard file."""
        if not self._buffer:
            return

        self._shard_index += 1
        # Use .tmp extension until we know total shard count
        shard_file = self.output_dir / f"{self.base_name}-{self._shard_index:05d}.safetensors.tmp"

        logger.debug(
            f"Writing shard {self._shard_index}: {len(self._buffer)} tensors, "
            f"{self._buffer_bytes / 1e9:.2f} GB"
        )
        save_file(self._buffer, shard_file)
        self._shard_files.append(shard_file)

        # Record weight locations (will update names in finalize)
        for key in self._buffer:
            self._weight_map[key] = shard_file.name

        # Clear buffer
        self._buffer.clear()
        self._buffer_bytes = 0

    def _finalize(self) -> Path:
        """Flush remaining tensors and write the index file.

        Returns the path to the index file (or single safetensor file if only one shard).
        """
        if self._finalized:
            return self._result_path

        # Flush any remaining tensors
        self._flush()
        self._finalized = True

        total_shards = len(self._shard_files)

        if total_shards == 0:
            raise ValueError("No tensors were written")

        # Rename temp files to final names with correct shard count
        final_names: dict[str, str] = {}
        for i, tmp_file in enumerate(self._shard_files, 1):
            if total_shards == 1:
                # Single shard: just use model.safetensors
                final_name = f"{self.base_name}.safetensors"
            else:
                final_name = f"{self.base_name}-{i:05d}-of-{total_shards:05d}.safetensors"

            final_path = self.output_dir / final_name
            tmp_file.rename(final_path)
            final_names[tmp_file.name] = final_name
            logger.info(f"Saved {final_path.name}")

        # Update weight_map with final names
        for key in self._weight_map:
            old_name = self._weight_map[key]
            self._weight_map[key] = final_names[old_name]

        # Write index file if sharded
        if total_shards > 1:
            index = {
                "metadata": {"total_size": self._total_bytes},
                "weight_map": self._weight_map,
            }
            index_file = self.output_dir / f"{self.base_name}.safetensors.index.json"
            with open(index_file, "w") as f:
                json.dump(index, f, indent=2, sort_keys=True)
            logger.info(f"Saved index: {index_file.name}")
            self._result_path = index_file
        else:
            self._result_path = self.output_dir / f"{self.base_name}.safetensors"

        return self._result_path

    @property
    def result_path(self) -> Path:
        """Get the path to the result file (available after finalization)."""
        if not self._finalized:
            raise RuntimeError("Result path not available until finalized")
        return self._result_path


# =============================================================================
# Plan Builders
# =============================================================================


def plan_llava_to_apriel2(llava_config: dict) -> ExprPlan:
    """Build an expression plan for Llava to Apriel2 conversion.

    This is a pure mapping (all Ref expressions) since Llava→Apriel2
    is just renaming keys.
    """
    mappings: dict[str, Expr] = {}

    num_text_layers = llava_config.get("text_config", {}).get("num_hidden_layers", 0)
    num_vision_layers = llava_config.get("vision_config", {}).get("num_hidden_layers", 0)

    # Static mappings (must match convert_from_llava._STATIC_WEIGHT_MAP)
    static_mappings = [
        (W("language_model", "model", "embed_tokens", "weight"), W("model", "embed_tokens", "weight")),
        (W("language_model", "lm_head", "weight"), W("lm_head", "weight")),
        (W("language_model", "model", "norm", "weight"), W("model", "norm", "weight")),
        (
            W("vision_tower", "patch_conv", "weight"),
            W("model", "vision_encoder", "patch_convolution", "conv", "weight"),
        ),
        (W("vision_tower", "ln_pre", "weight"), W("model", "vision_encoder", "patch_convolution", "norm", "weight")),
        (
            W("multi_modal_projector", "linear_1", "weight"),
            W("model", "vision_encoder", "adapter", "linear_1", "weight"),
        ),
        (W("multi_modal_projector", "linear_1", "bias"), W("model", "vision_encoder", "adapter", "linear_1", "bias")),
        (
            W("multi_modal_projector", "linear_2", "weight"),
            W("model", "vision_encoder", "adapter", "linear_2", "weight"),
        ),
        (W("multi_modal_projector", "linear_2", "bias"), W("model", "vision_encoder", "adapter", "linear_2", "bias")),
    ]

    for src, tgt in static_mappings:
        mappings[tgt] = Ref(key=src)

    # Text decoder layers
    for layer in range(num_text_layers):
        llava_layer = W("language_model", "model", "layers", layer)
        apriel_layer = W("model", "decoder", "blocks", layer)

        # Attention projections
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            src = llava_layer / "self_attn" / proj / "weight"
            tgt = apriel_layer / "mixer" / "self_attn" / proj / "weight"
            mappings[tgt] = Ref(key=src)

        # MLP projections
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            src = llava_layer / "mlp" / proj / "weight"
            tgt = apriel_layer / "mlp" / proj / "weight"
            mappings[tgt] = Ref(key=src)

        # Layer norms
        mappings[apriel_layer / "input_layernorm" / "weight"] = Ref(key=llava_layer / "input_layernorm" / "weight")
        mappings[apriel_layer / "post_attention_layernorm" / "weight"] = Ref(
            key=llava_layer / "post_attention_layernorm" / "weight"
        )

    # Vision encoder layers
    for layer in range(num_vision_layers):
        llava_layer = W("vision_tower", "transformer", "layers", layer)
        apriel_layer = W("model", "vision_encoder", "encoder", "blocks", layer)

        # Attention projections
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            src = llava_layer / "attention" / proj / "weight"
            tgt = apriel_layer / "mixer" / "self_attn" / proj / "weight"
            mappings[tgt] = Ref(key=src)

        # MLP projections (llava uses feed_forward, apriel uses mlp)
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            src = llava_layer / "feed_forward" / proj / "weight"
            tgt = apriel_layer / "mlp" / proj / "weight"
            mappings[tgt] = Ref(key=src)

        # Layer norms (different naming)
        mappings[apriel_layer / "input_layernorm" / "weight"] = Ref(key=llava_layer / "attention_norm" / "weight")
        mappings[apriel_layer / "post_attention_layernorm" / "weight"] = Ref(key=llava_layer / "ffn_norm" / "weight")

    return ExprPlan(
        mappings=mappings,
        source_format="llava",
        target_format="apriel2",
        metadata={
            "num_text_layers": num_text_layers,
            "num_vision_layers": num_vision_layers,
        },
    )


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
    dt_init_floor: float = 1e-4,
    source_prefix: W | str = "",
    target_prefix: W | str = "",
) -> dict[str, Expr]:
    """Build MIL expressions for one layer.

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
    in_proj_expr = Concat(
        exprs=(
            Init(shape=(d_inner, hidden_size), init_type="kaiming"),  # z: random
            Slice(expr=Ref(key=src / "v_proj" / "weight"), slices=((0, d_xb, None), (None, None, None))),  # x <- V
            Slice(expr=Ref(key=src / "k_proj" / "weight"), slices=((0, d_xb, None), (None, None, None))),  # B <- K
            Slice(expr=Ref(key=src / "q_proj" / "weight"), slices=((0, d_inner, None), (None, None, None))),  # C <- Q
        ),
        dim=0,
    )

    # Conv1d channels depend on repeat_kv_before_conv
    conv_channels = d_inner if repeat_kv_before_conv else d_xb

    result = {
        # Core projections
        tgt / "in_proj" / "weight": in_proj_expr,
        tgt / "out_proj" / "weight": Ref(key=src / "o_proj" / "weight"),
        # dt projections
        tgt / "dt_in_proj" / "weight": Init(shape=(dt_rank, hidden_size), init_type="kaiming"),
        tgt / "dt_proj" / "weight": Init(shape=(d_inner, dt_rank), init_type="kaiming"),
        # Conv1d
        tgt / "conv1d" / "weight": Init(shape=(conv_channels, 1, d_conv), init_type="kaiming"),
        # SSM parameters
        tgt / "A_log": Init(shape=(d_inner, d_state), init_type="s4d"),  # S4D initialization
        tgt / "D": Init(shape=(d_inner,), init_type="ones"),
    }

    # Optional biases
    if dt_bias:
        result[tgt / "dt_proj" / "bias"] = Init(
            shape=(d_inner,), init_type="dt_bias", init_params={"dt_min": dt_min, "dt_max": dt_max, "dt_init_floor": dt_init_floor}
        )

    if conv_bias:
        result[tgt / "conv1d" / "bias"] = Init(shape=(conv_channels,), init_type="zeros")

    return result


def plan_attention_to_gated_delta_net(
    hidden_size: int,
    num_v_heads: int,
    num_k_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel_size: int = 4,
    source_prefix: W | str = "",
    target_prefix: W | str = "",
) -> dict[str, Expr]:
    """Build expressions to convert attention weights to GatedDeltaNet.

    This is a "DIL" (Delta-net Initialization from LLM) approach that:
    - Maps Q/K/V/O projections from attention to GDN's in_proj_qkvz and out_proj
    - Initializes Z (gating) to zeros for neutral behavior
    - Initializes conv1d as identity (delta at last position)
    - Initializes beta/alpha projection to zeros (β=0.5, neutral gating)
    - Initializes A_log for slow decay (~10 step half-life)
    - Initializes dt_bias to zeros

    At init, the converted block behaves like linearized attention with
    slow-decaying state accumulation, making distillation much easier.

    GatedDeltaNet in_proj_qkvz layout: [Q, K, V, Z]
    - Q: size key_dim = num_k_heads * head_k_dim (but queries use num_v_heads!)
    - K: size key_dim
    - V: size value_dim = num_v_heads * head_v_dim
    - Z: size value_dim

    Note: In Qwen's GDN, queries use num_v_heads but head_k_dim, so
    q_dim = num_v_heads * head_k_dim, not num_k_heads * head_k_dim.

    Args:
        hidden_size: Model hidden size.
        num_v_heads: Number of value heads in GDN.
        num_k_heads: Number of key heads in GDN.
        head_k_dim: Key head dimension.
        head_v_dim: Value head dimension.
        conv_kernel_size: Convolution kernel size (default 4).
        source_prefix: Prefix for source attention keys (includes self_attn).
        target_prefix: Prefix for target GDN keys (e.g., layer.mixer.gdn).

    Returns:
        Dict mapping target keys to expressions.
    """
    # Convert to W for consistent path handling
    src = W(source_prefix) if source_prefix else W()
    # Apriel2GatedDeltaNet wraps the actual GDN module as 'gdn'
    tgt = (W(target_prefix) if target_prefix else W()) / "gdn"

    # GDN dimensions
    # Note: In Qwen's GDN, q_dim uses num_v_heads (not num_k_heads) but head_k_dim
    q_dim = num_v_heads * head_k_dim
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim
    conv_dim = key_dim * 2 + value_dim  # Q/K use key_dim after fix_query_key_value_ordering

    # in_proj_qkvz layout: [Q, K, V, Z]
    # Total size: q_dim + key_dim + value_dim + value_dim
    # But wait - looking at Qwen code, after fix_query_key_value_ordering:
    # - Q gets reshaped to (B, T, num_k_heads, head_k_dim) - uses key_dim
    # - K gets reshaped to (B, T, num_k_heads, head_k_dim) - uses key_dim
    # - V gets reshaped to (B, T, num_v_heads, head_v_dim) - uses value_dim
    # - Z gets reshaped to (B, T, num_v_heads, head_v_dim) - uses value_dim
    # So in_proj_qkvz total = key_dim + key_dim + value_dim + value_dim = 2*key_dim + 2*value_dim

    # Slices in in_proj_qkvz.weight (shape: [proj_size, hidden_size])
    q_slice = (0, key_dim, None)
    k_slice = (key_dim, 2 * key_dim, None)
    v_slice = (2 * key_dim, 2 * key_dim + value_dim, None)
    z_slice = (2 * key_dim + value_dim, 2 * key_dim + 2 * value_dim, None)

    # Build in_proj_qkvz from attention Q/K/V + zeros for Z
    in_proj_qkvz_expr = Concat(
        exprs=(
            # Q block: slice attention Q to match key_dim
            Slice(
                expr=Ref(key=src / "q_proj" / "weight"),
                slices=(q_slice, (None, None, None)),
            ),
            # K block: slice attention K to match key_dim
            Slice(
                expr=Ref(key=src / "k_proj" / "weight"),
                slices=((0, key_dim, None), (None, None, None)),
            ),
            # V block: slice attention V to match value_dim
            Slice(
                expr=Ref(key=src / "v_proj" / "weight"),
                slices=((0, value_dim, None), (None, None, None)),
            ),
            # Z block: zeros for neutral gating
            Init(shape=(value_dim, hidden_size), init_type="zeros"),
        ),
        dim=0,
    )

    # in_proj_ba: zeros → b=a=0 → β=sigmoid(0)=0.5 (neutral)
    # Shape: (2 * head_k_dim, hidden_size) - one beta and one alpha per head
    ba_dim = 2 * head_k_dim

    result = {
        # Combined Q/K/V/Z projection
        tgt / "in_proj_qkvz" / "weight": in_proj_qkvz_expr,
        # Beta/alpha projection: zeros for neutral gating
        tgt / "in_proj_ba" / "weight": Init(shape=(ba_dim, hidden_size), init_type="zeros"),
        # Output projection: copy from attention O
        tgt / "out_proj" / "weight": Ref(key=src / "o_proj" / "weight"),
        # Conv1d: identity kernel (delta at last position)
        # Shape: (conv_dim, 1, kernel_size) - depthwise conv
        tgt / "conv1d" / "weight": Init(
            shape=(conv_dim, 1, conv_kernel_size),
            init_type="identity_conv",
        ),
        # A_log: small value for slow decay (~10 step half-life)
        # exp(A_log) ≈ 0.1, combined with dt_bias=0 gives g ≈ -0.07, exp(g) ≈ 0.93
        tgt / "A_log": Init(shape=(num_v_heads,), init_type="slow_decay"),
        # dt_bias: zeros
        tgt / "dt_bias": Init(shape=(num_v_heads,), init_type="zeros"),
        # Norm: ones (neutral RMSNorm-like behavior)
        tgt / "norm" / "weight": Init(shape=(head_v_dim,), init_type="ones"),
    }

    return result


def _plan_non_decoder_weights(config: dict) -> dict[str, Expr]:
    """Build passthrough mappings for non-decoder weights.

    These weights are typically unchanged during surgery:
    - Embeddings
    - LM head
    - Final norm
    - Vision encoder (if present)

    Returns:
        Dict mapping target keys to expressions.
    """
    mappings: dict[str, Expr] = {}

    # Core model weights (passthrough as identity)
    embed = W("model", "embed_tokens", "weight")
    mappings[embed] = Ref(key=embed)

    head = W("lm_head", "weight")
    mappings[head] = Ref(key=head)

    norm = W("model", "norm", "weight")
    mappings[norm] = Ref(key=norm)

    # Vision encoder (if present)
    if "vision_encoder" in config:
        vision_config = config["vision_encoder"]
        vision = W("model", "vision_encoder")

        # Patch convolution
        patch_conv = vision / "patch_convolution" / "conv" / "weight"
        mappings[patch_conv] = Ref(key=patch_conv)

        patch_norm = vision / "patch_convolution" / "norm" / "weight"
        mappings[patch_norm] = Ref(key=patch_norm)

        # Vision encoder blocks
        encoder_config = vision_config.get("encoder", {})
        num_vision_layers = encoder_config.get("num_blocks", 0)

        for layer in range(num_vision_layers):
            block = vision / "encoder" / "blocks" / layer

            # Attention projections
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                key = block / "mixer" / "self_attn" / proj / "weight"
                mappings[key] = Ref(key=key)

            # MLP projections
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                key = block / "mlp" / proj / "weight"
                mappings[key] = Ref(key=key)

            # Layer norms
            for norm_name in ["input_layernorm", "post_attention_layernorm"]:
                key = block / norm_name / "weight"
                mappings[key] = Ref(key=key)

        # Adapter
        adapter_config = vision_config.get("adapter", {})
        add_biases = adapter_config.get("add_linear_biases", False)
        adapter = vision / "adapter"

        for proj in ["linear_1", "linear_2"]:
            weight_key = adapter / proj / "weight"
            mappings[weight_key] = Ref(key=weight_key)
            if add_biases:
                bias_key = adapter / proj / "bias"
                mappings[bias_key] = Ref(key=bias_key)

    return mappings


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
    mappings: dict[str, Expr] = {}

    hidden_size = target_config.get("hidden_size", source_config.get("hidden_size"))

    source_decoder = source_config.get("decoder", {})
    target_decoder = target_config.get("decoder", {})

    num_source_layers = source_decoder.get("num_blocks", 0)
    # Inherit num_blocks from source if not specified in target
    num_target_layers = target_decoder.get("num_blocks", num_source_layers)

    # Non-decoder weights: passthrough as Ref(key)
    mappings.update(_plan_non_decoder_weights(source_config))

    # Process decoder layers
    for target_layer_idx in range(num_target_layers):
        source_layer_idx = target_layer_idx % num_source_layers if num_source_layers > 0 else 0

        source_block = _get_block_config(source_decoder, source_layer_idx)
        target_block = _get_block_config(target_decoder, target_layer_idx)

        # Mixer conversion
        mappings.update(
            _plan_mixer(
                target_layer_idx,
                source_layer_idx,
                source_block.get("mixer", {}),
                target_block.get("mixer", {}),
                hidden_size,
            )
        )

        # MLP conversion (usually passthrough)
        mappings.update(
            _plan_mlp(
                target_layer_idx,
                source_layer_idx,
                source_block.get("mlp", {}),
                target_block.get("mlp", {}),
                hidden_size,
            )
        )

        # Norm conversion (usually passthrough)
        mappings.update(
            _plan_norms(
                target_layer_idx,
                source_layer_idx,
                source_block,
                target_block,
                hidden_size,
            )
        )

    return ExprPlan(mappings=mappings, source_format="apriel2", target_format="apriel2")


def _plan_mixer(
    target_layer_idx: int,
    source_layer_idx: int,
    source_mixer: dict,
    target_mixer: dict,
    hidden_size: int,
) -> dict[str, Expr]:
    """Build mixer conversion expressions.

    Returns:
        Dict mapping target keys to expressions.
    """
    mappings: dict[str, Expr] = {}

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

    # Handle target - parse init mode once, then dispatch to the right function
    if target_type == "stochastic":
        for sub_name, sub_config in target_mixer.get("mixers", {}).items():
            sub_type = sub_config.get("type", "attention")
            target_prefix = target_layer / "mixer" / "mixers" / sub_name

            # Parse init mode and dispatch
            if sub_config.get("init") == "random":
                mappings.update(
                    _plan_random_mixer(target_prefix, sub_type, sub_config, hidden_size)
                )
            else:
                # Default is transfer - fail fast if no converter
                mappings.update(
                    _plan_mixer_transfer(
                        actual_source_type,
                        sub_type,
                        actual_source,
                        sub_config,
                        source_prefix,
                        target_prefix,
                        hidden_size,
                    )
                )
    else:
        target_prefix = target_layer / "mixer"

        # Parse init mode and dispatch
        if target_mixer.get("init") == "random":
            mappings.update(
                _plan_random_mixer(target_prefix, target_type, target_mixer, hidden_size)
            )
        else:
            # Default is transfer - fail fast if no converter
            mappings.update(
                _plan_mixer_transfer(
                    actual_source_type,
                    target_type,
                    actual_source,
                    target_mixer,
                    source_prefix,
                    target_prefix,
                    hidden_size,
                )
            )

    return mappings


def _plan_mixer_transfer(
    source_type: str,
    target_type: str,
    source_config: dict,
    target_config: dict,
    source_prefix: W,
    target_prefix: W,
    hidden_size: int,
) -> dict[str, Expr]:
    """Build expressions for transferring weights between mixer types.

    This function only handles transfer (not random init). Call _plan_random_mixer
    for random initialization.

    Note: source_prefix already includes self_attn for attention types.

    Raises:
        ValueError: If no converter exists for this source->target type pair.
    """
    mappings: dict[str, Expr] = {}

    # Attention -> Attention (including sliding window variants)
    if source_type in ("attention", "sliding_window") and target_type in ("attention", "sliding_window"):
        # Attention to attention: direct copy
        # Source prefix already includes self_attn, target needs it added
        target_attn = target_prefix / "self_attn"
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            mappings[target_attn / proj / "weight"] = Ref(key=source_prefix / proj / "weight")

    elif source_type in ("attention", "sliding_window") and target_type == "mamba":
        # Attention to Mamba: MIL conversion
        # Mamba dimensions - derive from hidden_size if not specified
        d_inner = target_config.get("d_inner", 2 * hidden_size)
        dt_rank = target_config.get("dt_rank", hidden_size // 16)
        d_xb = target_config.get("d_xb", hidden_size // 4)
        # These require explicit values (no sensible derivation)
        d_state = target_config["d_state"]
        d_conv = target_config["d_conv"]
        repeat_kv_before_conv = target_config["repeat_kv_before_conv"]
        conv_bias = target_config["conv_bias"]
        dt_bias = target_config["dt_proj_bias"]
        dt_min = target_config["dt_min"]
        dt_max = target_config["dt_max"]
        dt_init_floor = target_config["dt_init_floor"]

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
            dt_init_floor=dt_init_floor,
            source_prefix=source_prefix,
            target_prefix=target_prefix,
        )
        mappings.update(mil_exprs)

    elif source_type == "mamba" and target_type == "mamba":
        # Mamba to Mamba: direct copy (including conv1d)
        for name in [
            "in_proj.weight",
            "out_proj.weight",
            "dt_in_proj.weight",
            "dt_proj.weight",
            "dt_proj.bias",
            "conv1d.weight",
            "conv1d.bias",
            "A_log",
            "D",
        ]:
            mappings[target_prefix / name] = Ref(key=source_prefix / name)

    elif source_type in ("attention", "sliding_window") and target_type == "gated_delta_net":
        # Attention to GatedDeltaNet: DIL conversion
        # Get source attention params
        source_heads = source_config["heads"]
        source_kv_heads = source_config["head_groups"]
        source_head_size = source_config["head_size"]

        # GDN dimensions - derive from source attention if not specified
        num_v_heads = target_config.get("num_value_heads", source_heads)
        num_k_heads = target_config.get("num_key_heads", source_kv_heads)
        head_k_dim = target_config.get("key_head_dim", source_head_size)
        head_v_dim = target_config.get("value_head_dim", source_head_size)
        # conv_kernel_size requires explicit value (no derivation)
        conv_kernel_size = target_config["conv_kernel_size"]

        dil_exprs = plan_attention_to_gated_delta_net(
            hidden_size=hidden_size,
            num_v_heads=num_v_heads,
            num_k_heads=num_k_heads,
            head_k_dim=head_k_dim,
            head_v_dim=head_v_dim,
            conv_kernel_size=conv_kernel_size,
            source_prefix=source_prefix,
            target_prefix=target_prefix,
        )
        mappings.update(dil_exprs)

    elif source_type == "gated_delta_net" and target_type == "gated_delta_net":
        # GatedDeltaNet to GatedDeltaNet: direct copy
        for name in [
            "gdn.in_proj_qkvz.weight",
            "gdn.in_proj_ba.weight",
            "gdn.out_proj.weight",
            "gdn.conv1d.weight",
            "gdn.conv1d.bias",
            "gdn.A_log",
            "gdn.dt_bias",
            "gdn.norm.weight",
        ]:
            mappings[target_prefix / name] = Ref(key=source_prefix / name)

    else:
        raise ValueError(
            f"No converter available for {source_type} -> {target_type}. "
            f"Use 'init: random' to initialize randomly, or implement a converter."
        )

    return mappings


def _plan_random_mixer(
    prefix: W,
    mixer_type: str,
    config: dict,
    hidden_size: int,
) -> dict[str, Expr]:
    """Build random initialization expressions for a mixer.

    Returns:
        Dict mapping target keys to expressions.
    """
    mappings: dict[str, Expr] = {}

    if mixer_type in ("attention", "sliding_window"):
        heads = config["heads"]
        head_groups = config["head_groups"]
        head_size = config["head_size"]
        q_size = heads * head_size
        kv_size = head_groups * head_size

        attn = prefix / "self_attn"
        mappings[attn / "q_proj" / "weight"] = Init(shape=(q_size, hidden_size), init_type="kaiming")
        mappings[attn / "k_proj" / "weight"] = Init(shape=(kv_size, hidden_size), init_type="kaiming")
        mappings[attn / "v_proj" / "weight"] = Init(shape=(kv_size, hidden_size), init_type="kaiming")
        mappings[attn / "o_proj" / "weight"] = Init(shape=(hidden_size, q_size), init_type="kaiming")

    elif mixer_type == "mamba":
        d_inner = config["d_inner"]
        d_state = config["d_state"]
        dt_rank = config["dt_rank"]
        d_xb = config["d_xb"]
        d_conv = config["d_conv"]
        repeat_kv_before_conv = config["repeat_kv_before_conv"]
        conv_bias = config["conv_bias"]
        dt_bias = config["dt_proj_bias"]
        dt_min = config["dt_min"]
        dt_max = config["dt_max"]
        dt_init_floor = config["dt_init_floor"]

        # Conv1d channels depend on repeat_kv_before_conv
        conv_channels = d_inner if repeat_kv_before_conv else d_xb

        # Core projections
        mappings[prefix / "in_proj" / "weight"] = Init(
            shape=(2 * d_inner + 2 * d_xb, hidden_size), init_type="kaiming"
        )
        mappings[prefix / "out_proj" / "weight"] = Init(shape=(hidden_size, d_inner), init_type="kaiming")

        # dt projections
        mappings[prefix / "dt_in_proj" / "weight"] = Init(shape=(dt_rank, hidden_size), init_type="kaiming")
        mappings[prefix / "dt_proj" / "weight"] = Init(shape=(d_inner, dt_rank), init_type="kaiming")
        # Conv1d
        mappings[prefix / "conv1d" / "weight"] = Init(shape=(conv_channels, 1, d_conv), init_type="kaiming")
        if conv_bias:
            mappings[prefix / "conv1d" / "bias"] = Init(shape=(conv_channels,), init_type="zeros")
        # dt_proj bias with proper initialization
        if dt_bias:
            mappings[prefix / "dt_proj" / "bias"] = Init(
                shape=(d_inner,), init_type="dt_bias", init_params={"dt_min": dt_min, "dt_max": dt_max, "dt_init_floor": dt_init_floor}
            )

        # SSM parameters - S4D initialization for A_log
        mappings[prefix / "A_log"] = Init(shape=(d_inner, d_state), init_type="s4d")
        mappings[prefix / "D"] = Init(shape=(d_inner,), init_type="ones")

    elif mixer_type == "gated_delta_net":
        # GatedDeltaNet random initialization
        num_v_heads = config["num_value_heads"]
        num_k_heads = config["num_key_heads"]
        head_k_dim = config["key_head_dim"]
        head_v_dim = config["value_head_dim"]
        conv_kernel_size = config.get("conv_kernel_size", 4)

        # GDN dimensions
        key_dim = head_k_dim * num_k_heads
        value_dim = head_v_dim * num_v_heads
        q_dim = head_k_dim * num_v_heads  # Queries use num_v_heads but head_k_dim
        conv_dim = key_dim * 2 + value_dim

        gdn = prefix / "gdn"

        # Combined Q/K/V/Z projection
        qkvz_size = q_dim + key_dim + value_dim * 2  # Q + K + V + Z
        mappings[gdn / "in_proj_qkvz" / "weight"] = Init(shape=(qkvz_size, hidden_size), init_type="kaiming")

        # Beta/alpha projection
        mappings[gdn / "in_proj_ba" / "weight"] = Init(shape=(key_dim * 2, hidden_size), init_type="zeros")

        # Output projection
        mappings[gdn / "out_proj" / "weight"] = Init(shape=(hidden_size, value_dim), init_type="kaiming")

        # Conv1d (depthwise, no bias)
        mappings[gdn / "conv1d" / "weight"] = Init(
            shape=(conv_dim, 1, conv_kernel_size), init_type="identity_conv"
        )

        # A_log for slow decay
        mappings[gdn / "A_log"] = Init(shape=(num_v_heads,), init_type="slow_decay")

        # dt_bias
        mappings[gdn / "dt_bias"] = Init(shape=(num_v_heads,), init_type="zeros")

        # Norm
        mappings[gdn / "norm" / "weight"] = Init(shape=(value_dim,), init_type="ones")

    return mappings


def _plan_mlp(
    target_layer_idx: int,
    source_layer_idx: int,
    source_mlp: dict,
    target_mlp: dict,
    hidden_size: int,
) -> dict[str, Expr]:
    """Build MLP conversion expressions.

    Parses init mode and dispatches to _plan_mlp_transfer or _plan_random_mlp.
    """
    # Parse init mode and dispatch
    if target_mlp.get("init") == "random":
        return _plan_random_mlp(target_layer_idx, target_mlp, hidden_size)
    else:
        # Default is transfer
        return _plan_mlp_transfer(
            target_layer_idx, source_layer_idx, source_mlp, target_mlp, hidden_size
        )


def _plan_mlp_transfer(
    target_layer_idx: int,
    source_layer_idx: int,
    source_mlp: dict,
    target_mlp: dict,
    hidden_size: int,
) -> dict[str, Expr]:
    """Build MLP transfer expressions. Fails if types differ."""
    mappings: dict[str, Expr] = {}

    source_mlp_path = W("model", "decoder", "blocks", source_layer_idx, "mlp")
    target_mlp_path = W("model", "decoder", "blocks", target_layer_idx, "mlp")

    source_type = source_mlp.get("type", "mlp")
    target_type = target_mlp.get("type", "mlp")

    if source_type != target_type:
        raise ValueError(
            f"Cannot transfer MLP weights: source type '{source_type}' != target type '{target_type}'. "
            f"Use 'init: random' to initialize randomly."
        )

    for proj in ["gate_proj", "up_proj", "down_proj"]:
        mappings[target_mlp_path / proj / "weight"] = Ref(key=source_mlp_path / proj / "weight")

    return mappings


def _plan_random_mlp(
    target_layer_idx: int,
    target_mlp: dict,
    hidden_size: int,
) -> dict[str, Expr]:
    """Build random MLP initialization expressions."""
    mappings: dict[str, Expr] = {}

    target_mlp_path = W("model", "decoder", "blocks", target_layer_idx, "mlp")
    intermediate_size = target_mlp["intermediate_size"]

    mappings[target_mlp_path / "gate_proj" / "weight"] = Init(
        shape=(intermediate_size, hidden_size), init_type="kaiming"
    )
    mappings[target_mlp_path / "up_proj" / "weight"] = Init(
        shape=(intermediate_size, hidden_size), init_type="kaiming"
    )
    mappings[target_mlp_path / "down_proj" / "weight"] = Init(
        shape=(hidden_size, intermediate_size), init_type="kaiming"
    )

    return mappings


def _plan_norms(
    target_layer_idx: int,
    source_layer_idx: int,
    source_block: dict,
    target_block: dict,
    hidden_size: int,
) -> dict[str, Expr]:
    """Build normalization conversion expressions.

    Parses init mode and dispatches to transfer or random init.
    """
    target_norm = target_block.get("normalization", {})

    # Parse init mode and dispatch
    if target_norm.get("init") == "random":
        return _plan_random_norms(target_layer_idx, hidden_size)
    else:
        # Default is transfer
        return _plan_norms_transfer(
            target_layer_idx, source_layer_idx, source_block, target_block, hidden_size
        )


def _plan_norms_transfer(
    target_layer_idx: int,
    source_layer_idx: int,
    source_block: dict,
    target_block: dict,
    hidden_size: int,
) -> dict[str, Expr]:
    """Build norm transfer expressions. Fails if types differ."""
    mappings: dict[str, Expr] = {}

    source_layer = W("model", "decoder", "blocks", source_layer_idx)
    target_layer = W("model", "decoder", "blocks", target_layer_idx)

    source_norm = source_block.get("normalization", {})
    target_norm = target_block.get("normalization", {})

    source_type = source_norm.get("type", "rms_norm")
    target_type = target_norm.get("type", "rms_norm")

    if source_type != target_type:
        raise ValueError(
            f"Cannot transfer norm weights: source type '{source_type}' != target type '{target_type}'. "
            f"Use 'init: random' to initialize randomly."
        )

    for norm_name in ["input_layernorm", "post_attention_layernorm"]:
        source_norm_path = source_layer / norm_name
        target_norm_path = target_layer / norm_name
        mappings[target_norm_path / "weight"] = Ref(key=source_norm_path / "weight")

    return mappings


def _plan_random_norms(
    target_layer_idx: int,
    hidden_size: int,
) -> dict[str, Expr]:
    """Build random norm initialization expressions."""
    mappings: dict[str, Expr] = {}

    target_layer = W("model", "decoder", "blocks", target_layer_idx)

    for norm_name in ["input_layernorm", "post_attention_layernorm"]:
        target_norm_path = target_layer / norm_name
        mappings[target_norm_path / "weight"] = Init(shape=(hidden_size,), init_type="ones")

    return mappings
