"""Plan tree rendering for visualization.

Renders an ExprPlan as a hierarchical tree with pattern collapsing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fast_llm_external_models.apriel2.conversion.expr import Expr, ExprPlan

from fast_llm_external_models.apriel2.conversion.expr import (
    Concat,
    Init,
    Ref,
    Reshape,
    Slice,
)


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
    child_sigs = tuple(sorted((name, _tree_structure_signature(child)) for name, child in node.children.items()))
    return ("node", child_sigs)


def _merge_sibling_trees(nodes: list[tuple[str, PlanTreeNode]]) -> PlanTreeNode:
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
            return PlanTreeNode(values=[(key, expr) for _, expr in node.values])
        else:
            return PlanTreeNode(
                children={name: _merge_sibling_trees([(key, child)]) for name, child in node.children.items()}
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
    result_children = {name: _collapse_siblings(child) for name, child in merged_children.items()}

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
        suffix = suffix[len(prefix) + len(suffix) - len(strings[0]) :]

    # Extract varying parts
    varying = []
    for s in strings:
        end_idx = len(s) - len(suffix) if suffix else len(s)
        varying.append(s[len(prefix) : end_idx])

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
        │           │   ├── q_proj/
        │           │   │   └── weight ← ...layers.[0..47]...q_proj.weight
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
        part_exprs = [(key, expr.exprs[i]) for key, expr in values if isinstance(expr, Concat) and len(expr.exprs) > i]
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
