"""Expression-based plan system for weight transformations.

Core expression types (Pydantic discriminated union):
- Ref(key): Reference to a source tensor
- Slice(expr, slices): Slice an expression
- Concat(exprs, dim): Concatenate expressions along a dimension
- Init(shape, init_type): Random/constant initialization
- Reshape(expr, shape): Reshape an expression

Weight path utilities:
- W: Builder for structured weight key paths
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Annotated, Any, Callable, Iterator, Literal, TypedDict, Union, Unpack

import torch
from pydantic import BaseModel, ConfigDict, Field, GetCoreSchemaHandler, TypeAdapter
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema, core_schema
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

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source: type[Any],
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        """Parse as a string, then call cls(value) which runs __new__."""
        return core_schema.no_info_after_validator_function(
            cls,
            core_schema.str_schema(),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        schema: CoreSchema,
        handler: Callable[[CoreSchema], JsonSchemaValue],
    ) -> JsonSchemaValue:
        """Emit as a string in JSON schema."""
        json_schema = handler(schema)
        json_schema["type"] = "string"
        return json_schema


# =============================================================================
# Expression Types (Pydantic Discriminated Union)
# =============================================================================


class EvalKwargs(TypedDict):
    """Keyword arguments for expression evaluation."""

    device: torch.device
    dtype: torch.dtype
    generator: torch.Generator


class Ref(BaseModel):
    """Reference to a source tensor by key."""

    model_config = ConfigDict(frozen=True)

    type: Literal["ref"] = "ref"
    key: W

    def find_refs(self) -> set[W]:
        return {self.key}

    def evaluate(self, sources: dict[W, Tensor], **kwargs: Unpack[EvalKwargs]) -> Tensor:
        if self.key not in sources:
            raise KeyError(f"Source key not found: {self.key}")
        # Preserve source device/dtype - no conversion
        return sources[self.key].clone()

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

    def find_refs(self) -> set[W]:
        return self.expr.find_refs()

    def evaluate(self, sources: dict[W, Tensor], **kwargs: Unpack[EvalKwargs]) -> Tensor:
        tensor = self.expr.evaluate(sources, **kwargs)
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

    def find_refs(self) -> set[W]:
        refs = set()
        for expr in self.exprs:
            refs.update(expr.find_refs())
        return refs

    def evaluate(self, sources: dict[W, Tensor], **kwargs: Unpack[EvalKwargs]) -> Tensor:
        tensors = [e.evaluate(sources, **kwargs) for e in self.exprs]
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

    def find_refs(self) -> set[W]:
        return set()  # Init has no dependencies

    def evaluate(self, sources: dict[W, Tensor], **kwargs: Unpack[EvalKwargs]) -> Tensor:
        device, dtype, gen = kwargs["device"], kwargs["dtype"], kwargs["generator"]

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

        elif self.init_type == "scaled_identity_conv":
            # Scaled identity kernel for depthwise conv followed by SiLU
            # Uses 0.5 at last position to stay in SiLU's linear regime
            # Shape: (channels, 1, kernel_size)
            if len(self.shape) != 3 or self.shape[1] != 1:
                raise ValueError(f"scaled_identity_conv requires shape (C, 1, K), got {self.shape}")
            channels, _, kernel_size = self.shape
            tensor = torch.zeros(self.shape, device=device, dtype=dtype)
            tensor[:, 0, -1] = 0.5  # Scaled delta for SiLU linearity
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

    def find_refs(self) -> set[W]:
        return self.expr.find_refs()

    def evaluate(self, sources: dict[W, Tensor], **kwargs: Unpack[EvalKwargs]) -> Tensor:
        tensor = self.expr.evaluate(sources, **kwargs)
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

    mappings: dict[W, Expr] = Field(default_factory=dict)
    source_format: str = ""
    target_format: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.mappings)

    def __iter__(self) -> Iterator[tuple[W, Expr]]:
        return iter(self.mappings.items())

    def __getitem__(self, key: W) -> Expr:
        return self.mappings[key]

    def __contains__(self, key: W) -> bool:
        return key in self.mappings

    def __or__(self, other: "ExprPlan") -> "ExprPlan":
        """Compose plans: self | other means self (A→B) then other (B→C) = (A→C)."""
        return compose(self, other)

    def __add__(self, other: "ExprPlan") -> "ExprPlan":
        """Merge plans with disjoint targets: combine parallel sub-plans."""
        return merge(self, other)

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

    def fuse(self) -> "ExprPlan":
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
        from fast_llm_external_models.apriel2.conversion.render import render_tree

        return render_tree(self, collapse_layers=collapse_layers)


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


def merge(plan1: ExprPlan, plan2: ExprPlan) -> ExprPlan:
    """Merge two plans with disjoint targets.

    Unlike compose (which chains A→B→C), merge combines parallel sub-plans
    that produce different targets from the same source.

    Args:
        plan1: First plan.
        plan2: Second plan (must have disjoint targets).

    Returns:
        Merged plan with all targets from both plans.

    Raises:
        ValueError: If plans have overlapping target keys.
    """
    overlap = plan1.target_keys() & plan2.target_keys()
    if overlap:
        raise ValueError(f"Cannot merge plans with overlapping targets: {overlap}")

    return ExprPlan(
        mappings={**plan1.mappings, **plan2.mappings},
        source_format=plan1.source_format or plan2.source_format,
        target_format=plan1.target_format or plan2.target_format,
        metadata={
            "merged_from": [plan1.metadata, plan2.metadata],
        },
    )
