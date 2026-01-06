"""Expression-based plan system for weight transformations.

This module defines the core expression types and plan class for declarative
weight transformations. Expressions are Pydantic models (JSON-serializable,
immutable, type-safe) that form an AST describing how to compute target tensors.

Expression Types
================

**Ref(key)**
    Reference to a source tensor by key. The fundamental leaf node.

**Slice(expr, slices)**
    Slice an expression along dimensions. Used for extracting subsets
    (e.g., taking first N rows of a weight matrix).

**Concat(exprs, dim)**
    Concatenate multiple expressions along a dimension. Used for building
    composite tensors (e.g., Mamba's fused in_proj from Q/K/V slices).

**Init(shape, init_type)**
    Random or constant initialization. Types include: zeros, ones, kaiming,
    normal, s4d (Mamba A_log), dt_bias (Mamba dt_proj.bias).

**Reshape(expr, shape)**
    Reshape an expression. Used for layout transformations.

Plan Composition
================

Plans compose via the `|` operator:

    full_plan = plan_a | plan_b  # plan_a produces B, plan_b consumes B

Composition works by substitution: Ref expressions in plan_b are replaced
with their producing expressions from plan_a. This is declarative composition
(substitution), not operational composition (function application).

Weight Paths
============

The `W` class builds structured weight key paths:

    layer = W("model", "decoder", "blocks", 0)
    q_weight = layer / "mixer" / "q_proj" / "weight"
    # Result: "model.decoder.blocks.0.mixer.q_proj.weight"

W is a string subclass, so it can be used directly as a dict key.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterator
from typing import Annotated, Any, Callable, Literal, TypedDict, Union, Unpack

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
        q = mixer / "q_proj" / "weight"
        # Result: "model.decoder.blocks.0.mixer.q_proj.weight"

        # Use directly - it's already a string!
        mappings[q] = Ref(key=source_q)
    """

    def __new__(cls, *parts) -> W:
        # Join parts, stripping any leading/trailing dots from each
        cleaned = []
        for p in parts:
            if p is None:
                continue
            s = str(p).strip(".")
            if s:
                cleaned.append(s)
        return super().__new__(cls, ".".join(cleaned))

    def __truediv__(self, other) -> W:
        if isinstance(other, (list, tuple)):
            return W(self, *other)
        return W(self, other)

    def __rtruediv__(self, other) -> W:
        return W(other, self)

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source: type[Any],
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
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
        json_schema = handler(schema)
        json_schema["type"] = "string"
        return json_schema


# =============================================================================
# Expression Types (Pydantic Discriminated Union)
# =============================================================================


class EvalKwargs(TypedDict):
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
        return sources[self.key].clone()

    def __repr__(self) -> str:
        return f"Ref(key={self.key!r})"


class Slice(BaseModel):
    """Slice an expression. slices: tuple of (start, stop, step) per dimension."""

    model_config = ConfigDict(frozen=True)

    type: Literal["slice"] = "slice"
    expr: Expr
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
    model_config = ConfigDict(frozen=True)

    type: Literal["concat"] = "concat"
    exprs: tuple[Expr, ...]
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
    """Initialize a tensor. init_type: zeros, ones, kaiming, normal, s4d, dt_bias,
    identity_conv, scaled_identity_conv, slow_decay.
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["init"] = "init"
    shape: tuple[int, ...]
    init_type: str = "kaiming"
    init_params: dict[str, Any] | None = None

    def find_refs(self) -> set[W]:
        return set()

    def evaluate(self, sources: dict[W, Tensor], **kwargs: Unpack[EvalKwargs]) -> Tensor:
        device, dtype, gen = kwargs["device"], kwargs["dtype"], kwargs["generator"]

        if self.init_type == "zeros":
            return torch.zeros(self.shape, device=device, dtype=dtype)

        elif self.init_type == "ones":
            return torch.ones(self.shape, device=device, dtype=dtype)

        elif self.init_type == "kaiming":
            tensor = torch.empty(self.shape, device=device, dtype=dtype)
            if len(self.shape) >= 2:
                fan_in = self.shape[1]
                bound = math.sqrt(1.0 / fan_in)
                tensor.uniform_(-bound, bound, generator=gen)
            else:
                tensor.normal_(0, 0.02, generator=gen)
            return tensor

        elif self.init_type == "normal":
            tensor = torch.empty(self.shape, device=device, dtype=dtype)
            tensor.normal_(0, 0.02, generator=gen)
            return tensor

        elif self.init_type == "s4d":
            # S4D real init for Mamba A_log: log(1..d_state) expanded to (d_inner, d_state)
            if len(self.shape) != 2:
                raise ValueError(f"s4d requires 2D shape, got {self.shape}")
            d_inner, d_state = self.shape
            A = torch.arange(1, d_state + 1, device=device, dtype=torch.float32)
            A = A.unsqueeze(0).expand(d_inner, -1).contiguous()
            return torch.log(A).to(dtype)

        elif self.init_type == "dt_bias":
            # Mamba dt_proj.bias: inverse-softplus of log-uniform samples in [dt_min, dt_max]
            if not self.init_params:
                raise ValueError("dt_bias requires init_params: dt_min, dt_max, dt_init_floor")
            dt_min = self.init_params["dt_min"]
            dt_max = self.init_params["dt_max"]
            dt_init_floor = self.init_params["dt_init_floor"]

            if len(self.shape) != 1:
                raise ValueError(f"dt_bias requires 1D shape, got {self.shape}")
            d_inner = self.shape[0]

            tensor = torch.empty(d_inner, device=device, dtype=dtype)
            tensor.uniform_(generator=gen)
            dt = torch.exp(tensor * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
            dt = dt.clamp(min=dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            return inv_dt

        elif self.init_type == "identity_conv":
            # Delta at last position: identity for causal depthwise conv
            if len(self.shape) != 3 or self.shape[1] != 1:
                raise ValueError(f"identity_conv requires shape (C, 1, K), got {self.shape}")
            tensor = torch.zeros(self.shape, device=device, dtype=dtype)
            tensor[:, 0, -1] = 1.0
            return tensor

        elif self.init_type == "scaled_identity_conv":
            # 0.5 at last position: identity scaled for SiLU's linear regime
            if len(self.shape) != 3 or self.shape[1] != 1:
                raise ValueError(f"scaled_identity_conv requires shape (C, 1, K), got {self.shape}")
            tensor = torch.zeros(self.shape, device=device, dtype=dtype)
            tensor[:, 0, -1] = 0.5
            return tensor

        elif self.init_type == "slow_decay":
            # GDN A_log: log(0.1) gives ~10-step half-life
            A = torch.full(self.shape, 0.1, device=device, dtype=torch.float32)
            return torch.log(A).to(dtype)

        else:
            raise ValueError(f"Unknown init type: {self.init_type}")

    def __repr__(self) -> str:
        if self.init_params:
            return f"Init(shape={self.shape}, init_type={self.init_type!r}, {self.init_params!r})"
        return f"Init(shape={self.shape}, init_type={self.init_type!r})"


class Reshape(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["reshape"] = "reshape"
    expr: Expr
    shape: tuple[int, ...]

    def find_refs(self) -> set[W]:
        return self.expr.find_refs()

    def evaluate(self, sources: dict[W, Tensor], **kwargs: Unpack[EvalKwargs]) -> Tensor:
        tensor = self.expr.evaluate(sources, **kwargs)
        return tensor.reshape(self.shape)

    def __repr__(self) -> str:
        return f"Reshape({self.expr}, {self.shape})"


Expr = Annotated[
    Union[Ref, Slice, Concat, Init, Reshape],
    Field(discriminator="type"),
]

Slice.model_rebuild()
Concat.model_rebuild()
Reshape.model_rebuild()

ExprAdapter: TypeAdapter[Expr] = TypeAdapter(Expr)


# =============================================================================
# Slice Helpers
# =============================================================================


def slice_spec(
    start: int | None = None,
    stop: int | None = None,
    step: int | None = None,
) -> tuple[int | None, int | None, int | None]:
    return (start, stop, step)


def full_slice() -> tuple[int | None, int | None, int | None]:
    """Equivalent to `:`."""
    return (None, None, None)


def make_slice(expr: Expr, dim_slices: list[tuple[int | None, int | None, int | None]]) -> Slice:
    return Slice(expr=expr, slices=tuple(dim_slices))


# =============================================================================
# Expression Utilities
# =============================================================================


def substitute(expr: Expr, bindings: dict[str, Expr]) -> Expr:
    """Replace Ref(key) with bindings[key]. Core of plan composition."""
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
    """Flatten nested Concat, collapse nested Reshape."""
    match expr:
        case Ref():
            return expr

        case Slice(expr=inner, slices=slices):
            return Slice(expr=fuse(inner), slices=slices)

        case Concat(exprs=exprs, dim=dim):
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

    Example:
        plan = ExprPlan(mappings={
            "out.weight": Ref(key="in.weight"),
            "out.bias": Init(shape=(10,), init_type="zeros"),
        })
        full_pipeline = plan1 | plan2 | plan3  # compose with |
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

    def __or__(self, other: ExprPlan) -> ExprPlan:
        return compose(self, other)

    def __add__(self, other: ExprPlan) -> ExprPlan:
        return merge(self, other)

    def source_keys(self) -> set[str]:
        refs = set()
        for expr in self.mappings.values():
            refs.update(expr.find_refs())
        return refs

    def target_keys(self) -> set[str]:
        return set(self.mappings.keys())

    def summary(self) -> dict[str, Any]:
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
        return ExprPlan(
            mappings={k: fuse(v) for k, v in self.mappings.items()},
            source_format=self.source_format,
            target_format=self.target_format,
            metadata=self.metadata,
        )

    def render_tree(self, collapse_layers: bool = True) -> str:
        """If collapse_layers, blocks.0, blocks.1, ... becomes blocks.[0..N]."""
        from fast_llm_external_models.apriel2.conversion.render import render_tree

        return render_tree(self, collapse_layers=collapse_layers)


# =============================================================================
# Plan Composition
# =============================================================================


def compose(plan1: ExprPlan, plan2: ExprPlan) -> ExprPlan:
    """plan1 (A→B) | plan2 (B→C) = (A→C). Substitutes plan2's Refs with plan1's expressions."""
    bindings = plan1.mappings

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

    return composed.fuse()


def merge(plan1: ExprPlan, plan2: ExprPlan) -> ExprPlan:
    """Combine parallel sub-plans with disjoint targets."""
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
