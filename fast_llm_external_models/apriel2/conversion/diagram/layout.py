"""Composable layout engine with two-pass measure-then-render architecture.

Design principles:
1. All spacing is declarative — no manual cur_y arithmetic
2. Two-pass protocol: measure(th) -> Size, render(bb, th) -> Iterator[Element]
3. Composable — any Renderable wraps any other
4. Overlap detection built into LayoutRoot
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import svg as svglib

if TYPE_CHECKING:
    from fast_llm_external_models.apriel2.conversion.diagram.style import Theme


# ═══════════════════════════════════════════════════════════════════════
# Core types
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Size:
    w: float
    h: float


@dataclass(frozen=True)
class BBox:
    x: float
    y: float
    w: float
    h: float

    @property
    def center_x(self) -> float:
        return self.x + self.w / 2

    @property
    def center_y(self) -> float:
        return self.y + self.h / 2

    @property
    def cx(self) -> float:
        return self.center_x

    @property
    def cy(self) -> float:
        return self.center_y

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h

    def inset(self, dx: float, dy: float | None = None) -> BBox:
        """Return a new BBox inset by dx horizontally and dy vertically."""
        if dy is None:
            dy = dx
        return BBox(
            x=self.x + dx,
            y=self.y + dy,
            w=self.w - 2 * dx,
            h=self.h - 2 * dy,
        )


class Align(Enum):
    START = "start"
    CENTER = "center"
    END = "end"
    STRETCH = "stretch"


@runtime_checkable
class Renderable(Protocol):
    def measure(self, th: Theme) -> Size: ...
    def render(self, bb: BBox, th: Theme) -> Iterator[svglib.Element]: ...


def _align_offset(container: float, child: float, align: Align) -> float:
    """Compute offset for aligning child within container along one axis."""
    if align == Align.START or align == Align.STRETCH:
        return 0.0
    elif align == Align.CENTER:
        return (container - child) / 2
    else:  # END
        return container - child


# ═══════════════════════════════════════════════════════════════════════
# Stack containers
# ═══════════════════════════════════════════════════════════════════════


class VStack:
    """Vertical arrangement. gap=None -> use th.geo.gap."""

    def __init__(
        self,
        children: list[Renderable],
        gap: float | None = None,
        align: Align = Align.CENTER,
    ) -> None:
        self.children = children
        self.gap = gap
        self.align = align

    def measure(self, th: Theme) -> Size:
        if not self.children:
            return Size(0, 0)
        g = self.gap if self.gap is not None else th.geo.gap
        sizes = [c.measure(th) for c in self.children]
        w = max(s.w for s in sizes)
        h = sum(s.h for s in sizes) + g * (len(sizes) - 1)
        return Size(w, h)

    def render(self, bb: BBox, th: Theme) -> Iterator[svglib.Element]:
        g = self.gap if self.gap is not None else th.geo.gap
        y = bb.y
        for child in self.children:
            child_size = child.measure(th)
            if self.align == Align.STRETCH:
                child_bb = BBox(bb.x, y, bb.w, child_size.h)
            else:
                x_offset = _align_offset(bb.w, child_size.w, self.align)
                child_bb = BBox(bb.x + x_offset, y, child_size.w, child_size.h)
            yield from child.render(child_bb, th)
            y += child_size.h + g


class HStack:
    """Horizontal arrangement. gap=None -> use th.geo.gap."""

    def __init__(
        self,
        children: list[Renderable],
        gap: float | None = None,
        align: Align = Align.START,
    ) -> None:
        self.children = children
        self.gap = gap
        self.align = align

    def measure(self, th: Theme) -> Size:
        if not self.children:
            return Size(0, 0)
        g = self.gap if self.gap is not None else th.geo.gap
        sizes = [c.measure(th) for c in self.children]
        w = sum(s.w for s in sizes) + g * (len(sizes) - 1)
        h = max(s.h for s in sizes)
        return Size(w, h)

    def render(self, bb: BBox, th: Theme) -> Iterator[svglib.Element]:
        g = self.gap if self.gap is not None else th.geo.gap
        x = bb.x
        for child in self.children:
            child_size = child.measure(th)
            if self.align == Align.STRETCH:
                child_bb = BBox(x, bb.y, child_size.w, bb.h)
            else:
                y_offset = _align_offset(bb.h, child_size.h, self.align)
                child_bb = BBox(x, bb.y + y_offset, child_size.w, child_size.h)
            yield from child.render(child_bb, th)
            x += child_size.w + g


class ZStack:
    """Overlay children at the same position (first=back, last=front)."""

    def __init__(
        self,
        children: list[Renderable],
        align_x: Align = Align.CENTER,
        align_y: Align = Align.CENTER,
    ) -> None:
        self.children = children
        self.align_x = align_x
        self.align_y = align_y

    def measure(self, th: Theme) -> Size:
        if not self.children:
            return Size(0, 0)
        sizes = [c.measure(th) for c in self.children]
        w = max(s.w for s in sizes)
        h = max(s.h for s in sizes)
        return Size(w, h)

    def render(self, bb: BBox, th: Theme) -> Iterator[svglib.Element]:
        for child in self.children:
            child_size = child.measure(th)
            x_offset = _align_offset(bb.w, child_size.w, self.align_x)
            y_offset = _align_offset(bb.h, child_size.h, self.align_y)
            child_bb = BBox(
                bb.x + x_offset,
                bb.y + y_offset,
                child_size.w,
                child_size.h,
            )
            yield from child.render(child_bb, th)


# ═══════════════════════════════════════════════════════════════════════
# Spacing and sizing wrappers
# ═══════════════════════════════════════════════════════════════════════


class Padded:
    """Inset padding on each edge."""

    def __init__(
        self,
        child: Renderable,
        top: float = 0,
        right: float = 0,
        bottom: float = 0,
        left: float = 0,
    ) -> None:
        self.child = child
        self.top = top
        self.right = right
        self.bottom = bottom
        self.left = left

    @classmethod
    def uniform(cls, child: Renderable, padding: float) -> Padded:
        return cls(child, padding, padding, padding, padding)

    def measure(self, th: Theme) -> Size:
        child_size = self.child.measure(th)
        return Size(
            w=child_size.w + self.left + self.right,
            h=child_size.h + self.top + self.bottom,
        )

    def render(self, bb: BBox, th: Theme) -> Iterator[svglib.Element]:
        child_bb = BBox(
            x=bb.x + self.left,
            y=bb.y + self.top,
            w=bb.w - self.left - self.right,
            h=bb.h - self.top - self.bottom,
        )
        yield from self.child.render(child_bb, th)


class Spacer:
    """Fixed-size invisible gap."""

    def __init__(self, w: float = 0, h: float = 0) -> None:
        self.w = w
        self.h = h

    def measure(self, th: Theme) -> Size:  # noqa: ARG002
        return Size(self.w, self.h)

    def render(self, bb: BBox, th: Theme) -> Iterator[svglib.Element]:  # noqa: ARG002
        yield from ()


class FixedSize:
    """Override intrinsic width and/or height."""

    def __init__(self, child: Renderable, w: float | None = None, h: float | None = None) -> None:
        self.child = child
        self._w = w
        self._h = h

    def measure(self, th: Theme) -> Size:
        child_size = self.child.measure(th)
        return Size(
            self._w if self._w is not None else child_size.w,
            self._h if self._h is not None else child_size.h,
        )

    def render(self, bb: BBox, th: Theme) -> Iterator[svglib.Element]:
        yield from self.child.render(bb, th)


# Keep FixedWidth as an alias for backwards compatibility during migration
class FixedWidth:
    """Wraps a child with an overridden width. Deprecated: use FixedSize(child, w=...)."""

    def __init__(self, child: Renderable, width: float) -> None:
        self.child = child
        self.width = width

    def measure(self, th: Theme) -> Size:
        child_size = self.child.measure(th)
        return Size(self.width, child_size.h)

    def render(self, bb: BBox, th: Theme) -> Iterator[svglib.Element]:
        yield from self.child.render(bb, th)


class MinSize:
    """Floor constraint on width and/or height."""

    def __init__(self, child: Renderable, min_w: float = 0, min_h: float = 0) -> None:
        self.child = child
        self.min_w = min_w
        self.min_h = min_h

    def measure(self, th: Theme) -> Size:
        child_size = self.child.measure(th)
        return Size(max(child_size.w, self.min_w), max(child_size.h, self.min_h))

    def render(self, bb: BBox, th: Theme) -> Iterator[svglib.Element]:
        yield from self.child.render(bb, th)


class Clamped:
    """Constrain size to [min, max] range on each axis."""

    def __init__(
        self,
        child: Renderable,
        min_w: float = 0,
        max_w: float = math.inf,
        min_h: float = 0,
        max_h: float = math.inf,
    ) -> None:
        self.child = child
        self.min_w = min_w
        self.max_w = max_w
        self.min_h = min_h
        self.max_h = max_h

    def measure(self, th: Theme) -> Size:
        child_size = self.child.measure(th)
        return Size(
            max(self.min_w, min(child_size.w, self.max_w)),
            max(self.min_h, min(child_size.h, self.max_h)),
        )

    def render(self, bb: BBox, th: Theme) -> Iterator[svglib.Element]:
        yield from self.child.render(bb, th)


# ═══════════════════════════════════════════════════════════════════════
# Positioning containers
# ═══════════════════════════════════════════════════════════════════════


class Offset:
    """Translate child by (dx, dy). Measured size unchanged."""

    def __init__(self, child: Renderable, dx: float = 0, dy: float = 0) -> None:
        self.child = child
        self.dx = dx
        self.dy = dy

    def measure(self, th: Theme) -> Size:
        return self.child.measure(th)

    def render(self, bb: BBox, th: Theme) -> Iterator[svglib.Element]:
        yield from self.child.render(BBox(bb.x + self.dx, bb.y + self.dy, bb.w, bb.h), th)


class Aligned:
    """Position child within its allocated BBox by alignment."""

    def __init__(
        self,
        child: Renderable,
        align_x: Align = Align.CENTER,
        align_y: Align = Align.CENTER,
    ) -> None:
        self.child = child
        self.align_x = align_x
        self.align_y = align_y

    def measure(self, th: Theme) -> Size:
        return self.child.measure(th)

    def render(self, bb: BBox, th: Theme) -> Iterator[svglib.Element]:
        child_size = self.child.measure(th)
        x_offset = _align_offset(bb.w, child_size.w, self.align_x)
        y_offset = _align_offset(bb.h, child_size.h, self.align_y)
        child_bb = BBox(
            bb.x + x_offset,
            bb.y + y_offset,
            child_size.w if self.align_x != Align.STRETCH else bb.w,
            child_size.h if self.align_y != Align.STRETCH else bb.h,
        )
        yield from self.child.render(child_bb, th)


class Background:
    """Render a background rect behind child content."""

    def __init__(
        self,
        child: Renderable,
        css_class: str = "",
        rx: float = 0,
        padding: float = 0,
    ) -> None:
        self.child = child
        self.css_class = css_class
        self.rx = rx
        self.padding = padding

    def measure(self, th: Theme) -> Size:
        child_size = self.child.measure(th)
        p2 = self.padding * 2
        return Size(child_size.w + p2, child_size.h + p2)

    def render(self, bb: BBox, th: Theme) -> Iterator[svglib.Element]:
        yield svglib.Rect(
            x=bb.x, y=bb.y, width=bb.w, height=bb.h,
            rx=self.rx,
            class_=[self.css_class] if self.css_class else None,
        )
        inner = BBox(
            bb.x + self.padding,
            bb.y + self.padding,
            bb.w - self.padding * 2,
            bb.h - self.padding * 2,
        )
        yield from self.child.render(inner, th)


# ═══════════════════════════════════════════════════════════════════════
# Responsive containers
# ═══════════════════════════════════════════════════════════════════════


class Responsive:
    """Choose child layout based on available width.
    breakpoints: [(min_width, renderable)] sorted descending.
    Falls through to the first breakpoint <= available width."""

    def __init__(
        self,
        breakpoints: list[tuple[float, Renderable]],
        fallback: Renderable,
    ) -> None:
        self.breakpoints = sorted(breakpoints, key=lambda bp: bp[0], reverse=True)
        self.fallback = fallback

    def _select(self, width: float) -> Renderable:
        for min_w, child in self.breakpoints:
            if width >= min_w:
                return child
        return self.fallback

    def measure(self, th: Theme) -> Size:
        # Use largest breakpoint's child for intrinsic size
        if self.breakpoints:
            return self.breakpoints[0][1].measure(th)
        return self.fallback.measure(th)

    def render(self, bb: BBox, th: Theme) -> Iterator[svglib.Element]:
        child = self._select(bb.w)
        yield from child.render(bb, th)


class AspectFit:
    """Maintain width/height ratio within allocated space."""

    def __init__(self, child: Renderable, ratio: float) -> None:
        self.child = child
        self.ratio = ratio

    def measure(self, th: Theme) -> Size:
        child_size = self.child.measure(th)
        # Adjust to satisfy ratio
        w = child_size.w
        h = child_size.h
        if w / max(h, 0.001) > self.ratio:
            h = w / self.ratio
        else:
            w = h * self.ratio
        return Size(w, h)

    def render(self, bb: BBox, th: Theme) -> Iterator[svglib.Element]:
        # Fit within bb preserving ratio
        if bb.w / max(bb.h, 0.001) > self.ratio:
            # Height-constrained
            fitted_w = bb.h * self.ratio
            fitted_h = bb.h
        else:
            # Width-constrained
            fitted_w = bb.w
            fitted_h = bb.w / self.ratio
        x = bb.x + (bb.w - fitted_w) / 2
        y = bb.y + (bb.h - fitted_h) / 2
        yield from self.child.render(BBox(x, y, fitted_w, fitted_h), th)


# ═══════════════════════════════════════════════════════════════════════
# Overlap detection and resolution
# ═══════════════════════════════════════════════════════════════════════


def detect_overlaps(bboxes: list[BBox], min_distance: float = 0) -> list[tuple[int, int]]:
    """Return pairs of indices where BBoxes overlap or are closer than min_distance."""
    overlaps: list[tuple[int, int]] = []
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            a, b = bboxes[i], bboxes[j]
            if (a.x - min_distance < b.right and b.x - min_distance < a.right
                    and a.y - min_distance < b.bottom and b.y - min_distance < a.bottom):
                overlaps.append((i, j))
    return overlaps


def resolve_overlaps(
    bboxes: list[BBox],
    axis: Literal["x", "y"] = "y",
    min_distance: float = 0,
) -> list[BBox]:
    """Nudge overlapping BBoxes apart along axis."""
    if not bboxes:
        return []

    if axis == "y":
        indexed = sorted(enumerate(bboxes), key=lambda ib: ib[1].y)
        result = list(bboxes)
        for k in range(1, len(indexed)):
            prev_idx = indexed[k - 1][0]
            curr_idx = indexed[k][0]
            prev = result[prev_idx]
            curr = result[curr_idx]
            needed = prev.bottom + min_distance
            if curr.y < needed:
                result[curr_idx] = BBox(curr.x, needed, curr.w, curr.h)
        return result
    else:  # x
        indexed = sorted(enumerate(bboxes), key=lambda ib: ib[1].x)
        result = list(bboxes)
        for k in range(1, len(indexed)):
            prev_idx = indexed[k - 1][0]
            curr_idx = indexed[k][0]
            prev = result[prev_idx]
            curr = result[curr_idx]
            needed = prev.right + min_distance
            if curr.x < needed:
                result[curr_idx] = BBox(needed, curr.y, curr.w, curr.h)
        return result


class LayoutRoot:
    """Top-level wrapper providing viewport awareness and overlap resolution."""

    def __init__(
        self,
        child: Renderable,
        viewport: Size | None = None,
        min_distance: float = 4.0,
    ) -> None:
        self.child = child
        self.viewport = viewport
        self.min_distance = min_distance

    def measure(self, th: Theme) -> Size:
        child_size = self.child.measure(th)
        if self.viewport is not None:
            return Size(
                min(child_size.w, self.viewport.w),
                min(child_size.h, self.viewport.h),
            )
        return child_size

    def render(self, bb: BBox, th: Theme) -> Iterator[svglib.Element]:
        yield from self.child.render(bb, th)


# ═══════════════════════════════════════════════════════════════════════
# Connector utilities
# ═══════════════════════════════════════════════════════════════════════


class AnchorSide(Enum):
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"


def anchor_point(bb: BBox, side: AnchorSide) -> tuple[float, float]:
    """Get the midpoint of a BBox edge."""
    if side == AnchorSide.LEFT:
        return (bb.x, bb.cy)
    elif side == AnchorSide.RIGHT:
        return (bb.right, bb.cy)
    elif side == AnchorSide.TOP:
        return (bb.cx, bb.y)
    else:  # BOTTOM
        return (bb.cx, bb.bottom)


def render_connector(
    from_bb: BBox,
    to_bb: BBox,
    from_side: AnchorSide = AnchorSide.RIGHT,
    to_side: AnchorSide = AnchorSide.LEFT,
    style: Literal["bezier", "straight"] = "bezier",
) -> Iterator[svglib.Element]:
    """Render a dashed connector between two BBoxes."""
    x1, y1 = anchor_point(from_bb, from_side)
    x2, y2 = anchor_point(to_bb, to_side)

    if style == "straight":
        yield svglib.Line(x1=x1, y1=y1, x2=x2, y2=y2, class_=["connector"])
    else:  # bezier
        mx = (x1 + x2) / 2
        yield svglib.Path(
            d=[svglib.MoveTo(x1, y1), svglib.CubicBezier(x1=mx, y1=y1, x2=mx, y2=y2, x=x2, y=y2)],
            class_=["connector"],
        )


def render_brace(
    bb: BBox,
    side: AnchorSide = AnchorSide.LEFT,
    inset: float = 5,
) -> Iterator[svglib.Element]:
    """Render a curly brace along one edge of a BBox."""
    if side == AnchorSide.LEFT:
        x = bb.x
        y = bb.y + inset
        h = bb.h - inset * 2
        mid = y + h / 2
        yield svglib.Path(
            d=[
                svglib.MoveTo(x, y),
                svglib.CubicBezier(x1=x - 12, y1=y, x2=x - 12, y2=mid, x=x - 18, y=mid),
                svglib.CubicBezier(x1=x - 12, y1=mid, x2=x - 12, y2=y + h, x=x, y=y + h),
            ],
            class_=["brace"],
        )
    elif side == AnchorSide.RIGHT:
        x = bb.right
        y = bb.y + inset
        h = bb.h - inset * 2
        mid = y + h / 2
        yield svglib.Path(
            d=[
                svglib.MoveTo(x, y),
                svglib.CubicBezier(x1=x + 12, y1=y, x2=x + 12, y2=mid, x=x + 18, y=mid),
                svglib.CubicBezier(x1=x + 12, y1=mid, x2=x + 12, y2=y + h, x=x, y=y + h),
            ],
            class_=["brace"],
        )
    elif side == AnchorSide.TOP:
        y = bb.y
        x_start = bb.x + inset
        w = bb.w - inset * 2
        mid = x_start + w / 2
        yield svglib.Path(
            d=[
                svglib.MoveTo(x_start, y),
                svglib.CubicBezier(x1=x_start, y1=y - 12, x2=mid, y2=y - 12, x=mid, y=y - 18),
                svglib.CubicBezier(x1=mid, y1=y - 12, x2=x_start + w, y2=y - 12, x=x_start + w, y=y),
            ],
            class_=["brace"],
        )
    else:  # BOTTOM
        y = bb.bottom
        x_start = bb.x + inset
        w = bb.w - inset * 2
        mid = x_start + w / 2
        yield svglib.Path(
            d=[
                svglib.MoveTo(x_start, y),
                svglib.CubicBezier(x1=x_start, y1=y + 12, x2=mid, y2=y + 12, x=mid, y=y + 18),
                svglib.CubicBezier(x1=mid, y1=y + 12, x2=x_start + w, y2=y + 12, x=x_start + w, y=y),
            ],
            class_=["brace"],
        )
