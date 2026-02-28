"""Renderable elements for architecture diagrams.

Primitives: Box, Symbol, Arrow, Label
Composites: DecoderBlock, BlockGroup, detail panels per mixer type

All visual styling goes through CSS class names — no inline fills/strokes.
All render methods are generators yielding svg.Element.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import svg as S

from fast_llm_external_models.apriel2.conversion.diagram.layout import (
    BBox,
    HStack,
    Size,
    VStack,
    Align,
)
from fast_llm_external_models.apriel2.conversion.diagram.model import (
    AttentionDisplayConfig,
    GDNDisplayConfig,
    KDADisplayConfig,
    MLPDisplayConfig,
    MixerSpec,
    StochasticMixerSpec,
)
from fast_llm_external_models.apriel2.conversion.diagram.style import Theme


# ═══════════════════════════════════════════════════════════════════════
# Primitives
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class Box:
    """Rounded rect with centered label. Styled entirely by css_class."""

    label: str
    css: str
    w: float | None = None
    h: float | None = None
    sublabel: str = ""
    bold: bool = False

    def measure(self, th: Theme) -> Size:
        w = self.w or th.geo.inner_w
        h = self.h or (th.geo.box_h if not self.sublabel else th.geo.box_h + 14)
        return Size(w, h)

    def render(self, bb: BBox, th: Theme) -> Iterator[S.Element]:
        txt_cls = ["t-label-bold"] if self.bold else ["t-label"]
        els: list[S.Element] = [
            S.Rect(x=bb.x, y=bb.y, width=bb.w, height=bb.h, rx=th.geo.rx),
        ]
        if self.sublabel:
            els.append(S.Text(x=bb.cx, y=bb.cy - 7, text=self.label, class_=txt_cls))
            els.append(S.Text(x=bb.cx, y=bb.cy + 8, text=self.sublabel, class_=["t-small"]))
        else:
            els.append(S.Text(x=bb.cx, y=bb.cy, text=self.label, class_=txt_cls))
        yield S.G(class_=["box", self.css], elements=els)


@dataclass
class Symbol:
    """Circle with plus or cross lines."""

    kind: str = "plus"
    r: float | None = None

    def measure(self, th: Theme) -> Size:
        r = self.r or th.geo.symbol_r
        return Size(r * 2, r * 2)

    def render(self, bb: BBox, th: Theme) -> Iterator[S.Element]:
        r = self.r or th.geo.symbol_r
        cx, cy = bb.cx, bb.cy
        d = r * 0.5
        lines: list[S.Element] = []
        if self.kind == "plus":
            lines = [
                S.Line(x1=cx - d, y1=cy, x2=cx + d, y2=cy),
                S.Line(x1=cx, y1=cy - d, x2=cx, y2=cy + d),
            ]
        else:
            d2 = r * 0.4
            lines = [
                S.Line(x1=cx - d2, y1=cy - d2, x2=cx + d2, y2=cy + d2),
                S.Line(x1=cx - d2, y1=cy + d2, x2=cx + d2, y2=cy - d2),
            ]
        yield S.G(class_=["symbol"], elements=[S.Circle(cx=cx, cy=cy, r=r), *lines])


@dataclass
class Arrow:
    """Vertical arrow (down or up). Fixed height."""

    direction: str = "down"
    length: float = 20

    def measure(self, th: Theme) -> Size:  # noqa: ARG002
        return Size(0, self.length)

    def render(self, bb: BBox, th: Theme) -> Iterator[S.Element]:  # noqa: ARG002
        x = bb.cx
        if self.direction == "down":
            yield S.Line(
                x1=x, y1=bb.y, x2=x, y2=bb.bottom - 4,
                class_=["arrow"], marker_end="url(#arr-d)",
            )
        else:
            yield S.Line(
                x1=x, y1=bb.bottom, x2=x, y2=bb.y + 4,
                class_=["arrow"], marker_end="url(#arr-u)",
            )


@dataclass
class Label:
    """Standalone text label for annotations."""

    text: str
    css: str = "t-ann"
    anchor: str = "start"

    def measure(self, th: Theme) -> Size:
        sz = {
            "t-ann": th.typo.sz_ann, "t-dim": th.typo.sz_ann,
            "t-count": th.typo.sz_ann, "t-note": th.typo.sz_ann,
            "t-small": th.typo.sz_small, "t-title": th.typo.sz_title,
            "t-sub": th.typo.sz_subtitle, "t-label": th.typo.sz_label,
        }.get(self.css, th.typo.sz_ann)
        return Size(len(self.text) * sz * 0.6, sz + 4)

    def render(self, bb: BBox, th: Theme) -> Iterator[S.Element]:  # noqa: ARG002
        if self.anchor == "middle":
            x = bb.cx
        elif self.anchor == "end":
            x = bb.right
        else:
            x = bb.x
        yield S.Text(
            x=x, y=bb.cy, text=self.text,
            class_=[self.css], text_anchor=self.anchor,
            dominant_baseline="central",
        )


# ═══════════════════════════════════════════════════════════════════════
# SVG definitions (markers, filters)
# ═══════════════════════════════════════════════════════════════════════


def defs(th: Theme) -> S.Defs:  # noqa: ARG001
    """Arrow markers + drop shadow filter."""
    vb = S.ViewBoxSpec(0, 0, 10, 10)
    return S.Defs(elements=[
        S.Marker(
            id="arr-d", viewBox=vb, refX=5, refY=10,
            markerWidth=8, markerHeight=8, orient="auto-start-reverse",
            elements=[S.Path(d=[S.MoveTo(0, 0), S.LineTo(5, 10), S.LineTo(10, 0)], class_=["arrow"])],
        ),
        S.Marker(
            id="arr-u", viewBox=vb, refX=5, refY=0,
            markerWidth=8, markerHeight=8, orient="auto-start-reverse",
            elements=[S.Path(d=[S.MoveTo(0, 10), S.LineTo(5, 0), S.LineTo(10, 10)], class_=["arrow"])],
        ),
        S.Filter(
            id="shadow", x="-5%", y="-5%", width="110%", height="110%",
            elements=[S.FeDropShadow(dx=0, dy=1, stdDeviation=2, flood_opacity=0.08)],
        ),
    ])


# ═══════════════════════════════════════════════════════════════════════
# Connectors and braces (convenience wrappers around layout functions)
# ═══════════════════════════════════════════════════════════════════════


def connector_bezier(x1: float, y1: float, x2: float, y2: float) -> S.Path:
    """Dashed cubic Bezier from (x1,y1) to (x2,y2)."""
    mx = (x1 + x2) / 2
    return S.Path(
        d=[S.MoveTo(x1, y1), S.CubicBezier(x1=mx, y1=y1, x2=mx, y2=y2, x=x2, y=y2)],
        class_=["connector"],
    )


def brace_left(x: float, y: float, h: float) -> S.Path:
    """Left curly brace."""
    mid = y + h / 2
    return S.Path(
        d=[
            S.MoveTo(x, y),
            S.CubicBezier(x1=x - 12, y1=y, x2=x - 12, y2=mid, x=x - 18, y=mid),
            S.CubicBezier(x1=x - 12, y1=mid, x2=x - 12, y2=y + h, x=x, y=y + h),
        ],
        class_=["brace"],
    )


# ═══════════════════════════════════════════════════════════════════════
# Decoder block
# ═══════════════════════════════════════════════════════════════════════


def _mixer_box(mixer: MixerSpec | StochasticMixerSpec, th: Theme) -> Box:
    """Create the correct Box for a mixer spec."""
    if isinstance(mixer, StochasticMixerSpec):
        return Box("Stochastic Mixer", "box-stochastic", bold=True)

    css_map = {
        "attention": "box-attention",
        "sliding_window": "box-attention",
        "gdn": "box-ssm",
        "kda": "box-ssm",
        "mamba": "box-ssm",
    }
    css = css_map.get(mixer.mixer_type, "box-linear")

    if mixer.mixer_type in ("attention", "sliding_window"):
        display = mixer.display
        if isinstance(display, AttentionDisplayConfig) and display.window_size:
            return Box("Sliding window", css, sublabel="multi-head attention", bold=True)
        return Box("Grouped-query", css, sublabel="attention", bold=True)
    elif mixer.mixer_type == "gdn":
        return Box("Gated DeltaNet", css, bold=True)
    elif mixer.mixer_type == "kda":
        return Box("Kimi Delta", css, sublabel="Attention", bold=True)
    elif mixer.mixer_type == "mamba":
        return Box("Mamba SSM", css, bold=True)
    return Box(mixer.mixer_type, css)


@dataclass
class DecoderBlock:
    """Full decoder block: mixer + residual + norm + MLP + residual + norm."""

    mixer: MixerSpec | StochasticMixerSpec
    norm_type: str = "RMSNorm"
    block_w: float | None = None

    def measure(self, th: Theme) -> Size:
        w = self.block_w or th.geo.block_w
        g = th.geo.gap
        bh = th.geo.box_h
        mixer_box = _mixer_box(self.mixer, th)
        mh = mixer_box.measure(th).h
        h = 2 * th.geo.pad_block + mh + g + (bh - 4) + g + 18 + (bh - 4) + g + bh + g + 18
        return Size(w, h)

    def render(self, bb: BBox, th: Theme) -> Iterator[S.Element]:
        g = th.geo.gap
        bh = th.geo.box_h
        pad = th.geo.pad_block
        iw = bb.w - 2 * pad - 25
        ix = bb.x + pad
        cur_y = bb.y + pad

        elements: list[S.Element] = []

        def _collect(it: Iterator[S.Element]) -> None:
            elements.extend(it)

        # Residual symbol 2 (output)
        res_x = bb.right - pad - 2
        _collect(Symbol("plus").render(
            BBox(res_x - th.geo.symbol_r, cur_y, th.geo.symbol_r * 2, th.geo.symbol_r * 2), th))
        cur_y += th.geo.symbol_r * 2 + g - 4

        # Post-attention norm 2
        _collect(Box(self.norm_type, "box-norm", w=iw, h=bh - 4).render(
            BBox(ix, cur_y, iw, bh - 4), th))
        cur_y += bh - 4 + g

        # Feed forward
        _collect(Box("Feed forward", "box-mlp", w=iw, h=bh).render(
            BBox(ix, cur_y, iw, bh), th))
        cur_y += bh + g + 4

        # Residual symbol 1
        _collect(Symbol("plus").render(
            BBox(res_x - th.geo.symbol_r, cur_y - 2, th.geo.symbol_r * 2, th.geo.symbol_r * 2), th))
        cur_y += th.geo.symbol_r * 2 + g - 6

        # Pre-attention norm 1
        _collect(Box(self.norm_type, "box-norm", w=iw, h=bh - 4).render(
            BBox(ix, cur_y, iw, bh - 4), th))
        cur_y += bh - 4 + g

        # Mixer
        mixer = _mixer_box(self.mixer, th)
        msz = mixer.measure(th)
        _collect(mixer.render(BBox(ix, cur_y, iw, msz.h), th))
        cur_y += msz.h + pad

        # Background rect (behind everything)
        actual_h = cur_y - bb.y
        yield S.Rect(x=bb.x, y=bb.y, width=bb.w, height=actual_h, class_=["block-bg"])
        yield from elements


@dataclass
class BlockGroup:
    """Collapsed block group with count badge + left brace."""

    block: DecoderBlock
    count: int
    label: str = ""

    def measure(self, th: Theme) -> Size:
        return self.block.measure(th)

    def render(self, bb: BBox, th: Theme) -> Iterator[S.Element]:
        yield from self.block.render(bb, th)
        yield brace_left(bb.x - 8, bb.y + 5, bb.h - 10)
        yield S.Text(
            x=bb.x - 30, y=bb.cy, text=f"{self.count} \u00d7",
            class_=["t-count"], text_anchor="end", dominant_baseline="central",
            font_size=th.typo.sz_subtitle, font_weight="600",
        )
        if self.label:
            yield S.Text(
                x=bb.x - 8, y=bb.cy + 18, text=self.label,
                class_=["t-note"], text_anchor="end", dominant_baseline="central",
            )


# ═══════════════════════════════════════════════════════════════════════
# Detail panels — one per mixer type
# ═══════════════════════════════════════════════════════════════════════


MixerDetail = "AttentionDetail | GDNDetail | KDADetail"


@dataclass
class AttentionDetail:
    """Detail panel for attention (full or sliding window)."""

    config: AttentionDisplayConfig
    w: float = 220

    def _title_line1(self) -> str:
        return "Sliding Window" if self.config.window_size else "Grouped-Query"

    def _title_line2(self) -> str:
        return "Multi-Head Attention" if self.config.window_size else "Attention"

    def _build(self, th: Theme) -> list[tuple[BBox, Box | Symbol | HStack]]:
        bh = th.geo.box_h - 2
        proj_w = 55
        children: list[Box | Symbol | HStack] = [
            Box("Linear", "box-linear", w=70, h=bh - 2),
            Symbol("cross"),
            Box(self._title_line1(), "box-attention", w=self.w - 30, h=bh + 6,
                sublabel=self._title_line2(), bold=True),
            HStack([Box("q", "box-norm", w=65, h=bh - 4),
                    Box("k", "box-norm", w=65, h=bh - 4)], gap=10),
            HStack([Box("Linear", "box-linear", w=proj_w, h=bh - 4),
                    Box("Linear", "box-linear", w=proj_w, h=bh - 4),
                    Box("Linear", "box-linear", w=proj_w, h=bh - 4)], gap=8),
        ]
        return _detail_layout(children, th)

    def measure(self, th: Theme) -> Size:
        items = self._build(th)
        total_h = items[-1][0].bottom if items else 0
        return Size(self.w + 100, total_h)

    def render(self, bb: BBox, th: Theme) -> Iterator[S.Element]:
        items = self._build(th)
        for child_bb, child in items:
            shifted = BBox(bb.x + child_bb.x, bb.y + child_bb.y, child_bb.w, child_bb.h)
            yield from child.render(shifted, th)

        # Side annotations
        attn_bb = BBox(bb.x + items[2][0].x, bb.y + items[2][0].y, items[2][0].w, items[2][0].h)
        heads = self.config.heads
        kv_heads = self.config.kv_heads or heads
        ax = bb.x + self.w - 10
        yield S.Text(x=ax, y=attn_bb.cy - 8, text=f"{heads} attention heads", class_=["t-count"])
        yield S.Text(x=ax, y=attn_bb.cy + 6, text=f"{kv_heads} key & value heads", class_=["t-count"])
        if self.config.window_size:
            yield S.Text(x=ax, y=attn_bb.cy + 20,
                         text=f"window = {self.config.window_size:,}", class_=["t-dim"])

        # "RoPE" label
        qk_bb = BBox(bb.x + items[3][0].x, bb.y + items[3][0].y, items[3][0].w, items[3][0].h)
        yield S.Text(x=qk_bb.x - 8, y=qk_bb.cy, text="RoPE", class_=["t-note"],
                     text_anchor="end", dominant_baseline="central")

        # q k v labels
        proj_row_bb = BBox(bb.x + items[4][0].x, bb.y + items[4][0].y, items[4][0].w, items[4][0].h)
        proj_w, proj_gap = 55, 8
        for i, lbl in enumerate("qkv"):
            px = proj_row_bb.x + i * (proj_w + proj_gap) + proj_w / 2
            yield S.Text(x=px, y=proj_row_bb.bottom + 10, text=lbl, class_=["t-note"],
                         text_anchor="middle")


@dataclass
class GDNDetail:
    """Detail panel for Gated DeltaNet."""

    config: GDNDisplayConfig
    w: float = 240

    def _build(self, th: Theme) -> list[tuple[BBox, Box | Symbol | HStack]]:
        bh = th.geo.box_h - 2
        pw = (self.w - 60) // 2
        children: list[Box | Symbol | HStack] = [
            Box("Linear", "box-linear", w=70, h=bh - 2),
            Symbol("cross"),
            Box("Gated RMSNorm", "box-norm", w=self.w - 40, h=bh),
            Box("Gated Delta Rule", "box-ssm", w=self.w - 30, h=bh + 10, bold=True),
            Box("CausalConv1d", "box-conv", w=self.w - 60, h=bh),
            HStack([
                Box("in_proj_qkvz", "box-linear", w=pw + 10, h=bh),
                Box("in_proj_\u03b2\u03b1", "box-gate", w=pw - 5, h=bh),
            ], gap=10),
        ]
        return _detail_layout(children, th)

    def measure(self, th: Theme) -> Size:
        items = self._build(th)
        total_h = items[-1][0].bottom if items else 0
        return Size(self.w + 80, total_h)

    def render(self, bb: BBox, th: Theme) -> Iterator[S.Element]:
        items = self._build(th)
        for child_bb, child in items:
            shifted = BBox(bb.x + child_bb.x, bb.y + child_bb.y, child_bb.w, child_bb.h)
            yield from child.render(shifted, th)

        gdr_bb = BBox(bb.x + items[3][0].x, bb.y + items[3][0].y, items[3][0].w, items[3][0].h)
        conv_bb = BBox(bb.x + items[4][0].x, bb.y + items[4][0].y, items[4][0].w, items[4][0].h)
        ax = bb.x + self.w - 15
        v_heads = self.config.value_heads
        k_heads = self.config.key_heads
        k_dim = self.config.key_head_dim
        v_dim = self.config.value_head_dim
        conv_k = self.config.conv_kernel
        yield S.Text(x=ax, y=gdr_bb.cy - 14, text=f"{v_heads} value heads", class_=["t-count"])
        yield S.Text(x=ax, y=gdr_bb.cy, text=f"{k_heads} key heads", class_=["t-count"])
        yield S.Text(x=ax, y=gdr_bb.cy + 14, text=f"dim: {k_dim}\u00d7{v_dim}", class_=["t-dim"])
        yield S.Text(x=ax, y=conv_bb.cy, text=f"kernel = {conv_k}", class_=["t-dim"])


@dataclass
class KDADetail:
    """Detail panel for Kimi Delta Attention."""

    config: KDADisplayConfig
    w: float = 260

    def _build(self, th: Theme) -> list[tuple[BBox, Box | Symbol | HStack]]:
        bh = th.geo.box_h - 2
        cw = (self.w - 80) // 3
        children: list[Box | Symbol | HStack] = [
            Box("Linear", "box-linear", w=70, h=bh - 2),
            Symbol("cross"),
            Box("Gated RMSNorm", "box-norm", w=self.w - 50, h=bh),
            Box("Kimi Delta Attention", "box-ssm", w=self.w - 30, h=bh + 10, bold=True),
            HStack([Box("Conv", "box-conv", w=cw, h=bh - 4),
                    Box("Conv", "box-conv", w=cw, h=bh - 4),
                    Box("Conv", "box-conv", w=cw, h=bh - 4)], gap=5),
            HStack([Box("Linear", "box-linear", w=cw, h=bh - 4),
                    Box("Linear", "box-linear", w=cw, h=bh - 4),
                    Box("Linear", "box-linear", w=cw, h=bh - 4)], gap=5),
        ]
        return _detail_layout(children, th)

    def measure(self, th: Theme) -> Size:
        items = self._build(th)
        total_h = items[-1][0].bottom + 12 if items else 0
        return Size(self.w + 90, total_h)

    def render(self, bb: BBox, th: Theme) -> Iterator[S.Element]:
        items = self._build(th)
        for child_bb, child in items:
            shifted = BBox(bb.x + child_bb.x, bb.y + child_bb.y, child_bb.w, child_bb.h)
            yield from child.render(shifted, th)

        kda_bb = BBox(bb.x + items[3][0].x, bb.y + items[3][0].y, items[3][0].w, items[3][0].h)
        ax = bb.x + self.w - 10
        ay = kda_bb.y - 5
        yield S.Text(x=ax, y=ay, text="Gate kernel:", class_=["t-note"])
        yield S.Text(x=ax, y=ay + 13, text="f_a \u2192 f_b (low-rank)", class_=["t-note"],
                     font_size=th.typo.sz_small)
        yield S.Text(x=ax, y=ay + 28, text="Output gate:", class_=["t-note"])
        yield S.Text(x=ax, y=ay + 41, text="g_a \u2192 g_b (low-rank)", class_=["t-note"],
                     font_size=th.typo.sz_small)
        n_heads = self.config.heads
        head_dim = self.config.head_dim
        yield S.Text(x=ax, y=ay + 60, text=f"{n_heads} heads, dim {head_dim}", class_=["t-count"])

        # q k v labels under projections (item 5)
        proj_bb = BBox(bb.x + items[5][0].x, bb.y + items[5][0].y, items[5][0].w, items[5][0].h)
        cw = (self.w - 80) // 3
        for i, lbl in enumerate("qkv"):
            px = proj_bb.x + i * (cw + 5) + cw / 2
            yield S.Text(x=px, y=proj_bb.bottom + 10, text=lbl, class_=["t-note"],
                         text_anchor="middle")


@dataclass
class MLPDetail:
    """Detail panel for the MLP / Feed Forward sub-block."""

    config: MLPDisplayConfig
    w: float = 180

    def _build(self, th: Theme) -> list[tuple[BBox, Box | Symbol | HStack]]:
        bh = th.geo.box_h - 4
        act = self.config.activation.upper() if self.config.activation else "SILU"
        if self.config.gated:
            half = (self.w - 30) // 2
            children: list[Box | Symbol | HStack] = [
                Box("Linear", "box-linear", w=self.w - 20, h=bh),
                Symbol("cross"),
                HStack([Box(act, "box-activation", w=half, h=bh),
                        Box("Linear", "box-linear", w=half, h=bh)], gap=6),
                Box("Linear", "box-linear", w=self.w - 20, h=bh),
            ]
        else:
            children = [
                Box("Linear", "box-linear", w=self.w - 20, h=bh),
                Box(act, "box-activation", w=self.w - 20, h=bh),
                Box("Linear", "box-linear", w=self.w - 20, h=bh),
            ]
        return _detail_layout(children, th)

    def measure(self, th: Theme) -> Size:
        items = self._build(th)
        total_h = items[-1][0].bottom if items else 0
        return Size(self.w + 60, total_h)

    def render(self, bb: BBox, th: Theme) -> Iterator[S.Element]:
        items = self._build(th)
        for child_bb, child in items:
            shifted = BBox(bb.x + child_bb.x, bb.y + child_bb.y, child_bb.w, child_bb.h)
            yield from child.render(shifted, th)

        intermediate = self.config.intermediate_size
        if intermediate:
            yield S.Text(x=bb.x + self.w - 10, y=bb.y + 8, text="Intermediate size", class_=["t-note"])
            yield S.Text(x=bb.x + self.w - 10, y=bb.y + 22, text=f"= {intermediate:,}", class_=["t-dim"])


def detail_for_mixer(mixer: MixerSpec) -> AttentionDetail | GDNDetail | KDADetail | None:
    """Factory: return the right detail panel for a mixer spec."""
    if isinstance(mixer.display, AttentionDisplayConfig):
        return AttentionDetail(mixer.display)
    elif isinstance(mixer.display, GDNDisplayConfig):
        return GDNDetail(mixer.display)
    elif isinstance(mixer.display, KDADisplayConfig):
        return KDADetail(mixer.display)
    return None


def _detail_layout(children: list, th: Theme, gap: float | None = None) -> list[tuple[BBox, object]]:
    """Measure children vertically and compute per-child BBoxes (relative coords)."""
    g = gap if gap is not None else th.geo.gap - 2
    sizes = [c.measure(th) for c in children]
    max_w = max((s.w for s in sizes), default=0)
    items: list[tuple[BBox, object]] = []
    y = 0.0
    for child, sz in zip(children, sizes, strict=True):
        x = (max_w - sz.w) / 2
        items.append((BBox(x, y, sz.w, sz.h), child))
        y += sz.h + g
    return items
