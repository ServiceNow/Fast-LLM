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
    ArchitectureModel,
    AttentionDisplayConfig,
    BlockSpec,
    GDNDisplayConfig,
    KDADisplayConfig,
    MambaDisplayConfig,
    MLPDisplayConfig,
    MixerSpec,
    StochasticMixerSpec,
    VisionEncoderSpec,
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
            markerWidth=8, markerHeight=8,
            elements=[S.Path(d=[S.MoveTo(0, 0), S.LineTo(5, 10), S.LineTo(10, 0)], class_=["arrow"])],
        ),
        S.Marker(
            id="arr-u", viewBox=vb, refX=5, refY=0,
            markerWidth=8, markerHeight=8,
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


def _arrow_up(x: float, y: float, sz: float = 4) -> S.Path:
    """Upward arrowhead tip at (x, y): a small V opening downward."""
    return S.Path(
        d=[S.MoveTo(x - sz, y + sz), S.LineTo(x, y), S.LineTo(x + sz, y + sz)],
        class_=["arrow"],
    )


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
# Layer stack column
# ═══════════════════════════════════════════════════════════════════════


def mixer_css_class(spec: BlockSpec) -> str:
    """Return the CSS class name for a block spec's mixer type."""
    mixer = spec.mixer
    if isinstance(mixer, StochasticMixerSpec):
        return "box-stochastic"
    css_map = {
        "attention": "box-attention",
        "sliding_window": "box-attention",
        "gdn": "box-ssm",
        "kda": "box-ssm",
        "mamba": "box-ssm",
    }
    return css_map.get(mixer.mixer_type, "box-linear")


def _mixer_short_name(spec: BlockSpec) -> str:
    """Derive a short display label for a block spec's mixer type."""
    mixer = spec.mixer
    if isinstance(mixer, StochasticMixerSpec):
        return "stochastic"
    return {
        "attention": "attn",
        "sliding_window": "SWA",
        "gdn": "GDN",
        "kda": "KDA",
        "mamba": "Mamba",
    }.get(mixer.mixer_type, mixer.mixer_type)


@dataclass
class _OverviewItem:
    """A single cell in the architecture overview column."""

    label: str  # text inside the cell
    sublabel: str  # second line (e.g. "V=131,072")
    css: str  # box CSS class
    spec: BlockSpec | None  # for connector drawing (None for non-decoder cells)
    range_label: str  # left annotation (e.g. "0..7")


@dataclass
class ArchitectureOverview:
    """Full architecture pipeline: Embedding → decoder blocks → Norm → LM Head."""

    arch: ArchitectureModel

    def _build_items(self) -> list[_OverviewItem]:
        """Build items in bottom-to-top order (Embedding first, LM Head last)."""
        arch = self.arch
        items: list[_OverviewItem] = []

        # 1. Embedding (bottom)
        sublabel = f"V={arch.vocab_size:,}" if arch.vocab_size else ""
        items.append(_OverviewItem(
            label="Embedding", sublabel=sublabel,
            css="box-embedding", spec=None, range_label="",
        ))

        # 2. Decoder block groups (in order: layer 0 at bottom → higher layers at top)
        for group in arch.block_groups:
            name = group.block_name or _mixer_short_name(group.block_spec)
            label = f"{name} \u00d7{group.count}" if group.count > 1 else name
            if group.count == 1:
                range_label = str(group.start_index)
            else:
                range_label = f"{group.start_index}..{group.end_index}"
            items.append(_OverviewItem(
                label=label, sublabel="",
                css=mixer_css_class(group.block_spec),
                spec=group.block_spec, range_label=range_label,
            ))

        # 3. Norm (use the norm_type from the first block group)
        norm_type = "RMSNorm"
        if arch.block_groups:
            norm_type = arch.block_groups[0].block_spec.norm_type
        items.append(_OverviewItem(
            label=norm_type, sublabel="",
            css="box-norm", spec=None, range_label="",
        ))

        # 4. LM Head (top)
        sublabel = ""
        if arch.hidden_size and arch.vocab_size:
            sublabel = f"{arch.hidden_size} \u2192 {arch.vocab_size:,}"
        items.append(_OverviewItem(
            label="LM Head", sublabel=sublabel,
            css="box-linear", spec=None, range_label="",
        ))

        return items

    def measure(self, th: Theme) -> Size:
        items = self._build_items()
        n = len(items)
        if n == 0:
            return Size(0, 0)
        range_label_w = 50  # space for range labels to the left
        total_h = n * th.geo.stack_cell_h + (n - 1) * th.geo.stack_cell_gap
        return Size(th.geo.stack_w + range_label_w, total_h)

    def render(self, bb: BBox, th: Theme) -> Iterator[S.Element]:
        items = self._build_items()
        range_label_w = 50
        box_x = bb.x + range_label_w
        box_w = th.geo.stack_w
        cell_h = th.geo.stack_cell_h
        gap = th.geo.stack_cell_gap
        cx = box_x + box_w / 2

        y = bb.y
        for i, item in enumerate(items):
            # Determine actual box height (taller for sublabels)
            bx = Box(item.label, item.css, w=box_w, h=cell_h, sublabel=item.sublabel)
            actual_h = bx.measure(th).h
            # Center the (possibly taller) box within the cell_h slot
            box_y = y + (cell_h - actual_h) / 2
            yield from bx.render(BBox(box_x, box_y, box_w, actual_h), th)

            # Range label to the left (only for decoder cells)
            if item.range_label:
                idx_x = box_x - 4
                yield S.Text(
                    x=idx_x, y=y + cell_h / 2,
                    text=item.range_label,
                    class_=["stack-label"], text_anchor="end",
                    dominant_baseline="central",
                )

            # Upward flow line between cells (arrow from current top to next bottom)
            if i < len(items) - 1:
                line_y1 = y + cell_h  # bottom of current cell
                line_y2 = y + cell_h + gap  # top of next cell
                if line_y2 - line_y1 > 1:
                    yield S.Line(x1=cx, y1=line_y1, x2=cx, y2=line_y2, class_=["arrow"])

            y += cell_h + gap

        # Tied weights: dashed connector from Embedding (bottom) to LM Head (top)
        if self.arch.tie_word_embeddings and len(items) >= 2:
            embed_cy = bb.y + cell_h / 2
            lm_head_cy = bb.y + (len(items) - 1) * (cell_h + gap) + cell_h / 2
            tied_x = box_x + box_w + 12  # right side of the boxes
            yield S.Path(
                d=[
                    S.MoveTo(box_x + box_w, embed_cy),
                    S.LineTo(tied_x, embed_cy),
                    S.LineTo(tied_x, lm_head_cy),
                    S.LineTo(box_x + box_w, lm_head_cy),
                ],
                class_=["connector"],
            )
            mid_y = (embed_cy + lm_head_cy) / 2
            yield S.Text(
                x=tied_x + 4, y=mid_y,
                text="tied weights",
                class_=["t-small"],
                dominant_baseline="central",
            )

    def cell_bboxes(self, bb: BBox, th: Theme) -> list[tuple[BBox, BlockSpec | None]]:
        """Return (bbox, spec) for each cell. Only decoder cells have non-None spec."""
        items = self._build_items()
        range_label_w = 50
        box_x = bb.x + range_label_w
        box_w = th.geo.stack_w
        cell_h = th.geo.stack_cell_h
        gap = th.geo.stack_cell_gap

        result: list[tuple[BBox, BlockSpec | None]] = []
        y = bb.y
        for item in items:
            result.append((BBox(box_x, y, box_w, cell_h), item.spec))
            y += cell_h + gap
        return result


@dataclass
class VisionEncoderColumn:
    """Vision encoder pipeline: Patch Embed → Encoder → Adapter."""

    vision: VisionEncoderSpec

    def _build_items(self) -> list[_OverviewItem]:
        """Build 3 items in bottom-to-top order: Patch Embed, Encoder, Adapter."""
        v = self.vision
        patch_h, patch_w = v.patch_size
        return [
            _OverviewItem(
                label=f"Patch Embed {patch_h}\u00d7{patch_w}", sublabel="",
                css="box-embedding", spec=None, range_label="",
            ),
            _OverviewItem(
                label=f"Vision Enc \u00d7{v.num_blocks}" if v.num_blocks > 1 else "Vision Enc",
                sublabel="",
                css=mixer_css_class(v.block_spec),
                spec=v.block_spec, range_label="",
            ),
            _OverviewItem(
                label="Adapter MLP", sublabel="",
                css="box-mlp", spec=None, range_label="",
            ),
        ]

    def measure(self, th: Theme) -> Size:
        return Size(th.geo.stack_w, 3 * th.geo.stack_cell_h + 2 * th.geo.stack_cell_gap)

    def render(self, bb: BBox, th: Theme) -> Iterator[S.Element]:
        items = self._build_items()
        box_w = th.geo.stack_w
        cell_h = th.geo.stack_cell_h
        gap = th.geo.stack_cell_gap
        cx = bb.x + box_w / 2

        y = bb.y
        for i, item in enumerate(items):
            bx = Box(item.label, item.css, w=box_w, h=cell_h)
            yield from bx.render(BBox(bb.x, y, box_w, cell_h), th)

            # Upward flow line between cells
            if i < len(items) - 1:
                line_y1 = y + cell_h
                line_y2 = y + cell_h + gap
                if line_y2 - line_y1 > 1:
                    yield S.Line(x1=cx, y1=line_y1, x2=cx, y2=line_y2, class_=["arrow"])

            y += cell_h + gap

    def cell_bboxes(self, bb: BBox, th: Theme) -> list[tuple[BBox, BlockSpec | None]]:
        """Return (bbox, spec) for each cell. Only the encoder cell (middle) has a non-None spec."""
        items = self._build_items()
        box_w = th.geo.stack_w
        cell_h = th.geo.stack_cell_h
        gap = th.geo.stack_cell_gap

        result: list[tuple[BBox, BlockSpec | None]] = []
        y = bb.y
        for item in items:
            result.append((BBox(bb.x, y, box_w, cell_h), item.spec))
            y += cell_h + gap
        return result


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
            return Box("Sliding-window attention", css, bold=True)
        return Box("Grouped-query attention", css, bold=True)
    elif mixer.mixer_type == "gdn":
        return Box("Gated DeltaNet", css, bold=True)
    elif mixer.mixer_type == "kda":
        return Box("Kimi Delta Attention", css, bold=True)
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
        sr = th.geo.symbol_r
        mixer_box = _mixer_box(self.mixer, th)
        mh = mixer_box.measure(th).h
        bg = g * 2  # branch gap — extra space where bypass lines leave the center line
        h = 2 * th.geo.pad_block + sr * 2 + g + (bh - 4) + g + bh + bg + sr * 2 + g + (bh - 4) + g + mh + g
        return Size(w, h)

    def render(self, bb: BBox, th: Theme) -> Iterator[S.Element]:
        g = th.geo.gap
        bg = g * 2  # branch gap — extra space where bypass lines leave the center line
        bh = th.geo.box_h
        pad = th.geo.pad_block
        sr = th.geo.symbol_r
        iw = bb.w - 2 * pad - g - 6  # reserve right margin for bypass lines
        ix = bb.x + pad
        cx = ix + iw / 2  # center x for flow lines
        bypass_x = bb.right - pad - 6  # single bypass x for both residuals
        cur_y = bb.y + pad

        elements: list[S.Element] = []
        arrows: list[S.Element] = []

        # ── Manual arrowhead helpers ──────────────────────────────
        def _arrow_left(x: float, y: float, sz: float = 4) -> S.Path:
            """Left-pointing arrowhead tip at (x, y): a small V opening rightward."""
            return S.Path(
                d=[S.MoveTo(x + sz, y - sz), S.LineTo(x, y), S.LineTo(x + sz, y + sz)],
                class_=["arrow"],
            )

        def _collect(it: Iterator[S.Element]) -> None:
            elements.extend(it)

        # ── Residual add 2 (output side) ──────────────────────────
        res2_bb = BBox(cx - sr, cur_y, sr * 2, sr * 2)
        _collect(Symbol("plus").render(res2_bb, th))
        cur_y += sr * 2 + g

        # ── Post-MLP norm ─────────────────────────────────────────
        norm2_bb = BBox(ix, cur_y, iw, bh - 4)
        _collect(Box(self.norm_type, "box-norm", w=iw, h=bh - 4).render(norm2_bb, th))
        cur_y += bh - 4 + g

        # ── Feed forward (MLP) ────────────────────────────────────
        mlp_bb = BBox(ix, cur_y, iw, bh)
        _collect(Box("Feed forward", "box-mlp", w=iw, h=bh).render(mlp_bb, th))
        cur_y += bh + bg

        # ── Residual add 1 ────────────────────────────────────────
        res1_bb = BBox(cx - sr, cur_y, sr * 2, sr * 2)
        _collect(Symbol("plus").render(res1_bb, th))
        cur_y += sr * 2 + g

        # ── Pre-mixer norm ────────────────────────────────────────
        norm1_bb = BBox(ix, cur_y, iw, bh - 4)
        _collect(Box(self.norm_type, "box-norm", w=iw, h=bh - 4).render(norm1_bb, th))
        cur_y += bh - 4 + g

        # ── Mixer ─────────────────────────────────────────────────
        mixer = _mixer_box(self.mixer, th)
        msz = mixer.measure(th)
        mixer_bb = BBox(ix, cur_y, iw, msz.h)
        _collect(mixer.render(mixer_bb, th))
        cur_y += msz.h + pad + g

        # ── Vertical flow arrows through center ───────────────────
        flow_segments = [
            (res2_bb.bottom, norm2_bb.y),
            (norm2_bb.bottom, mlp_bb.y),
            (mlp_bb.bottom, res1_bb.y),
            (res1_bb.bottom, norm1_bb.y),
            (norm1_bb.bottom, mixer_bb.y),
        ]
        for y1, y2 in flow_segments:
            if y2 - y1 > 1:
                arrows.append(S.Line(x1=cx, y1=y1, x2=cx, y2=y2, class_=["arrow"]))

        # ── Entry / exit arrow stubs (manual arrowheads) ──────────
        # Output stub: line from res2+ upward, arrowhead at top
        arrows.append(S.Line(x1=cx, y1=res2_bb.y, x2=cx, y2=res2_bb.y - 8, class_=["arrow"]))
        arrows.append(_arrow_up(cx, res2_bb.y - 8))

        # Input stub: line from below mixer upward, arrowhead at mixer bottom
        arrows.append(S.Line(x1=cx, y1=mixer_bb.bottom + g + 8, x2=cx, y2=mixer_bb.bottom, class_=["arrow"]))
        arrows.append(_arrow_up(cx, mixer_bb.bottom))

        # ── Residual bypass lines ─────────────────────────────────
        # Residual 1: wraps Mixer + Norm1 (bypass from input to res1+)
        input_y = mixer_bb.bottom + g
        arrows.append(S.Line(x1=cx, y1=input_y, x2=bypass_x, y2=input_y, class_=["arrow"]))
        arrows.append(S.Line(x1=bypass_x, y1=input_y, x2=bypass_x, y2=res1_bb.cy, class_=["arrow"]))
        arrows.append(S.Line(x1=bypass_x, y1=res1_bb.cy, x2=cx + sr, y2=res1_bb.cy, class_=["arrow"]))
        arrows.append(_arrow_left(cx + sr, res1_bb.cy))

        # Residual 2: wraps FFN + Norm2 (bypass from above res1+ to res2+)
        branch2_y = (mlp_bb.bottom + res1_bb.y) / 2
        arrows.append(S.Line(x1=cx, y1=branch2_y, x2=bypass_x, y2=branch2_y, class_=["arrow"]))
        arrows.append(S.Line(x1=bypass_x, y1=branch2_y, x2=bypass_x, y2=res2_bb.cy, class_=["arrow"]))
        arrows.append(S.Line(x1=bypass_x, y1=res2_bb.cy, x2=cx + sr, y2=res2_bb.cy, class_=["arrow"]))
        arrows.append(_arrow_left(cx + sr, res2_bb.cy))

        # ── Background rect (behind everything) ──────────────────
        actual_h = cur_y - bb.y
        yield S.Rect(x=bb.x, y=bb.y, width=bb.w, height=actual_h, class_=["block-bg"])
        yield from arrows
        yield from elements

    def mlp_bbox(self, bb: BBox, th: Theme) -> BBox:
        """Return the BBox of the 'Feed forward' box within this block."""
        g = th.geo.gap
        bh = th.geo.box_h
        pad = th.geo.pad_block
        sr = th.geo.symbol_r
        iw = bb.w - 2 * pad - g - 6
        ix = bb.x + pad
        cur_y = bb.y + pad
        cur_y += sr * 2 + g      # res2 symbol
        cur_y += (bh - 4) + g    # norm2
        return BBox(ix, cur_y, iw, bh)


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


MixerDetail = "AttentionDetail | GDNDetail | KDADetail | MambaDetail"

_ACTIVATION_DISPLAY: dict[str, str] = {"silu": "SiLU", "gelu": "GELU", "relu": "ReLU"}


@dataclass
class AttentionDetail:
    """Detail panel for attention (full or sliding window)."""

    config: AttentionDisplayConfig
    w: float = 220

    def _title(self) -> str:
        return "Sliding-window attention" if self.config.window_size else "Grouped-query attention"

    def _build(self, th: Theme) -> list[tuple[BBox, Box | Symbol | HStack]]:
        bh = th.geo.box_h - 2
        children: list[Box | Symbol | HStack] = [
            Box("o_proj", "box-linear", w=70, h=bh - 2),
            Box(self._title(), "box-attention", w=self.w - 30, h=bh,
                bold=True),
            HStack([Box("q", "box-norm", w=65, h=bh - 4),
                    Box("k", "box-norm", w=65, h=bh - 4)], gap=10),
            Box("qkv_proj", "box-linear", w=self.w - 60, h=bh - 2),
        ]
        return _detail_layout(children, th)

    def measure(self, th: Theme) -> Size:
        items = self._build(th)
        total_h = items[-1][0].bottom if items else 0
        return Size(self.w + 100, total_h)

    def render(self, bb: BBox, th: Theme) -> Iterator[S.Element]:
        items = self._build(th)
        yield from _DetailArrows.render(items, bb)
        for child_bb, child in items:
            shifted = BBox(bb.x + child_bb.x, bb.y + child_bb.y, child_bb.w, child_bb.h)
            yield from child.render(shifted, th)

        # Side annotations (item 1 = attention box)
        attn_bb = BBox(bb.x + items[1][0].x, bb.y + items[1][0].y, items[1][0].w, items[1][0].h)
        heads = self.config.heads
        kv_heads = self.config.kv_heads or heads
        ax = bb.x + self.w - 10
        yield S.Text(x=ax, y=attn_bb.cy - 8, text=f"{heads} attention heads", class_=["t-count"])
        yield S.Text(x=ax, y=attn_bb.cy + 6, text=f"{kv_heads} key & value heads", class_=["t-count"])
        if self.config.window_size:
            yield S.Text(x=ax, y=attn_bb.cy + 20,
                         text=f"window = {self.config.window_size:,}", class_=["t-dim"])

        # "RoPE" label (item 2 = q/k RoPE row)
        qk_bb = BBox(bb.x + items[2][0].x, bb.y + items[2][0].y, items[2][0].w, items[2][0].h)
        yield S.Text(x=qk_bb.x - 8, y=qk_bb.cy, text="RoPE", class_=["t-note"],
                     text_anchor="end", dominant_baseline="central")

        # q k v labels under qkv_proj (item 3)
        proj_bb = BBox(bb.x + items[3][0].x, bb.y + items[3][0].y, items[3][0].w, items[3][0].h)
        third = proj_bb.w / 3
        for i, lbl in enumerate("qkv"):
            px = proj_bb.x + (i + 0.5) * third
            yield S.Text(x=px, y=proj_bb.bottom + 10, text=lbl, class_=["t-note"],
                         text_anchor="middle")


@dataclass
class GDNDetail:
    """Detail panel for Gated DeltaNet."""

    config: GDNDisplayConfig
    w: float = 240

    def _build(self, th: Theme) -> list[tuple[BBox, Box | Symbol | HStack]]:
        bh = th.geo.box_h - 2
        pw = (self.w - 70) // 2
        children: list[Box | Symbol | HStack] = [
            Box("out_proj", "box-linear", w=70, h=bh - 2),
            Box("Gated RMSNorm", "box-norm", w=self.w - 40, h=bh),
            Box("Gated Delta Rule", "box-ssm", w=self.w - 30, h=bh + 10, bold=True),
            Box("CausalConv1d", "box-conv", w=self.w - 60, h=bh),
            HStack([
                Box("in_proj_qkvz", "box-linear", w=pw, h=bh),
                Box("in_proj_\u03b2\u03b1", "box-gate", w=pw, h=bh),
            ], gap=10),
        ]
        return _detail_layout(children, th)

    def measure(self, th: Theme) -> Size:
        items = self._build(th)
        total_h = items[-1][0].bottom if items else 0
        return Size(self.w + 80, total_h)

    def render(self, bb: BBox, th: Theme) -> Iterator[S.Element]:
        items = self._build(th)
        yield from _DetailArrows.render(items, bb)
        for child_bb, child in items:
            shifted = BBox(bb.x + child_bb.x, bb.y + child_bb.y, child_bb.w, child_bb.h)
            yield from child.render(shifted, th)

        gdr_bb = BBox(bb.x + items[2][0].x, bb.y + items[2][0].y, items[2][0].w, items[2][0].h)
        conv_bb = BBox(bb.x + items[3][0].x, bb.y + items[3][0].y, items[3][0].w, items[3][0].h)
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
            Box("o_proj", "box-linear", w=70, h=bh - 2),
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
        yield from _DetailArrows.render(items, bb)
        for child_bb, child in items:
            shifted = BBox(bb.x + child_bb.x, bb.y + child_bb.y, child_bb.w, child_bb.h)
            yield from child.render(shifted, th)

        kda_bb = BBox(bb.x + items[2][0].x, bb.y + items[2][0].y, items[2][0].w, items[2][0].h)
        ax = bb.x + self.w - 10
        ay = kda_bb.y - 5
        yield S.Text(x=ax, y=ay, text="\u03b2: b_proj", class_=["t-note"])
        yield S.Text(x=ax, y=ay + 15, text="Gate kernel:", class_=["t-note"])
        yield S.Text(x=ax, y=ay + 28, text="f_a \u2192 f_b (low-rank)", class_=["t-note"],
                     font_size=th.typo.sz_small)
        yield S.Text(x=ax, y=ay + 43, text="Output gate:", class_=["t-note"])
        yield S.Text(x=ax, y=ay + 56, text="g_a \u2192 g_b (low-rank)", class_=["t-note"],
                     font_size=th.typo.sz_small)
        n_heads = self.config.heads
        head_dim = self.config.head_dim
        yield S.Text(x=ax, y=ay + 75, text=f"{n_heads} heads, dim {head_dim}", class_=["t-count"])

        # q k v labels under projections (item 4)
        proj_bb = BBox(bb.x + items[4][0].x, bb.y + items[4][0].y, items[4][0].w, items[4][0].h)
        cw = (self.w - 80) // 3
        for i, lbl in enumerate("qkv"):
            px = proj_bb.x + i * (cw + 5) + cw / 2
            yield S.Text(x=px, y=proj_bb.bottom + 10, text=lbl, class_=["t-note"],
                         text_anchor="middle")


@dataclass
class MambaDetail:
    """Detail panel for Mamba SSM."""

    config: MambaDisplayConfig
    w: float = 220

    def _build(self, th: Theme) -> list[tuple[BBox, Box | Symbol | HStack]]:
        bh = th.geo.box_h - 2
        children: list[Box | Symbol | HStack] = [
            Box("out_proj", "box-linear", w=70, h=bh - 2),
            Box("Gated RMSNorm", "box-norm", w=self.w - 40, h=bh),
            Box("Mamba SSM", "box-ssm", w=self.w - 30, h=bh + 10, bold=True),
            Box("CausalConv1d", "box-conv", w=self.w - 60, h=bh),
            Box("in_proj", "box-linear", w=self.w - 60, h=bh),
        ]
        return _detail_layout(children, th)

    def measure(self, th: Theme) -> Size:
        items = self._build(th)
        total_h = items[-1][0].bottom if items else 0
        return Size(self.w + 80, total_h)

    def render(self, bb: BBox, th: Theme) -> Iterator[S.Element]:
        items = self._build(th)
        yield from _DetailArrows.render(items, bb)
        for child_bb, child in items:
            shifted = BBox(bb.x + child_bb.x, bb.y + child_bb.y, child_bb.w, child_bb.h)
            yield from child.render(shifted, th)

        # Side annotations
        ssm_bb = BBox(bb.x + items[2][0].x, bb.y + items[2][0].y, items[2][0].w, items[2][0].h)
        ax = bb.x + self.w - 15
        if self.config.d_state:
            yield S.Text(x=ax, y=ssm_bb.cy - 8, text=f"d_state = {self.config.d_state}", class_=["t-dim"])
        if self.config.d_inner:
            yield S.Text(x=ax, y=ssm_bb.cy + 6, text=f"d_inner = {self.config.d_inner}", class_=["t-dim"])
        conv_bb = BBox(bb.x + items[3][0].x, bb.y + items[3][0].y, items[3][0].w, items[3][0].h)
        if self.config.d_conv:
            yield S.Text(x=ax, y=conv_bb.cy, text=f"d_conv = {self.config.d_conv}", class_=["t-dim"])


@dataclass
class MLPDetail:
    """Detail panel for the MLP / Feed Forward sub-block."""

    config: MLPDisplayConfig
    w: float = 180

    def _build(self, th: Theme) -> list[tuple[BBox, Box | Symbol | HStack]]:
        bh = th.geo.box_h - 4
        act = _ACTIVATION_DISPLAY.get(self.config.activation, self.config.activation.upper()) if self.config.activation else "SiLU"
        if self.config.gated:
            half = (self.w - 30) // 2
            children: list[Box | Symbol | HStack] = [
                Box("down_proj", "box-linear", w=self.w - 20, h=bh),
                Symbol("cross"),
                HStack([Box(act, "box-activation", w=half, h=bh),
                        Box("gate_proj", "box-linear", w=half, h=bh)], gap=6),
                Box("up_proj", "box-linear", w=self.w - 20, h=bh),
            ]
        else:
            children = [
                Box("down_proj", "box-linear", w=self.w - 20, h=bh),
                Box(act, "box-activation", w=self.w - 20, h=bh),
                Box("up_proj", "box-linear", w=self.w - 20, h=bh),
            ]
        gap = th.geo.gap * 2 if self.config.gated else None
        return _detail_layout(children, th, gap=gap)

    def measure(self, th: Theme) -> Size:
        items = self._build(th)
        total_h = items[-1][0].bottom if items else 0
        pad = th.geo.gap
        return Size(self.w + 2 * pad + 60, total_h + 2 * pad)

    def _render_arrows(
        self,
        items: list[tuple[BBox, Box | Symbol | HStack]],
        bb: BBox,
        th: Theme,
    ) -> Iterator[S.Element]:
        """Yield arrows between items. For gated MLP, fork/merge around the HStack."""
        if not self.config.gated or len(items) < 4:
            # Ungated: simple vertical arrows
            yield from _DetailArrows.render(items, bb)
            return

        # Gated layout (top-to-bottom): down_proj, ×, HStack[act, gate], up_proj
        down_bb, cross_bb, hstack_bb, up_bb = (it[0] for it in items)
        hstack = items[2][1]  # the HStack object

        # Compute HStack child center-x positions (relative to hstack_bb)
        assert isinstance(hstack, HStack)
        child_sizes = [c.measure(th) for c in hstack.children]
        g = hstack.gap if hstack.gap is not None else th.geo.gap
        child0_cx = hstack_bb.x + child_sizes[0].w / 2
        child1_cx = hstack_bb.x + child_sizes[0].w + g + child_sizes[1].w / 2

        # Absolute positions
        cross_cx = bb.x + (cross_bb.x + cross_bb.w / 2)
        up_cx = bb.x + (up_bb.x + up_bb.w / 2)
        down_cx = bb.x + (down_bb.x + down_bb.w / 2)
        abs_child0_cx = bb.x + child0_cx
        abs_child1_cx = bb.x + child1_cx

        clearance = 3

        # down_proj → × : single upward arrow
        y1 = bb.y + down_bb.bottom
        y2 = bb.y + cross_bb.y
        if y2 - y1 > 1:
            tip_y = y1 + clearance
            yield S.Line(x1=down_cx, y1=y2, x2=down_cx, y2=tip_y, class_=["arrow"])
            yield _arrow_up(down_cx, tip_y)

        # × → HStack : two converging arrows from each child center up to × center
        y1 = bb.y + cross_bb.bottom
        y2 = bb.y + hstack_bb.y
        if y2 - y1 > 1:
            mid_y = (y1 + y2) / 2
            tip_y = y1 + clearance
            # Left child (act) → ×
            yield S.Path(
                d=[
                    S.MoveTo(abs_child0_cx, y2),
                    S.LineTo(abs_child0_cx, mid_y),
                    S.LineTo(cross_cx, mid_y),
                    S.LineTo(cross_cx, tip_y),
                ],
                class_=["arrow"],
            )
            yield _arrow_up(cross_cx, tip_y)
            # Right child (gate_proj) → ×
            yield S.Path(
                d=[
                    S.MoveTo(abs_child1_cx, y2),
                    S.LineTo(abs_child1_cx, mid_y),
                    S.LineTo(cross_cx, mid_y),
                    S.LineTo(cross_cx, tip_y),
                ],
                class_=["arrow"],
            )

        # HStack → up_proj : two diverging arrows from up_proj center to each child center
        y1 = bb.y + hstack_bb.bottom
        y2 = bb.y + up_bb.y
        if y2 - y1 > 1:
            mid_y = (y1 + y2) / 2
            # up_proj → left child (act)
            tip_y = y1 + clearance
            yield S.Path(
                d=[
                    S.MoveTo(up_cx, y2),
                    S.LineTo(up_cx, mid_y),
                    S.LineTo(abs_child0_cx, mid_y),
                    S.LineTo(abs_child0_cx, tip_y),
                ],
                class_=["arrow"],
            )
            yield _arrow_up(abs_child0_cx, tip_y)
            # up_proj → right child (gate_proj)
            yield S.Path(
                d=[
                    S.MoveTo(up_cx, y2),
                    S.LineTo(up_cx, mid_y),
                    S.LineTo(abs_child1_cx, mid_y),
                    S.LineTo(abs_child1_cx, tip_y),
                ],
                class_=["arrow"],
            )
            yield _arrow_up(abs_child1_cx, tip_y)

    def render(self, bb: BBox, th: Theme) -> Iterator[S.Element]:
        items = self._build(th)
        content_h = items[-1][0].bottom if items else 0
        pad = th.geo.gap

        # Background box with padding
        yield S.Rect(
            x=bb.x, y=bb.y, width=self.w + 2 * pad, height=content_h + 2 * pad,
            class_=["detail-bg"],
        )

        # Offset content by padding
        inner_bb = BBox(bb.x + pad, bb.y + pad, bb.w - 2 * pad, bb.h - 2 * pad)
        yield from self._render_arrows(items, inner_bb, th)
        for child_bb, child in items:
            shifted = BBox(inner_bb.x + child_bb.x, inner_bb.y + child_bb.y, child_bb.w, child_bb.h)
            yield from child.render(shifted, th)

        intermediate = self.config.intermediate_size
        if intermediate:
            ax = bb.x + self.w + pad - 10
            yield S.Text(x=ax, y=bb.y + pad + 8, text="Intermediate size", class_=["t-note"])
            yield S.Text(x=ax, y=bb.y + pad + 22, text=f"= {intermediate:,}", class_=["t-dim"])


def _mixer_spec_css(mixer: MixerSpec) -> str:
    """Return the CSS class for a single MixerSpec."""
    css_map = {
        "attention": "box-attention",
        "sliding_window": "box-attention",
        "gdn": "box-ssm",
        "kda": "box-ssm",
        "mamba": "box-ssm",
    }
    return css_map.get(mixer.mixer_type, "box-linear")


@dataclass
class StochasticMixerPanel:
    """Stochastic mixer dispatch breakdown showing sub-mixer options."""

    spec: StochasticMixerSpec
    w: float | None = None

    def _box_h(self, th: Theme) -> float:
        return th.geo.box_h

    def _gap(self) -> float:
        return 8.0

    def _pad(self) -> float:
        return 12.0

    def _title_h(self) -> float:
        return 20.0

    def measure(self, th: Theme) -> Size:
        w = self.w or th.geo.stack_w
        pad = self._pad()
        title_h = self._title_h()
        bh = self._box_h(th)
        gap = self._gap()
        n = len(self.spec.sub_mixers)
        content_h = n * bh + max(0, n - 1) * gap
        total_h = pad + title_h + content_h + pad
        return Size(w, total_h)

    def render(self, bb: BBox, th: Theme) -> Iterator[S.Element]:
        pad = self._pad()
        title_h = self._title_h()
        bh = self._box_h(th)
        gap = self._gap()
        inner_w = bb.w - 2 * pad

        # Dashed container background
        yield S.Rect(
            x=bb.x, y=bb.y, width=bb.w, height=bb.h,
            class_=["detail-bg"],
        )

        # Title
        yield S.Text(
            x=bb.x + bb.w / 2, y=bb.y + pad + title_h / 2,
            text="Stochastic Dispatch",
            class_=["t-note"], text_anchor="middle",
            dominant_baseline="central",
        )

        # Sub-mixer boxes
        box_y = bb.y + pad + title_h
        for name, sub_mixer in self.spec.sub_mixers:
            css = _mixer_spec_css(sub_mixer)
            label = name
            if name == self.spec.main_mixer_name:
                label += " \u2605"
            box = Box(label, css, w=inner_w, h=bh)
            box_bb = BBox(bb.x + pad, box_y, inner_w, bh)
            yield from box.render(box_bb, th)
            box_y += bh + gap

    def sub_mixer_bboxes(self, bb: BBox, th: Theme) -> list[tuple[BBox, MixerSpec]]:
        """Return (bbox, MixerSpec) for each sub-mixer box."""
        pad = self._pad()
        title_h = self._title_h()
        bh = self._box_h(th)
        gap = self._gap()
        inner_w = bb.w - 2 * pad

        result: list[tuple[BBox, MixerSpec]] = []
        box_y = bb.y + pad + title_h
        for _name, sub_mixer in self.spec.sub_mixers:
            result.append((BBox(bb.x + pad, box_y, inner_w, bh), sub_mixer))
            box_y += bh + gap
        return result


def detail_for_mixer(mixer: MixerSpec) -> AttentionDetail | GDNDetail | KDADetail | MambaDetail | None:
    """Factory: return the right detail panel for a mixer spec."""
    if isinstance(mixer.display, AttentionDisplayConfig):
        return AttentionDetail(mixer.display)
    elif isinstance(mixer.display, GDNDisplayConfig):
        return GDNDetail(mixer.display)
    elif isinstance(mixer.display, KDADisplayConfig):
        return KDADetail(mixer.display)
    elif isinstance(mixer.display, MambaDisplayConfig):
        return MambaDetail(mixer.display)
    return None


def _detail_layout(children: list, th: Theme, gap: float | None = None) -> list[tuple[BBox, object]]:
    """Measure children vertically and compute per-child BBoxes (relative coords)."""
    g = gap if gap is not None else th.geo.gap
    sizes = [c.measure(th) for c in children]
    max_w = max((s.w for s in sizes), default=0)
    items: list[tuple[BBox, object]] = []
    y = 0.0
    for child, sz in zip(children, sizes, strict=True):
        x = (max_w - sz.w) / 2
        items.append((BBox(x, y, sz.w, sz.h), child))
        y += sz.h + g
    return items


class _DetailArrows:
    """Draws vertical arrows between detail-panel children."""

    @staticmethod
    def render(items: list[tuple[BBox, object]], bb: BBox) -> Iterator[S.Element]:
        """Yield upward vertical arrows between consecutive items."""
        for i in range(len(items) - 1):
            cur_bb = items[i][0]
            next_bb = items[i + 1][0]
            cx = bb.x + (cur_bb.x + cur_bb.w / 2 + next_bb.x + next_bb.w / 2) / 2
            y1 = bb.y + cur_bb.bottom
            y2 = bb.y + next_bb.y
            if y2 - y1 > 1:
                tip_y = y1 + 3
                yield S.Line(x1=cx, y1=y2, x2=cx, y2=tip_y, class_=["arrow"])
                yield _arrow_up(cx, tip_y)
