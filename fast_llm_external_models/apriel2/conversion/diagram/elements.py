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
    Spacer,
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

_ARROW_CLR = 1  # px clearance between arrowhead tip and target box edge


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
            tip_y = bb.bottom - _ARROW_CLR
            yield S.Line(x1=x, y1=bb.y, x2=x, y2=tip_y, class_=["arrow"])
            yield _arrow_down(x, tip_y)
        else:
            tip_y = bb.y + _ARROW_CLR
            yield S.Line(x1=x, y1=bb.bottom, x2=x, y2=tip_y, class_=["arrow"])
            yield _arrow_up(x, tip_y)


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


def _arrow_down(x: float, y: float, sz: float = 4) -> S.Path:
    """Downward arrowhead tip at (x, y): a small V opening upward."""
    return S.Path(
        d=[S.MoveTo(x - sz, y - sz), S.LineTo(x, y), S.LineTo(x + sz, y - sz)],
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
        "sliding_window": "box-swa",
        "gdn": "box-gdn",
        "kda": "box-kda",
        "mamba": "box-mamba",
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
    in_decoder: bool = True  # False for pre-decoder items outside the frame
    is_data_label: bool = False  # True → render as plain text, not a box


@dataclass
class ArchitectureOverview:
    """Full architecture pipeline: Embedding → decoder blocks → Norm → LM Head."""

    arch: ArchitectureModel

    def _build_items(self) -> list[_OverviewItem]:
        """Build items in render order: top of screen first, bottom last.

        Signal flows upward (bottom-to-top): Embedding feeds into block layers
        which feed into Norm → LM Head.  Pre-decoder items (Text tokens)
        sit below the decoder frame.
        """
        arch = self.arch
        items: list[_OverviewItem] = []

        # 1. LM Head (top, inside decoder)
        sublabel = ""
        if arch.hidden_size and arch.vocab_size:
            sublabel = f"{arch.hidden_size} \u2192 {arch.vocab_size:,}"
        items.append(_OverviewItem(
            label="LM Head", sublabel=sublabel,
            css="box-linear", spec=None, range_label="",
        ))

        # 2. Norm (inside decoder)
        norm_type = "RMSNorm"
        if arch.block_groups:
            norm_type = arch.block_groups[0].block_spec.norm_type
        items.append(_OverviewItem(
            label=norm_type, sublabel="",
            css="box-norm", spec=None, range_label="",
        ))

        # 3. Block groups REVERSED (highest layers at top, layer 0 at bottom)
        for group in reversed(arch.block_groups):
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

        # 4. Embedding (bottom of decoder)
        sublabel = f"V={arch.vocab_size:,}" if arch.vocab_size else ""
        items.append(_OverviewItem(
            label="Embedding", sublabel=sublabel,
            css="box-embedding", spec=None, range_label="",
        ))

        # 5. Pre-decoder items (outside frame, below decoder)
        items.append(_OverviewItem(
            label="Text tokens", sublabel="",
            css="box-norm", spec=None, range_label="",
            in_decoder=False, is_data_label=True,
        ))

        return items

    def measure(self, th: Theme) -> Size:
        items = self._build_items()
        n = len(items)
        if n == 0:
            return Size(0, 0)
        range_label_w = 50  # space for range labels to the left
        n_dec = sum(1 for i in items if i.in_decoder)
        n_pre = n - n_dec
        cell_h = th.geo.stack_cell_h
        gap = th.geo.stack_cell_gap
        g = th.geo.gap
        title_h = th.geo.title_h

        # Decoder frame: title strip + padding + cells
        dec_h = title_h + 2 * g + n_dec * cell_h + (n_dec - 1) * gap
        # Pre-decoder cells below frame
        pre_h = n_pre * cell_h + (n_pre - 1) * gap if n_pre > 0 else 0
        # Gap between frame and pre-decoder
        frame_gap = gap if n_pre > 0 else 0

        total_h = dec_h + frame_gap + pre_h
        return Size(th.geo.stack_w + range_label_w, total_h)

    def render(self, bb: BBox, th: Theme) -> Iterator[S.Element]:
        items = self._build_items()
        range_label_w = 50
        box_x = bb.x + range_label_w
        box_w = th.geo.stack_w
        cell_h = th.geo.stack_cell_h
        gap = th.geo.stack_cell_gap
        g = th.geo.gap
        title_h = th.geo.title_h
        cx = box_x + box_w / 2

        dec_items = [it for it in items if it.in_decoder]
        pre_items = [it for it in items if not it.in_decoder]
        n_dec = len(dec_items)

        # ── Decoder frame ───────────────────────────────────────────
        dec_h = title_h + 2 * g + n_dec * cell_h + (n_dec - 1) * gap
        frame, content_bb = _render_detail_frame(
            box_x - g, bb.y, box_w + 2 * g, dec_h,
            "Decoder", "detail-decoder", th,
        )
        yield frame

        # ── Decoder cells inside frame ──────────────────────────────
        y = content_bb.y
        for i, item in enumerate(dec_items):
            bx = Box(item.label, item.css, w=box_w, h=cell_h, sublabel=item.sublabel)
            actual_h = bx.measure(th).h
            box_y = y + (cell_h - actual_h) / 2
            yield from bx.render(BBox(box_x, box_y, box_w, actual_h), th)

            # Range label to the left
            if item.range_label:
                idx_x = box_x - 4
                yield S.Text(
                    x=idx_x, y=y + cell_h / 2,
                    text=item.range_label,
                    class_=["stack-label"], text_anchor="end",
                    dominant_baseline="central",
                )

            # Upward flow arrow between decoder cells
            if i < n_dec - 1:
                line_y1 = y + cell_h
                line_y2 = y + cell_h + gap
                if line_y2 - line_y1 > 1:
                    tip_y = line_y1 + _ARROW_CLR
                    yield S.Line(x1=cx, y1=line_y2, x2=cx, y2=tip_y, class_=["arrow"])
                    yield _arrow_up(cx, tip_y)

            y += cell_h + gap

        # ── Pre-decoder cells below frame ───────────────────────────
        pre_y = bb.y + dec_h + g  # gap between frame and pre-decoder
        # Arrow from first pre-decoder to last decoder (crossing frame boundary)
        if pre_items and dec_items:
            last_dec_bottom = content_bb.y + (n_dec - 1) * (cell_h + gap) + cell_h
            first_pre_top = pre_y
            if first_pre_top - last_dec_bottom > 1:
                tip_y = last_dec_bottom + _ARROW_CLR
                yield S.Line(x1=cx, y1=first_pre_top, x2=cx, y2=tip_y, class_=["arrow"])
                yield _arrow_up(cx, tip_y)

        for i, item in enumerate(pre_items):
            if item.is_data_label:
                # Transparent box for data annotation labels
                bx = Box(item.label, "box-transparent", w=box_w, h=cell_h)
                yield from bx.render(BBox(box_x, pre_y, box_w, cell_h), th)
            else:
                bx = Box(item.label, item.css, w=box_w, h=cell_h, sublabel=item.sublabel)
                actual_h = bx.measure(th).h
                box_y = pre_y + (cell_h - actual_h) / 2
                yield from bx.render(BBox(box_x, box_y, box_w, actual_h), th)

            # Upward flow arrow between pre-decoder cells
            if i < len(pre_items) - 1:
                line_y1 = pre_y + cell_h
                line_y2 = pre_y + cell_h + gap
                if line_y2 - line_y1 > 1:
                    tip_y = line_y1 + _ARROW_CLR
                    yield S.Line(x1=cx, y1=line_y2, x2=cx, y2=tip_y, class_=["arrow"])
                    yield _arrow_up(cx, tip_y)

            pre_y += cell_h + gap

        # ── Tied weights: Embedding → LM Head ───────────────────────
        # In render order: LM Head is dec_items[0], Embedding is dec_items[-1]
        if self.arch.tie_word_embeddings and n_dec >= 2:
            lm_head_cy = content_bb.y + cell_h / 2
            embed_cy = content_bb.y + (n_dec - 1) * (cell_h + gap) + cell_h / 2
            tied_x = box_x + box_w + 12
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
        """Return (bbox, spec) for each cell. Only decoder block cells have non-None spec."""
        items = self._build_items()
        range_label_w = 50
        box_x = bb.x + range_label_w
        box_w = th.geo.stack_w
        cell_h = th.geo.stack_cell_h
        gap = th.geo.stack_cell_gap
        g = th.geo.gap
        title_h = th.geo.title_h

        dec_items = [it for it in items if it.in_decoder]
        pre_items = [it for it in items if not it.in_decoder]
        n_dec = len(dec_items)
        dec_h = title_h + 2 * g + n_dec * cell_h + (n_dec - 1) * gap

        result: list[tuple[BBox, BlockSpec | None]] = []

        # Decoder cells (inside frame content area)
        content_y = bb.y + title_h + g
        y = content_y
        for item in dec_items:
            result.append((BBox(box_x, y, box_w, cell_h), item.spec))
            y += cell_h + gap

        # Pre-decoder cells (below frame)
        pre_y = bb.y + dec_h + g
        for item in pre_items:
            result.append((BBox(box_x, pre_y, box_w, cell_h), item.spec))
            pre_y += cell_h + gap

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
        "sliding_window": "box-swa",
        "gdn": "box-gdn",
        "kda": "box-kda",
        "mamba": "box-mamba",
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
    title: str = ""

    def _title_offset(self, th: Theme) -> float:
        return th.geo.title_h if self.title else 0

    def measure(self, th: Theme) -> Size:
        w = self.block_w or th.geo.block_w
        g = th.geo.gap
        bh = th.geo.box_h
        sr = th.geo.symbol_r
        mixer_box = _mixer_box(self.mixer, th)
        mh = mixer_box.measure(th).h
        bg = g * 2  # branch gap — extra space where bypass lines leave the center line
        h = 3 * g + sr * 2 + g + (bh - 4) + g + bh + bg + sr * 2 + g + (bh - 4) + g + mh + g
        h += self._title_offset(th)
        return Size(w, h)

    def render(self, bb: BBox, th: Theme) -> Iterator[S.Element]:
        g = th.geo.gap
        bg = g * 2  # branch gap — extra space where bypass lines leave the center line
        bh = th.geo.box_h
        sr = th.geo.symbol_r
        title_offset = self._title_offset(th)
        iw = bb.w - 3 * g  # reserve right margin for bypass lines
        ix = bb.x + g
        cx = ix + iw / 2  # center x for flow lines
        bypass_x = bb.right - g  # single bypass x for both residuals
        cur_y = bb.y + 2 * g + title_offset

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
        cur_y += msz.h + 3 * g

        # ── Vertical flow arrows through center ───────────────────
        flow_segments = [
            (res2_bb.bottom, norm2_bb.y),
            (norm2_bb.bottom, mlp_bb.y),
            (mlp_bb.bottom, res1_bb.y),
            (res1_bb.bottom, norm1_bb.y),
            (norm1_bb.bottom, mixer_bb.y),
        ]
        for y1, y2 in flow_segments:
            if y2 - y1 > 2 * _ARROW_CLR:
                arrows.append(S.Line(x1=cx, y1=y1 + _ARROW_CLR, x2=cx, y2=y2 - _ARROW_CLR, class_=["arrow"]))

        # ── Entry / exit arrow stubs (manual arrowheads) ──────────
        # Output stub: line from res2+ upward, arrowhead at top
        arrows.append(S.Line(x1=cx, y1=res2_bb.y, x2=cx, y2=res2_bb.y - g, class_=["arrow"]))
        arrows.append(_arrow_up(cx, res2_bb.y - g))

        # Input stub: line from below mixer upward, arrowhead at mixer bottom
        arrows.append(S.Line(x1=cx, y1=mixer_bb.bottom + 2 * g, x2=cx, y2=mixer_bb.bottom + _ARROW_CLR, class_=["arrow"]))
        arrows.append(_arrow_up(cx, mixer_bb.bottom + _ARROW_CLR))

        # ── Residual bypass lines ─────────────────────────────────
        # Residual 1: wraps Mixer + Norm1 (bypass from input to res1+)
        input_y = mixer_bb.bottom + g
        arrows.append(S.Line(x1=cx, y1=input_y, x2=bypass_x, y2=input_y, class_=["arrow"]))
        arrows.append(S.Line(x1=bypass_x, y1=input_y, x2=bypass_x, y2=res1_bb.cy, class_=["arrow"]))
        arrows.append(S.Line(x1=bypass_x, y1=res1_bb.cy, x2=cx + sr + _ARROW_CLR, y2=res1_bb.cy, class_=["arrow"]))
        arrows.append(_arrow_left(cx + sr + _ARROW_CLR, res1_bb.cy))

        # Residual 2: wraps FFN + Norm2 (bypass from above res1+ to res2+)
        branch2_y = (mlp_bb.bottom + res1_bb.y) / 2
        arrows.append(S.Line(x1=cx, y1=branch2_y, x2=bypass_x, y2=branch2_y, class_=["arrow"]))
        arrows.append(S.Line(x1=bypass_x, y1=branch2_y, x2=bypass_x, y2=res2_bb.cy, class_=["arrow"]))
        arrows.append(S.Line(x1=bypass_x, y1=res2_bb.cy, x2=cx + sr + _ARROW_CLR, y2=res2_bb.cy, class_=["arrow"]))
        arrows.append(_arrow_left(cx + sr + _ARROW_CLR, res2_bb.cy))

        # ── Background frame (behind everything) ─────────────────
        actual_h = cur_y - bb.y
        frame, _ = _render_detail_frame(bb.x, bb.y, bb.w, actual_h, self.title, "block-bg", th)
        yield frame

        yield from arrows
        yield from elements

    def mlp_bbox(self, bb: BBox, th: Theme) -> BBox:
        """Return the BBox of the 'Feed forward' box within this block."""
        g = th.geo.gap
        bh = th.geo.box_h
        sr = th.geo.symbol_r
        iw = bb.w - 3 * g
        ix = bb.x + g
        cur_y = bb.y + 2 * g + self._title_offset(th)
        cur_y += sr * 2 + g      # res2 symbol
        cur_y += (bh - 4) + g    # norm2
        return BBox(ix, cur_y, iw, bh)

    def mixer_bbox(self, bb: BBox, th: Theme) -> BBox:
        """Return the BBox of the mixer box within this block."""
        g = th.geo.gap
        bg = g * 2  # branch gap
        bh = th.geo.box_h
        sr = th.geo.symbol_r
        iw = bb.w - 3 * g
        ix = bb.x + g
        cur_y = bb.y + 2 * g + self._title_offset(th)
        cur_y += sr * 2 + g      # res2 symbol
        cur_y += (bh - 4) + g    # norm2
        cur_y += bh + bg          # FFN + branch gap
        cur_y += sr * 2 + g      # res1 symbol
        cur_y += (bh - 4) + g    # norm1
        mixer = _mixer_box(self.mixer, th)
        msz = mixer.measure(th)
        return BBox(ix, cur_y, iw, msz.h)


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


def _render_detail_frame(
    x: float, y: float, w: float, h: float,
    title: str, css: str, th: Theme,
) -> tuple[S.G, BBox]:
    """Titled outer box: colored title strip + white content area.

    Returns (frame_group, content_bb) where content_bb is the inset region
    for child content (padded by ``g`` on all sides below the title strip).
    """
    rx = th.geo.rx_block
    g = th.geo.gap
    title_h = th.geo.title_h if title else 0
    content_bb = BBox(x + g, y + title_h + g, w - 2 * g, h - title_h - 2 * g)
    if not title:
        # Untitled: single rounded rect, white content
        return S.G(class_=[css], elements=[
            S.Rect(x=x, y=y, width=w, height=h, rx=rx, class_=["detail-content"]),
        ]), content_bb
    # Title strip: rounded top corners, square bottom
    title_path = S.Path(
        d=[
            S.MoveTo(x, y + title_h),
            S.LineTo(x, y + rx),
            S.Arc(rx, rx, 0, False, True, x + rx, y),
            S.LineTo(x + w - rx, y),
            S.Arc(rx, rx, 0, False, True, x + w, y + rx),
            S.LineTo(x + w, y + title_h),
            S.ClosePath(),
        ],
        class_=["detail-title"],
    )
    # Content area: square top, rounded bottom corners
    content_path = S.Path(
        d=[
            S.MoveTo(x, y + title_h),
            S.LineTo(x + w, y + title_h),
            S.LineTo(x + w, y + h - rx),
            S.Arc(rx, rx, 0, False, True, x + w - rx, y + h),
            S.LineTo(x + rx, y + h),
            S.Arc(rx, rx, 0, False, True, x, y + h - rx),
            S.ClosePath(),
        ],
        class_=["detail-content"],
    )
    return S.G(class_=[css], elements=[
        title_path,
        content_path,
        S.Text(x=x + w / 2, y=y + title_h / 2, text=title,
               class_=["t-label-bold"], text_anchor="middle",
               dominant_baseline="central"),
    ]), content_bb


MixerDetail = "AttentionDetail | GDNDetail | KDADetail | MambaDetail"

_ACTIVATION_DISPLAY: dict[str, str] = {"silu": "SiLU", "gelu": "GELU", "relu": "ReLU"}


@dataclass
class AttentionDetail:
    """Detail panel for attention (full or sliding window).

    Layout (data flows bottom → top)::

        o_proj                          [0] box-linear
          ↑
        Scaled dot-product attention    [1] box-attention, bold
          ↑       ↑       ↑
          q       k       v            [2] HStack, box-transparent
          ↑       ↑       |
        RoPE    RoPE    (spacer)       [3] HStack (RoPE, RoPE, Spacer)
          ↑       ↑       ↑
        q_proj  k_proj  v_proj         [4] HStack, box-linear
           \\      |      /
            fork point
    """

    config: AttentionDisplayConfig
    w: float = 250

    def _title(self) -> str:
        return "Sliding-window attention" if self.config.window_size else "Grouped-query attention"

    @staticmethod
    def _sdpa_title() -> str:
        return "Scaled dot-product attention"

    _PROJ_W = 55
    _PROJ_GAP = 8
    _FORK_SPACE = 12  # vertical space for entry fork below proj row

    def _build(self, th: Theme) -> list[tuple[BBox, Box | Symbol | HStack]]:
        bh = th.geo.box_h - 2
        box_css = "box-swa" if self.config.window_size else "box-attention"
        pw, pg = self._PROJ_W, self._PROJ_GAP
        children: list[Box | Symbol | HStack] = [
            Box("o_proj", "box-linear", w=70, h=bh - 2),                                           # [0]
            Box(self._sdpa_title(), box_css, w=self.w - 30, h=bh, bold=True),                      # [1]
            HStack([Box("q", "box-transparent", w=pw, h=bh - 4),
                    Box("k", "box-transparent", w=pw, h=bh - 4),
                    Box("v", "box-transparent", w=pw, h=bh - 4)], gap=pg),                         # [2]
            HStack([Box("RoPE", "box-norm", w=pw, h=bh - 4),
                    Box("RoPE", "box-norm", w=pw, h=bh - 4),
                    Spacer(w=pw, h=bh - 4)], gap=pg),                                              # [3]
            HStack([Box("q_proj", "box-linear", w=pw, h=bh - 2),
                    Box("k_proj", "box-linear", w=pw, h=bh - 2),
                    Box("v_proj", "box-linear", w=pw, h=bh - 2)], gap=pg),                         # [4]
        ]
        return _detail_layout(children, th)

    def _col_centers(self, items: list[tuple[BBox, Box | Symbol | HStack]], th: Theme) -> list[float]:
        """Compute center-x of each of the three columns from the proj HStack (item 4)."""
        proj_hstack = items[4][1]
        assert isinstance(proj_hstack, HStack)
        sizes = [c.measure(th) for c in proj_hstack.children]
        hg = proj_hstack.gap if proj_hstack.gap is not None else th.geo.gap
        proj_bb = items[4][0]
        centers: list[float] = []
        x = proj_bb.x
        for sz in sizes:
            centers.append(x + sz.w / 2)
            x += sz.w + hg
        return centers

    def measure(self, th: Theme) -> Size:
        g = th.geo.gap
        items = self._build(th)
        total_h = items[-1][0].bottom if items else 0
        content_w = max((bb.right for bb, _ in items), default=0)
        frame_h = total_h + self._FORK_SPACE + 2 * g + th.geo.title_h
        return Size(content_w + 2 * g + 100, frame_h)

    def _render_arrows(
        self,
        items: list[tuple[BBox, Box | Symbol | HStack]],
        bb: BBox,
        th: Theme,
    ) -> Iterator[S.Element]:
        """Yield arrows between items with three parallel columns."""
        clearance = _ARROW_CLR
        g = th.geo.gap
        cols = self._col_centers(items, th)

        # [0] o_proj → [1] SDPA : single vertical arrow (centred on o_proj)
        o_bb = items[0][0]
        sdpa_bb = items[1][0]
        cx = bb.x + o_bb.cx
        y1 = bb.y + o_bb.bottom
        y2 = bb.y + sdpa_bb.y
        if y2 - y1 > 1:
            tip_y = y1 + clearance
            yield S.Line(x1=cx, y1=y2, x2=cx, y2=tip_y, class_=["arrow"])
            yield _arrow_up(cx, tip_y)

        # [1] SDPA → [2] labels : three parallel arrows
        label_bb = items[2][0]
        y1 = bb.y + sdpa_bb.bottom
        y2 = bb.y + label_bb.y
        if y2 - y1 > 1:
            for col_x in cols:
                acx = bb.x + col_x
                tip_y = y1 + clearance
                yield S.Line(x1=acx, y1=y2, x2=acx, y2=tip_y, class_=["arrow"])
                yield _arrow_up(acx, tip_y)

        # [2] labels → [3] RoPE/Spacer : three parallel arrows (v column is pass-through)
        rope_bb = items[3][0]
        y1 = bb.y + label_bb.bottom
        y2 = bb.y + rope_bb.y
        if y2 - y1 > 1:
            for col_x in cols:
                acx = bb.x + col_x
                tip_y = y1 + clearance
                yield S.Line(x1=acx, y1=y2, x2=acx, y2=tip_y, class_=["arrow"])
                yield _arrow_up(acx, tip_y)

        # [3] RoPE/Spacer → [4] projs : three parallel arrows
        proj_bb = items[4][0]
        y1 = bb.y + rope_bb.bottom
        y2 = bb.y + proj_bb.y
        if y2 - y1 > 1:
            for col_x in cols:
                acx = bb.x + col_x
                tip_y = y1 + clearance
                yield S.Line(x1=acx, y1=y2, x2=acx, y2=tip_y, class_=["arrow"])
                yield _arrow_up(acx, tip_y)

    def _render_entry_fork(
        self,
        items: list[tuple[BBox, Box | Symbol | HStack]],
        frame_bb: BBox,
        content_bb: BBox,
        th: Theme,
    ) -> Iterator[S.Element]:
        """Fork from single entry point to three proj columns (L-shaped paths)."""
        clearance = _ARROW_CLR
        g = th.geo.gap
        cols = self._col_centers(items, th)
        proj_bb = items[4][0]

        # Fork midpoint: centered below proj row
        content_cx = content_bb.x + content_bb.w / 2
        midpoint_y = content_bb.y + proj_bb.bottom + g

        for col_x in cols:
            acx = content_bb.x + col_x
            tip_y = content_bb.y + proj_bb.bottom + clearance
            yield S.Path(
                d=[
                    S.MoveTo(content_cx, midpoint_y),
                    S.LineTo(acx, midpoint_y),
                    S.LineTo(acx, tip_y),
                ],
                class_=["arrow"],
            )
            yield _arrow_up(acx, tip_y)

        # Entry line from frame bottom to fork midpoint
        y_start = frame_bb.y + frame_bb.h
        tip_y = midpoint_y + clearance
        if y_start - tip_y > 1:
            yield S.Line(x1=content_cx, y1=y_start, x2=content_cx, y2=tip_y, class_=["arrow"])

    def render(self, bb: BBox, th: Theme) -> Iterator[S.Element]:
        g = th.geo.gap
        title_h = th.geo.title_h
        items = self._build(th)
        content_h = items[-1][0].bottom if items else 0
        content_w = max((bb.right for bb, _ in items), default=0)

        # Titled outer frame
        frame_h = content_h + self._FORK_SPACE + 2 * g + title_h
        detail_css = "detail-swa" if self.config.window_size else "detail-attention"
        frame_bb = BBox(bb.x, bb.y, content_w + 2 * g, frame_h)
        frame, content_bb = _render_detail_frame(
            frame_bb.x, frame_bb.y, frame_bb.w, frame_bb.h,
            self._title(), detail_css, th,
        )
        yield frame

        # Exit arrow (top item to title strip)
        yield from _DetailArrows.render_exit_arrow(items, frame_bb, content_bb, title_h)

        # Custom inter-item arrows (three parallel columns)
        yield from self._render_arrows(items, content_bb, th)

        # Entry fork (single point fans out to three projections)
        yield from self._render_entry_fork(items, frame_bb, content_bb, th)

        # Render children
        for child_bb, child in items:
            shifted = BBox(content_bb.x + child_bb.x, content_bb.y + child_bb.y, child_bb.w, child_bb.h)
            yield from child.render(shifted, th)

        # Side annotations (item 1 = SDPA box)
        attn_bb = BBox(content_bb.x + items[1][0].x, content_bb.y + items[1][0].y, items[1][0].w, items[1][0].h)
        heads = self.config.heads
        kv_heads = self.config.kv_heads or heads
        ax = content_bb.right - 10
        yield S.Text(x=ax, y=attn_bb.cy - 8, text=f"{heads} attention heads", class_=["t-count"])
        yield S.Text(x=ax, y=attn_bb.cy + 6, text=f"{kv_heads} key & value heads", class_=["t-count"])
        if self.config.window_size:
            yield S.Text(x=ax, y=attn_bb.cy + 20,
                         text=f"window = {self.config.window_size:,}", class_=["t-dim"])


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
            Box("Gated Delta Rule", "box-gdn", w=self.w - 30, h=bh + 10, bold=True),
            Box("CausalConv1d", "box-conv", w=self.w - 60, h=bh),
            HStack([
                Box("in_proj_qkvz", "box-linear", w=pw, h=bh),
                Box("in_proj_\u03b2\u03b1", "box-gate", w=pw, h=bh),
            ], gap=10),
        ]
        return _detail_layout(children, th)

    def measure(self, th: Theme) -> Size:
        g = th.geo.gap
        items = self._build(th)
        total_h = items[-1][0].bottom if items else 0
        content_w = max((bb.right for bb, _ in items), default=0)
        return Size(content_w + 2 * g + 80, total_h + 2 * g + th.geo.title_h)

    def render(self, bb: BBox, th: Theme) -> Iterator[S.Element]:
        g = th.geo.gap
        title_h = th.geo.title_h
        items = self._build(th)
        content_h = items[-1][0].bottom if items else 0
        content_w = max((bb.right for bb, _ in items), default=0)

        # Titled outer frame
        frame_bb = BBox(bb.x, bb.y, content_w + 2 * g, content_h + 2 * g + title_h)
        frame, content_bb = _render_detail_frame(
            frame_bb.x, frame_bb.y, frame_bb.w, frame_bb.h,
            "Gated DeltaNet", "detail-gdn", th,
        )
        yield frame

        # Content inside padded area
        yield from _DetailArrows.render(items, content_bb)
        yield from _DetailArrows.render_exit_arrow(items, frame_bb, content_bb, title_h)
        for child_bb, child in items:
            shifted = BBox(content_bb.x + child_bb.x, content_bb.y + child_bb.y, child_bb.w, child_bb.h)
            yield from child.render(shifted, th)

        gdr_bb = BBox(content_bb.x + items[2][0].x, content_bb.y + items[2][0].y, items[2][0].w, items[2][0].h)
        conv_bb = BBox(content_bb.x + items[3][0].x, content_bb.y + items[3][0].y, items[3][0].w, items[3][0].h)
        ax = content_bb.right - 15
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
            Box("Kimi Delta Attention", "box-kda", w=self.w - 30, h=bh + 10, bold=True),
            HStack([Box("Conv", "box-conv", w=cw, h=bh - 4),
                    Box("Conv", "box-conv", w=cw, h=bh - 4),
                    Box("Conv", "box-conv", w=cw, h=bh - 4)], gap=5),
            HStack([Box("Linear", "box-linear", w=cw, h=bh - 4),
                    Box("Linear", "box-linear", w=cw, h=bh - 4),
                    Box("Linear", "box-linear", w=cw, h=bh - 4)], gap=5),
        ]
        return _detail_layout(children, th)

    _EXT_LABEL_H = 25  # extra height for external data labels below the frame

    def measure(self, th: Theme) -> Size:
        g = th.geo.gap
        items = self._build(th)
        total_h = items[-1][0].bottom if items else 0
        content_w = max((bb.right for bb, _ in items), default=0)
        return Size(content_w + 2 * g + 90, total_h + 2 * g + th.geo.title_h + self._EXT_LABEL_H)

    def render(self, bb: BBox, th: Theme) -> Iterator[S.Element]:
        g = th.geo.gap
        title_h = th.geo.title_h
        items = self._build(th)
        content_h = items[-1][0].bottom if items else 0
        content_w = max((bb.right for bb, _ in items), default=0)

        # Titled outer frame (does not include the external label area)
        frame_h = content_h + 2 * g + title_h
        frame_bb = BBox(bb.x, bb.y, content_w + 2 * g, frame_h)
        frame, content_bb = _render_detail_frame(
            frame_bb.x, frame_bb.y, frame_bb.w, frame_bb.h,
            "Kimi Delta Attention", "detail-kda", th,
        )
        yield frame

        # Content inside padded area
        yield from _DetailArrows.render(items, content_bb)
        yield from _DetailArrows.render_exit_arrow(items, frame_bb, content_bb, title_h)
        yield from _DetailArrows.render_entry_arrow(items, frame_bb, content_bb)
        for child_bb, child in items:
            shifted = BBox(content_bb.x + child_bb.x, content_bb.y + child_bb.y, child_bb.w, child_bb.h)
            yield from child.render(shifted, th)

        kda_bb = BBox(content_bb.x + items[2][0].x, content_bb.y + items[2][0].y, items[2][0].w, items[2][0].h)
        ax = content_bb.right - 10
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

        # External data labels below the frame as transparent boxes
        frame_bottom = frame_bb.y + frame_bb.h
        label_box_h = 16
        label_y = frame_bottom + self._EXT_LABEL_H - label_box_h

        # q k v labels (centered under projection columns)
        proj_bb = BBox(content_bb.x + items[4][0].x, content_bb.y + items[4][0].y, items[4][0].w, items[4][0].h)
        cw = (self.w - 80) // 3
        for i, lbl in enumerate("qkv"):
            px = proj_bb.x + i * (cw + 5) + cw / 2
            bw = cw - 2
            bx = Box(lbl, "box-transparent", w=bw, h=label_box_h)
            yield from bx.render(BBox(px - bw / 2, label_y, bw, label_box_h), th)


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
            Box("Mamba SSM", "box-mamba", w=self.w - 30, h=bh + 10, bold=True),
            Box("CausalConv1d", "box-conv", w=self.w - 60, h=bh),
            Box("in_proj", "box-linear", w=self.w - 60, h=bh),
        ]
        return _detail_layout(children, th)

    def measure(self, th: Theme) -> Size:
        g = th.geo.gap
        items = self._build(th)
        total_h = items[-1][0].bottom if items else 0
        content_w = max((bb.right for bb, _ in items), default=0)
        return Size(content_w + 2 * g + 80, total_h + 2 * g + th.geo.title_h)

    def render(self, bb: BBox, th: Theme) -> Iterator[S.Element]:
        g = th.geo.gap
        title_h = th.geo.title_h
        items = self._build(th)
        content_h = items[-1][0].bottom if items else 0
        content_w = max((bb.right for bb, _ in items), default=0)

        # Titled outer frame
        frame_bb = BBox(bb.x, bb.y, content_w + 2 * g, content_h + 2 * g + title_h)
        frame, content_bb = _render_detail_frame(
            frame_bb.x, frame_bb.y, frame_bb.w, frame_bb.h,
            "Mamba SSM", "detail-mamba", th,
        )
        yield frame

        # Content inside padded area
        yield from _DetailArrows.render(items, content_bb)
        yield from _DetailArrows.render_exit_arrow(items, frame_bb, content_bb, title_h)
        for child_bb, child in items:
            shifted = BBox(content_bb.x + child_bb.x, content_bb.y + child_bb.y, child_bb.w, child_bb.h)
            yield from child.render(shifted, th)

        # Side annotations
        ssm_bb = BBox(content_bb.x + items[2][0].x, content_bb.y + items[2][0].y, items[2][0].w, items[2][0].h)
        ax = content_bb.right - 15
        if self.config.d_state:
            yield S.Text(x=ax, y=ssm_bb.cy - 8, text=f"d_state = {self.config.d_state}", class_=["t-dim"])
        if self.config.d_inner:
            yield S.Text(x=ax, y=ssm_bb.cy + 6, text=f"d_inner = {self.config.d_inner}", class_=["t-dim"])
        conv_bb = BBox(content_bb.x + items[3][0].x, content_bb.y + items[3][0].y, items[3][0].w, items[3][0].h)
        if self.config.d_conv:
            yield S.Text(x=ax, y=conv_bb.cy, text=f"d_conv = {self.config.d_conv}", class_=["t-dim"])


@dataclass
class MLPDetail:
    """Detail panel for the MLP / Feed Forward sub-block."""

    config: MLPDisplayConfig
    w: float = 180

    _EXT_LABEL_H = 24  # vertical space for one external "hidden states" label
    _LABEL_BOX_H = 16  # height of the transparent label box

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
                Box("up_proj", "box-linear", w=half, h=bh),
            ]
        else:
            children = [
                Box("down_proj", "box-linear", w=self.w - 20, h=bh),
                Box(act, "box-activation", w=self.w - 20, h=bh),
                Box("up_proj", "box-linear", w=self.w - 20, h=bh),
            ]
        if self.config.gated:
            g = th.geo.gap
            gap = [g, g * 2, g]  # × sits close to down_proj; wider spacing below HStack
        else:
            gap = None
        items = _detail_layout(children, th, gap=gap)
        # Left-align up_proj under the act box in the HStack
        if self.config.gated and len(items) >= 4:
            hstack_bb = items[2][0]
            up_bb = items[3][0]
            items[3] = (BBox(hstack_bb.x, up_bb.y, up_bb.w, up_bb.h), items[3][1])
        return items

    def measure(self, th: Theme) -> Size:
        items = self._build(th)
        total_h = items[-1][0].bottom if items else 0
        g = th.geo.gap
        content_w = max((bb.right for bb, _ in items), default=0)
        fork_space = g if self.config.gated else 0  # space for fork midpoint
        frame_h = total_h + fork_space + 2 * g + th.geo.title_h
        return Size(content_w + 2 * g + 60, frame_h + 2 * self._EXT_LABEL_H)

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
        hg = hstack.gap if hstack.gap is not None else th.geo.gap
        child0_cx = hstack_bb.x + child_sizes[0].w / 2
        child1_cx = hstack_bb.x + child_sizes[0].w + hg + child_sizes[1].w / 2

        # Absolute positions
        cross_cx = bb.x + (cross_bb.x + cross_bb.w / 2)
        down_cx = bb.x + (down_bb.x + down_bb.w / 2)
        abs_child0_cx = bb.x + child0_cx
        abs_child1_cx = bb.x + child1_cx

        clearance = _ARROW_CLR
        g = th.geo.gap

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

        # up_proj → act : straight vertical arrow (up_proj is left-aligned under act)
        y1 = bb.y + hstack_bb.bottom
        y2 = bb.y + up_bb.y
        if y2 - y1 > 1:
            tip_y = y1 + clearance
            yield S.Line(x1=abs_child0_cx, y1=y2, x2=abs_child0_cx, y2=tip_y, class_=["arrow"])
            yield _arrow_up(abs_child0_cx, tip_y)

        # Fork midpoint: centered below up_proj (clean split, no dot)
        content_cx = bb.x + bb.w / 2
        midpoint_y = bb.y + up_bb.bottom + g

        # Midpoint → up_proj: L-path left then up
        up_proj_tip_y = bb.y + up_bb.bottom + clearance
        yield S.Path(
            d=[
                S.MoveTo(content_cx, midpoint_y),
                S.LineTo(abs_child0_cx, midpoint_y),
                S.LineTo(abs_child0_cx, up_proj_tip_y),
            ],
            class_=["arrow"],
        )
        yield _arrow_up(abs_child0_cx, up_proj_tip_y)

        # Midpoint → gate_proj: L-path right then up
        gate_tip_y = bb.y + hstack_bb.bottom + clearance
        yield S.Path(
            d=[
                S.MoveTo(content_cx, midpoint_y),
                S.LineTo(abs_child1_cx, midpoint_y),
                S.LineTo(abs_child1_cx, gate_tip_y),
            ],
            class_=["arrow"],
        )
        yield _arrow_up(abs_child1_cx, gate_tip_y)

    def _render_exit_arrow(
        self,
        items: list[tuple[BBox, Box | Symbol | HStack]],
        frame_bb: BBox,
        content_bb: BBox,
        g: float,
    ) -> Iterator[S.Element]:
        """Exit arrow from top item up through (skipping) title bar to output label."""
        if not items:
            return
        top_bb = items[0][0]
        cx = content_bb.x + top_bb.cx

        # Segment 1: from top of down_proj to bottom of title strip (within content area)
        y_from = content_bb.y + top_bb.y
        y_title_bottom = content_bb.y - g  # = frame_bb.y + title_h
        if y_from - y_title_bottom > 1:
            yield S.Line(x1=cx, y1=y_from, x2=cx, y2=y_title_bottom, class_=["arrow"])

        # Segment 2: from top of frame to above-frame label area
        y_frame_top = frame_bb.y
        tip_y = frame_bb.y - g + _ARROW_CLR
        if y_frame_top - tip_y > 1:
            yield S.Line(x1=cx, y1=y_frame_top, x2=cx, y2=tip_y, class_=["arrow"])
            yield _arrow_up(cx, tip_y)

    def _render_entry_arrow(
        self,
        items: list[tuple[BBox, Box | Symbol | HStack]],
        frame_bb: BBox,
        content_bb: BBox,
        g: float,
    ) -> Iterator[S.Element]:
        """Entry arrow from input label up to bottom item or fork midpoint."""
        if not items:
            return
        clearance = _ARROW_CLR

        if self.config.gated and len(items) >= 4:
            # Arrow to fork midpoint (centered)
            up_bb = items[-1][0]
            midpoint_y = content_bb.y + up_bb.bottom + g
            cx = content_bb.x + content_bb.w / 2
            tip_y = midpoint_y + clearance
        else:
            # Arrow to bottom item
            bottom_bb = items[-1][0]
            cx = content_bb.x + bottom_bb.cx
            tip_y = content_bb.y + bottom_bb.bottom + clearance

        y_start = frame_bb.bottom + self._EXT_LABEL_H - self._LABEL_BOX_H - clearance
        if y_start - tip_y > 1:
            yield S.Line(x1=cx, y1=y_start, x2=cx, y2=tip_y, class_=["arrow"])

    def _render_external_labels(
        self,
        frame_bb: BBox,
        content_bb: BBox,
        th: Theme,
    ) -> Iterator[S.Element]:
        """Render "hidden states" labels above and below the frame as transparent boxes."""
        cx = content_bb.cx
        w, h = 90, self._LABEL_BOX_H

        # Output label: top-aligned in the _EXT_LABEL_H space above the frame
        out_box = Box("hidden states", "box-transparent", w=w, h=h)
        yield from out_box.render(BBox(cx - w / 2, frame_bb.y - self._EXT_LABEL_H, w, h), th)

        # Input label: bottom-aligned in the _EXT_LABEL_H space below the frame
        in_box = Box("hidden states", "box-transparent", w=w, h=h)
        in_y = frame_bb.bottom + self._EXT_LABEL_H - h
        yield from in_box.render(BBox(cx - w / 2, in_y, w, h), th)

    def render(self, bb: BBox, th: Theme) -> Iterator[S.Element]:
        items = self._build(th)
        content_h = items[-1][0].bottom if items else 0
        g = th.geo.gap
        title_h = th.geo.title_h
        content_w = max((cbb.right for cbb, _ in items), default=0)
        fork_space = g if self.config.gated else 0

        # Frame is inset from bb by _EXT_LABEL_H on top and bottom
        frame_x = bb.x
        frame_y = bb.y + self._EXT_LABEL_H
        frame_w = content_w + 2 * g
        frame_h = content_h + fork_space + 2 * g + title_h
        frame_bb = BBox(frame_x, frame_y, frame_w, frame_h)

        frame, content_bb = _render_detail_frame(
            frame_x, frame_y, frame_w, frame_h,
            "Feed forward", "detail-mlp", th,
        )

        # 1. Exit arrow step 1 — title-bar region only (BEFORE frame, hidden by title strip)
        if items:
            top_bb = items[0][0]
            cx = content_bb.x + top_bb.cx
            y_title_bottom = frame_bb.y + title_h  # = content_bb.y - g
            y_title_top = frame_bb.y
            if y_title_bottom - y_title_top > 1:
                yield S.Line(x1=cx, y1=y_title_bottom, x2=cx, y2=y_title_top, class_=["arrow"])

        # 2. Frame group (title bar paints on top, hiding step-1 segment)
        yield frame

        # 3. Exit arrow step 2 — content-area segment (AFTER frame, visible)
        if items:
            top_bb = items[0][0]
            cx = content_bb.x + top_bb.cx
            y_from = content_bb.y + top_bb.y
            y_title_bottom = frame_bb.y + title_h
            if y_from - y_title_bottom > 1:
                yield S.Line(x1=cx, y1=y_from, x2=cx, y2=y_title_bottom, class_=["arrow"])

        # 3b. Exit arrow step 3 — above-frame segment + arrowhead (AFTER frame, visible)
        if items:
            top_bb = items[0][0]
            cx = content_bb.x + top_bb.cx
            y_frame_top = frame_bb.y
            tip_y = frame_bb.y - self._EXT_LABEL_H + self._LABEL_BOX_H + _ARROW_CLR
            if y_frame_top - tip_y > 1:
                yield S.Line(x1=cx, y1=y_frame_top, x2=cx, y2=tip_y, class_=["arrow"])
                yield _arrow_up(cx, tip_y)

        # 4. Internal arrows (fork midpoint, merge paths, inter-item arrows)
        yield from self._render_arrows(items, content_bb, th)

        # 5. Entry arrow (from input label to bottom item or fork midpoint)
        yield from self._render_entry_arrow(items, frame_bb, content_bb, g)

        # 6. Children (boxes, symbols)
        for child_bb, child in items:
            shifted = BBox(content_bb.x + child_bb.x, content_bb.y + child_bb.y, child_bb.w, child_bb.h)
            yield from child.render(shifted, th)

        # 7. Annotations
        intermediate = self.config.intermediate_size
        if intermediate:
            ax = content_bb.right - 10
            yield S.Text(x=ax, y=content_bb.y + 8, text="Intermediate size", class_=["t-note"])
            yield S.Text(x=ax, y=content_bb.y + 22, text=f"= {intermediate:,}", class_=["t-dim"])

        # 8. External "hidden states" labels
        yield from self._render_external_labels(frame_bb, content_bb, th)


def _mixer_spec_css(mixer: MixerSpec) -> str:
    """Return the CSS class for a single MixerSpec."""
    css_map = {
        "attention": "box-attention",
        "sliding_window": "box-swa",
        "gdn": "box-gdn",
        "kda": "box-kda",
        "mamba": "box-mamba",
    }
    return css_map.get(mixer.mixer_type, "box-linear")


@dataclass
class StochasticMixerPanel:
    """Stochastic mixer dispatch breakdown showing sub-mixer options."""

    spec: StochasticMixerSpec
    w: float | None = None

    def _box_h(self, th: Theme) -> float:
        return th.geo.box_h

    def _inner_gap(self) -> float:
        return 8.0

    def measure(self, th: Theme) -> Size:
        w = self.w or th.geo.stack_w
        g = th.geo.gap
        title_h = th.geo.title_h
        bh = self._box_h(th)
        gap = self._inner_gap()
        n = len(self.spec.sub_mixers)
        content_h = n * bh + max(0, n - 1) * gap
        total_h = 2 * g + title_h + content_h
        return Size(w, total_h)

    def render(self, bb: BBox, th: Theme) -> Iterator[S.Element]:
        bh = self._box_h(th)
        gap = self._inner_gap()

        # Titled outer frame
        frame, content_bb = _render_detail_frame(
            bb.x, bb.y, bb.w, bb.h,
            "Stochastic mixer dispatch", "detail-stochastic", th,
        )
        yield frame

        # Sub-mixer boxes
        inner_w = content_bb.w
        box_y = content_bb.y
        for name, sub_mixer in self.spec.sub_mixers:
            css = _mixer_spec_css(sub_mixer)
            label = name
            if name == self.spec.main_mixer_name:
                label += " \u2605"
            box = Box(label, css, w=inner_w, h=bh)
            box_bb = BBox(content_bb.x, box_y, inner_w, bh)
            yield from box.render(box_bb, th)
            box_y += bh + gap

    def sub_mixer_bboxes(self, bb: BBox, th: Theme) -> list[tuple[BBox, MixerSpec]]:
        """Return (bbox, MixerSpec) for each sub-mixer box."""
        g = th.geo.gap
        title_h = th.geo.title_h
        bh = self._box_h(th)
        gap = self._inner_gap()
        inner_w = bb.w - 2 * g

        result: list[tuple[BBox, MixerSpec]] = []
        box_y = bb.y + title_h + g
        for _name, sub_mixer in self.spec.sub_mixers:
            result.append((BBox(bb.x + g, box_y, inner_w, bh), sub_mixer))
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


def _detail_layout(children: list, th: Theme, gap: float | list[float] | None = None) -> list[tuple[BBox, object]]:
    """Measure children vertically and compute per-child BBoxes (relative coords)."""
    default_g = th.geo.gap
    gaps: list[float] | None = gap if isinstance(gap, list) else None
    uniform_g = gap if isinstance(gap, (int, float)) else default_g
    sizes = [c.measure(th) for c in children]
    max_w = max((s.w for s in sizes), default=0)
    items: list[tuple[BBox, object]] = []
    y = 0.0
    for i, (child, sz) in enumerate(zip(children, sizes, strict=True)):
        x = (max_w - sz.w) / 2
        items.append((BBox(x, y, sz.w, sz.h), child))
        g = gaps[i] if gaps and i < len(gaps) else uniform_g
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
                tip_y = y1 + _ARROW_CLR
                yield S.Line(x1=cx, y1=y2, x2=cx, y2=tip_y, class_=["arrow"])
                yield _arrow_up(cx, tip_y)

    @staticmethod
    def render_exit_arrow(
        items: list[tuple[BBox, object]], bb: BBox, content_bb: BBox, title_h: float,
    ) -> Iterator[S.Element]:
        """Upward arrow from top item to the frame's title strip bottom."""
        if not items:
            return
        top_bb = items[0][0]
        cx = content_bb.x + top_bb.x + top_bb.w / 2
        y_start = content_bb.y + top_bb.y
        y_end = bb.y + title_h + _ARROW_CLR
        if y_start - y_end > 2 * _ARROW_CLR:
            yield S.Line(x1=cx, y1=y_start, x2=cx, y2=y_end, class_=["arrow"])
            yield _arrow_up(cx, y_end)

    @staticmethod
    def render_entry_arrow(
        items: list[tuple[BBox, object]], bb: BBox, content_bb: BBox,
    ) -> Iterator[S.Element]:
        """Upward arrow from below the frame into the bottom item."""
        if not items:
            return
        bottom_bb = items[-1][0]
        cx = content_bb.x + bottom_bb.x + bottom_bb.w / 2
        y_start = bb.y + bb.h
        y_end = content_bb.y + bottom_bb.bottom + _ARROW_CLR
        if y_start - y_end > 2 * _ARROW_CLR:
            yield S.Line(x1=cx, y1=y_start, x2=cx, y2=y_end, class_=["arrow"])
            yield _arrow_up(cx, y_end)
