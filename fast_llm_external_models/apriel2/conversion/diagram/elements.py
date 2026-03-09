"""Renderable elements for architecture diagrams.

Primitives: Box, Symbol, Arrow, Label
Composites: DecoderBlock, detail panels per mixer type

All visual styling goes through CSS class names — no inline fills/strokes.
All render methods are generators yielding svg.Element.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, assert_never

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
    bold: bool = False

    def measure(self, th: Theme) -> Size:
        w = self.w or th.geo.inner_w
        h = self.h or th.geo.box_h
        return Size(w, h)

    def render(self, bb: BBox, th: Theme) -> Iterator[S.Element]:
        txt_cls = ["t-label-bold"] if self.bold else ["t-label"]
        is_muted = self.css in (
            "box-norm", "box-linear", "box-mlp",
            "box-embedding", "box-activation", "box-transparent",
        )
        shadow_cls = "box-shadow-muted" if is_muted else "box-shadow"
        els: list[S.Element] = [
            S.Rect(x=bb.x + 3, y=bb.y + 3, width=bb.w, height=bb.h,
                   rx=th.geo.rx, class_=[shadow_cls]),
            S.Rect(x=bb.x, y=bb.y, width=bb.w, height=bb.h, rx=th.geo.rx),
            S.Rect(x=bb.x + 2, y=bb.y + 1, width=bb.w - 4, height=6,
                   rx=5, class_=["box-sheen"]),
            S.Text(x=bb.cx, y=bb.cy, text=self.label, class_=txt_cls),
        ]
        yield S.G(class_=["box", self.css], elements=els)


@dataclass
class Symbol:
    """Circle with plus or cross lines."""

    kind: Literal["plus", "cross"] = "plus"
    r: float | None = None

    def measure(self, th: Theme) -> Size:
        r = self.r or th.geo.symbol_r
        return Size(r * 2, r * 2)

    def render(self, bb: BBox, th: Theme) -> Iterator[S.Element]:
        r = self.r or th.geo.symbol_r
        cx, cy = bb.cx, bb.cy
        clip_id = f"sym-clip-{id(self)}"
        arm = r * 0.45  # half-length of each stroke
        if self.kind == "plus":
            strokes: list[S.Element] = [
                S.Line(x1=cx - arm, y1=cy, x2=cx + arm, y2=cy, class_=["symbol-stroke"]),
                S.Line(x1=cx, y1=cy - arm, x2=cx, y2=cy + arm, class_=["symbol-stroke"]),
            ]
        elif self.kind == "cross":
            strokes = [
                S.Line(x1=cx - arm, y1=cy - arm, x2=cx + arm, y2=cy + arm, class_=["symbol-stroke"]),
                S.Line(x1=cx - arm, y1=cy + arm, x2=cx + arm, y2=cy - arm, class_=["symbol-stroke"]),
            ]
        else:
            assert_never(self.kind)
        yield S.G(class_=["symbol"], elements=[
            S.Circle(cx=cx + 2, cy=cy + 2, r=r, class_=["box-shadow-muted"]),
            S.Circle(cx=cx, cy=cy, r=r),
            S.Defs(elements=[
                S.ClipPath(id=clip_id, elements=[
                    S.Circle(cx=cx, cy=cy, r=r),
                ]),
            ]),
            S.Ellipse(cx=cx, cy=cy - r * 0.45, rx=r * 0.7, ry=r * 0.4,
                       clip_path=f"url(#{clip_id})", class_=["box-sheen"]),
            *strokes,
        ])


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
            "t-small": th.typo.sz_small, "t-label": th.typo.sz_label,
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


@dataclass
class ValueLabel:
    """Data-value annotation (e.g. "hidden states", "q", "k", "v").

    Auto-sizes width from text; unified height via ``th.geo.value_label_h``.
    Pass explicit ``w`` to override (e.g. for HStack column alignment).
    """

    text: str
    w: float | None = None
    _H_PAD: float = 8

    def measure(self, th: Theme) -> Size:
        if self.w is not None:
            w = self.w
        else:
            w = len(self.text) * th.typo.sz_ann * 0.6 + 2 * self._H_PAD
        return Size(w, th.geo.value_label_h)

    def render(self, bb: BBox, th: Theme) -> Iterator[S.Element]:
        sz = self.measure(th)
        rx = bb.x + (bb.w - sz.w) / 2
        ry = bb.y + (bb.h - sz.h) / 2
        yield S.G(class_=["box", "box-transparent"], elements=[
            S.Rect(x=rx + 3, y=ry + 3, width=sz.w, height=sz.h,
                   rx=th.geo.rx, class_=["box-shadow-muted"]),
            S.Rect(x=rx, y=ry, width=sz.w, height=sz.h, rx=th.geo.rx),
            S.Rect(x=rx + 2, y=ry + 1, width=sz.w - 4, height=6,
                   rx=5, class_=["box-sheen"]),
            S.Text(x=bb.cx, y=bb.cy, text=self.text, class_=["t-label"]),
        ])


# ═══════════════════════════════════════════════════════════════════════
# SVG definitions (markers, filters)
# ═══════════════════════════════════════════════════════════════════════


def defs(th: Theme) -> S.Defs:
    """Dot-grid background pattern."""
    return S.Defs(elements=[
        S.Pattern(
            id="dotgrid", width=20, height=20,
            patternUnits="userSpaceOnUse",
            elements=[S.Circle(cx=10, cy=10, r=0.8, fill="#d4cfc8", opacity=0.5)],
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


def _arrow_left(x: float, y: float, sz: float = 4) -> S.Path:
    """Left-pointing arrowhead tip at (x, y): a small V opening rightward."""
    return S.Path(
        d=[S.MoveTo(x + sz, y - sz), S.LineTo(x, y), S.LineTo(x + sz, y + sz)],
        class_=["arrow"],
    )


def connector_bezier(x1: float, y1: float, x2: float, y2: float) -> S.Path:
    """Dashed cubic Bezier from (x1,y1) to (x2,y2)."""
    mx = (x1 + x2) / 2
    return S.Path(
        d=[S.MoveTo(x1, y1), S.CubicBezier(x1=mx, y1=y1, x2=mx, y2=y2, x=x2, y=y2)],
        class_=["connector"],
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
    css: str  # box CSS class
    spec: BlockSpec | None  # for connector drawing (None for non-decoder cells)
    in_decoder: bool = True  # False for pre-decoder items outside the frame
    is_data_label: bool = False  # True → render as plain text, not a box


@dataclass
class ArchitectureOverview:
    """Full architecture pipeline: Embedding → decoder blocks → Norm → LM Head."""

    arch: ArchitectureModel

    _POST_LABEL_H = 30  # vertical space for "token probabilities" label above frame

    def _build_items(self) -> list[_OverviewItem]:
        """Build items in render order: top of screen first, bottom last.

        Signal flows upward (bottom-to-top): Embedding feeds into block layers
        which feed into Norm → LM Head.  Pre-decoder items (text tokens)
        sit below the decoder frame.
        """
        arch = self.arch
        items: list[_OverviewItem] = []

        # 1. LM Head (top, inside decoder)
        items.append(_OverviewItem(
            label="LM Head",
            css="box-linear", spec=None,
        ))

        # 2. Norm (inside decoder)
        norm_type = "RMSNorm"
        if arch.block_groups:
            norm_type = arch.block_groups[0].block_spec.norm_type
        items.append(_OverviewItem(
            label=norm_type,
            css="box-norm", spec=None,
        ))

        # 3. Block groups REVERSED (highest layers at top, layer 0 at bottom)
        for group in reversed(arch.block_groups):
            name = group.block_name or _mixer_short_name(group.block_spec)
            label = f"{name} \u00d7{group.count}" if group.count > 1 else name
            items.append(_OverviewItem(
                label=label,
                css=mixer_css_class(group.block_spec),
                spec=group.block_spec,
            ))

        # 4. Embedding (bottom of decoder)
        items.append(_OverviewItem(
            label="Embedding",
            css="box-embedding", spec=None,
        ))

        # 5. Pre-decoder items (outside frame, below decoder)
        items.append(_OverviewItem(
            label="text tokens",
            css="box-norm", spec=None,
            in_decoder=False, is_data_label=True,
        ))

        return items

    def measure(self, th: Theme) -> Size:
        items = self._build_items()
        n = len(items)
        if n == 0:
            return Size(0, 0)
        n_dec = sum(1 for i in items if i.in_decoder)
        n_pre = n - n_dec
        cell_h = th.geo.stack_cell_h
        gap = th.geo.stack_cell_gap
        g = th.geo.gap
        title_h = th.geo.title_h

        # Decoder frame: title strip + padding + cells
        dec_h = title_h + 2 * g + n_dec * cell_h + (n_dec - 1) * gap
        # Pre-decoder cells below frame (data labels use value_label_h)
        value_h = th.geo.value_label_h
        pre_items = [it for it in items if not it.in_decoder]
        pre_h = sum(value_h if it.is_data_label else cell_h for it in pre_items) + max(0, n_pre - 1) * gap if n_pre > 0 else 0
        # Gap between frame and pre-decoder
        frame_gap = g if n_pre > 0 else 0

        total_h = self._POST_LABEL_H + dec_h + frame_gap + pre_h
        w = th.geo.stack_w
        if self.arch.tie_word_embeddings:
            w += g
        return Size(w, total_h)

    def render(self, bb: BBox, th: Theme) -> Iterator[S.Element]:
        items = self._build_items()
        box_x = bb.x
        box_w = th.geo.stack_w
        cell_h = th.geo.stack_cell_h
        gap = th.geo.stack_cell_gap
        g = th.geo.gap
        title_h = th.geo.title_h
        cx = box_x + box_w / 2

        dec_items = [it for it in items if it.in_decoder]
        pre_items = [it for it in items if not it.in_decoder]
        n_dec = len(dec_items)

        # ── Post-decoder label ("token probabilities") above frame ──
        post_label = ValueLabel("token probabilities")
        post_sz = post_label.measure(th)
        frame_top = bb.y + self._POST_LABEL_H

        # Exit arrow step 1: line through title bar region (drawn BEFORE frame so title hides it)
        yield S.Line(x1=cx, y1=frame_top + title_h, x2=cx, y2=frame_top, class_=["arrow"])

        # ── Decoder frame ───────────────────────────────────────────
        dec_h = title_h + 2 * g + n_dec * cell_h + (n_dec - 1) * gap
        frame_w = bb.w + 2 * g if self.arch.tie_word_embeddings else box_w + 2 * g
        frame, content_bb = _render_detail_frame(
            box_x - g, frame_top, frame_w, dec_h,
            "Decoder", "detail-decoder", th,
        )
        yield frame

        # Exit arrow step 2: above frame to label (drawn AFTER frame, visible)
        label_bottom = bb.y + self._POST_LABEL_H - g
        tip_y = label_bottom + _ARROW_CLR
        yield S.Line(x1=cx, y1=frame_top, x2=cx, y2=tip_y, class_=["arrow"])
        yield _arrow_up(cx, tip_y)

        # Render the label itself
        yield from post_label.render(
            BBox(cx - post_sz.w / 2, bb.y, post_sz.w, post_sz.h), th,
        )

        # Exit arrow step 3: from LM Head to title bar (inside content, drawn AFTER frame)
        lm_head_top = content_bb.y
        y_title_bottom = frame_top + title_h
        if y_title_bottom - lm_head_top < -1:
            yield S.Line(x1=cx, y1=lm_head_top, x2=cx, y2=y_title_bottom, class_=["arrow"])

        # ── Decoder cells inside frame ──────────────────────────────
        y = content_bb.y
        for i, item in enumerate(dec_items):
            bx = Box(item.label, item.css, w=box_w, h=cell_h)
            yield from bx.render(BBox(box_x, y, box_w, cell_h), th)

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
        frame_bottom = frame_top + dec_h
        pre_y = frame_bottom + g  # gap between frame and pre-decoder

        for i, item in enumerate(pre_items):
            if item.is_data_label:
                bx = ValueLabel(item.label)
                bx_sz = bx.measure(th)
                bx_h = bx_sz.h
                yield from bx.render(BBox(box_x, pre_y, box_w, bx_h), th)
            else:
                bx_h = cell_h
                bx = Box(item.label, item.css, w=box_w, h=cell_h)
                yield from bx.render(BBox(box_x, pre_y, box_w, cell_h), th)

            # Arrow from this pre-decoder item to last decoder (first item only)
            if i == 0 and dec_items:
                last_dec_bottom = content_bb.y + (n_dec - 1) * (cell_h + gap) + cell_h
                arrow_start = pre_y
                if arrow_start - last_dec_bottom > 1:
                    tip_y = last_dec_bottom + _ARROW_CLR
                    yield S.Line(x1=cx, y1=arrow_start, x2=cx, y2=tip_y, class_=["arrow"])
                    yield _arrow_up(cx, tip_y)

            # Upward flow arrow between pre-decoder cells
            if i < len(pre_items) - 1:
                line_y1 = pre_y + bx_h
                line_y2 = pre_y + bx_h + gap
                if line_y2 - line_y1 > 1:
                    tip_y = line_y1 + _ARROW_CLR
                    yield S.Line(x1=cx, y1=line_y2, x2=cx, y2=tip_y, class_=["arrow"])
                    yield _arrow_up(cx, tip_y)

            pre_y += bx_h + gap

        # ── Tied weights: Embedding → LM Head ───────────────────────
        # In render order: LM Head is dec_items[0], Embedding is dec_items[-1]
        if self.arch.tie_word_embeddings and n_dec >= 2:
            lm_head_cy = content_bb.y + cell_h / 2
            embed_cy = content_bb.y + (n_dec - 1) * (cell_h + gap) + cell_h / 2
            bypass_x = box_x + box_w + g
            yield S.Path(
                d=[
                    S.MoveTo(box_x + box_w, embed_cy),
                    S.LineTo(bypass_x, embed_cy),
                    S.LineTo(bypass_x, lm_head_cy),
                    S.LineTo(box_x + box_w, lm_head_cy),
                ],
                class_=["connector"],
            )

    def cell_bboxes(self, bb: BBox, th: Theme) -> list[tuple[BBox, BlockSpec | None]]:
        """Return (bbox, spec) for each cell. Only decoder block cells have non-None spec."""
        items = self._build_items()
        box_x = bb.x
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
                label=f"Patch Embed {patch_h}\u00d7{patch_w}",
                css="box-embedding", spec=None,
            ),
            _OverviewItem(
                label=f"Vision Enc \u00d7{v.num_blocks}" if v.num_blocks > 1 else "Vision Enc",
                css=mixer_css_class(v.block_spec),
                spec=v.block_spec,
            ),
            _OverviewItem(
                label="Adapter MLP",
                css="box-mlp", spec=None,
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

    def _envelope(self) -> DetailEnvelope:
        return DetailEnvelope(
            self.title, "block-bg",
            output_labels=[ExternalLabel("hidden states")],
            input_labels=[ExternalLabel("hidden states")],
        )

    def _content_h(self, th: Theme) -> float:
        """Height of internal content (without frame/title/label areas)."""
        g = th.geo.gap
        bh = th.geo.box_h
        sr = th.geo.symbol_r
        mixer_box = _mixer_box(self.mixer, th)
        mh = mixer_box.measure(th).h
        bg = g * 2
        return sr * 2 + g + bh + g + (bh - 4) + bg + sr * 2 + g + mh + g + (bh - 4) + g

    def measure(self, th: Theme) -> Size:
        w = self.block_w or th.geo.block_w
        envelope = self._envelope()
        content_h = self._content_h(th)
        # Content width = block_w - 2*g (frame padding)
        content_w = w - 2 * th.geo.gap
        return envelope.measure_envelope(content_w, content_h, th)

    def render(self, bb: BBox, th: Theme) -> Iterator[S.Element]:
        g = th.geo.gap
        bg = g * 2
        bh = th.geo.box_h
        sr = th.geo.symbol_r
        w = self.block_w or th.geo.block_w

        envelope = self._envelope()
        content_h = self._content_h(th)
        content_w = w - 2 * g
        env = envelope.render_envelope(bb, content_w, content_h, th)

        # The frame bb determines the block area
        frame_bb = env.frame_bb
        title_offset = self._title_offset(th)
        iw = frame_bb.w - 3 * g  # reserve right margin for bypass lines
        ix = frame_bb.x + g
        cx = ix + iw / 2  # center x for flow lines
        env.spine_cx = cx  # shift envelope arrows left to match internal flow
        bypass_x = frame_bb.right - g
        cur_y = frame_bb.y + g + title_offset

        elements: list[S.Element] = []
        arrows: list[S.Element] = []

        def _collect(it: Iterator[S.Element]) -> None:
            elements.extend(it)

        # ── Residual add 2 (output side) ──────────────────────────
        res2_bb = BBox(cx - sr, cur_y, sr * 2, sr * 2)
        _collect(Symbol("plus").render(res2_bb, th))
        cur_y += sr * 2 + g

        # ── Feed forward (MLP) ────────────────────────────────────
        mlp_bb = BBox(ix, cur_y, iw, bh)
        _collect(Box("Feed forward", "box-mlp", w=iw, h=bh).render(mlp_bb, th))
        cur_y += bh + g

        # ── Pre-FFN norm ──────────────────────────────────────────
        norm2_bb = BBox(ix, cur_y, iw, bh - 4)
        _collect(Box(self.norm_type, "box-norm", w=iw, h=bh - 4).render(norm2_bb, th))
        cur_y += bh - 4 + bg

        # ── Residual add 1 ────────────────────────────────────────
        res1_bb = BBox(cx - sr, cur_y, sr * 2, sr * 2)
        _collect(Symbol("plus").render(res1_bb, th))
        cur_y += sr * 2 + g

        # ── Mixer ─────────────────────────────────────────────────
        mixer = _mixer_box(self.mixer, th)
        msz = mixer.measure(th)
        mixer_bb = BBox(ix, cur_y, iw, msz.h)
        _collect(mixer.render(mixer_bb, th))
        cur_y += msz.h + g

        # ── Pre-mixer norm (input layernorm) ──────────────────────
        norm1_bb = BBox(ix, cur_y, iw, bh - 4)
        _collect(Box(self.norm_type, "box-norm", w=iw, h=bh - 4).render(norm1_bb, th))
        cur_y += bh - 4 + g

        # ── Vertical flow arrows through center ───────────────────
        flow_segments = [
            (res2_bb.bottom, mlp_bb.y),
            (mlp_bb.bottom, norm2_bb.y),
            (norm2_bb.bottom, res1_bb.y),
            (res1_bb.bottom, mixer_bb.y),
            (mixer_bb.bottom, norm1_bb.y),
        ]
        for y1, y2 in flow_segments:
            if y2 - y1 > 2 * _ARROW_CLR:
                arrows.append(S.Line(x1=cx, y1=y1 + _ARROW_CLR, x2=cx, y2=y2 - _ARROW_CLR, class_=["arrow"]))

        # ── Internal connector stubs (plain lines, no arrowheads) ──
        arrows.append(S.Line(x1=cx, y1=res2_bb.y, x2=cx, y2=frame_bb.y + title_offset, class_=["arrow"]))

        arrows.append(S.Line(x1=cx, y1=frame_bb.bottom, x2=cx, y2=norm1_bb.bottom, class_=["arrow"]))

        # ── Residual bypass lines ─────────────────────────────────
        tip_x = cx + sr + _ARROW_CLR

        input_y = norm1_bb.bottom + g
        arrows.append(S.Polyline(
            points=[S.Point(cx, input_y), S.Point(bypass_x, input_y),
                    S.Point(bypass_x, res1_bb.cy), S.Point(tip_x, res1_bb.cy)],
            class_=["arrow"],
        ))
        arrows.append(_arrow_left(tip_x, res1_bb.cy))

        branch2_y = (norm2_bb.bottom + res1_bb.y) / 2
        arrows.append(S.Polyline(
            points=[S.Point(cx, branch2_y), S.Point(bypass_x, branch2_y),
                    S.Point(bypass_x, res2_bb.cy), S.Point(tip_x, res2_bb.cy)],
            class_=["arrow"],
        ))
        arrows.append(_arrow_left(tip_x, res2_bb.cy))

        # ── Render in correct Z-order ──────────────────────────────
        # Phase 1: behind title
        yield from env.phase1_behind_title()
        # Phase 2: frame (behind everything)
        yield from env.phase2_frame()
        # Phase 3: exit arrow + output labels
        yield from env.phase3_exit_arrow_and_output_labels()

        yield from arrows
        yield from elements

        # Phase 4: entry arrow + input labels
        yield from env.phase4_entry_arrow_and_input_labels()

    def mlp_bbox(self, bb: BBox, th: Theme) -> BBox:
        """Return the BBox of the 'Feed forward' box within this block."""
        g = th.geo.gap
        bh = th.geo.box_h
        sr = th.geo.symbol_r
        output_area_h = self._envelope()._output_area_h(th)
        iw = bb.w - 3 * g
        ix = bb.x + g
        cur_y = bb.y + output_area_h + 2 * g + self._title_offset(th)
        cur_y += sr * 2 + g      # res2 symbol
        return BBox(ix, cur_y, iw, bh)

    def mixer_bbox(self, bb: BBox, th: Theme) -> BBox:
        """Return the BBox of the mixer box within this block."""
        g = th.geo.gap
        bg = g * 2  # branch gap
        bh = th.geo.box_h
        sr = th.geo.symbol_r
        output_area_h = self._envelope()._output_area_h(th)
        iw = bb.w - 3 * g
        ix = bb.x + g
        cur_y = bb.y + output_area_h + 2 * g + self._title_offset(th)
        cur_y += sr * 2 + g      # res2 symbol
        cur_y += bh + g           # FFN
        cur_y += (bh - 4) + bg    # norm2 + branch gap
        cur_y += sr * 2 + g      # res1 symbol
        mixer = _mixer_box(self.mixer, th)
        msz = mixer.measure(th)
        return BBox(ix, cur_y, iw, msz.h)


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


@dataclass
class ExternalLabel:
    """Spec for an I/O label outside a container frame."""

    text: str
    w: float | None = None  # None = auto-size


@dataclass
class EnvelopeResult:
    """Positions computed by DetailEnvelope.render_envelope().

    Provides phased rendering for correct Z-ordering:
    - phase1: arrow segment behind title bar (drawn BEFORE frame)
    - phase2: the frame group itself
    - phase3: exit arrow above frame + output labels
    - phase4: entry arrow below frame + input labels
    """

    frame_bb: BBox
    content_bb: BBox
    frame_group: S.G
    spine_cx: float
    _th: Theme
    _title_h: float
    _output_labels: list[ExternalLabel] | None
    _input_labels: list[ExternalLabel] | None
    _output_area_h: float
    _input_area_h: float

    def phase1_behind_title(self) -> Iterator[S.Element]:
        """Arrow segment through title bar region (drawn BEFORE frame so title hides it)."""
        if self._title_h < 1:
            return
        y_title_bottom = self.frame_bb.y + self._title_h
        y_title_top = self.frame_bb.y
        if y_title_bottom - y_title_top > 1:
            yield S.Line(x1=self.spine_cx, y1=y_title_bottom, x2=self.spine_cx, y2=y_title_top, class_=["arrow"])

    def phase2_frame(self) -> Iterator[S.Element]:
        """The frame group (title bar paints on top, hiding phase1 segment)."""
        yield self.frame_group

    def phase3_exit_arrow_and_output_labels(self) -> Iterator[S.Element]:
        """Exit arrow from content top through title bar to output labels."""
        g = self._th.geo.gap

        if self._output_labels and self._output_area_h > 0:
            # Arrow from top of frame to output label area
            y_frame_top = self.frame_bb.y
            label_box_h = self._th.geo.value_label_h
            tip_y = self.frame_bb.y - self._output_area_h + label_box_h + _ARROW_CLR
            if y_frame_top - tip_y > 1:
                yield S.Line(x1=self.spine_cx, y1=y_frame_top, x2=self.spine_cx, y2=tip_y, class_=["arrow"])
                yield _arrow_up(self.spine_cx, tip_y)

            # Render output labels centered above frame
            yield from self._render_label_elements(
                self._output_labels,
                y_base=self.frame_bb.y - self._output_area_h,
            )

    def phase4_entry_arrow_and_input_labels(self, target_y: float | None = None) -> Iterator[S.Element]:
        """Entry arrow from input labels up into the frame bottom.

        Args:
            target_y: Y coordinate where the entry arrow should terminate (tip).
                       If None, arrow goes to the frame bottom.
        """
        if not self._input_labels or self._input_area_h <= 0:
            return

        frame_bottom = self.frame_bb.y + self.frame_bb.h
        label_box_h = self._th.geo.value_label_h

        if len(self._input_labels) == 1:
            # Single label: centered under frame
            lbl = self._input_labels[0]
            vl = ValueLabel(lbl.text, w=lbl.w)
            sz = vl.measure(self._th)
            label_y = frame_bottom + self._input_area_h - sz.h
            yield from vl.render(
                BBox(self.spine_cx - sz.w / 2, label_y, sz.w, sz.h), self._th,
            )
            # Entry arrow from label top to target
            arrow_start_y = label_y - _ARROW_CLR
            arrow_end_y = target_y if target_y is not None else frame_bottom
            if arrow_start_y - arrow_end_y > 1:
                yield S.Line(x1=self.spine_cx, y1=arrow_start_y, x2=self.spine_cx, y2=arrow_end_y, class_=["arrow"])
        else:
            # Multiple labels: HStack layout centered on frame
            value_labels = [ValueLabel(lbl.text, w=lbl.w) for lbl in self._input_labels]
            hstack = HStack(value_labels, gap=5)
            hsz = hstack.measure(self._th)
            label_y = frame_bottom + self._input_area_h - hsz.h
            hstack_x = self.spine_cx - hsz.w / 2
            yield from hstack.render(
                BBox(hstack_x, label_y, hsz.w, hsz.h), self._th,
            )

    def _render_label_elements(
        self, labels: list[ExternalLabel], y_base: float,
    ) -> Iterator[S.Element]:
        """Render output labels (above frame)."""
        if len(labels) == 1:
            lbl = labels[0]
            vl = ValueLabel(lbl.text, w=lbl.w)
            sz = vl.measure(self._th)
            yield from vl.render(
                BBox(self.spine_cx - sz.w / 2, y_base, sz.w, sz.h), self._th,
            )
        else:
            value_labels = [ValueLabel(lbl.text, w=lbl.w) for lbl in labels]
            hstack = HStack(value_labels, gap=5)
            hsz = hstack.measure(self._th)
            hstack_x = self.spine_cx - hsz.w / 2
            yield from hstack.render(
                BBox(hstack_x, y_base, hsz.w, hsz.h), self._th,
            )


@dataclass
class DetailEnvelope:
    """Handles titled frame + external I/O labels + through-bar arrows.

    Composition-based helper: containers delegate to this rather than inheriting.
    """

    title: str
    css: str
    output_labels: list[ExternalLabel] | None = None
    input_labels: list[ExternalLabel] | None = None

    def label_area_h(self, th: Theme) -> float:
        """Height of one label area (label box + gap)."""
        return th.geo.value_label_h + th.geo.gap

    def _output_area_h(self, th: Theme) -> float:
        return self.label_area_h(th) if self.output_labels else 0

    def _input_area_h(self, th: Theme) -> float:
        return self.label_area_h(th) if self.input_labels else 0

    def measure_envelope(self, content_w: float, content_h: float, th: Theme) -> Size:
        """Total size including frame + label areas."""
        g = th.geo.gap
        title_h = th.geo.title_h if self.title else 0
        frame_h = content_h + 2 * g + title_h
        total_h = self._output_area_h(th) + frame_h + self._input_area_h(th)
        frame_w = content_w + 2 * g
        return Size(frame_w, total_h)

    def render_envelope(
        self, bb: BBox, content_w: float, content_h: float, th: Theme,
    ) -> EnvelopeResult:
        """Compute positions and create the frame. Returns EnvelopeResult for phased rendering."""
        g = th.geo.gap
        title_h = th.geo.title_h if self.title else 0
        output_area_h = self._output_area_h(th)
        input_area_h = self._input_area_h(th)

        frame_w = content_w + 2 * g
        frame_h = content_h + 2 * g + title_h
        frame_x = bb.x
        frame_y = bb.y + output_area_h

        frame_bb = BBox(frame_x, frame_y, frame_w, frame_h)
        frame_group, content_bb = _render_detail_frame(
            frame_x, frame_y, frame_w, frame_h,
            self.title, self.css, th,
        )

        cx = content_bb.cx

        return EnvelopeResult(
            frame_bb=frame_bb,
            content_bb=content_bb,
            frame_group=frame_group,
            spine_cx=cx,
            _th=th,
            _title_h=title_h,
            _output_labels=self.output_labels,
            _input_labels=self.input_labels,
            _output_area_h=output_area_h,
            _input_area_h=input_area_h,
        )


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
          q       k       v            [2] HStack, value labels
          ↑       ↑       |
        RoPE    RoPE      |            [3] HStack (2 RoPE boxes only)
          |       |       |
        qkv_proj                       [4] box-linear (single merged)
          ↑
        (entry arrow)

    Three lines emerge from the top of qkv_proj at the column
    positions (split, not fork).  The v line bypasses the RoPE row.
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

    def _build(self, th: Theme) -> list[tuple[BBox, Box | Symbol | HStack]]:
        bh = th.geo.box_h - 2
        box_css = "box-swa" if self.config.window_size else "box-attention"
        pw, pg = self._PROJ_W, self._PROJ_GAP
        children: list[Box | Symbol | HStack] = [
            Box("o_proj", "box-linear", w=70, h=bh - 2),                                           # [0]
            Box(self._sdpa_title(), box_css, w=self.w - 30, h=bh, bold=True),                      # [1]
            HStack([ValueLabel("q", w=pw),
                    ValueLabel("k", w=pw),
                    ValueLabel("v", w=pw)], gap=pg),                                                # [2]
            HStack([Box("RoPE", "box-norm", w=pw, h=bh - 4),
                    Box("RoPE", "box-norm", w=pw, h=bh - 4),
                    Spacer(w=pw, h=bh - 4)], gap=pg),                                              # [3]
            Box("qkv_proj", "box-linear", w=self.w - 30, h=bh - 2),                                # [4]
        ]
        return _detail_layout(children, th)

    def _col_centers(self, items: list[tuple[BBox, Box | Symbol | HStack]], th: Theme) -> list[float]:
        """Compute center-x of each of the three columns from the value labels HStack (item 2)."""
        label_hstack = items[2][1]
        assert isinstance(label_hstack, HStack)
        sizes = [c.measure(th) for c in label_hstack.children]
        hg = label_hstack.gap if label_hstack.gap is not None else th.geo.gap
        label_bb = items[2][0]
        centers: list[float] = []
        x = label_bb.x
        for sz in sizes:
            centers.append(x + sz.w / 2)
            x += sz.w + hg
        return centers

    def _envelope(self) -> DetailEnvelope:
        detail_css = "detail-swa" if self.config.window_size else "detail-attention"
        return DetailEnvelope(
            self._title(), detail_css,
            output_labels=[ExternalLabel("hidden states")],
            input_labels=[ExternalLabel("hidden states")],
        )

    def measure(self, th: Theme) -> Size:
        items = self._build(th)
        total_h = items[-1][0].bottom if items else 0
        content_w = max((bb.right for bb, _ in items), default=0)
        return self._envelope().measure_envelope(content_w, total_h, th)

    def _render_arrows(
        self,
        items: list[tuple[BBox, Box | Symbol | HStack]],
        bb: BBox,
        th: Theme,
    ) -> Iterator[S.Element]:
        """Yield arrows between items with three parallel columns.

        q and k flow: qkv_proj → RoPE → label → SDPA
        v flows:      qkv_proj → label → SDPA  (bypasses RoPE row)
        """
        clearance = _ARROW_CLR
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

        rope_bb = items[3][0]
        qkv_bb = items[4][0]

        # [2] labels → [3] RoPE : plain lines for q and k columns (no arrowhead)
        y1 = bb.y + label_bb.bottom
        y2 = bb.y + rope_bb.y
        if y2 - y1 > 1:
            for col_x in cols[:2]:
                acx = bb.x + col_x
                yield S.Line(x1=acx, y1=y2, x2=acx, y2=y1, class_=["arrow"])

        # [3] RoPE → [4] qkv_proj : arrows for q and k columns
        y1 = bb.y + rope_bb.bottom
        y2 = bb.y + qkv_bb.y
        if y2 - y1 > 1:
            for col_x in cols[:2]:
                acx = bb.x + col_x
                tip_y = y1 + clearance
                yield S.Line(x1=acx, y1=y2, x2=acx, y2=tip_y, class_=["arrow"])
                yield _arrow_up(acx, tip_y)

        # v column: continuous line from qkv_proj top to v label bottom (bypasses RoPE, no arrowhead)
        v_cx = bb.x + cols[2]
        v_end_y = bb.y + label_bb.bottom
        v_start_y = bb.y + qkv_bb.y
        if v_start_y - v_end_y > 1:
            yield S.Line(x1=v_cx, y1=v_start_y, x2=v_cx, y2=v_end_y, class_=["arrow"])

    def render(self, bb: BBox, th: Theme) -> Iterator[S.Element]:
        items = self._build(th)
        content_h = items[-1][0].bottom if items else 0
        content_w = max((cbb.right for cbb, _ in items), default=0)

        envelope = self._envelope()
        env = envelope.render_envelope(bb, content_w, content_h, th)
        title_h = th.geo.title_h
        if items:
            top_bb = items[0][0]
            env.spine_cx = env.content_bb.x + top_bb.x + top_bb.w / 2

        yield from env.phase1_behind_title()
        yield from env.phase2_frame()
        yield from env.phase3_exit_arrow_and_output_labels()

        # Exit line from top item to title bar (no arrowhead)
        if items:
            top_bb = items[0][0]
            cx = env.content_bb.x + top_bb.x + top_bb.w / 2
            y_start = env.content_bb.y + top_bb.y
            y_end = env.frame_bb.y + title_h
            if y_start - y_end > 1:
                yield S.Line(x1=cx, y1=y_start, x2=cx, y2=y_end, class_=["arrow"])

        # Entry arrow from frame bottom to bottom item
        yield from _DetailArrows.render_entry_arrow(items, env.frame_bb, env.content_bb)

        # Custom inter-item arrows (three parallel columns)
        yield from self._render_arrows(items, env.content_bb, th)

        # Render children
        for child_bb, child in items:
            shifted = BBox(env.content_bb.x + child_bb.x, env.content_bb.y + child_bb.y, child_bb.w, child_bb.h)
            yield from child.render(shifted, th)

        # Input/output labels
        yield from env.phase4_entry_arrow_and_input_labels()


@dataclass
class GDNDetail:
    """Detail panel for Gated DeltaNet.

    Layout (data flows bottom → top)::

             out_proj                              [0] box-linear
               |                                        line only, no arrowhead
             Gated RMSNorm                         [1] box-norm
      ↑        ↑                 ↑ z bypass
      β,α    Gated Delta Rule  [spacer]            [2] HStack(DeltaRule, Spacer)
      bypass   ↑   ↑   ↑
      [spacer] CausalConv1d [spacer]               [3] HStack(Spacer, Conv, Spacer)
               |   |   |
      β   α    q   k   v    z                      [4] HStack, value labels (reordered)
      |   |    |   |   |    |
      in_proj_βα | in_proj_qkvz                    [5] HStack
             ↑       ↑
              ╰──┬──╯                                   fork from hidden states
                  |

    β, α at far left bypass CausalConv1d, feeding into Gated Delta Rule.
    q, k, v in middle flow through CausalConv1d → Gated Delta Rule.
    z at far right bypasses both conv and delta rule, into Gated RMSNorm.
    """

    config: GDNDisplayConfig
    w: float = 280

    _PROJ_W = 32
    _PROJ_GAP = 4

    def _build(self, th: Theme) -> list[tuple[BBox, Box | Symbol | HStack]]:
        bh = th.geo.box_h - 2
        pw, pg = self._PROJ_W, self._PROJ_GAP
        conv_w = 3 * pw + 2 * pg                                    # 104
        left_spacer_w = 2 * (pw + pg)                               # 72
        labels_total_w = 6 * pw + 5 * pg                            # 212
        right_spacer_w = labels_total_w - left_spacer_w - conv_w    # 36
        proj_left_w = 2 * pw + pg            # in_proj_βα  (2 outputs)
        proj_right_w = 4 * pw + 3 * pg      # in_proj_qkvz (4 outputs)
        delta_w = 5 * pw + 4 * pg                                       # 176
        delta_spacer_w = labels_total_w - delta_w                        # 36
        children: list[Box | Symbol | HStack] = [
            Box("out_proj", "box-linear", w=70, h=bh - 2),                              # [0]
            Box("Gated RMSNorm", "box-norm", w=labels_total_w, h=bh),                     # [1]
            HStack([Box("Gated Delta Rule", "box-gdn", w=delta_w, h=bh + 10, bold=True),  # [2]
                    Spacer(delta_spacer_w, bh + 10)], gap=0),
            HStack([Spacer(left_spacer_w, bh),                                           # [3]
                    Box("CausalConv1d", "box-conv", w=conv_w, h=bh),
                    Spacer(right_spacer_w, bh)], gap=0),
            HStack([ValueLabel("\u03b2", w=pw), ValueLabel("\u03b1", w=pw),              # [4]
                    ValueLabel("q", w=pw), ValueLabel("k", w=pw),
                    ValueLabel("v", w=pw), ValueLabel("z", w=pw)], gap=pg),
            HStack([                                                                      # [5]
                Box("in_proj_\u03b2\u03b1", "box-gate", w=proj_left_w, h=bh),
                Box("in_proj_qkvz", "box-linear", w=proj_right_w, h=bh),
            ], gap=pg),
        ]
        return _detail_layout(children, th)

    def _col_centers(self, items: list[tuple[BBox, Box | Symbol | HStack]], th: Theme) -> list[float]:
        """Compute center-x of each of the 6 value label columns (β,α,q,k,v,z) from item [4]."""
        label_hstack = items[4][1]
        assert isinstance(label_hstack, HStack)
        sizes = [c.measure(th) for c in label_hstack.children]
        hg = label_hstack.gap if label_hstack.gap is not None else th.geo.gap
        label_bb = items[4][0]
        centers: list[float] = []
        x = label_bb.x
        for sz in sizes:
            centers.append(x + sz.w / 2)
            x += sz.w + hg
        return centers

    def _envelope(self) -> DetailEnvelope:
        return DetailEnvelope(
            "Gated DeltaNet", "detail-gdn",
            output_labels=[ExternalLabel("hidden states")],
            input_labels=[ExternalLabel("hidden states")],
        )

    def measure(self, th: Theme) -> Size:
        items = self._build(th)
        content_h = items[-1][0].bottom if items else 0
        content_w = max((bb.right for bb, _ in items), default=0)
        fork_space = th.geo.gap  # space below proj boxes for fork
        content_h += fork_space
        return self._envelope().measure_envelope(content_w, content_h, th)

    def _render_arrows(
        self,
        items: list[tuple[BBox, Box | Symbol | HStack]],
        bb: BBox,
        th: Theme,
    ) -> Iterator[S.Element]:
        """Yield arrows with bypass paths for z and β/α.

        β, α: proj labels → Gated Delta Rule (bypasses conv)
        q, k, v: proj labels → CausalConv1d → Gated Delta Rule (standard path)
        z:       proj label  → Gated RMSNorm (bypasses conv and delta rule)
        """
        clr = _ARROW_CLR
        cols = self._col_centers(items, th)
        # cols: [β, α, q, k, v, z]

        o_bb = items[0][0]       # out_proj
        norm_bb = items[1][0]    # Gated RMSNorm
        delta_bb = items[2][0]   # Gated Delta Rule
        conv_bb = items[3][0]    # CausalConv1d (HStack with spacers)
        label_bb = items[4][0]   # value labels
        proj_bb = items[5][0]    # projection boxes

        # --- Standard vertical arrows (centred) ---

        # [0] out_proj ← [1] Gated RMSNorm
        cx = bb.x + o_bb.cx
        y1 = bb.y + o_bb.bottom
        y2 = bb.y + norm_bb.y
        if y2 - y1 > 1:
            tip_y = y1 + clr
            yield S.Line(x1=cx, y1=y2, x2=cx, y2=tip_y, class_=["arrow"])
            yield _arrow_up(cx, tip_y)

        # [1] Gated RMSNorm ← [2] Gated Delta Rule (centred)
        cx = bb.x + (norm_bb.cx + delta_bb.cx) / 2
        y1 = bb.y + norm_bb.bottom
        y2 = bb.y + delta_bb.y
        if y2 - y1 > 1:
            tip_y = y1 + clr
            yield S.Line(x1=cx, y1=y2, x2=cx, y2=tip_y, class_=["arrow"])
            yield _arrow_up(cx, tip_y)

        # --- q, k, v columns (indices 2,3,4): conv → delta rule arrows ---
        y1 = bb.y + delta_bb.bottom
        y2 = bb.y + conv_bb.y
        if y2 - y1 > 1:
            for col_x in cols[2:5]:
                acx = bb.x + col_x
                tip_y = y1 + clr
                yield S.Line(x1=acx, y1=y2, x2=acx, y2=tip_y, class_=["arrow"])
                yield _arrow_up(acx, tip_y)

        # --- q, k, v columns (indices 2,3,4): labels → conv arrows ---
        y1 = bb.y + conv_bb.bottom
        y2 = bb.y + label_bb.y
        if y2 - y1 > 1:
            for col_x in cols[2:5]:
                acx = bb.x + col_x
                tip_y = y1 + clr
                yield S.Line(x1=acx, y1=y2, x2=acx, y2=tip_y, class_=["arrow"])
                yield _arrow_up(acx, tip_y)

        # --- All 6 columns: labels ← projections (plain lines, no arrowheads) ---
        y1 = bb.y + label_bb.bottom
        y2 = bb.y + proj_bb.y
        if y2 - y1 > 1:
            for col_x in cols:
                acx = bb.x + col_x
                yield S.Line(x1=acx, y1=y1, x2=acx, y2=y2, class_=["arrow"])

        # --- z bypass (index 5): label → Gated RMSNorm (skipping conv and delta rule) ---
        z_cx = bb.x + cols[5]
        z_tip_y = bb.y + norm_bb.bottom + clr
        z_start_y = bb.y + label_bb.y
        if z_start_y - z_tip_y > 1:
            yield S.Line(x1=z_cx, y1=z_start_y, x2=z_cx, y2=z_tip_y, class_=["arrow"])
            yield _arrow_up(z_cx, z_tip_y)

        # --- β, α bypass (indices 0,1): labels → Gated Delta Rule (skipping conv) ---
        for col_x in cols[0:2]:
            acx = bb.x + col_x
            tip_y = bb.y + delta_bb.bottom + clr
            start_y = bb.y + label_bb.y
            if start_y - tip_y > 1:
                yield S.Line(x1=acx, y1=start_y, x2=acx, y2=tip_y, class_=["arrow"])
                yield _arrow_up(acx, tip_y)

        # --- Fork below projections ---
        # Compute proj box centers from the proj HStack children layout
        proj_hstack = items[5][1]
        assert isinstance(proj_hstack, HStack)
        proj_sizes = [c.measure(th) for c in proj_hstack.children]
        proj_hg = proj_hstack.gap if proj_hstack.gap is not None else th.geo.gap
        proj_left_cx = proj_bb.x + proj_sizes[0].w / 2
        proj_right_cx = proj_bb.x + proj_sizes[0].w + proj_hg + proj_sizes[1].w / 2

        abs_proj_left_cx = bb.x + proj_left_cx
        abs_proj_right_cx = bb.x + proj_right_cx
        content_cx = (abs_proj_left_cx + abs_proj_right_cx) / 2
        g = th.geo.gap
        midpoint_y = bb.y + proj_bb.bottom + g

        # Left branch: center → in_proj_βα center → up with arrowhead
        proj_left_tip_y = bb.y + proj_bb.bottom + clr
        yield S.Path(
            d=[
                S.MoveTo(content_cx, midpoint_y),
                S.LineTo(abs_proj_left_cx, midpoint_y),
                S.LineTo(abs_proj_left_cx, proj_left_tip_y),
            ],
            class_=["arrow"],
        )
        yield _arrow_up(abs_proj_left_cx, proj_left_tip_y)

        # Right branch: center → in_proj_qkvz center → up with arrowhead
        proj_right_tip_y = bb.y + proj_bb.bottom + clr
        yield S.Path(
            d=[
                S.MoveTo(content_cx, midpoint_y),
                S.LineTo(abs_proj_right_cx, midpoint_y),
                S.LineTo(abs_proj_right_cx, proj_right_tip_y),
            ],
            class_=["arrow"],
        )
        yield _arrow_up(abs_proj_right_cx, proj_right_tip_y)

    def render(self, bb: BBox, th: Theme) -> Iterator[S.Element]:
        items = self._build(th)
        content_h = items[-1][0].bottom if items else 0
        content_w = max((bb.right for bb, _ in items), default=0)
        g = th.geo.gap
        fork_space = g  # space below proj boxes for fork
        content_h += fork_space

        envelope = self._envelope()
        env = envelope.render_envelope(bb, content_w, content_h, th)
        if items:
            top_bb = items[0][0]
            env.spine_cx = env.content_bb.x + top_bb.x + top_bb.w / 2

        # Phase 1: arrow behind title bar
        yield from env.phase1_behind_title()
        # Phase 2: frame
        yield from env.phase2_frame()
        # Phase 3: exit arrow + output labels
        yield from env.phase3_exit_arrow_and_output_labels()

        # Custom inter-item arrows (bypass paths for z and β/α, fork)
        yield from self._render_arrows(items, env.content_bb, th)

        # Exit line from top item to title bar (no arrowhead)
        title_h = th.geo.title_h
        if items:
            top_bb = items[0][0]
            cx = env.content_bb.x + top_bb.x + top_bb.w / 2
            y_start = env.content_bb.y + top_bb.y
            y_end = env.frame_bb.y + title_h + _ARROW_CLR
            if y_start - y_end > 2 * _ARROW_CLR:
                yield S.Line(x1=cx, y1=y_start, x2=cx, y2=y_end, class_=["arrow"])

        # Children
        for child_bb, child in items:
            shifted = BBox(env.content_bb.x + child_bb.x, env.content_bb.y + child_bb.y, child_bb.w, child_bb.h)
            yield from child.render(shifted, th)

        # Phase 4: entry arrow + input labels
        if items:
            bottom_bb = items[-1][0]
            target_y = env.content_bb.y + bottom_bb.bottom + fork_space + _ARROW_CLR
        else:
            target_y = None
        yield from env.phase4_entry_arrow_and_input_labels(target_y)


@dataclass
class KDADetail:
    """Detail panel for Kimi Delta Attention.

    Layout (data flows bottom → top)::

             o_proj                                 [0] box-linear
               |                                         line only, no arrowhead
             Gated RMSNorm                          [1] box-norm
      ↑        ↑                 ↑ g₂ bypass
      β,g₁   Kimi Delta Attn  [spacer]             [2] HStack(KDA, Spacer)
      bypass   ↑   ↑   ↑
      [spacer] Conv Conv Conv  [spacer]             [3] HStack(Spacer, 3×Conv, Spacer)
               |   |   |
      β  g₁   q   k   v   g₂                       [4] HStack, value labels
      |  |    |   |   |    |
      proj_βg₁  proj_qkv  proj_g₂                  [5] HStack of 3 proj boxes
             ↑       ↑       ↑
              ╰──────┼───────╯                           3-way fork from hidden states
                     |

    β, g₁ at far left bypass CausalConv1d, feeding into KDA core.
    q, k, v in middle flow through CausalConv1d → KDA core.
    g₂ at far right bypasses both conv and KDA core, into Gated RMSNorm.
    """

    config: KDADisplayConfig
    w: float = 400

    _PROJ_W = 32
    _PROJ_GAP = 4

    def _build(self, th: Theme) -> list[tuple[BBox, Box | Symbol | HStack]]:
        bh = th.geo.box_h - 2
        pw, pg = self._PROJ_W, self._PROJ_GAP
        qkv_w = 2 * pw + pg                                         # 68 (q/k/v each span 2 cols)
        labels_total_w = (pw + pg + pw + pg                          # β, g₁
                          + qkv_w + pg + qkv_w + pg + qkv_w + pg    # q, k, v
                          + (2 * pw + pg))                           # g₂  → 356
        kda_w = pw + pg + pw + pg + qkv_w + pg + qkv_w + pg + qkv_w  # 284 (β..v)
        kda_spacer_w = labels_total_w - kda_w                        # 72
        proj_left_w = 2 * pw + pg            # proj_βg₁  (2 outputs)  68
        proj_center_w = 3 * qkv_w + 2 * pg  # proj_qkv  (3 outputs)  212
        proj_right_w = 2 * pw + pg           # proj_g₂   (2 columns)  68
        stagger_step = bh + th.geo.gap
        spacer_h = bh + 2 * stagger_step     # vertical space for 3 staggered conv boxes
        children: list[Box | Symbol | HStack] = [
            Box("o_proj", "box-linear", w=70, h=bh - 2),                                  # [0]
            Box("Gated RMSNorm", "box-norm", w=labels_total_w, h=bh),                     # [1]
            HStack([Box("Kimi Delta Attention", "box-kda", w=kda_w, h=bh + 10, bold=True),  # [2]
                    Spacer(kda_spacer_w, bh + 10)], gap=0),
            Spacer(labels_total_w, spacer_h),                                              # [3] reserved for staggered conv boxes
            HStack([ValueLabel("\u03b2", w=pw), ValueLabel("g\u2081", w=pw),               # [4]
                    ValueLabel("q", w=qkv_w), ValueLabel("k", w=qkv_w),
                    ValueLabel("v", w=qkv_w), ValueLabel("g\u2082", w=2*pw+pg)], gap=pg),
            HStack([                                                                        # [5]
                Box("proj_\u03b2g\u2081", "box-gate", w=proj_left_w, h=bh),
                Box("proj_qkv", "box-linear", w=proj_center_w, h=bh),
                Box("proj_g\u2082", "box-gate", w=proj_right_w, h=bh),
            ], gap=pg),
        ]
        return _detail_layout(children, th)

    def _col_centers(self, items: list[tuple[BBox, Box | Symbol | HStack]], th: Theme) -> list[float]:
        """Compute center-x of each of the 6 value label columns (β,g₁,q,k,v,g₂) from item [4]."""
        label_hstack = items[4][1]
        assert isinstance(label_hstack, HStack)
        sizes = [c.measure(th) for c in label_hstack.children]
        hg = label_hstack.gap if label_hstack.gap is not None else th.geo.gap
        label_bb = items[4][0]
        centers: list[float] = []
        x = label_bb.x
        for sz in sizes:
            centers.append(x + sz.w / 2)
            x += sz.w + hg
        return centers

    def _envelope(self) -> DetailEnvelope:
        return DetailEnvelope(
            "Kimi Delta Attention", "detail-kda",
            output_labels=[ExternalLabel("hidden states")],
            input_labels=[ExternalLabel("hidden states")],
        )

    def measure(self, th: Theme) -> Size:
        items = self._build(th)
        content_h = items[-1][0].bottom if items else 0
        content_w = max((bb.right for bb, _ in items), default=0)
        fork_space = th.geo.gap
        content_h += fork_space
        return self._envelope().measure_envelope(content_w, content_h, th)

    def _render_arrows(
        self,
        items: list[tuple[BBox, Box | Symbol | HStack]],
        bb: BBox,
        th: Theme,
    ) -> Iterator[S.Element]:
        """Yield arrows with bypass paths for g₂ and β/g₁.

        β, g₁: proj labels → KDA core (bypasses conv)
        q, k, v: proj labels → CausalConv1d (staggered) → KDA core
        g₂:      proj label  → Gated RMSNorm (bypasses conv and KDA core)
        """
        clr = _ARROW_CLR
        cols = self._col_centers(items, th)
        # cols: [β, g₁, q, k, v, g₂]

        bh = th.geo.box_h - 2
        pw, pg = self._PROJ_W, self._PROJ_GAP

        o_bb = items[0][0]       # out_proj
        norm_bb = items[1][0]    # Gated RMSNorm
        kda_bb = items[2][0]     # Kimi Delta Attention
        spacer_bb = items[3][0]  # Spacer (reserved for staggered conv boxes)
        label_bb = items[4][0]   # value labels
        proj_bb = items[5][0]    # projection boxes

        # --- Render 3 staggered CausalConv1d boxes ---
        kda_conv_w = 2 * pw + pg + 24  # 92 — fits "CausalConv1d" label with comfortable padding
        stagger_step = bh + th.geo.gap  # vertical gap between boxes = g
        conv_boxes: list[Box] = []
        conv_bboxes: list[BBox] = []
        for i, col_idx in enumerate((2, 3, 4)):  # q, k, v
            box = Box("CausalConv1d", "box-conv", w=kda_conv_w, h=bh)
            col_cx = cols[col_idx]
            box_x = bb.x + col_cx - kda_conv_w / 2
            box_y = bb.y + spacer_bb.y + i * stagger_step
            box_bb = BBox(box_x, box_y, kda_conv_w, bh)
            conv_boxes.append(box)
            conv_bboxes.append(box_bb)
            yield from box.render(box_bb, th)

        # --- Standard vertical arrows (centred) ---

        # [0] out_proj ← [1] Gated RMSNorm
        cx = bb.x + o_bb.cx
        y1 = bb.y + o_bb.bottom
        y2 = bb.y + norm_bb.y
        if y2 - y1 > 1:
            tip_y = y1 + clr
            yield S.Line(x1=cx, y1=y2, x2=cx, y2=tip_y, class_=["arrow"])
            yield _arrow_up(cx, tip_y)

        # [1] Gated RMSNorm ← [2] KDA core (centred)
        cx = bb.x + (norm_bb.cx + kda_bb.cx) / 2
        y1 = bb.y + norm_bb.bottom
        y2 = bb.y + kda_bb.y
        if y2 - y1 > 1:
            tip_y = y1 + clr
            yield S.Line(x1=cx, y1=y2, x2=cx, y2=tip_y, class_=["arrow"])
            yield _arrow_up(cx, tip_y)

        # --- q, k, v: conv → KDA core arrows (per staggered box) ---
        for i, col_idx in enumerate((2, 3, 4)):
            acx = bb.x + cols[col_idx]
            tip_y = bb.y + kda_bb.bottom + clr
            start_y = conv_bboxes[i].y  # top of this conv box
            if start_y - tip_y > 1:
                yield S.Line(x1=acx, y1=start_y, x2=acx, y2=tip_y, class_=["arrow"])
                yield _arrow_up(acx, tip_y)

        # --- q, k, v: labels → conv arrows (per staggered box) ---
        for i, col_idx in enumerate((2, 3, 4)):
            acx = bb.x + cols[col_idx]
            tip_y = conv_bboxes[i].bottom + clr  # bottom of this conv box
            start_y = bb.y + label_bb.y
            if start_y - tip_y > 1:
                yield S.Line(x1=acx, y1=start_y, x2=acx, y2=tip_y, class_=["arrow"])
                yield _arrow_up(acx, tip_y)

        # --- All 6 columns: labels ← projections (plain lines, no arrowheads) ---
        y1 = bb.y + label_bb.bottom
        y2 = bb.y + proj_bb.y
        if y2 - y1 > 1:
            for col_x in cols:
                acx = bb.x + col_x
                yield S.Line(x1=acx, y1=y1, x2=acx, y2=y2, class_=["arrow"])

        # --- g₂ bypass (index 5): label → Gated RMSNorm (skipping conv and KDA core) ---
        g2_cx = bb.x + cols[5]
        g2_tip_y = bb.y + norm_bb.bottom + clr
        g2_start_y = bb.y + label_bb.y
        if g2_start_y - g2_tip_y > 1:
            yield S.Line(x1=g2_cx, y1=g2_start_y, x2=g2_cx, y2=g2_tip_y, class_=["arrow"])
            yield _arrow_up(g2_cx, g2_tip_y)

        # --- β, g₁ bypass (indices 0,1): labels → KDA core (skipping conv) ---
        for col_x in cols[0:2]:
            acx = bb.x + col_x
            tip_y = bb.y + kda_bb.bottom + clr
            start_y = bb.y + label_bb.y
            if start_y - tip_y > 1:
                yield S.Line(x1=acx, y1=start_y, x2=acx, y2=tip_y, class_=["arrow"])
                yield _arrow_up(acx, tip_y)

        # --- 3-way fork below projections ---
        proj_hstack = items[5][1]
        assert isinstance(proj_hstack, HStack)
        proj_sizes = [c.measure(th) for c in proj_hstack.children]
        proj_hg = proj_hstack.gap if proj_hstack.gap is not None else th.geo.gap
        proj_left_cx = proj_bb.x + proj_sizes[0].w / 2
        proj_center_cx = proj_bb.x + proj_sizes[0].w + proj_hg + proj_sizes[1].w / 2
        proj_right_cx = (proj_bb.x + proj_sizes[0].w + proj_hg
                         + proj_sizes[1].w + proj_hg + proj_sizes[2].w / 2)

        abs_proj_left_cx = bb.x + proj_left_cx
        abs_proj_center_cx = bb.x + proj_center_cx
        abs_proj_right_cx = bb.x + proj_right_cx
        content_cx = abs_proj_center_cx
        g = th.geo.gap
        midpoint_y = bb.y + proj_bb.bottom + g

        # Left branch: center → proj_βg₁ center → up
        proj_tip_y = bb.y + proj_bb.bottom + clr
        yield S.Path(
            d=[
                S.MoveTo(content_cx, midpoint_y),
                S.LineTo(abs_proj_left_cx, midpoint_y),
                S.LineTo(abs_proj_left_cx, proj_tip_y),
            ],
            class_=["arrow"],
        )
        yield _arrow_up(abs_proj_left_cx, proj_tip_y)

        # Center branch: straight up
        yield S.Path(
            d=[
                S.MoveTo(content_cx, midpoint_y),
                S.LineTo(content_cx, proj_tip_y),
            ],
            class_=["arrow"],
        )
        yield _arrow_up(content_cx, proj_tip_y)

        # Right branch: center → proj_g₂ center → up
        yield S.Path(
            d=[
                S.MoveTo(content_cx, midpoint_y),
                S.LineTo(abs_proj_right_cx, midpoint_y),
                S.LineTo(abs_proj_right_cx, proj_tip_y),
            ],
            class_=["arrow"],
        )
        yield _arrow_up(abs_proj_right_cx, proj_tip_y)

    def render(self, bb: BBox, th: Theme) -> Iterator[S.Element]:
        items = self._build(th)
        content_h = items[-1][0].bottom if items else 0
        content_w = max((bb.right for bb, _ in items), default=0)
        g = th.geo.gap
        fork_space = g
        content_h += fork_space

        envelope = self._envelope()
        env = envelope.render_envelope(bb, content_w, content_h, th)
        if items:
            top_bb = items[0][0]
            env.spine_cx = env.content_bb.x + top_bb.x + top_bb.w / 2

        # Phase 1: arrow behind title bar
        yield from env.phase1_behind_title()
        # Phase 2: frame
        yield from env.phase2_frame()
        # Phase 3: exit arrow + output labels
        yield from env.phase3_exit_arrow_and_output_labels()

        # Custom inter-item arrows (bypass paths for g₂ and β/g₁, fork)
        yield from self._render_arrows(items, env.content_bb, th)

        # Exit line from top item to title bar (no arrowhead)
        title_h = th.geo.title_h
        if items:
            top_bb = items[0][0]
            cx = env.content_bb.x + top_bb.x + top_bb.w / 2
            y_start = env.content_bb.y + top_bb.y
            y_end = env.frame_bb.y + title_h + _ARROW_CLR
            if y_start - y_end > 2 * _ARROW_CLR:
                yield S.Line(x1=cx, y1=y_start, x2=cx, y2=y_end, class_=["arrow"])

        # Children
        for child_bb, child in items:
            shifted = BBox(env.content_bb.x + child_bb.x, env.content_bb.y + child_bb.y, child_bb.w, child_bb.h)
            yield from child.render(shifted, th)

        # Phase 4: entry arrow + input labels
        if items:
            bottom_bb = items[-1][0]
            target_y = env.content_bb.y + bottom_bb.bottom + fork_space + _ARROW_CLR
        else:
            target_y = None
        yield from env.phase4_entry_arrow_and_input_labels(target_y)


@dataclass
class MambaDetail:
    """Detail panel for Mamba SSM.

    Layout (data flows bottom → top)::

             out_proj                              [0] box-linear (narrow, centered)
               ↑
             Selective Scan                         [1] box-mamba (wide, spans 6 cols = 208px)
         ↑   ↑   ↑   ↑   ↑                             receives z, x, B, C, Δ
         z   │        │   Δ  bypass conv (in spacer zones)
             │   B,C  │      bypass conv (through conv zone, single-segment arrows)
             │        │
         [sp] CausalConv1d [sp]                   [2] HStack(Spacer, Conv, Spacer)
              ↑                                         only x actually convolved
         z   x   B   C   Δ                        [3] HStack of 5 ValueLabels
         └───┼───┼───┘   │
          in_proj    dt_proj                       [4] HStack of 2 projection boxes
              ↑         ↑
               ╰───┼───╯                                2-way fork from hidden states
                   ↑

    z bypasses CausalConv1d, fed to SSM as output gate (y * silu(z)).
    x flows through CausalConv1d → SSM (input signal).
    B, C bypass CausalConv1d, fed to SSM as state-space parameters.
    Δ comes from a separate dt_proj, fed to SSM as time-step parameter.
    No Gated RMSNorm — Mamba has zero normalization layers.
    """

    config: MambaDisplayConfig
    w: float = 340

    _PROJ_W = 32
    _PROJ_GAP = 4

    def _build(self, th: Theme) -> list[tuple[BBox, Box | Symbol | HStack]]:
        bh = th.geo.box_h - 2
        pw, pg = self._PROJ_W, self._PROJ_GAP
        x_w = 3 * pw + 2 * pg                               # 104 (x spans 3 columns)
        labels_total_w = pw + pg + x_w + pg + pw + pg + pw + pg + (2 * pw + pg)  # 284
        left_spacer_w = pw + pg                              # 36  (z bypass zone)
        conv_w = 3 * pw + 2 * pg                             # 104 (covers x only)
        right_spacer_w = labels_total_w - left_spacer_w - conv_w  # 144  (B, C, Δ bypass zone)
        ssm_w = labels_total_w                               # 284
        in_proj_w = pw + pg + x_w + pg + pw + pg + pw        # 212 (z, x, B, C)
        dt_proj_w = 2 * pw + pg                              # 68  (Δ)

        children: list[Box | Symbol | HStack] = [
            Box("out_proj", "box-linear", w=70, h=bh - 2),                        # [0]
            Box("Selective Scan", "box-mamba", w=ssm_w, h=bh + 10, bold=True),     # [1]
            HStack([Spacer(left_spacer_w, bh),                                     # [2]
                    Box("CausalConv1d", "box-conv", w=conv_w, h=bh),
                    Spacer(right_spacer_w, bh)], gap=0),
            HStack([ValueLabel("z", w=pw), ValueLabel("x", w=x_w),                # [3]
                    ValueLabel("B", w=pw), ValueLabel("C", w=pw),
                    ValueLabel("\u0394", w=2 * pw + pg)], gap=pg),
            HStack([Box("in_proj", "box-linear", w=in_proj_w, h=bh),              # [4]
                    Box("dt_proj", "box-linear", w=dt_proj_w, h=bh)], gap=pg),
        ]
        return _detail_layout(children, th)

    def _col_centers(self, items: list[tuple[BBox, Box | Symbol | HStack]], th: Theme) -> list[float]:
        """Compute center-x of each of the 5 value label columns (z,x,B,C,Δ) from item [3]."""
        label_hstack = items[3][1]
        assert isinstance(label_hstack, HStack)
        sizes = [c.measure(th) for c in label_hstack.children]
        hg = label_hstack.gap if label_hstack.gap is not None else th.geo.gap
        label_bb = items[3][0]
        centers: list[float] = []
        x = label_bb.x
        for sz in sizes:
            centers.append(x + sz.w / 2)
            x += sz.w + hg
        return centers

    def _envelope(self) -> DetailEnvelope:
        return DetailEnvelope(
            "Mamba", "detail-mamba",
            output_labels=[ExternalLabel("hidden states")],
            input_labels=[ExternalLabel("hidden states")],
        )

    def measure(self, th: Theme) -> Size:
        items = self._build(th)
        content_h = items[-1][0].bottom if items else 0
        content_w = max((bb.right for bb, _ in items), default=0)
        fork_space = th.geo.gap  # space below proj boxes for fork
        content_h += fork_space
        return self._envelope().measure_envelope(content_w, content_h, th)

    def _render_arrows(
        self,
        items: list[tuple[BBox, Box | Symbol | HStack]],
        bb: BBox,
        th: Theme,
    ) -> Iterator[S.Element]:
        """Yield arrows with bypass paths for z, B, C, and Δ.

        x:    label → CausalConv1d → Mamba SSM (standard path through conv)
        z:    label → Mamba SSM (bypasses conv, output gate)
        B, C: labels → Mamba SSM (bypass conv, SSM parameters)
        Δ:    label → Mamba SSM (bypasses conv, time-step from dt_proj)
        """
        clr = _ARROW_CLR
        cols = self._col_centers(items, th)
        # cols: [z, x, B, C, Δ]

        o_bb = items[0][0]       # out_proj
        ssm_bb = items[1][0]     # Mamba SSM
        conv_bb = items[2][0]    # CausalConv1d (HStack with spacers)
        label_bb = items[3][0]   # value labels
        proj_bb = items[4][0]    # projection boxes

        # --- [0] out_proj ← [1] Mamba SSM ---
        cx = bb.x + o_bb.cx
        y1 = bb.y + o_bb.bottom
        y2 = bb.y + ssm_bb.y
        if y2 - y1 > 1:
            tip_y = y1 + clr
            yield S.Line(x1=cx, y1=y2, x2=cx, y2=tip_y, class_=["arrow"])
            yield _arrow_up(cx, tip_y)

        # --- x column (index 1): conv → SSM arrow ---
        y1 = bb.y + ssm_bb.bottom
        y2 = bb.y + conv_bb.y
        if y2 - y1 > 1:
            acx = bb.x + cols[1]
            tip_y = y1 + clr
            yield S.Line(x1=acx, y1=y2, x2=acx, y2=tip_y, class_=["arrow"])
            yield _arrow_up(acx, tip_y)

        # --- x column (index 1): label → conv arrow ---
        y1 = bb.y + conv_bb.bottom
        y2 = bb.y + label_bb.y
        if y2 - y1 > 1:
            acx = bb.x + cols[1]
            tip_y = y1 + clr
            yield S.Line(x1=acx, y1=y2, x2=acx, y2=tip_y, class_=["arrow"])
            yield _arrow_up(acx, tip_y)

        # --- z, B, C, Δ bypass (indices 0, 2, 3, 4): labels → SSM (skipping conv) ---
        for col_idx in (0, 2, 3, 4):
            acx = bb.x + cols[col_idx]
            tip_y = bb.y + ssm_bb.bottom + clr
            start_y = bb.y + label_bb.y
            if start_y - tip_y > 1:
                yield S.Line(x1=acx, y1=start_y, x2=acx, y2=tip_y, class_=["arrow"])
                yield _arrow_up(acx, tip_y)

        # --- All 5 columns: labels ← projections (plain lines, no arrowheads) ---
        y1 = bb.y + label_bb.bottom
        y2 = bb.y + proj_bb.y
        if y2 - y1 > 1:
            for col_x in cols:
                acx = bb.x + col_x
                yield S.Line(x1=acx, y1=y1, x2=acx, y2=y2, class_=["arrow"])

        # --- 2-way fork below projections ---
        proj_hstack = items[4][1]
        assert isinstance(proj_hstack, HStack)
        proj_sizes = [c.measure(th) for c in proj_hstack.children]
        proj_hg = proj_hstack.gap if proj_hstack.gap is not None else th.geo.gap
        proj_left_cx = proj_bb.x + proj_sizes[0].w / 2
        proj_right_cx = proj_bb.x + proj_sizes[0].w + proj_hg + proj_sizes[1].w / 2

        abs_proj_left_cx = bb.x + proj_left_cx
        abs_proj_right_cx = bb.x + proj_right_cx
        content_cx = (abs_proj_left_cx + abs_proj_right_cx) / 2
        g = th.geo.gap
        midpoint_y = bb.y + proj_bb.bottom + g

        # Left branch: center → in_proj center → up with arrowhead
        proj_left_tip_y = bb.y + proj_bb.bottom + clr
        yield S.Path(
            d=[
                S.MoveTo(content_cx, midpoint_y),
                S.LineTo(abs_proj_left_cx, midpoint_y),
                S.LineTo(abs_proj_left_cx, proj_left_tip_y),
            ],
            class_=["arrow"],
        )
        yield _arrow_up(abs_proj_left_cx, proj_left_tip_y)

        # Right branch: center → dt_proj center → up with arrowhead
        proj_right_tip_y = bb.y + proj_bb.bottom + clr
        yield S.Path(
            d=[
                S.MoveTo(content_cx, midpoint_y),
                S.LineTo(abs_proj_right_cx, midpoint_y),
                S.LineTo(abs_proj_right_cx, proj_right_tip_y),
            ],
            class_=["arrow"],
        )
        yield _arrow_up(abs_proj_right_cx, proj_right_tip_y)

    def render(self, bb: BBox, th: Theme) -> Iterator[S.Element]:
        items = self._build(th)
        content_h = items[-1][0].bottom if items else 0
        content_w = max((bb.right for bb, _ in items), default=0)
        g = th.geo.gap
        fork_space = g  # space below proj boxes for fork
        content_h += fork_space

        envelope = self._envelope()
        env = envelope.render_envelope(bb, content_w, content_h, th)
        if items:
            top_bb = items[0][0]
            env.spine_cx = env.content_bb.x + top_bb.x + top_bb.w / 2

        # Phase 1: arrow behind title bar
        yield from env.phase1_behind_title()
        # Phase 2: frame
        yield from env.phase2_frame()
        # Phase 3: exit arrow + output labels
        yield from env.phase3_exit_arrow_and_output_labels()

        # Custom inter-item arrows (bypass paths for z, B, C, Δ, fork)
        yield from self._render_arrows(items, env.content_bb, th)

        # Exit line from top item to title bar (no arrowhead)
        title_h = th.geo.title_h
        if items:
            top_bb = items[0][0]
            cx = env.content_bb.x + top_bb.x + top_bb.w / 2
            y_start = env.content_bb.y + top_bb.y
            y_end = env.frame_bb.y + title_h + _ARROW_CLR
            if y_start - y_end > 2 * _ARROW_CLR:
                yield S.Line(x1=cx, y1=y_start, x2=cx, y2=y_end, class_=["arrow"])

        # Children
        for child_bb, child in items:
            shifted = BBox(env.content_bb.x + child_bb.x, env.content_bb.y + child_bb.y, child_bb.w, child_bb.h)
            yield from child.render(shifted, th)

        # Phase 4: entry arrow + input labels
        if items:
            bottom_bb = items[-1][0]
            target_y = env.content_bb.y + bottom_bb.bottom + fork_space + _ARROW_CLR
        else:
            target_y = None
        yield from env.phase4_entry_arrow_and_input_labels(target_y)


@dataclass
class MLPDetail:
    """Detail panel for the MLP / Feed Forward sub-block."""

    config: MLPDisplayConfig
    w: float = 180

    def _envelope(self) -> DetailEnvelope:
        return DetailEnvelope(
            "Feed forward", "detail-mlp",
            output_labels=[ExternalLabel("hidden states")],
            input_labels=[ExternalLabel("hidden states")],
        )

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
        fork_space = g if self.config.gated else 0
        content_h = total_h + fork_space
        return self._envelope().measure_envelope(content_w, content_h, th)

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

    def render(self, bb: BBox, th: Theme) -> Iterator[S.Element]:
        items = self._build(th)
        total_h = items[-1][0].bottom if items else 0
        g = th.geo.gap
        content_w = max((cbb.right for cbb, _ in items), default=0)
        fork_space = g if self.config.gated else 0
        content_h = total_h + fork_space

        envelope = self._envelope()
        env = envelope.render_envelope(bb, content_w, content_h, th)
        title_h = th.geo.title_h
        if items:
            top_bb = items[0][0]
            env.spine_cx = env.content_bb.x + top_bb.cx

        # Phase 1: arrow behind title bar
        yield from env.phase1_behind_title()
        # Phase 2: frame
        yield from env.phase2_frame()
        # Phase 3: exit arrow + output labels
        yield from env.phase3_exit_arrow_and_output_labels()

        # Exit arrow from top item to title bar (inside content)
        if items:
            top_bb = items[0][0]
            cx = env.content_bb.x + top_bb.cx
            y_from = env.content_bb.y + top_bb.y
            y_title_bottom = env.frame_bb.y + title_h
            if y_from - y_title_bottom > 1:
                yield S.Line(x1=cx, y1=y_from, x2=cx, y2=y_title_bottom, class_=["arrow"])

        # Internal arrows (fork midpoint, merge paths, inter-item arrows)
        yield from self._render_arrows(items, env.content_bb, th)

        # Children (boxes, symbols)
        for child_bb, child in items:
            shifted = BBox(env.content_bb.x + child_bb.x, env.content_bb.y + child_bb.y, child_bb.w, child_bb.h)
            yield from child.render(shifted, th)

        # Entry arrow + input labels
        if items:
            if self.config.gated and len(items) >= 4:
                up_bb = items[-1][0]
                target_y = env.content_bb.y + up_bb.bottom + g + _ARROW_CLR
            else:
                bottom_bb = items[-1][0]
                target_y = env.content_bb.y + bottom_bb.bottom + _ARROW_CLR
        else:
            target_y = None
        yield from env.phase4_entry_arrow_and_input_labels(target_y)


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

    def _envelope(self) -> DetailEnvelope:
        return DetailEnvelope(
            "Stochastic mixer dispatch", "detail-stochastic",
            output_labels=[ExternalLabel("hidden states")],
            input_labels=[ExternalLabel("hidden states")],
        )

    def _content_size(self, th: Theme) -> tuple[float, float]:
        """Return (content_w, content_h) for the sub-mixer boxes."""
        w = self.w or th.geo.stack_w
        g = th.geo.gap
        bh = self._box_h(th)
        gap = self._inner_gap()
        n = len(self.spec.sub_mixers)
        content_h = n * bh + max(0, n - 1) * gap
        content_w = w - 2 * g
        return content_w, content_h

    def measure(self, th: Theme) -> Size:
        content_w, content_h = self._content_size(th)
        return self._envelope().measure_envelope(content_w, content_h, th)

    def render(self, bb: BBox, th: Theme) -> Iterator[S.Element]:
        bh = self._box_h(th)
        gap = self._inner_gap()
        content_w, content_h = self._content_size(th)

        envelope = self._envelope()
        env = envelope.render_envelope(bb, content_w, content_h, th)

        yield from env.phase1_behind_title()
        yield from env.phase2_frame()
        yield from env.phase3_exit_arrow_and_output_labels()

        # Sub-mixer boxes
        inner_w = env.content_bb.w
        box_y = env.content_bb.y
        for name, sub_mixer in self.spec.sub_mixers:
            css = _mixer_spec_css(sub_mixer)
            label = name
            if name == self.spec.main_mixer_name:
                label += " \u2605"
            box = Box(label, css, w=inner_w, h=bh)
            box_bb = BBox(env.content_bb.x, box_y, inner_w, bh)
            yield from box.render(box_bb, th)
            box_y += bh + gap

        # Entry arrow + input labels
        yield from env.phase4_entry_arrow_and_input_labels()

    def sub_mixer_bboxes(self, bb: BBox, th: Theme) -> list[tuple[BBox, MixerSpec]]:
        """Return (bbox, MixerSpec) for each sub-mixer box."""
        g = th.geo.gap
        title_h = th.geo.title_h
        bh = self._box_h(th)
        gap = self._inner_gap()
        inner_w = bb.w - 2 * g
        output_area_h = self._envelope()._output_area_h(th)

        result: list[tuple[BBox, MixerSpec]] = []
        box_y = bb.y + output_area_h + title_h + g
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
