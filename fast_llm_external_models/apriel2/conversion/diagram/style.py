"""Theme → CSS stylesheet for detailed architecture diagrams.

All color, typography, and decorative properties are declared here and
emitted as an SVG ``<style>`` block.  The layout engine and element
renderers reference *CSS class names only* — they never set fill, stroke,
font-*, etc. inline.

Geometry constants (widths, gaps, padding) are also on the Theme so the
layout engine can read them, but they are consumed as numbers, not CSS.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Palette:
    # mixer types — derived from matplotlib Tab10
    attention: str = "#8fbbda"
    attn_text: str = "#13476c"
    swa: str = "#ffbf87"
    swa_text: str = "#994c08"
    kda: str = "#96d096"
    kda_text: str = "#1a601a"
    gdn: str = "#eb9394"
    gdn_text: str = "#801718"
    mamba: str = "#cab3de"
    mamba_text: str = "#593e71"
    conv: str = "#f0d0e0"
    conv_text: str = "#7a2a4a"
    gate: str = "#e8d8f0"
    gate_text: str = "#5a3a7a"
    stochastic: str = "#f0e8d8"
    stoch_text: str = "#6a5a3a"

    # infrastructure
    embedding: str = "#3d6b6b"
    emb_text: str = "#ffffff"
    norm: str = "#d4e8d4"
    norm_text: str = "#2d5a2d"
    linear: str = "#e8e8e8"
    linear_text: str = "#444444"
    activation: str = "#fff3c0"
    act_text: str = "#7a6a00"
    mlp: str = "#e0e0e0"
    mlp_text: str = "#444444"

    # chrome
    block_bg: str = "#f2f2f2"
    block_stroke: str = "#cccccc"
    detail_bg: str = "#fafafa"
    detail_stroke: str = "#bbbbbb"

    # wires
    wire: str = "#555555"
    dash: str = "#999999"

    # annotation semantic colors
    dim: str = "#2a8a5a"
    count: str = "#c04070"
    note: str = "#666666"
    body: str = "#333333"

    def mixer(self, kind: str) -> tuple[str, str]:
        """(fill, text) for a mixer type."""
        return {
            "attention": (self.attention, self.attn_text),
            "sliding_window": (self.swa, self.swa_text),
            "gdn": (self.gdn, self.gdn_text),
            "kda": (self.kda, self.kda_text),
            "mamba": (self.mamba, self.mamba_text),
            "stochastic": (self.stochastic, self.stoch_text),
        }.get(kind, (self.linear, self.linear_text))


@dataclass(frozen=True)
class Typography:
    family: str = "Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif"
    mono: str = "'SF Mono', 'Fira Code', Consolas, monospace"
    sz_title: float = 22
    sz_subtitle: float = 16
    sz_label: float = 13
    sz_ann: float = 11
    sz_small: float = 10


@dataclass(frozen=True)
class Geometry:
    block_w: float = 220
    inner_w: float = 185
    detail_w: float = 240
    box_h: float = 32
    box_h_sm: float = 28
    rx: float = 6
    rx_block: float = 8

    gap: float = 10
    gap_block: float = 6
    gap_detail: float = 24
    pad_outer: float = 40

    connector_gap: float = 80
    residual_r: float = 9
    symbol_r: float = 9

    # Layer stack / architecture overview column
    stack_w: float = 185  # matches inner_w for consistency with block boxes
    stack_cell_h: float = 32  # matches box_h
    stack_cell_gap: float = 10  # matches gap — room for vertical flow lines
    stack_conn_gap: float = 60  # gap between overview right edge and definition area

    title_h: float = 24  # height of title strip in titled panels

    stroke: float = 1.2
    stroke_arrow: float = 1.5
    dash: list[int] = field(default_factory=lambda: [6, 4])


@dataclass(frozen=True)
class Theme:
    pal: Palette = field(default_factory=Palette)
    typo: Typography = field(default_factory=Typography)
    geo: Geometry = field(default_factory=Geometry)

    def stylesheet(self) -> str:
        p, t, g = self.pal, self.typo, self.geo
        dash = " ".join(str(d) for d in g.dash)
        return _CSS.format(
            font=t.family, mono=t.mono,
            rx=g.rx, rx_blk=g.rx_block, sw=g.stroke,
            body=p.body, note=p.note, dim=p.dim, count=p.count,
            wire=p.wire, dash_col=p.dash, dash=dash,
            emb=p.embedding, emb_t=p.emb_text,
            norm=p.norm, norm_t=p.norm_text,
            lin=p.linear, lin_t=p.linear_text,
            act=p.activation, act_t=p.act_text,
            mlp=p.mlp, mlp_t=p.mlp_text,
            attn=p.attention, attn_t=p.attn_text,
            swa=p.swa, swa_t=p.swa_text,
            kda=p.kda, kda_t=p.kda_text,
            gdn=p.gdn, gdn_t=p.gdn_text,
            mamba=p.mamba, mamba_t=p.mamba_text,
            conv=p.conv, conv_t=p.conv_text,
            gate=p.gate, gate_t=p.gate_text,
            stoch=p.stochastic, stoch_t=p.stoch_text,
            blk_bg=p.block_bg, blk_s=p.block_stroke,
            det_bg=p.detail_bg, det_s=p.detail_stroke,
            sz_title=t.sz_title, sz_sub=t.sz_subtitle,
            sz_label=t.sz_label, sz_ann=t.sz_ann, sz_sm=t.sz_small,
        )


_CSS = """\
/* ── base ───────────────────────────────── */
text {{ font-family: {font}; fill: {body}; }}
text.mono {{ font-family: {mono}; }}

/* ── text roles ─────────────────────────── */
.t-title  {{ font-size: {sz_title}px; font-weight: 700; fill: {count}; }}
.t-sub    {{ font-size: {sz_sub}px; font-weight: 600; }}
.t-label  {{ font-size: {sz_label}px; text-anchor: middle; dominant-baseline: central; }}
.t-label-bold {{ font-size: {sz_label}px; font-weight: 600;
                 text-anchor: middle; dominant-baseline: central; }}
.t-ann    {{ font-size: {sz_ann}px; }}
.t-dim    {{ font-size: {sz_ann}px; fill: {dim}; }}
.t-count  {{ font-size: {sz_ann}px; fill: {count}; }}
.t-note   {{ font-size: {sz_ann}px; fill: {note}; }}
.t-small  {{ font-size: {sz_sm}px; fill: {note}; }}

/* ── box base (rx via attribute, not CSS) ─ */
.box      {{ stroke-width: {sw}; }}

/* ── component boxes ────────────────────── */
.box-embedding {{ fill: {emb}; stroke: none; }}
.box-embedding > text {{ fill: {emb_t}; }}
.box-norm      {{ fill: {norm}; stroke: none; }}
.box-norm > text {{ fill: {norm_t}; }}
.box-linear    {{ fill: {lin}; stroke: none; }}
.box-linear > text {{ fill: {lin_t}; }}
.box-activation {{ fill: {act}; stroke: none; }}
.box-activation > text {{ fill: {act_t}; }}
.box-mlp       {{ fill: {mlp}; stroke: none; }}
.box-mlp > text {{ fill: {mlp_t}; }}

/* ── mixer boxes ────────────────────────── */
.box-attention {{ fill: {attn}; stroke: none; }}
.box-attention > text {{ fill: {attn_t}; }}
.box-swa       {{ fill: {swa}; stroke: none; }}
.box-swa > text {{ fill: {swa_t}; }}
.box-gdn       {{ fill: {gdn}; stroke: none; }}
.box-gdn > text {{ fill: {gdn_t}; }}
.box-kda       {{ fill: {kda}; stroke: none; }}
.box-kda > text {{ fill: {kda_t}; }}
.box-mamba     {{ fill: {mamba}; stroke: none; }}
.box-mamba > text {{ fill: {mamba_t}; }}
.box-conv      {{ fill: {conv}; stroke: none; }}
.box-conv > text {{ fill: {conv_t}; }}
.box-gate      {{ fill: {gate}; stroke: none; }}
.box-gate > text {{ fill: {gate_t}; }}
.box-stochastic {{
  fill: {stoch}; stroke: {stoch_t}; stroke-dasharray: 6 3;
}}
.box-stochastic > text {{ fill: {stoch_t}; stroke: none; }}

/* ── structural boxes ───────────────────── */
.block-bg {{
  fill: {blk_bg}; stroke: {blk_s};
  rx: {rx_blk}; stroke-width: {sw};
}}
.detail-bg {{
  fill: {det_bg}; stroke: {det_s};
  rx: {rx_blk}; stroke-width: {sw}; stroke-dasharray: {dash};
}}

/* ── detail panel frames ───────────────── */
.detail-attention {{ fill: {attn}; stroke: {attn}; stroke-width: {sw}; }}
.detail-attention > text {{ fill: {attn_t}; stroke: none; }}

.detail-swa {{ fill: {swa}; stroke: {swa}; stroke-width: {sw}; }}
.detail-swa > text {{ fill: {swa_t}; stroke: none; }}
.detail-gdn {{ fill: {gdn}; stroke: {gdn}; stroke-width: {sw}; }}
.detail-gdn > text {{ fill: {gdn_t}; stroke: none; }}
.detail-kda {{ fill: {kda}; stroke: {kda}; stroke-width: {sw}; }}
.detail-kda > text {{ fill: {kda_t}; stroke: none; }}
.detail-mamba {{ fill: {mamba}; stroke: {mamba}; stroke-width: {sw}; }}
.detail-mamba > text {{ fill: {mamba_t}; stroke: none; }}

.detail-stochastic {{ fill: {stoch}; stroke: {stoch}; stroke-width: {sw}; stroke-dasharray: 6 3; }}
.detail-stochastic > text {{ fill: {stoch_t}; stroke: none; }}

.detail-mlp {{ fill: {mlp}; stroke: {mlp}; stroke-width: {sw}; }}
.detail-mlp > text {{ fill: {mlp_t}; stroke: none; }}

.detail-decoder {{ fill: {blk_bg}; stroke: {blk_s}; stroke-width: {sw}; }}
.detail-decoder > text {{ fill: {note}; stroke: none; }}

.detail-content {{ fill: white; }}

/* ── wires ──────────────────────────────── */
.arrow     {{ stroke: {wire}; stroke-width: 1.5; fill: none; }}
.connector {{ stroke: {dash_col}; stroke-width: 1;
              stroke-dasharray: {dash}; fill: none; }}
.brace     {{ stroke: {note}; stroke-width: 1.5; fill: none; }}

/* ── layer stack ───────────────────────── */
.stack-label {{ font-size: {sz_sm}px; fill: {note}; dominant-baseline: central; }}
.symbol    {{ stroke: {wire}; stroke-width: 1.2; fill: white; }}
.symbol line {{ stroke: {wire}; stroke-width: 1.5; }}
.box-transparent {{ fill: none; stroke: #cccccc; stroke-width: 0.5; }}
.box-transparent > text {{ font-size: {sz_ann}px; fill: {note}; }}

/* ── background ────────────────────────── */
.background {{ fill: white; }}
"""
