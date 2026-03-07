"""Architecture diagram generator for Apriel2 models.

Public API::

    from fast_llm_external_models.apriel2.conversion.diagram import generate_diagram

    svg_string = generate_diagram(config)
    generate_diagram(config, output_path="/tmp/detailed.svg")
"""

from __future__ import annotations

import logging
import re

import svg as S

from fast_llm_external_models.apriel2.conversion.diagram.elements import (
    ArchitectureOverview,
    BlockGroup,
    DecoderBlock,
    MambaDetail,
    MLPDetail,
    StochasticMixerPanel,
    VisionEncoderColumn,
    connector_bezier,
    defs,
    detail_for_mixer,
)
from fast_llm_external_models.apriel2.conversion.diagram.layout import BBox
from fast_llm_external_models.apriel2.conversion.diagram.model import (
    ArchitectureModel,
    BlockSpec,
    MLPDisplayConfig,
    StochasticMixerSpec,
    extract_model,
)
from fast_llm_external_models.apriel2.conversion.diagram.style import Theme

logger = logging.getLogger(__name__)


def _fix_xml_text(svg: str) -> str:
    """Escape bare ``&`` inside SVG text nodes.

    The ``svg`` library does not XML-escape text content, so ``&`` in
    labels like "key & value heads" produces invalid XML.  This function
    replaces any ``&`` that is NOT already part of an entity reference
    (``&amp;``, ``&#123;``, etc.) with ``&amp;``.
    """
    return re.sub(r"&(?!amp;|lt;|gt;|quot;|apos;|#)", "&amp;", svg)


def generate_diagram(
    config: dict,
    output_path: str | None = None,
    theme: Theme | None = None,
) -> str:
    """Generate an architecture diagram SVG.

    Args:
        config: Complete Apriel2 config dict.
        output_path: Optional path to write the SVG file.
        theme: Optional Theme. Defaults to Theme().

    Returns:
        SVG string.
    """
    th = theme or Theme()
    arch = extract_model(config)

    body, content_w, content_h = _layout(arch, th)

    pad = th.geo.pad_outer
    total_w = content_w + pad * 2
    total_h = content_h + pad * 2

    svg_elements: list[S.Element] = [
        S.Style(text=th.stylesheet()),
        defs(th),
        S.Rect(x=0, y=0, width=total_w, height=total_h, class_=["background"]),
    ]
    svg_elements.extend(body)

    canvas = S.SVG(
        viewBox=S.ViewBoxSpec(0, 0, total_w, total_h),
        width=total_w,
        height=total_h,
        elements=svg_elements,
    )

    svg_str = _fix_xml_text(str(canvas))

    if output_path is not None:
        with open(output_path, "w") as f:
            f.write(svg_str)
        logger.info(f"Diagram saved to {output_path}")

    return svg_str


def _get_mixer_detail(spec: BlockSpec) -> object | None:
    """Return a single detail panel for a non-stochastic block spec's mixer."""
    mixer = spec.mixer
    if isinstance(mixer, StochasticMixerSpec):
        return None
    return detail_for_mixer(mixer)


def _layout(
    arch: ArchitectureModel,
    th: Theme,
) -> tuple[list[S.Element], float, float]:
    """Build element list and compute content size.

    Returns:
        (elements, content_width, content_height)
    """
    row_gap = th.geo.gap_detail * 2
    conn_gap = th.geo.connector_gap

    # ── Compute horizontal zones ───────────────────────────────────
    range_label_w = 50  # space for "0..47" labels to the left of the overview
    vision_col_w = th.geo.stack_w + 30 if arch.vision_encoder else 0  # vision column + gap
    overview_x = vision_col_w + range_label_w
    brace_margin = 50  # room for brace + count badge
    left_x = overview_x + th.geo.stack_w + th.geo.stack_conn_gap + brace_margin

    # Determine if any block spec uses a stochastic mixer
    has_stochastic = any(
        isinstance(spec.mixer, StochasticMixerSpec)
        for _, spec in arch.unique_block_specs
    )
    stochastic_col_w = (th.geo.stack_w + conn_gap) if has_stochastic else 0

    body: list[S.Element] = []
    content_y = 0.0
    max_x = 0.0

    # ── Title ──────────────────────────────────────────────────────
    body.append(S.Text(
        x=left_x, y=content_y + th.typo.sz_title,
        text=arch.model_name, class_=["t-title"],
    ))
    content_y += th.typo.sz_title + 6

    subtitle_parts = [f"h={arch.hidden_size}"]
    if arch.vocab_size:
        subtitle_parts.append(f"V={arch.vocab_size:,}")
    subtitle_parts.append(f"{arch.total_blocks} blocks")
    body.append(S.Text(
        x=left_x, y=content_y + th.typo.sz_subtitle,
        text="  ".join(subtitle_parts), class_=["t-sub"],
    ))
    content_y += th.typo.sz_subtitle + row_gap

    # ── Architecture overview column ───────────────────────────────
    overview = ArchitectureOverview(arch)
    overview_sz = overview.measure(th)
    overview_bb = BBox(overview_x, content_y, overview_sz.w, overview_sz.h)
    body.extend(overview.render(overview_bb, th))
    cell_bboxes = overview.cell_bboxes(overview_bb, th)

    # ── Vision encoder column (if multimodal) ──────────────────────
    if arch.vision_encoder:
        vision_col = VisionEncoderColumn(arch.vision_encoder)
        vision_sz = vision_col.measure(th)
        # Embedding is the last decoder cell, 3rd from end (before Text tokens + Sample input)
        embed_bb = cell_bboxes[-3][0]
        # Vision column top (adapter) aligns with embedding center
        vision_top_y = embed_bb.cy - vision_sz.h / 2
        vision_bb = BBox(range_label_w, vision_top_y, vision_sz.w, vision_sz.h)
        body.extend(vision_col.render(vision_bb, th))

        # Horizontal merge arrow: adapter (top cell of vision) → embedding
        adapter_cy = vision_bb.y + 2 * (th.geo.stack_cell_h + th.geo.stack_cell_gap) + th.geo.stack_cell_h / 2
        body.append(connector_bezier(
            vision_bb.x + vision_sz.w, adapter_cy,
            embed_bb.x, embed_bb.cy,
        ))
        body.append(S.Text(
            x=(vision_bb.x + vision_sz.w + embed_bb.x) / 2,
            y=adapter_cy - 8,
            text="replace at [IMG]",
            class_=["t-note"], text_anchor="middle",
        ))

    # ── Block type rows ────────────────────────────────────────────
    spec_positions: dict[BlockSpec, BBox] = {}
    # Track Feed Forward box positions for MLP deduplication
    mlp_usages: dict[MLPDisplayConfig, list[BBox]] = {}

    for label, spec in arch.unique_block_specs:
        # Total count for this spec across all block groups
        count = sum(g.count for g in arch.block_groups if g.block_spec == spec)
        if count == 0 and arch.vision_encoder and arch.vision_encoder.block_spec == spec:
            count = arch.vision_encoder.num_blocks

        # Decoder block (with brace + count)
        block = DecoderBlock(spec.mixer, norm_type=spec.norm_type, title=label)
        group = BlockGroup(block, count, label if count > 1 else "")
        group_sz = group.measure(th)
        group_bb = BBox(left_x, content_y, group_sz.w, group_sz.h)
        body.extend(group.render(group_bb, th))

        # Track position for overview connectors
        spec_positions[spec] = group_bb

        # Track Feed Forward box position for MLP detail connectors
        ff_bb = block.mlp_bbox(group_bb, th)
        mlp_usages.setdefault(spec.mlp, []).append(ff_bb)

        is_stochastic = isinstance(spec.mixer, StochasticMixerSpec)
        # Detail x accounts for stochastic column width
        detail_x = group_bb.right + conn_gap + stochastic_col_w
        detail_y = content_y

        if is_stochastic:
            assert isinstance(spec.mixer, StochasticMixerSpec)
            # Render stochastic dispatch panel
            stoch_panel = StochasticMixerPanel(spec.mixer)
            stoch_sz = stoch_panel.measure(th)
            stoch_x = group_bb.right + conn_gap
            stoch_bb = BBox(stoch_x, content_y, stoch_sz.w, stoch_sz.h)
            body.extend(stoch_panel.render(stoch_bb, th))

            # Connector: mixer box → stochastic panel
            mixer_bb = block.mixer_bbox(group_bb, th)
            body.append(connector_bezier(
                mixer_bb.right, mixer_bb.cy,
                stoch_bb.x, stoch_bb.cy,
            ))

            # For each sub-mixer, render its detail and connect
            sub_bboxes = stoch_panel.sub_mixer_bboxes(stoch_bb, th)
            for sub_bb, sub_mixer in sub_bboxes:
                detail = detail_for_mixer(sub_mixer)
                if detail is not None:
                    detail_sz = detail.measure(th)
                    detail_bb = BBox(detail_x, detail_y, detail_sz.w, detail_sz.h)
                    body.extend(detail.render(detail_bb, th))
                    # Connector: sub-mixer box → detail panel
                    body.append(connector_bezier(
                        sub_bb.right, sub_bb.cy,
                        detail_bb.x, detail_bb.cy,
                    ))
                    detail_y += detail_sz.h + th.geo.gap_detail
                    max_x = max(max_x, detail_bb.right)

            max_x = max(max_x, stoch_bb.right)
        else:
            # Non-stochastic: render single mixer detail
            detail = _get_mixer_detail(spec)
            if detail is not None:
                detail_sz = detail.measure(th)
                detail_bb = BBox(detail_x, detail_y, detail_sz.w, detail_sz.h)
                body.extend(detail.render(detail_bb, th))
                # Connector: mixer box → detail
                mixer_bb = block.mixer_bbox(group_bb, th)
                body.append(connector_bezier(
                    mixer_bb.right, mixer_bb.cy,
                    detail_bb.x, detail_bb.cy,
                ))
                detail_y += detail_sz.h + th.geo.gap_detail
                max_x = max(max_x, detail_bb.right)

        row_h = max(group_sz.h, detail_y - content_y)
        content_y += row_h + row_gap
        max_x = max(max_x, group_bb.right)

    # ── Shared MLP detail panels (deduplicated) ───────────────────
    for mlp_config, ff_bboxes in mlp_usages.items():
        if not (mlp_config.intermediate_size or mlp_config.activation):
            continue
        mlp_detail = MLPDetail(mlp_config)
        mlp_sz = mlp_detail.measure(th)
        # Place MLP detail to the right of block definitions
        mlp_detail_x = left_x + th.geo.block_w + conn_gap + stochastic_col_w
        mlp_bb = BBox(mlp_detail_x, content_y, mlp_sz.w, mlp_sz.h)
        body.extend(mlp_detail.render(mlp_bb, th))
        # Connector from each block's "Feed forward" box to the shared MLP detail
        for ff_bb in ff_bboxes:
            body.append(connector_bezier(
                ff_bb.right, ff_bb.cy,
                mlp_bb.x, mlp_bb.cy,
            ))
        content_y += mlp_sz.h + row_gap
        max_x = max(max_x, mlp_bb.right)

    # ── Overview → definition connectors ───────────────────────────
    for cell_bb, cell_spec in cell_bboxes:
        if cell_spec is not None and cell_spec in spec_positions:
            def_bb = spec_positions[cell_spec]
            body.append(connector_bezier(
                cell_bb.right, cell_bb.cy,
                def_bb.x - 8, def_bb.cy,
            ))

    # ── Vision encoder → definition connectors ─────────────────────
    if arch.vision_encoder:
        vision_col = VisionEncoderColumn(arch.vision_encoder)
        vision_cells = vision_col.cell_bboxes(vision_bb, th)  # type: ignore[possibly-undefined]
        for cell_bb, cell_spec in vision_cells:
            if cell_spec is not None and cell_spec in spec_positions:
                def_bb = spec_positions[cell_spec]
                body.append(connector_bezier(
                    cell_bb.right, cell_bb.cy,
                    def_bb.x - 8, def_bb.cy,
                ))

    content_w = max(max_x, left_x + th.geo.block_w)
    content_h = max(content_y, overview_bb.bottom)

    return body, content_w, content_h
