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
    BlockGroup,
    DecoderBlock,
    MLPDetail,
    connector_bezier,
    defs,
    detail_for_mixer,
)
from fast_llm_external_models.apriel2.conversion.diagram.layout import BBox
from fast_llm_external_models.apriel2.conversion.diagram.model import (
    ArchitectureModel,
    BlockSpec,
    MixerSpec,
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


def _get_mixer_details(spec: BlockSpec, th: Theme) -> list:
    """Return detail panel(s) for a block spec's mixer.

    For stochastic mixers, returns a detail for each sub-mixer.
    """
    mixer = spec.mixer
    if isinstance(mixer, StochasticMixerSpec):
        details = []
        for _name, sub_mixer in mixer.sub_mixers:
            d = detail_for_mixer(sub_mixer)
            if d is not None:
                details.append(d)
        return details
    d = detail_for_mixer(mixer)
    return [d] if d is not None else []


def _layout(
    arch: ArchitectureModel,
    th: Theme,
) -> tuple[list[S.Element], float, float]:
    """Build element list and compute content size.

    Returns:
        (elements, content_width, content_height)
    """
    left_margin = 50  # room for brace + count labels
    row_gap = th.geo.gap_detail * 2
    conn_gap = th.geo.connector_gap

    body: list[S.Element] = []
    content_y = 0.0
    max_x = 0.0
    left_x = left_margin

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

    # ── Block type rows ────────────────────────────────────────────
    for label, spec in arch.unique_block_specs:
        # Total count for this spec across all block groups
        count = sum(g.count for g in arch.block_groups if g.block_spec == spec)

        # Section label
        body.append(S.Text(
            x=left_x, y=content_y + th.typo.sz_ann,
            text=f"Block type: {label}", class_=["t-note"],
        ))
        content_y += th.typo.sz_ann + 8

        # Decoder block (with brace + count)
        block = DecoderBlock(spec.mixer, norm_type=spec.norm_type)
        group = BlockGroup(block, count, label if count > 1 else "")
        group_sz = group.measure(th)
        group_bb = BBox(left_x, content_y, group_sz.w, group_sz.h)
        body.extend(group.render(group_bb, th))

        # Detail panels to the right
        detail_x = group_bb.right + conn_gap
        detail_y = content_y

        for detail in _get_mixer_details(spec, th):
            detail_sz = detail.measure(th)
            detail_bb = BBox(detail_x, detail_y, detail_sz.w, detail_sz.h)
            body.extend(detail.render(detail_bb, th))
            # Connector from block to detail
            body.append(connector_bezier(
                group_bb.right, group_bb.cy,
                detail_bb.x, detail_bb.cy,
            ))
            detail_y += detail_sz.h + th.geo.gap_detail
            max_x = max(max_x, detail_bb.right)

        # MLP detail
        if spec.mlp.intermediate_size or spec.mlp.activation:
            mlp_detail = MLPDetail(spec.mlp)
            mlp_sz = mlp_detail.measure(th)
            mlp_bb = BBox(detail_x, detail_y, mlp_sz.w, mlp_sz.h)
            body.extend(mlp_detail.render(mlp_bb, th))
            detail_y += mlp_sz.h
            max_x = max(max_x, mlp_bb.right)

        row_h = max(group_sz.h, detail_y - content_y)
        content_y += row_h + row_gap
        max_x = max(max_x, group_bb.right)

    content_w = max(max_x, left_x + th.geo.block_w)
    content_h = content_y

    return body, content_w, content_h
