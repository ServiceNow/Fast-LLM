"""Extract a typed diagram model from an Apriel2 config dict.

This module converts raw config dicts (the "State" type from config.py)
into typed dataclasses suitable for diagram generation. No SVG concerns here.
"""

from __future__ import annotations

from dataclasses import dataclass


# ─── Typed display configs ───────────────────────────────────────────


@dataclass(frozen=True)
class AttentionDisplayConfig:
    heads: int = 0
    kv_heads: int = 0
    head_dim: int = 0
    window_size: int | None = None


@dataclass(frozen=True)
class GDNDisplayConfig:
    value_heads: int = 0
    key_heads: int = 0
    key_head_dim: int = 0
    value_head_dim: int = 0
    conv_kernel: int = 4


@dataclass(frozen=True)
class KDADisplayConfig:
    heads: int = 0
    head_dim: int = 0
    conv_kernel: int | None = None


@dataclass(frozen=True)
class MambaDisplayConfig:
    d_state: int | None = None
    d_conv: int | None = None
    d_inner: int | None = None


MixerDisplayConfig = AttentionDisplayConfig | GDNDisplayConfig | KDADisplayConfig | MambaDisplayConfig


@dataclass(frozen=True)
class MLPDisplayConfig:
    gated: bool = False
    activation: str = ""
    intermediate_size: int = 0


# ─── Specs ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MixerSpec:
    mixer_type: str  # "attention", "gdn", "kda", "mamba", "sliding_window"
    label: str  # e.g. "Attention (32h, GQA 32/8, d=128)"
    display: MixerDisplayConfig = AttentionDisplayConfig()


@dataclass(frozen=True)
class StochasticMixerSpec:
    main_mixer_name: str
    sub_mixers: tuple[tuple[str, MixerSpec], ...]  # (name, spec) pairs, hashable


@dataclass(frozen=True)
class BlockSpec:
    mixer: MixerSpec | StochasticMixerSpec
    mlp: MLPDisplayConfig
    norm_type: str  # e.g. "RMSNorm"


@dataclass(frozen=True)
class BlockGroup:
    block_spec: BlockSpec
    block_name: str | None  # pattern block name, or None for fixed
    start_index: int
    count: int

    @property
    def end_index(self) -> int:
        return self.start_index + self.count - 1

    @property
    def range_label(self) -> str:
        if self.count == 1:
            return f"Block {self.start_index}"
        return f"Blocks {self.start_index}..{self.end_index}"


@dataclass(frozen=True)
class VisionEncoderSpec:
    hidden_size: int
    num_blocks: int
    patch_size: tuple[int, int]
    mixer: MixerSpec
    mlp_label: str
    adapter_label: str


@dataclass(frozen=True)
class ArchitectureModel:
    model_name: str
    hidden_size: int
    vocab_size: int
    block_groups: list[BlockGroup]
    unique_block_specs: list[tuple[str, BlockSpec]]  # (label, spec)
    vision_encoder: VisionEncoderSpec | None
    total_blocks: int
    tie_word_embeddings: bool


# ─── Extraction ──────────────────────────────────────────────────────


def extract_model(config: dict) -> ArchitectureModel:
    """Extract an ArchitectureModel from a complete Apriel2 config dict."""
    hidden_size = config.get("hidden_size", 0)
    vocab_size = config.get("vocab_size", 0)
    tie_word_embeddings = config.get("tie_word_embeddings", False)

    decoder = config.get("decoder", {})
    decoder_type = decoder.get("type", "fixed")
    num_blocks = decoder.get("num_blocks", 0)

    if decoder_type == "fixed":
        block_config = decoder.get("block", {})
        block_spec = _extract_block_spec(block_config, hidden_size)
        # For fixed decoder, if num_blocks not set, try to infer
        if num_blocks == 0:
            num_blocks = 1
        groups = [BlockGroup(block_spec=block_spec, block_name=None, start_index=0, count=num_blocks)]
    else:  # pattern
        pattern = decoder.get("pattern", [])
        blocks_config = decoder.get("blocks", {})
        if num_blocks == 0:
            num_blocks = len(pattern)

        # Expand pattern to full length
        if pattern:
            expanded = _expand_pattern(pattern, num_blocks)
        else:
            expanded = []

        # Resolve each block name to its spec
        resolved: list[tuple[str | None, BlockSpec]] = []
        for name in expanded:
            block_config = blocks_config.get(name, {})
            spec = _extract_block_spec(block_config, hidden_size)
            resolved.append((name, spec))

        groups = _run_length_encode(resolved)

    unique_specs = _identify_unique_specs(groups)

    # Vision encoder
    vision_encoder = None
    if "vision_encoder" in config:
        vision_encoder = _extract_vision_encoder(config["vision_encoder"])

    # Model name
    model_name = config.get("model_type", "Apriel2")
    architectures = config.get("architectures", [])
    if architectures:
        model_name = architectures[0].replace("ForCausalLM", "").replace("ForConditionalGeneration", "")

    return ArchitectureModel(
        model_name=model_name,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        block_groups=groups,
        unique_block_specs=unique_specs,
        vision_encoder=vision_encoder,
        total_blocks=num_blocks,
        tie_word_embeddings=tie_word_embeddings,
    )


def _expand_pattern(pattern: list[str], num_blocks: int) -> list[str]:
    """Expand a pattern cyclically to num_blocks length."""
    if not pattern:
        return []
    if len(pattern) >= num_blocks:
        return pattern[:num_blocks]
    # Repeat cyclically
    full_repeats = num_blocks // len(pattern)
    remainder = num_blocks % len(pattern)
    return pattern * full_repeats + pattern[:remainder]


def _extract_block_spec(block_config: dict, hidden_size: int) -> BlockSpec:
    """Extract a BlockSpec from a single block config dict."""
    mixer_config = block_config.get("mixer", {})
    mixer = _extract_mixer_spec(mixer_config, hidden_size)

    mlp_config = block_config.get("mlp", {})
    mlp = _extract_mlp_display_config(mlp_config)

    norm_config = block_config.get("normalization", {})
    norm_type = _make_norm_label(norm_config)

    return BlockSpec(mixer=mixer, mlp=mlp, norm_type=norm_type)


def _extract_mlp_display_config(mlp_config: dict) -> MLPDisplayConfig:
    """Extract MLPDisplayConfig from a raw MLP config dict."""
    return MLPDisplayConfig(
        gated=mlp_config.get("gated", False),
        activation=mlp_config.get("activation", ""),
        intermediate_size=mlp_config.get("intermediate_size", 0),
    )


def _extract_mixer_spec(mixer_config: dict, hidden_size: int) -> MixerSpec | StochasticMixerSpec:
    """Extract a MixerSpec or StochasticMixerSpec from a mixer config dict."""
    mixer_type = mixer_config.get("type", "attention")

    if mixer_type == "stochastic":
        main_mixer_name = mixer_config.get("main_mixer_name", "attention")
        sub_mixers_list: list[tuple[str, MixerSpec]] = []
        for name, sub_config in mixer_config.get("mixers", {}).items():
            if isinstance(sub_config, dict):
                spec = _extract_single_mixer_spec(sub_config, hidden_size)
                sub_mixers_list.append((name, spec))
        return StochasticMixerSpec(main_mixer_name=main_mixer_name, sub_mixers=tuple(sub_mixers_list))

    return _extract_single_mixer_spec(mixer_config, hidden_size)


def _extract_single_mixer_spec(mixer_config: dict, hidden_size: int) -> MixerSpec:
    """Extract a MixerSpec for a non-stochastic mixer."""
    mixer_type = mixer_config.get("type", "attention")

    # Determine effective type (sliding_window is attention with window_size)
    effective_type = mixer_type
    if mixer_type == "attention" and "window_size" in mixer_config:
        effective_type = "sliding_window"

    if effective_type in ("attention", "sliding_window"):
        heads = mixer_config.get("heads", 0)
        head_groups = mixer_config.get("head_groups", heads)
        head_size = mixer_config.get("head_size")
        if head_size is None and heads and hidden_size:
            head_size = hidden_size // heads
        window_size = mixer_config.get("window_size") if effective_type == "sliding_window" else None

        display = AttentionDisplayConfig(
            heads=heads or 0,
            kv_heads=head_groups or 0,
            head_dim=head_size or 0,
            window_size=window_size,
        )
        label = _make_attention_label(effective_type, heads or None, head_groups, head_size, mixer_config.get("window_size"))

    elif effective_type == "mamba":
        d_state = mixer_config.get("d_state")
        d_conv = mixer_config.get("d_conv")
        d_inner = mixer_config.get("d_inner")
        display = MambaDisplayConfig(d_state=d_state, d_conv=d_conv, d_inner=d_inner)
        label = _make_mamba_label(d_state, d_conv, d_inner)

    elif effective_type == "gdn":
        value_heads = mixer_config.get("value_heads", 0)
        key_heads = mixer_config.get("key_heads", 0)
        value_head_dim = mixer_config.get("value_head_dim", 0)
        key_head_dim = mixer_config.get("key_head_dim", 0)
        kernel = mixer_config.get("convolution_layer", {}).get("kernel_size", 4)
        display = GDNDisplayConfig(
            value_heads=value_heads,
            key_heads=key_heads,
            key_head_dim=key_head_dim,
            value_head_dim=value_head_dim,
            conv_kernel=kernel,
        )
        label = _make_gdn_label(value_heads or None, key_heads or None, value_head_dim or None, key_head_dim or None)

    elif effective_type == "kda":
        heads = mixer_config.get("heads", 0)
        head_dim = mixer_config.get("head_dim", 0)
        kernel = mixer_config.get("convolution_layer", {}).get("kernel_size")
        display = KDADisplayConfig(heads=heads, head_dim=head_dim, conv_kernel=kernel)
        label = _make_kda_label(heads or None, head_dim or None, kernel)

    else:
        display = AttentionDisplayConfig()
        label = effective_type.upper()

    return MixerSpec(mixer_type=effective_type, label=label, display=display)


# ─── Label formatters ────────────────────────────────────────────────


def _make_attention_label(
    effective_type: str,
    heads: int | None,
    head_groups: int | None,
    head_size: int | None,
    window_size: int | None,
) -> str:
    """Generate label for attention-type mixers."""
    parts: list[str] = []
    if heads:
        parts.append(f"{heads}h")
    if head_groups and head_groups != heads:
        parts.append(f"GQA {heads}/{head_groups}")
    if head_size:
        parts.append(f"d={head_size}")

    detail = ", ".join(parts)
    if effective_type == "sliding_window":
        prefix = f"SWA (w={window_size}" if window_size else "SWA ("
        if detail:
            return f"{prefix}, {detail})"
        return f"{prefix})"
    if detail:
        return f"Attention ({detail})"
    return "Attention"


def _make_mamba_label(
    d_state: int | None,
    d_conv: int | None,
    d_inner: int | None,
) -> str:
    parts: list[str] = []
    if d_state:
        parts.append(f"d_state={d_state}")
    if d_conv:
        parts.append(f"d_conv={d_conv}")
    if d_inner:
        parts.append(f"d_inner={d_inner}")
    if parts:
        return f"Mamba ({', '.join(parts)})"
    return "Mamba"


def _make_gdn_label(
    value_heads: int | None,
    key_heads: int | None,
    value_head_dim: int | None,
    key_head_dim: int | None,
) -> str:
    parts: list[str] = []
    if value_heads:
        parts.append(f"{value_heads}vh")
    if key_heads:
        parts.append(f"{key_heads}kh")
    if value_head_dim:
        parts.append(f"d={value_head_dim}")
    if parts:
        return f"GDN ({', '.join(parts)})"
    return "GDN"


def _make_kda_label(
    heads: int | None,
    head_dim: int | None,
    kernel_size: int | None,
) -> str:
    parts: list[str] = []
    if heads:
        parts.append(f"{heads}h")
    if head_dim:
        parts.append(f"d={head_dim}")
    if kernel_size:
        parts.append(f"k={kernel_size}")
    if parts:
        return f"KDA ({', '.join(parts)})"
    return "KDA"


def _make_mlp_label(mlp_config: dict) -> str:
    activation = mlp_config.get("activation", "")
    intermediate = mlp_config.get("intermediate_size")
    gated = mlp_config.get("gated", False)

    parts: list[str] = []
    if activation:
        name = activation.upper()
        if gated:
            name = f"Gated {name}"
        parts.append(name)
    if intermediate:
        parts.append(str(intermediate))
    if parts:
        return f"MLP ({', '.join(parts)})"
    return "MLP"


def mlp_label(mlp: MLPDisplayConfig) -> str:
    """Human-readable label from an MLPDisplayConfig."""
    parts: list[str] = []
    if mlp.activation:
        name = mlp.activation.upper()
        if mlp.gated:
            name = f"Gated {name}"
        parts.append(name)
    if mlp.intermediate_size:
        parts.append(str(mlp.intermediate_size))
    if parts:
        return f"MLP ({', '.join(parts)})"
    return "MLP"


def _make_norm_label(norm_config: dict) -> str:
    norm_type = norm_config.get("type", "rms_norm")
    if norm_type == "rms_norm":
        return "RMSNorm"
    elif norm_type == "layer_norm":
        return "LayerNorm"
    return norm_type


# ─── Run-length encoding ─────────────────────────────────────────────


def _run_length_encode(blocks: list[tuple[str | None, BlockSpec]]) -> list[BlockGroup]:
    """Collapse consecutive identical blocks into groups.

    Uses frozen dataclass __eq__ for structural comparison.
    """
    if not blocks:
        return []

    groups: list[BlockGroup] = []
    current_name, current_spec = blocks[0]
    start = 0
    count = 1

    for i in range(1, len(blocks)):
        name, spec = blocks[i]
        if spec == current_spec:
            count += 1
        else:
            groups.append(BlockGroup(
                block_spec=current_spec,
                block_name=current_name,
                start_index=start,
                count=count,
            ))
            current_name = name
            current_spec = spec
            start = i
            count = 1

    groups.append(BlockGroup(
        block_spec=current_spec,
        block_name=current_name,
        start_index=start,
        count=count,
    ))
    return groups


def _identify_unique_specs(groups: list[BlockGroup]) -> list[tuple[str, BlockSpec]]:
    """Identify unique block specs for detail panels.

    Returns (label, spec) pairs in order of first appearance.
    Labels come from block_name if available, otherwise generated.
    """
    seen: dict[BlockSpec, str] = {}
    result: list[tuple[str, BlockSpec]] = []

    for group in groups:
        spec = group.block_spec
        if spec not in seen:
            label = group.block_name if group.block_name else f"Block Type {len(seen) + 1}"
            seen[spec] = label
            result.append((label, spec))

    return result


# ─── Vision encoder ──────────────────────────────────────────────────


def _extract_vision_encoder(ve_config: dict) -> VisionEncoderSpec:
    """Extract VisionEncoderSpec from a vision_encoder config dict."""
    hidden_size = ve_config.get("hidden_size", 0)

    embeddings = ve_config.get("embeddings", {})
    patch_h = embeddings.get("patch_height", 16)
    patch_w = embeddings.get("patch_width", 16)

    encoder = ve_config.get("encoder", {})
    num_blocks = encoder.get("num_blocks", 0)
    block = encoder.get("block", {})
    mixer_config = block.get("mixer", {})
    mixer = _extract_single_mixer_spec(mixer_config, hidden_size)

    mlp_config = block.get("mlp", {})
    mlp_label_str = _make_mlp_label(mlp_config)

    adapter_config = ve_config.get("adapter", {})
    adapter_label = _make_mlp_label(adapter_config)

    return VisionEncoderSpec(
        hidden_size=hidden_size,
        num_blocks=num_blocks,
        patch_size=(patch_h, patch_w),
        mixer=mixer,
        mlp_label=mlp_label_str,
        adapter_label=adapter_label,
    )
