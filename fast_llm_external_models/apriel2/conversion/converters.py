"""Plan builders for weight conversion.

This module provides functions to build ExprPlan objects for different
conversion scenarios:
- plan_surgery: Apriel2 → Apriel2 architecture modification (e.g., adding Mamba)
- plan_mil_attention_to_mamba: Attention → Mamba (MIL conversion)
- plan_attention_to_gated_delta_net: Attention → GatedDeltaNet (DIL conversion)

For source-format-specific conversions (e.g., Llava → Apriel2), see the
respective subpackages (e.g., conversion.llava).
"""

from __future__ import annotations

from fast_llm_external_models.apriel2.conversion.expr import (
    Concat,
    Expr,
    ExprPlan,
    Init,
    Ref,
    Slice,
    W,
)


def plan_mil_attention_to_mamba(
    layer_idx: int,
    hidden_size: int,
    d_inner: int,
    d_xb: int,
    dt_rank: int,
    d_state: int,
    d_conv: int,
    repeat_kv_before_conv: bool,
    conv_bias: bool,
    dt_bias: bool,
    dt_min: float,
    dt_max: float,
    dt_init_floor: float,
    source_prefix: W,
    target_prefix: W,
) -> ExprPlan:
    """Build MIL expressions for one layer.

    MIL maps attention projections to Mamba's composite in_proj:
    - Q -> C (readout)
    - K -> B (input-dependent state transition)
    - V -> x (input)
    - z stays random
    - O -> out_proj

    Args:
        layer_idx: Layer index.
        hidden_size: Model hidden size.
        d_inner: Mamba inner dimension (usually 2 * hidden_size).
        d_xb: Mamba x/B dimension.
        dt_rank: Mamba dt rank.
        d_state: Mamba state dimension.
        d_conv: Convolution kernel size (default 4).
        repeat_kv_before_conv: If True, conv has d_inner channels; else d_xb.
        conv_bias: Whether conv1d has bias (default True).
        dt_bias: Whether dt_proj has bias (default True).
        dt_min: Minimum dt value for bias init (default 0.001).
        dt_max: Maximum dt value for bias init (default 0.1).
        source_prefix: Prefix for source attention keys (e.g. layer.mixer.self_attn).
        target_prefix: Prefix for target mamba keys (e.g. layer.mixer).

    Returns:
        ExprPlan mapping target keys to expressions.
    """
    # in_proj layout: [z, x, B, C] with sizes [d_inner, d_xb, d_xb, d_inner]
    # Total: 2*d_inner + 2*d_xb
    #
    # MIL requires source attention dimensions to match target Mamba dimensions:
    # - Q rows must equal d_inner (for C mapping)
    # - K/V rows must equal d_xb (for B/x mapping)
    in_proj_expr = Concat(
        exprs=(
            Init(shape=(d_inner, hidden_size), init_type="kaiming"),  # z: random
            Slice(
                expr=Ref(key=source_prefix / "v_proj" / "weight"), slices=((0, d_xb, None), (None, None, None))
            ),  # x <- V
            Slice(
                expr=Ref(key=source_prefix / "k_proj" / "weight"), slices=((0, d_xb, None), (None, None, None))
            ),  # B <- K
            Slice(
                expr=Ref(key=source_prefix / "q_proj" / "weight"), slices=((0, d_inner, None), (None, None, None))
            ),  # C <- Q
        ),
        dim=0,
    )

    # Conv1d channels depend on repeat_kv_before_conv
    conv_channels = d_inner if repeat_kv_before_conv else d_xb

    result = {
        # Core projections
        target_prefix / "in_proj" / "weight": in_proj_expr,
        target_prefix / "out_proj" / "weight": Ref(key=source_prefix / "o_proj" / "weight"),
        # dt projections
        target_prefix / "dt_in_proj" / "weight": Init(shape=(dt_rank, hidden_size), init_type="kaiming"),
        target_prefix / "dt_proj" / "weight": Init(shape=(d_inner, dt_rank), init_type="kaiming"),
        # Conv1d
        target_prefix / "conv1d" / "weight": Init(shape=(conv_channels, 1, d_conv), init_type="kaiming"),
        # SSM parameters
        target_prefix / "A_log": Init(shape=(d_inner, d_state), init_type="s4d"),  # S4D initialization
        target_prefix / "D": Init(shape=(d_inner,), init_type="ones"),
    }

    # Optional biases
    if dt_bias:
        result[target_prefix / "dt_proj" / "bias"] = Init(
            shape=(d_inner,),
            init_type="dt_bias",
            init_params={"dt_min": dt_min, "dt_max": dt_max, "dt_init_floor": dt_init_floor},
        )

    if conv_bias:
        result[target_prefix / "conv1d" / "bias"] = Init(shape=(conv_channels,), init_type="zeros")

    return ExprPlan(mappings=result)


def plan_attention_to_gated_delta_net(
    *,
    hidden_size: int,
    # Target GatedDeltaNet geometry
    num_v_heads: int,
    num_k_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel_size: int,
    # Source attention geometry (GQA)
    source_num_q_heads: int,
    source_num_kv_heads: int,
    source_head_dim: int,
    # Wiring
    source_prefix: W,
    target_prefix: W,
) -> ExprPlan:
    """Build expressions to convert an attention layer to a GatedDeltaNet block (GQA-aware).

    DIL (Delta-net Initialization from LLM):

    - Map teacher Q/K/V/O into GatedDeltaNet's:
        * in_proj_qkvz.weight  (flattened [Q, K, V, Z] over head groups)
        * out_proj.weight
    - Respect per-head grouping required by fix_query_key_value_ordering:
        For each key-head group g = 0..num_k_heads-1:
            [Q_g (head_k_dim rows),
             K_g (head_k_dim rows),
             V_group_g (v_heads_per_group * head_v_dim rows),
             Z_group_g (same shape as V_group_g, initialized to zeros)]
    - Handle GQA by *tiling* source heads:
        * Q_g comes from teacher Q head (g mod source_num_q_heads)
        * K_g comes from teacher KV head (g mod source_num_kv_heads)
        * V_group_g is built by tiling teacher V heads modulo source_num_kv_heads
    - Initialize Z to zeros (neutral gating input),
      in_proj_ba to zeros (b=a=0 → β≈0.5),
      A_log to small values (slow decay),
      dt_bias to zeros,
      conv1d as near-identity (delta at last position, scaled 0.5 for SiLU),
      norm.weight to ones.

    At init, the block behaves like a gently decaying linearized attention
    with teacher-shaped Q/K/V features.

    Args:
        hidden_size: Model hidden size.
        num_v_heads: Number of value heads in target GDN.
        num_k_heads: Number of key heads in target GDN.
        head_k_dim: Key head dimension in target GDN.
        head_v_dim: Value head dimension in target GDN.
        conv_kernel_size: Convolution kernel size (default 4).
        source_num_q_heads: Number of Q heads in source attention.
        source_num_kv_heads: Number of K/V heads in source attention (GQA).
        source_head_dim: Per-head dimension in source attention.
        source_prefix: Prefix for source attention keys.
        target_prefix: Prefix for target GDN keys.

    Returns:
        ExprPlan mapping target keys to expressions.
    """
    # Target dimensions
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim
    v_heads_per_group = num_v_heads // num_k_heads
    conv_dim = 2 * key_dim + value_dim  # Q + K + V channels

    # References to source weights (row-major: [rows, hidden_size])
    q_ref = Ref(key=source_prefix / "q_proj" / "weight")
    k_ref = Ref(key=source_prefix / "k_proj" / "weight")
    v_ref = Ref(key=source_prefix / "v_proj" / "weight")

    # --- Build per-group blocks for in_proj_qkvz.weight ---
    # Each group: [Q_g, K_g, V_group_g, Z_group_g]
    group_exprs: list[Expr] = []

    for g in range(num_k_heads):
        # Q_g: from teacher Q head (g mod source_num_q_heads)
        # Use source_head_dim for offset, head_k_dim for slice length
        q_head_idx = g % source_num_q_heads
        q_row_start = q_head_idx * source_head_dim
        q_rows = Slice(
            expr=q_ref,
            slices=((q_row_start, q_row_start + head_k_dim, None), (None, None, None)),
        )

        # K_g: from teacher KV head (g mod source_num_kv_heads)
        k_head_idx = g % source_num_kv_heads
        k_row_start = k_head_idx * source_head_dim
        k_rows = Slice(
            expr=k_ref,
            slices=((k_row_start, k_row_start + head_k_dim, None), (None, None, None)),
        )

        # V_group_g: v_heads_per_group target heads, tiled from source KV heads
        v_slices: list[Expr] = []
        for j in range(v_heads_per_group):
            v_head_idx = g * v_heads_per_group + j
            src_v_head_idx = v_head_idx % source_num_kv_heads
            v_row_start = src_v_head_idx * source_head_dim
            v_slices.append(
                Slice(
                    expr=v_ref,
                    slices=((v_row_start, v_row_start + head_v_dim, None), (None, None, None)),
                )
            )
        v_group: Expr = Concat(exprs=tuple(v_slices), dim=0) if len(v_slices) > 1 else v_slices[0]

        # Z_group_g: zeros, same shape as V_group_g
        z_group = Init(shape=(v_heads_per_group * head_v_dim, hidden_size), init_type="zeros")

        # Block for group g
        group_block = Concat(exprs=(q_rows, k_rows, v_group, z_group), dim=0)
        group_exprs.append(group_block)

    in_proj_qkvz_expr: Expr = Concat(exprs=tuple(group_exprs), dim=0)

    # in_proj_ba: zeros → b=a=0 → β = sigmoid(0) = 0.5, a=0
    in_proj_ba_expr = Init(shape=(2 * num_v_heads, hidden_size), init_type="zeros")

    # out_proj: copy from attention O
    out_proj_expr = Ref(key=source_prefix / "o_proj" / "weight")

    # conv1d: near-identity depthwise conv, scaled 0.5 for SiLU linearity
    conv_weight_expr = Init(shape=(conv_dim, 1, conv_kernel_size), init_type="scaled_identity_conv")

    # A_log: slow decay (~10 step half-life)
    # exp(A_log) ≈ 0.1 → g ≈ -0.07 with dt_bias=0 → exp(g) ≈ 0.93
    A_log_expr = Init(shape=(num_v_heads,), init_type="slow_decay")

    # dt_bias: zeros
    dt_bias_expr = Init(shape=(num_v_heads,), init_type="zeros")

    # norm.weight: ones (neutral RMSNorm-like behavior)
    norm_weight_expr = Init(shape=(head_v_dim,), init_type="ones")

    # Note: Apriel2GatedDeltaNet wraps the actual GDN in self.gdn, so paths need .gdn. segment
    gdn = target_prefix / "gdn"
    return ExprPlan(
        mappings={
            gdn / "in_proj_qkvz" / "weight": in_proj_qkvz_expr,
            gdn / "in_proj_ba" / "weight": in_proj_ba_expr,
            gdn / "out_proj" / "weight": out_proj_expr,
            gdn / "conv1d" / "weight": conv_weight_expr,
            gdn / "A_log": A_log_expr,
            gdn / "dt_bias": dt_bias_expr,
            gdn / "norm" / "weight": norm_weight_expr,
        }
    )


def _plan_non_decoder_weights(config: dict) -> ExprPlan:
    """Build passthrough mappings for non-decoder weights.

    These weights are typically unchanged during surgery:
    - Embeddings
    - LM head
    - Final norm
    - Vision encoder (if present)
    """
    mappings: dict[W, Expr] = {}

    # Core model weights (passthrough as identity)
    embed = W("model", "embed_tokens", "weight")
    mappings[embed] = Ref(key=embed)

    head = W("lm_head", "weight")
    mappings[head] = Ref(key=head)

    norm = W("model", "norm", "weight")
    mappings[norm] = Ref(key=norm)

    # Vision encoder (if present)
    if "vision_encoder" in config:
        vision_config = config["vision_encoder"]
        vision = W("model", "vision_encoder")

        # Patch convolution
        patch_conv = vision / "patch_convolution" / "conv" / "weight"
        mappings[patch_conv] = Ref(key=patch_conv)

        patch_norm = vision / "patch_convolution" / "norm" / "weight"
        mappings[patch_norm] = Ref(key=patch_norm)

        # Vision encoder blocks
        encoder_config = vision_config.get("encoder", {})
        num_vision_layers = encoder_config.get("num_blocks", 0)

        for layer in range(num_vision_layers):
            block = vision / "encoder" / "blocks" / layer

            # Attention projections
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                key = block / "mixer" / "self_attn" / proj / "weight"
                mappings[key] = Ref(key=key)

            # MLP projections
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                key = block / "mlp" / proj / "weight"
                mappings[key] = Ref(key=key)

            # Layer norms
            for norm_name in ["input_layernorm", "post_attention_layernorm"]:
                key = block / norm_name / "weight"
                mappings[key] = Ref(key=key)

        # Adapter
        adapter_config = vision_config.get("adapter", {})
        add_biases = adapter_config.get("add_linear_biases", False)
        adapter = vision / "adapter"

        for proj in ["linear_1", "linear_2"]:
            weight_key = adapter / proj / "weight"
            mappings[weight_key] = Ref(key=weight_key)
            if add_biases:
                bias_key = adapter / proj / "bias"
                mappings[bias_key] = Ref(key=bias_key)

    return ExprPlan(mappings=mappings)


def _get_block_config(decoder_config: dict, layer_idx: int) -> dict:
    """Get block config for a specific layer index.

    Supports both 'fixed' (single block config) and 'pattern' (multiple block configs).
    """
    decoder_type = decoder_config.get("type", "fixed")

    if decoder_type == "fixed":
        return decoder_config.get("block", {})
    elif decoder_type == "pattern":
        pattern = decoder_config.get("pattern", [])
        blocks = decoder_config.get("blocks", {})
        if pattern:
            block_name = pattern[layer_idx % len(pattern)]
            return blocks.get(block_name, {})
        return {}
    else:
        return {}


def plan_surgery(
    source_config: dict,
    target_config: dict,
) -> ExprPlan:
    """Build an expression plan for Apriel2 surgery.

    This handles converting between different Apriel2 architectures,
    including attention → mamba (MIL) and stochastic mixer wrapping.
    """
    hidden_size = target_config.get("hidden_size", source_config.get("hidden_size"))
    assert hidden_size is not None, "hidden_size must be specified in source or target config"

    source_decoder = source_config.get("decoder", {})
    target_decoder = target_config.get("decoder", {})

    num_source_layers = source_decoder.get("num_blocks", 0)
    # Inherit num_blocks from source if not specified in target
    num_target_layers = target_decoder.get("num_blocks", num_source_layers)

    # Non-decoder weights: passthrough as Ref(key)
    plan = _plan_non_decoder_weights(source_config)

    # Process decoder layers
    for target_layer_idx in range(num_target_layers):
        source_layer_idx = target_layer_idx % num_source_layers if num_source_layers > 0 else 0

        source_block = _get_block_config(source_decoder, source_layer_idx)
        target_block = _get_block_config(target_decoder, target_layer_idx)

        # Mixer conversion
        plan += _plan_mixer(
            target_layer_idx,
            source_layer_idx,
            source_block.get("mixer", {}),
            target_block.get("mixer", {}),
            hidden_size,
        )

        # MLP conversion (usually passthrough)
        plan += _plan_mlp(
            target_layer_idx,
            source_layer_idx,
            source_block.get("mlp", {}),
            target_block.get("mlp", {}),
            hidden_size,
        )

        # Norm conversion (usually passthrough)
        plan += _plan_norms(
            target_layer_idx,
            source_layer_idx,
            source_block,
            target_block,
            hidden_size,
        )

    # Set source/target formats
    return ExprPlan(
        mappings=plan.mappings,
        source_format="apriel2",
        target_format="apriel2",
        metadata=plan.metadata,
    )


def _plan_mixer(
    target_layer_idx: int,
    source_layer_idx: int,
    source_mixer: dict,
    target_mixer: dict,
    hidden_size: int,
) -> ExprPlan:
    """Build mixer conversion expressions."""
    source_type = source_mixer.get("type", "attention")
    target_type = target_mixer.get("type", "attention")

    source_layer = W("model", "decoder", "blocks", source_layer_idx)
    target_layer = W("model", "decoder", "blocks", target_layer_idx)

    # Unwrap stochastic source
    if source_type == "stochastic":
        main_name = source_mixer.get("main_mixer_name", "attention")
        actual_source = source_mixer.get("mixers", {}).get(main_name, {})
        actual_source_type = actual_source.get("type", "attention")
        source_mixer_base = source_layer / "mixer" / "mixers" / main_name
    else:
        actual_source = source_mixer
        actual_source_type = source_type
        source_mixer_base = source_layer / "mixer"

    # Add self_attn for attention types
    if actual_source_type in ("attention", "sliding_window"):
        source_prefix = source_mixer_base / "self_attn"
    else:
        source_prefix = source_mixer_base

    # Handle target - parse init mode once, then dispatch to the right function
    if target_type == "stochastic":
        plan = ExprPlan()
        for sub_name, sub_config in target_mixer.get("mixers", {}).items():
            sub_type = sub_config.get("type", "attention")
            target_prefix = target_layer / "mixer" / "mixers" / sub_name

            # Parse init mode and dispatch
            if sub_config.get("init") == "random":
                plan += _plan_random_mixer(target_prefix, sub_type, sub_config, hidden_size)
            else:
                # Default is transfer - fail fast if no converter
                plan += _plan_mixer_transfer(
                    actual_source_type,
                    sub_type,
                    actual_source,
                    sub_config,
                    source_prefix,
                    target_prefix,
                    hidden_size,
                )
        return plan
    else:
        target_prefix = target_layer / "mixer"

        # Parse init mode and dispatch
        if target_mixer.get("init") == "random":
            return _plan_random_mixer(target_prefix, target_type, target_mixer, hidden_size)
        else:
            # Default is transfer - fail fast if no converter
            return _plan_mixer_transfer(
                actual_source_type,
                target_type,
                actual_source,
                target_mixer,
                source_prefix,
                target_prefix,
                hidden_size,
            )


def _plan_mixer_transfer(
    source_type: str,
    target_type: str,
    source_config: dict,
    target_config: dict,
    source_prefix: W,
    target_prefix: W,
    hidden_size: int,
) -> ExprPlan:
    """Build expressions for transferring weights between mixer types.

    This function only handles transfer (not random init). Call _plan_random_mixer
    for random initialization.

    Note: source_prefix already includes self_attn for attention types.

    Raises:
        ValueError: If no converter exists for this source->target type pair.
    """
    # Attention -> Attention (including sliding window variants)
    if source_type in ("attention", "sliding_window") and target_type in ("attention", "sliding_window"):
        # Attention to attention: direct copy
        # Source prefix already includes self_attn, target needs it added
        target_attn = target_prefix / "self_attn"
        return ExprPlan(
            mappings={
                target_attn / proj / "weight": Ref(key=source_prefix / proj / "weight")
                for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]
            }
        )

    if source_type in ("attention", "sliding_window") and target_type == "mamba":
        # Attention to Mamba: MIL conversion
        # Mamba dimensions - derive from hidden_size if not specified
        d_inner = target_config.get("d_inner", 2 * hidden_size)
        dt_rank = target_config.get("dt_rank", hidden_size // 16)
        d_xb = target_config.get("d_xb", hidden_size // 4)
        # These require explicit values (no sensible derivation)
        d_state = target_config["d_state"]
        d_conv = target_config["d_conv"]
        repeat_kv_before_conv = target_config["repeat_kv_before_conv"]
        conv_bias = target_config["conv_bias"]
        dt_bias = target_config["dt_proj_bias"]
        dt_min = target_config["dt_min"]
        dt_max = target_config["dt_max"]
        dt_init_floor = target_config["dt_init_floor"]

        return plan_mil_attention_to_mamba(
            layer_idx=0,  # Not used, we provide prefixes
            hidden_size=hidden_size,
            d_inner=d_inner,
            d_xb=d_xb,
            dt_rank=dt_rank,
            d_state=d_state,
            d_conv=d_conv,
            repeat_kv_before_conv=repeat_kv_before_conv,
            conv_bias=conv_bias,
            dt_bias=dt_bias,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init_floor=dt_init_floor,
            source_prefix=source_prefix,
            target_prefix=target_prefix,
        )

    if source_type == "mamba" and target_type == "mamba":
        # Mamba to Mamba: direct copy (including conv1d)
        return ExprPlan(
            mappings={
                target_prefix / name: Ref(key=source_prefix / name)
                for name in [
                    "in_proj.weight",
                    "out_proj.weight",
                    "dt_in_proj.weight",
                    "dt_proj.weight",
                    "dt_proj.bias",
                    "conv1d.weight",
                    "conv1d.bias",
                    "A_log",
                    "D",
                ]
            }
        )

    if source_type in ("attention", "sliding_window") and target_type == "gated_delta_net":
        # Attention to GatedDeltaNet: DIL conversion
        # Get source attention params
        source_heads = source_config["heads"]
        source_kv_heads = source_config["head_groups"]
        source_head_size = source_config["head_size"]

        # GDN dimensions - derive from source attention if not specified
        num_v_heads = target_config.get("num_value_heads", source_heads)
        num_k_heads = target_config.get("num_key_heads", source_kv_heads)
        head_k_dim = target_config.get("key_head_dim", source_head_size)
        head_v_dim = target_config.get("value_head_dim", source_head_size)
        # conv_kernel_size requires explicit value (no derivation)
        conv_kernel_size = target_config["conv_kernel_size"]

        return plan_attention_to_gated_delta_net(
            hidden_size=hidden_size,
            num_v_heads=num_v_heads,
            num_k_heads=num_k_heads,
            head_k_dim=head_k_dim,
            head_v_dim=head_v_dim,
            conv_kernel_size=conv_kernel_size,
            source_num_q_heads=source_heads,
            source_num_kv_heads=source_kv_heads,
            source_head_dim=source_head_size,
            source_prefix=source_prefix,
            target_prefix=target_prefix,
        )

    if source_type == "gated_delta_net" and target_type == "gated_delta_net":
        # GatedDeltaNet to GatedDeltaNet: direct copy
        return ExprPlan(
            mappings={
                target_prefix / name: Ref(key=source_prefix / name)
                for name in [
                    "gdn.in_proj_qkvz.weight",
                    "gdn.in_proj_ba.weight",
                    "gdn.out_proj.weight",
                    "gdn.conv1d.weight",
                    "gdn.conv1d.bias",
                    "gdn.A_log",
                    "gdn.dt_bias",
                    "gdn.norm.weight",
                ]
            }
        )

    raise ValueError(
        f"No converter available for {source_type} -> {target_type}. "
        f"Use 'init: random' to initialize randomly, or implement a converter."
    )


def _plan_random_mixer(
    prefix: W,
    mixer_type: str,
    config: dict,
    hidden_size: int,
) -> ExprPlan:
    """Build random initialization expressions for a mixer."""
    mappings: dict[W, Expr] = {}

    if mixer_type in ("attention", "sliding_window"):
        heads = config["heads"]
        head_groups = config["head_groups"]
        head_size = config["head_size"]
        q_size = heads * head_size
        kv_size = head_groups * head_size

        attn = prefix / "self_attn"
        mappings[attn / "q_proj" / "weight"] = Init(shape=(q_size, hidden_size), init_type="kaiming")
        mappings[attn / "k_proj" / "weight"] = Init(shape=(kv_size, hidden_size), init_type="kaiming")
        mappings[attn / "v_proj" / "weight"] = Init(shape=(kv_size, hidden_size), init_type="kaiming")
        mappings[attn / "o_proj" / "weight"] = Init(shape=(hidden_size, q_size), init_type="kaiming")

    elif mixer_type == "mamba":
        d_inner = config["d_inner"]
        d_state = config["d_state"]
        dt_rank = config["dt_rank"]
        d_xb = config["d_xb"]
        d_conv = config["d_conv"]
        repeat_kv_before_conv = config["repeat_kv_before_conv"]
        conv_bias = config["conv_bias"]
        dt_bias = config["dt_proj_bias"]
        dt_min = config["dt_min"]
        dt_max = config["dt_max"]
        dt_init_floor = config["dt_init_floor"]

        # Conv1d channels depend on repeat_kv_before_conv
        conv_channels = d_inner if repeat_kv_before_conv else d_xb

        # Core projections
        mappings[prefix / "in_proj" / "weight"] = Init(
            shape=(2 * d_inner + 2 * d_xb, hidden_size), init_type="kaiming"
        )
        mappings[prefix / "out_proj" / "weight"] = Init(shape=(hidden_size, d_inner), init_type="kaiming")

        # dt projections
        mappings[prefix / "dt_in_proj" / "weight"] = Init(shape=(dt_rank, hidden_size), init_type="kaiming")
        mappings[prefix / "dt_proj" / "weight"] = Init(shape=(d_inner, dt_rank), init_type="kaiming")
        # Conv1d
        mappings[prefix / "conv1d" / "weight"] = Init(shape=(conv_channels, 1, d_conv), init_type="kaiming")
        if conv_bias:
            mappings[prefix / "conv1d" / "bias"] = Init(shape=(conv_channels,), init_type="zeros")
        # dt_proj bias with proper initialization
        if dt_bias:
            mappings[prefix / "dt_proj" / "bias"] = Init(
                shape=(d_inner,),
                init_type="dt_bias",
                init_params={"dt_min": dt_min, "dt_max": dt_max, "dt_init_floor": dt_init_floor},
            )

        # SSM parameters - S4D initialization for A_log
        mappings[prefix / "A_log"] = Init(shape=(d_inner, d_state), init_type="s4d")
        mappings[prefix / "D"] = Init(shape=(d_inner,), init_type="ones")

    elif mixer_type == "gated_delta_net":
        # GatedDeltaNet random initialization
        num_v_heads = config["num_value_heads"]
        num_k_heads = config["num_key_heads"]
        head_k_dim = config["key_head_dim"]
        head_v_dim = config["value_head_dim"]
        conv_kernel_size = config.get("conv_kernel_size", 4)

        # GDN dimensions
        key_dim = head_k_dim * num_k_heads
        value_dim = head_v_dim * num_v_heads
        q_dim = head_k_dim * num_v_heads  # Queries use num_v_heads but head_k_dim
        conv_dim = key_dim * 2 + value_dim

        gdn = prefix / "gdn"

        # Combined Q/K/V/Z projection
        qkvz_size = q_dim + key_dim + value_dim * 2  # Q + K + V + Z
        mappings[gdn / "in_proj_qkvz" / "weight"] = Init(shape=(qkvz_size, hidden_size), init_type="kaiming")

        # Beta/alpha projection
        mappings[gdn / "in_proj_ba" / "weight"] = Init(shape=(key_dim * 2, hidden_size), init_type="zeros")

        # Output projection
        mappings[gdn / "out_proj" / "weight"] = Init(shape=(hidden_size, value_dim), init_type="kaiming")

        # Conv1d (depthwise, no bias) - scaled for SiLU linearity
        mappings[gdn / "conv1d" / "weight"] = Init(
            shape=(conv_dim, 1, conv_kernel_size), init_type="scaled_identity_conv"
        )

        # A_log for slow decay
        mappings[gdn / "A_log"] = Init(shape=(num_v_heads,), init_type="slow_decay")

        # dt_bias
        mappings[gdn / "dt_bias"] = Init(shape=(num_v_heads,), init_type="zeros")

        # Norm
        mappings[gdn / "norm" / "weight"] = Init(shape=(value_dim,), init_type="ones")

    return ExprPlan(mappings=mappings)


def _plan_mlp(
    target_layer_idx: int,
    source_layer_idx: int,
    source_mlp: dict,
    target_mlp: dict,
    hidden_size: int,
) -> ExprPlan:
    """Build MLP conversion expressions.

    Parses init mode and dispatches to _plan_mlp_transfer or _plan_random_mlp.
    """
    # Parse init mode and dispatch
    if target_mlp.get("init") == "random":
        return _plan_random_mlp(target_layer_idx, target_mlp, hidden_size)
    else:
        # Default is transfer
        return _plan_mlp_transfer(target_layer_idx, source_layer_idx, source_mlp, target_mlp, hidden_size)


def _plan_mlp_transfer(
    target_layer_idx: int,
    source_layer_idx: int,
    source_mlp: dict,
    target_mlp: dict,
    hidden_size: int,
) -> ExprPlan:
    """Build MLP transfer expressions. Fails if types differ."""
    source_mlp_path = W("model", "decoder", "blocks", source_layer_idx, "mlp")
    target_mlp_path = W("model", "decoder", "blocks", target_layer_idx, "mlp")

    source_type = source_mlp.get("type", "mlp")
    target_type = target_mlp.get("type", "mlp")

    if source_type != target_type:
        raise ValueError(
            f"Cannot transfer MLP weights: source type '{source_type}' != target type '{target_type}'. "
            f"Use 'init: random' to initialize randomly."
        )

    mappings: dict[W, Expr] = {
        target_mlp_path / proj / "weight": Ref(key=source_mlp_path / proj / "weight")
        for proj in ["gate_proj", "up_proj", "down_proj"]
    }

    return ExprPlan(mappings=mappings)


def _plan_random_mlp(
    target_layer_idx: int,
    target_mlp: dict,
    hidden_size: int,
) -> ExprPlan:
    """Build random MLP initialization expressions."""
    target_mlp_path = W("model", "decoder", "blocks", target_layer_idx, "mlp")
    intermediate_size = target_mlp["intermediate_size"]

    mappings: dict[W, Expr] = {
        target_mlp_path / "gate_proj" / "weight": Init(shape=(intermediate_size, hidden_size), init_type="kaiming"),
        target_mlp_path / "up_proj" / "weight": Init(shape=(intermediate_size, hidden_size), init_type="kaiming"),
        target_mlp_path / "down_proj" / "weight": Init(shape=(hidden_size, intermediate_size), init_type="kaiming"),
    }

    return ExprPlan(mappings=mappings)


def _plan_norms(
    target_layer_idx: int,
    source_layer_idx: int,
    source_block: dict,
    target_block: dict,
    hidden_size: int,
) -> ExprPlan:
    """Build normalization conversion expressions.

    Parses init mode and dispatches to transfer or random init.
    """
    target_norm = target_block.get("normalization", {})

    # Parse init mode and dispatch
    if target_norm.get("init") == "random":
        return _plan_random_norms(target_layer_idx, hidden_size)
    else:
        # Default is transfer
        return _plan_norms_transfer(target_layer_idx, source_layer_idx, source_block, target_block, hidden_size)


def _plan_norms_transfer(
    target_layer_idx: int,
    source_layer_idx: int,
    source_block: dict,
    target_block: dict,
    hidden_size: int,
) -> ExprPlan:
    """Build norm transfer expressions. Fails if types differ."""
    source_layer = W("model", "decoder", "blocks", source_layer_idx)
    target_layer = W("model", "decoder", "blocks", target_layer_idx)

    source_norm = source_block.get("normalization", {})
    target_norm = target_block.get("normalization", {})

    source_type = source_norm.get("type", "rms_norm")
    target_type = target_norm.get("type", "rms_norm")

    if source_type != target_type:
        raise ValueError(
            f"Cannot transfer norm weights: source type '{source_type}' != target type '{target_type}'. "
            f"Use 'init: random' to initialize randomly."
        )

    mappings: dict[W, Expr] = {
        target_layer / norm_name / "weight": Ref(key=source_layer / norm_name / "weight")
        for norm_name in ["input_layernorm", "post_attention_layernorm"]
    }

    return ExprPlan(mappings=mappings)


def _plan_random_norms(
    target_layer_idx: int,
    hidden_size: int,
) -> ExprPlan:
    """Build random norm initialization expressions."""
    target_layer = W("model", "decoder", "blocks", target_layer_idx)

    mappings: dict[W, Expr] = {
        target_layer / norm_name / "weight": Init(shape=(hidden_size,), init_type="ones")
        for norm_name in ["input_layernorm", "post_attention_layernorm"]
    }

    return ExprPlan(mappings=mappings)
