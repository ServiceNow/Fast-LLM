"""Plan builders for weight conversion.

This module builds ExprPlan objects that define weight transformations. Plans are
declarative: each target key maps to an expression that computes its value from
source tensors and/or random initialization.

Main Entry Point
================

**plan_surgery(source_config, surgery_spec)**
    Build a plan to transform weights from source_config to the architecture
    defined by applying surgery_spec. This is the primary function for
    architecture modifications (adding Mamba layers, stochastic mixers, etc.).

    The surgery_spec's `init` field controls weight handling:
    - `init: transfer` → use converters (MIL, DIL, KIL, passthrough)
    - `init: random` → use random initialization

    If `init: transfer` is requested but no converter exists for the type pair
    (e.g., mamba → attention), a ValueError is raised.

Conversion Types
================

**Passthrough (same type)**
    Source and target have the same type (e.g., attention → attention).
    Weights are copied directly via Ref expressions.

**MIL (Mamba Initialization from LLM)**
    Converts attention → mamba by mapping:
    - Q → C (readout)
    - K → B (input-dependent state transition)
    - V → x (input)
    - O → out_proj
    - z, conv1d, dt_proj, A_log, D → random initialization

**DIL (Delta-net Initialization from LLM)**
    Converts attention → gated_delta_net by mapping Q/K/V/O projections
    to the fused in_proj_qkvz and out_proj, respecting GQA head grouping.

**KIL (Kimi Initialization from LLM)**
    Converts attention → kda by mapping Q/K/V/O projections directly,
    with random initialization for gates, convolutions, and learnable params.

Stochastic Mixer Handling
=========================

For stochastic mixers (multiple sub-mixers with runtime selection):

1. Each sub-mixer in the target spec gets its own conversion based on its `init` field
2. Sub-mixers with matching names in source inherit from that sub-mixer
3. New sub-mixers inherit from the source's "main" mixer
4. Source sub-mixers not mentioned in target spec are passed through (stochastic → stochastic)

Source-Specific Converters
==========================

For converting from external formats (e.g., Llava → Apriel2), see the
respective subpackages (e.g., `conversion.llava`).
"""

from __future__ import annotations

from fast_llm_external_models.apriel2.conversion.expr import Concat, Expr, ExprPlan, Init, Ref, Slice, W

# =============================================================================
# SECTION 1: Per-Mixer Plan Functions
# =============================================================================
# Each mixer type has ONE function that handles both random init and passthrough.
# This is the single source of truth for each mixer's weight schema.


def _plan_attention_mixer(
    *,
    prefix: W,
    config: dict,
    hidden_size: int,
    source_prefix: W | None = None,
) -> ExprPlan:
    """Plan for attention/sliding_window mixer.

    Weight schema:
    - q_proj.weight: (q_size, hidden_size)
    - k_proj.weight: (kv_size, hidden_size)
    - v_proj.weight: (kv_size, hidden_size)
    - o_proj.weight: (hidden_size, q_size)

    Args:
        prefix: Target weight path prefix.
        config: Mixer config dict.
        hidden_size: Model hidden size.
        source_prefix: If provided, passthrough from source. If None, random init.
    """
    if source_prefix is not None:
        # Passthrough
        return ExprPlan(
            mappings={
                prefix / proj / "weight": Ref(key=source_prefix / proj / "weight")
                for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]
            }
        )

    # Random init
    heads = config["heads"]
    head_groups = config["head_groups"]
    head_size = config["head_size"]
    q_size = heads * head_size
    kv_size = head_groups * head_size

    return ExprPlan(
        mappings={
            prefix / "q_proj" / "weight": Init(shape=(q_size, hidden_size), init_type="kaiming"),
            prefix / "k_proj" / "weight": Init(shape=(kv_size, hidden_size), init_type="kaiming"),
            prefix / "v_proj" / "weight": Init(shape=(kv_size, hidden_size), init_type="kaiming"),
            prefix / "o_proj" / "weight": Init(shape=(hidden_size, q_size), init_type="kaiming"),
        }
    )


def _plan_mamba_mixer(
    *,
    prefix: W,
    config: dict,
    hidden_size: int,
    source_prefix: W | None = None,
) -> ExprPlan:
    """Plan for mamba mixer.

    Weight schema:
    - in_proj.weight: (2*d_inner + 2*d_xb, hidden_size)
    - out_proj.weight: (hidden_size, d_inner)
    - dt_in_proj.weight: (dt_rank, hidden_size)
    - dt_proj.weight: (d_inner, dt_rank)
    - dt_proj.bias: (d_inner,) [optional]
    - conv1d.weight: (conv_channels, 1, d_conv)
    - conv1d.bias: (conv_channels,) [optional]
    - A_log: (d_inner, d_state)
    - D: (d_inner,)

    Args:
        prefix: Target weight path prefix.
        config: Mixer config dict.
        hidden_size: Model hidden size.
        source_prefix: If provided, passthrough from source. If None, random init.
    """
    if source_prefix is not None:
        # Passthrough - include all possible weights
        return ExprPlan(
            mappings={
                prefix / name: Ref(key=source_prefix / name)
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

    # Random init
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

    conv_channels = d_inner if repeat_kv_before_conv else d_xb

    mappings: dict[W, Expr] = {
        prefix / "in_proj" / "weight": Init(shape=(2 * d_inner + 2 * d_xb, hidden_size), init_type="kaiming"),
        prefix / "out_proj" / "weight": Init(shape=(hidden_size, d_inner), init_type="kaiming"),
        prefix / "dt_in_proj" / "weight": Init(shape=(dt_rank, hidden_size), init_type="kaiming"),
        prefix / "dt_proj" / "weight": Init(shape=(d_inner, dt_rank), init_type="kaiming"),
        prefix / "conv1d" / "weight": Init(shape=(conv_channels, 1, d_conv), init_type="kaiming"),
        prefix / "A_log": Init(shape=(d_inner, d_state), init_type="s4d"),
        prefix / "D": Init(shape=(d_inner,), init_type="ones"),
    }

    if conv_bias:
        mappings[prefix / "conv1d" / "bias"] = Init(shape=(conv_channels,), init_type="zeros")
    if dt_bias:
        mappings[prefix / "dt_proj" / "bias"] = Init(
            shape=(d_inner,),
            init_type="dt_bias",
            init_params={"dt_min": dt_min, "dt_max": dt_max, "dt_init_floor": dt_init_floor},
        )

    return ExprPlan(mappings=mappings)


def _plan_gdn_mixer(
    *,
    prefix: W,
    config: dict,
    hidden_size: int,
    source_prefix: W | None = None,
) -> ExprPlan:
    """Plan for gated_delta_net (GDN) mixer.

    Weight schema:
    - in_proj_qkvz.weight: (qkvz_size, hidden_size)
    - in_proj_ba.weight: (2*num_v_heads, hidden_size)
    - out_proj.weight: (hidden_size, value_dim)
    - convolution.weight: (conv_dim, 1, kernel_size)
    - A_log: (num_v_heads,)
    - dt_bias: (num_v_heads,)
    - norm.weight: (head_v_dim,)

    Args:
        prefix: Target weight path prefix.
        config: Mixer config dict.
        hidden_size: Model hidden size.
        source_prefix: If provided, passthrough from source. If None, random init.
    """
    if source_prefix is not None:
        # Passthrough
        return ExprPlan(
            mappings={
                prefix / name: Ref(key=source_prefix / name)
                for name in [
                    "in_proj_qkvz.weight",
                    "in_proj_ba.weight",
                    "out_proj.weight",
                    "convolution.weight",
                    "A_log",
                    "dt_bias",
                    "norm.weight",
                ]
            }
        )

    # Random init
    num_v_heads = config["value_heads"]
    num_k_heads = config["key_heads"]
    head_k_dim = config["key_head_dim"]
    head_v_dim = config["value_head_dim"]
    conv_kernel_size = config["convolution_layer"]["kernel_size"]

    key_dim = head_k_dim * num_k_heads
    value_dim = head_v_dim * num_v_heads
    conv_dim = key_dim * 2 + value_dim
    qkvz_size = key_dim * 2 + value_dim * 2  # Q, K both key_dim; V, Z both value_dim

    return ExprPlan(
        mappings={
            prefix / "in_proj_qkvz" / "weight": Init(shape=(qkvz_size, hidden_size), init_type="kaiming"),
            prefix / "in_proj_ba" / "weight": Init(shape=(num_v_heads * 2, hidden_size), init_type="zeros"),
            prefix / "out_proj" / "weight": Init(shape=(hidden_size, value_dim), init_type="kaiming"),
            prefix
            / "convolution"
            / "weight": Init(shape=(conv_dim, 1, conv_kernel_size), init_type="scaled_identity_conv"),
            prefix / "A_log": Init(shape=(num_v_heads,), init_type="slow_decay"),
            prefix / "dt_bias": Init(shape=(num_v_heads,), init_type="zeros"),
            prefix / "norm" / "weight": Init(shape=(head_v_dim,), init_type="ones"),
        }
    )


def _plan_kda_mixer(
    *,
    prefix: W,
    config: dict,
    hidden_size: int,
    source_prefix: W | None = None,
) -> ExprPlan:
    """Plan for Kimi Delta Attention (KDA) mixer.

    Weight schema:
    - q_proj.weight, k_proj.weight, v_proj.weight: (projection_size, hidden_size)
    - o_proj.weight: (hidden_size, projection_size)
    - q_conv.weight, k_conv.weight, v_conv.weight: (projection_size, 1, kernel_size)
    - f_a_proj.weight: (head_dim, hidden_size)
    - f_b_proj.weight: (projection_size, head_dim)
    - g_a_proj.weight: (head_dim, hidden_size)
    - g_b_proj.weight: (projection_size, head_dim)
    - beta_proj.weight: (num_heads, hidden_size)
    - A_log: (num_heads,)
    - dt_bias: (projection_size,)
    - norm.weight: (head_dim,)

    Args:
        prefix: Target weight path prefix.
        config: Mixer config dict.
        hidden_size: Model hidden size.
        source_prefix: If provided, passthrough from source. If None, random init.
    """
    if source_prefix is not None:
        # Passthrough
        return ExprPlan(
            mappings={
                prefix / name: Ref(key=source_prefix / name)
                for name in [
                    "q_proj.weight",
                    "k_proj.weight",
                    "v_proj.weight",
                    "o_proj.weight",
                    "q_conv.weight",
                    "k_conv.weight",
                    "v_conv.weight",
                    "f_a_proj.weight",
                    "f_b_proj.weight",
                    "g_a_proj.weight",
                    "g_b_proj.weight",
                    "beta_proj.weight",
                    "A_log",
                    "dt_bias",
                    "norm.weight",
                ]
            }
        )

    # Random init
    num_heads = config["heads"]
    head_dim = config["head_dim"]
    projection_size = num_heads * head_dim
    conv_kernel_size = config.get("convolution_layer", {}).get("kernel_size", 4)

    return ExprPlan(
        mappings={
            # Main projections
            prefix / "q_proj" / "weight": Init(shape=(projection_size, hidden_size), init_type="kaiming"),
            prefix / "k_proj" / "weight": Init(shape=(projection_size, hidden_size), init_type="kaiming"),
            prefix / "v_proj" / "weight": Init(shape=(projection_size, hidden_size), init_type="kaiming"),
            prefix / "o_proj" / "weight": Init(shape=(hidden_size, projection_size), init_type="kaiming"),
            # Convolutions
            prefix
            / "q_conv"
            / "weight": Init(shape=(projection_size, 1, conv_kernel_size), init_type="scaled_identity_conv"),
            prefix
            / "k_conv"
            / "weight": Init(shape=(projection_size, 1, conv_kernel_size), init_type="scaled_identity_conv"),
            prefix
            / "v_conv"
            / "weight": Init(shape=(projection_size, 1, conv_kernel_size), init_type="scaled_identity_conv"),
            # Gate kernels (low-rank factorization)
            prefix / "f_a_proj" / "weight": Init(shape=(head_dim, hidden_size), init_type="kaiming"),
            prefix / "f_b_proj" / "weight": Init(shape=(projection_size, head_dim), init_type="kaiming"),
            # Output gate (low-rank factorization)
            prefix / "g_a_proj" / "weight": Init(shape=(head_dim, hidden_size), init_type="kaiming"),
            prefix / "g_b_proj" / "weight": Init(shape=(projection_size, head_dim), init_type="kaiming"),
            # Beta projection
            prefix / "beta_proj" / "weight": Init(shape=(num_heads, hidden_size), init_type="kaiming"),
            # Learnable parameters
            prefix / "A_log": Init(shape=(num_heads,), init_type="slow_decay"),
            prefix / "dt_bias": Init(shape=(projection_size,), init_type="zeros"),
            # Normalization
            prefix / "norm" / "weight": Init(shape=(head_dim,), init_type="ones"),
        }
    )


# Dispatcher for per-mixer plan functions
_MIXER_PLANNERS = {
    "attention": _plan_attention_mixer,
    "sliding_window": _plan_attention_mixer,
    "mamba": _plan_mamba_mixer,
    "gdn": _plan_gdn_mixer,
    "kda": _plan_kda_mixer,
}

# Types that are attention-like (can be source for MIL/DIL/KIL)
_ATTENTION_TYPES = frozenset({"attention", "sliding_window"})


# =============================================================================
# SECTION 2: Cross-Type Converters (attention → X)
# =============================================================================
# These are public functions for converting from attention to other mixer types.
# They handle the complex logic of slicing/tiling attention weights.


def plan_mil_attention_to_mamba(
    *,
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
    """MIL: Mamba Initialization from LLM.

    Converts attention → mamba by mapping:
    - Q → C (readout)
    - K → B (input-dependent state transition)
    - V → x (input)
    - O → out_proj
    - z, conv1d, dt_proj, A_log, D → random initialization

    in_proj layout: [z, x, B, C] with sizes [d_inner, d_xb, d_xb, d_inner]
    """
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

    conv_channels = d_inner if repeat_kv_before_conv else d_xb

    mappings: dict[W, Expr] = {
        target_prefix / "in_proj" / "weight": in_proj_expr,
        target_prefix / "out_proj" / "weight": Ref(key=source_prefix / "o_proj" / "weight"),
        target_prefix / "dt_in_proj" / "weight": Init(shape=(dt_rank, hidden_size), init_type="kaiming"),
        target_prefix / "dt_proj" / "weight": Init(shape=(d_inner, dt_rank), init_type="kaiming"),
        target_prefix / "conv1d" / "weight": Init(shape=(conv_channels, 1, d_conv), init_type="kaiming"),
        target_prefix / "A_log": Init(shape=(d_inner, d_state), init_type="s4d"),
        target_prefix / "D": Init(shape=(d_inner,), init_type="ones"),
    }

    if dt_bias:
        mappings[target_prefix / "dt_proj" / "bias"] = Init(
            shape=(d_inner,),
            init_type="dt_bias",
            init_params={"dt_min": dt_min, "dt_max": dt_max, "dt_init_floor": dt_init_floor},
        )

    if conv_bias:
        mappings[target_prefix / "conv1d" / "bias"] = Init(shape=(conv_channels,), init_type="zeros")

    return ExprPlan(mappings=mappings)


def plan_dil_attention_to_gdn(
    *,
    hidden_size: int,
    num_v_heads: int,
    num_k_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel_size: int,
    source_num_q_heads: int,
    source_num_kv_heads: int,
    source_head_dim: int,
    source_prefix: W,
    target_prefix: W,
) -> ExprPlan:
    """DIL: Delta-net Initialization from LLM.

    Converts attention → gated_delta_net by mapping Q/K/V/O projections
    to the fused in_proj_qkvz and out_proj, respecting GQA head grouping.

    Produces FLAT layout for in_proj_qkvz: [Q_all | K_all | V_all | Z_all]
    This matches Apriel2/Fast-LLM's expected layout.
    """
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim
    v_heads_per_group = num_v_heads // num_k_heads
    conv_dim = 2 * key_dim + value_dim

    q_ref = Ref(key=source_prefix / "q_proj" / "weight")
    k_ref = Ref(key=source_prefix / "k_proj" / "weight")
    v_ref = Ref(key=source_prefix / "v_proj" / "weight")

    # Build FLAT layout: [Q_all | K_all | V_all | Z_all]
    q_slices: list[Expr] = []
    k_slices: list[Expr] = []
    v_slices: list[Expr] = []

    for g in range(num_k_heads):
        # Q_g from teacher Q head (g mod source_num_q_heads)
        q_head_idx = g % source_num_q_heads
        q_row_start = q_head_idx * source_head_dim
        q_slices.append(
            Slice(
                expr=q_ref,
                slices=((q_row_start, q_row_start + head_k_dim, None), (None, None, None)),
            )
        )

        # K_g from teacher KV head (g mod source_num_kv_heads)
        k_head_idx = g % source_num_kv_heads
        k_row_start = k_head_idx * source_head_dim
        k_slices.append(
            Slice(
                expr=k_ref,
                slices=((k_row_start, k_row_start + head_k_dim, None), (None, None, None)),
            )
        )

        # V_group_g: tile v_heads_per_group from source KV heads
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

    # Z is zeros - flat layout [Z_all]
    z_all = Init(shape=(value_dim, hidden_size), init_type="zeros")

    # Concatenate: [Q_all | K_all | V_all | Z_all]
    in_proj_qkvz_expr: Expr = Concat(
        exprs=(
            Concat(exprs=tuple(q_slices), dim=0),
            Concat(exprs=tuple(k_slices), dim=0),
            Concat(exprs=tuple(v_slices), dim=0),
            z_all,
        ),
        dim=0,
    )

    return ExprPlan(
        mappings={
            target_prefix / "in_proj_qkvz" / "weight": in_proj_qkvz_expr,
            target_prefix
            / "in_proj_ba"
            / "weight": Init(shape=(2 * num_v_heads, hidden_size), init_type="zeros"),  # b=a=0 → β=0.5
            target_prefix / "out_proj" / "weight": Ref(key=source_prefix / "o_proj" / "weight"),
            target_prefix
            / "convolution"
            / "weight": Init(shape=(conv_dim, 1, conv_kernel_size), init_type="scaled_identity_conv"),
            target_prefix / "A_log": Init(shape=(num_v_heads,), init_type="slow_decay"),
            target_prefix / "dt_bias": Init(shape=(num_v_heads,), init_type="zeros"),
            target_prefix / "norm" / "weight": Init(shape=(head_v_dim,), init_type="ones"),
        }
    )


def plan_kil_attention_to_kda(
    *,
    hidden_size: int,
    num_heads: int,
    head_dim: int,
    conv_kernel_size: int,
    source_num_q_heads: int,
    source_num_kv_heads: int,
    source_head_dim: int,
    source_prefix: W,
    target_prefix: W,
) -> ExprPlan:
    """KIL: Kimi Initialization from LLM.

    Converts attention → KDA by transferring Q/K/V/O projections directly.
    Gates, convolutions, and learnable parameters are randomly initialized.

    Transfer (with GQA tiling if needed):
    - q_proj: Transfer from attention.q_proj
    - k_proj: Transfer from attention.k_proj (tiled if GQA)
    - v_proj: Transfer from attention.v_proj (tiled if GQA)
    - o_proj: Transfer from attention.o_proj

    Random init (no attention analogue):
    - f_a_proj, f_b_proj: Gate kernel (low-rank factorization)
    - g_a_proj, g_b_proj: Output gate (low-rank factorization)
    - beta_proj: Per-head beta gating
    - q_conv, k_conv, v_conv: Causal convolutions (scaled identity)
    - A_log: State matrix log (slow decay)
    - dt_bias: Time step bias (zeros)
    - norm: Gated RMS normalization (ones)
    """
    projection_size = num_heads * head_dim
    source_q_size = source_num_q_heads * source_head_dim
    source_kv_size = source_num_kv_heads * source_head_dim

    q_ref = Ref(key=source_prefix / "q_proj" / "weight")
    k_ref = Ref(key=source_prefix / "k_proj" / "weight")
    v_ref = Ref(key=source_prefix / "v_proj" / "weight")

    # Q: tile source Q heads to fill target projection_size
    if source_q_size == projection_size:
        q_expr: Expr = q_ref
    else:
        q_slices: list[Expr] = []
        for h in range(num_heads):
            src_h = h % source_num_q_heads
            row_start = src_h * source_head_dim
            q_slices.append(Slice(expr=q_ref, slices=((row_start, row_start + head_dim, None), (None, None, None))))
        q_expr = Concat(exprs=tuple(q_slices), dim=0)

    # K: tile source KV heads to fill target projection_size
    if source_kv_size == projection_size:
        k_expr: Expr = k_ref
    else:
        k_slices: list[Expr] = []
        for h in range(num_heads):
            src_h = h % source_num_kv_heads
            row_start = src_h * source_head_dim
            k_slices.append(Slice(expr=k_ref, slices=((row_start, row_start + head_dim, None), (None, None, None))))
        k_expr = Concat(exprs=tuple(k_slices), dim=0)

    # V: tile source KV heads to fill target projection_size
    if source_kv_size == projection_size:
        v_expr: Expr = v_ref
    else:
        v_slices: list[Expr] = []
        for h in range(num_heads):
            src_h = h % source_num_kv_heads
            row_start = src_h * source_head_dim
            v_slices.append(Slice(expr=v_ref, slices=((row_start, row_start + head_dim, None), (None, None, None))))
        v_expr = Concat(exprs=tuple(v_slices), dim=0)

    return ExprPlan(
        mappings={
            # Transfer main projections
            target_prefix / "q_proj" / "weight": q_expr,
            target_prefix / "k_proj" / "weight": k_expr,
            target_prefix / "v_proj" / "weight": v_expr,
            target_prefix / "o_proj" / "weight": Ref(key=source_prefix / "o_proj" / "weight"),
            # Random init: convolutions (scaled identity for near-passthrough initially)
            target_prefix
            / "q_conv"
            / "weight": Init(shape=(projection_size, 1, conv_kernel_size), init_type="scaled_identity_conv"),
            target_prefix
            / "k_conv"
            / "weight": Init(shape=(projection_size, 1, conv_kernel_size), init_type="scaled_identity_conv"),
            target_prefix
            / "v_conv"
            / "weight": Init(shape=(projection_size, 1, conv_kernel_size), init_type="scaled_identity_conv"),
            # Random init: gate kernels (low-rank factorization)
            target_prefix / "f_a_proj" / "weight": Init(shape=(head_dim, hidden_size), init_type="kaiming"),
            target_prefix / "f_b_proj" / "weight": Init(shape=(projection_size, head_dim), init_type="kaiming"),
            # Random init: output gate (low-rank factorization)
            target_prefix / "g_a_proj" / "weight": Init(shape=(head_dim, hidden_size), init_type="kaiming"),
            target_prefix / "g_b_proj" / "weight": Init(shape=(projection_size, head_dim), init_type="kaiming"),
            # Random init: beta projection
            target_prefix / "beta_proj" / "weight": Init(shape=(num_heads, hidden_size), init_type="kaiming"),
            # Random init: learnable parameters
            target_prefix / "A_log": Init(shape=(num_heads,), init_type="slow_decay"),
            target_prefix / "dt_bias": Init(shape=(projection_size,), init_type="zeros"),
            # Random init: normalization
            target_prefix / "norm" / "weight": Init(shape=(head_dim,), init_type="ones"),
        }
    )


# =============================================================================
# SECTION 3: Dispatch Logic
# =============================================================================


def _plan_mixer_transfer(
    source_type: str,
    target_type: str,
    source_config: dict,
    target_config: dict,
    source_prefix: W,
    target_prefix: W,
    hidden_size: int,
) -> ExprPlan:
    """Transfer weights between mixer types.

    For same-type transfers, uses passthrough via per-mixer plan functions.
    For cross-type transfers, dispatches to MIL/DIL/KIL converters.
    Raises ValueError if no converter exists for the type pair.
    """
    # Same-type: passthrough via unified per-mixer function
    if source_type == target_type:
        planner = _MIXER_PLANNERS.get(target_type)
        if planner is not None:
            return planner(
                prefix=target_prefix,
                config=target_config,
                hidden_size=hidden_size,
                source_prefix=source_prefix,
            )

    # Attention variants are interchangeable
    if source_type in _ATTENTION_TYPES and target_type in _ATTENTION_TYPES:
        return _plan_attention_mixer(
            prefix=target_prefix,
            config=target_config,
            hidden_size=hidden_size,
            source_prefix=source_prefix,
        )

    # Attention → Mamba (MIL)
    if source_type in _ATTENTION_TYPES and target_type == "mamba":
        return plan_mil_attention_to_mamba(
            hidden_size=hidden_size,
            d_inner=target_config.get("d_inner", 2 * hidden_size),
            d_xb=target_config.get("d_xb", hidden_size // 4),
            dt_rank=target_config.get("dt_rank", hidden_size // 16),
            d_state=target_config["d_state"],
            d_conv=target_config["d_conv"],
            repeat_kv_before_conv=target_config["repeat_kv_before_conv"],
            conv_bias=target_config["conv_bias"],
            dt_bias=target_config["dt_proj_bias"],
            dt_min=target_config["dt_min"],
            dt_max=target_config["dt_max"],
            dt_init_floor=target_config["dt_init_floor"],
            source_prefix=source_prefix,
            target_prefix=target_prefix,
        )

    # Attention → GatedDeltaNet (DIL)
    if source_type in _ATTENTION_TYPES and target_type == "gdn":
        source_heads = source_config["heads"]
        source_kv_heads = source_config["head_groups"]
        source_head_size = source_config["head_size"]

        return plan_dil_attention_to_gdn(
            hidden_size=hidden_size,
            num_v_heads=target_config.get("value_heads", source_heads),
            num_k_heads=target_config.get("key_heads", source_kv_heads),
            head_k_dim=target_config.get("key_head_dim", source_head_size),
            head_v_dim=target_config.get("value_head_dim", source_head_size),
            conv_kernel_size=target_config["convolution_layer"]["kernel_size"],
            source_num_q_heads=source_heads,
            source_num_kv_heads=source_kv_heads,
            source_head_dim=source_head_size,
            source_prefix=source_prefix,
            target_prefix=target_prefix,
        )

    # Attention → KDA (KIL)
    if source_type in _ATTENTION_TYPES and target_type == "kda":
        source_heads = source_config["heads"]
        source_kv_heads = source_config["head_groups"]
        source_head_size = source_config["head_size"]

        return plan_kil_attention_to_kda(
            hidden_size=hidden_size,
            num_heads=target_config.get("heads", source_heads),
            head_dim=target_config.get("head_dim", source_head_size),
            conv_kernel_size=target_config.get("convolution_layer", {}).get("kernel_size", 4),
            source_num_q_heads=source_heads,
            source_num_kv_heads=source_kv_heads,
            source_head_dim=source_head_size,
            source_prefix=source_prefix,
            target_prefix=target_prefix,
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
    """Random initialization for any mixer type.

    Dispatches to the per-mixer plan function with source_prefix=None.
    """
    planner = _MIXER_PLANNERS.get(mixer_type)
    if planner is None:
        raise ValueError(f"Unknown mixer type: {mixer_type}")
    return planner(prefix=prefix, config=config, hidden_size=hidden_size, source_prefix=None)


# =============================================================================
# SECTION 4: Main Entry Point
# =============================================================================


def plan_surgery(
    source_config: dict,
    target_config: dict,
) -> ExprPlan:
    """Build plan for Apriel2→Apriel2 surgery (MIL, DIL, KIL, stochastic mixers, etc.)."""
    hidden_size = target_config.get("hidden_size", source_config.get("hidden_size"))
    assert hidden_size is not None, "hidden_size must be specified in source or target config"

    source_decoder = source_config.get("decoder", {})
    target_decoder = target_config.get("decoder", {})

    num_source_layers = source_decoder.get("num_blocks", 0)
    num_target_layers = target_decoder.get("num_blocks", num_source_layers)

    plan = _plan_non_decoder_weights(source_config)

    for target_layer_idx in range(num_target_layers):
        source_layer_idx = target_layer_idx % num_source_layers if num_source_layers > 0 else 0
        source_block = _get_block_config(source_decoder, source_layer_idx)
        target_block = _get_block_config(target_decoder, target_layer_idx)

        plan += _plan_mixer(
            target_layer_idx,
            source_layer_idx,
            source_block.get("mixer", {}),
            target_block.get("mixer", {}),
            hidden_size,
        )
        plan += _plan_mlp(
            target_layer_idx,
            source_layer_idx,
            source_block.get("mlp", {}),
            target_block.get("mlp", {}),
            hidden_size,
        )
        plan += _plan_norms(
            target_layer_idx,
            source_layer_idx,
            source_block,
            target_block,
            hidden_size,
        )

    return ExprPlan(
        mappings=plan.mappings,
        source_format="apriel2",
        target_format="apriel2",
        metadata=plan.metadata,
    )


# =============================================================================
# SECTION 5: Non-Mixer Helpers
# =============================================================================


def _plan_non_decoder_weights(config: dict) -> ExprPlan:
    """Passthrough for embeddings, lm_head, final norm, vision encoder."""
    mappings: dict[W, Expr] = {}

    embed = W("model", "embed_tokens", "weight")
    mappings[embed] = Ref(key=embed)

    head = W("lm_head", "weight")
    mappings[head] = Ref(key=head)

    norm = W("model", "norm", "weight")
    mappings[norm] = Ref(key=norm)

    if "vision_encoder" in config:
        vision_config = config["vision_encoder"]
        vision = W("model", "vision_encoder")

        patch_emb = vision / "embeddings" / "patch_embeddings" / "weight"
        mappings[patch_emb] = Ref(key=patch_emb)
        emb_norm = vision / "embeddings" / "normalization" / "weight"
        mappings[emb_norm] = Ref(key=emb_norm)

        encoder_config = vision_config.get("encoder", {})
        num_vision_layers = encoder_config.get("num_blocks", 0)

        for layer in range(num_vision_layers):
            block = vision / "encoder" / "blocks" / layer
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                key = block / "mixer" / proj / "weight"
                mappings[key] = Ref(key=key)
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                key = block / "mlp" / proj / "weight"
                mappings[key] = Ref(key=key)
            for norm_name in ["input_layernorm", "post_attention_layernorm"]:
                key = block / norm_name / "weight"
                mappings[key] = Ref(key=key)

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
    """Supports 'fixed' (single block) and 'pattern' (multiple blocks) decoder types."""
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


def _plan_mixer(
    target_layer_idx: int,
    source_layer_idx: int,
    source_mixer: dict,
    target_mixer: dict,
    hidden_size: int,
) -> ExprPlan:
    """Plan mixer weights, handling stochastic wrapper routing."""
    source_type = source_mixer.get("type", "attention")
    target_type = target_mixer.get("type", source_type)

    source_layer = W("model", "decoder", "blocks", source_layer_idx)
    target_layer = W("model", "decoder", "blocks", target_layer_idx)

    source_mixers = source_mixer.get("mixers", {}) if source_type == "stochastic" else {}
    main_name = source_mixer.get("main_mixer_name", "attention") if source_type == "stochastic" else None

    if source_type == "stochastic":
        main_source = source_mixers.get(main_name, {})
        main_source_type = main_source.get("type", "attention")
    else:
        main_source = source_mixer
        main_source_type = source_type

    if target_type == "stochastic":
        plan = ExprPlan()
        target_mixers_spec = target_mixer.get("mixers", {})

        for sub_name, sub_config in target_mixers_spec.items():
            sub_type = sub_config.get("type", "attention")
            target_prefix = target_layer / "mixer" / "mixers" / sub_name

            if sub_config.get("init") == "random":
                plan += _plan_random_mixer(target_prefix, sub_type, sub_config, hidden_size)
            else:
                # Match by name (stoch→stoch), else use main mixer
                if source_type == "stochastic" and sub_name in source_mixers:
                    matched_source = source_mixers[sub_name]
                    matched_source_type = matched_source.get("type", "attention")
                    source_mixer_base = source_layer / "mixer" / "mixers" / sub_name
                else:
                    matched_source = main_source
                    matched_source_type = main_source_type
                    if source_type == "stochastic":
                        source_mixer_base = source_layer / "mixer" / "mixers" / main_name
                    else:
                        source_mixer_base = source_layer / "mixer"

                source_prefix = source_mixer_base

                plan += _plan_mixer_transfer(
                    matched_source_type,
                    sub_type,
                    matched_source,
                    sub_config,
                    source_prefix,
                    target_prefix,
                    hidden_size,
                )

        # Passthrough source sub-mixers not in target spec
        if source_type == "stochastic":
            for sub_name, sub_config in source_mixers.items():
                if sub_name not in target_mixers_spec:
                    sub_type = sub_config.get("type", "attention")
                    source_prefix = source_layer / "mixer" / "mixers" / sub_name
                    target_prefix = target_layer / "mixer" / "mixers" / sub_name
                    plan += _plan_mixer_transfer(
                        sub_type,
                        sub_type,
                        sub_config,
                        sub_config,
                        source_prefix,
                        target_prefix,
                        hidden_size,
                    )

        return plan
    else:
        target_prefix = target_layer / "mixer"

        if target_mixer.get("init") == "random":
            return _plan_random_mixer(target_prefix, target_type, target_mixer, hidden_size)

        if source_type == "stochastic":
            source_prefix = source_layer / "mixer" / "mixers" / main_name
        else:
            source_prefix = source_layer / "mixer"

        return _plan_mixer_transfer(
            main_source_type,
            target_type,
            main_source,
            target_mixer,
            source_prefix,
            target_prefix,
            hidden_size,
        )


def _plan_mlp(
    target_layer_idx: int,
    source_layer_idx: int,
    source_mlp: dict,
    target_mlp: dict,
    hidden_size: int,
) -> ExprPlan:
    """Plan MLP weights."""
    if target_mlp.get("init") == "random":
        return _plan_random_mlp(target_layer_idx, target_mlp, hidden_size)
    return _plan_mlp_transfer(target_layer_idx, source_layer_idx, source_mlp, target_mlp, hidden_size)


def _plan_mlp_transfer(
    target_layer_idx: int,
    source_layer_idx: int,
    source_mlp: dict,
    target_mlp: dict,
    hidden_size: int,
) -> ExprPlan:
    """Passthrough for MLP weights."""
    source_mlp_path = W("model", "decoder", "blocks", source_layer_idx, "mlp")
    target_mlp_path = W("model", "decoder", "blocks", target_layer_idx, "mlp")

    source_type = source_mlp.get("type", "mlp")
    target_type = target_mlp.get("type", "mlp")

    if source_type != target_type:
        raise ValueError(
            f"Cannot transfer MLP weights: source type '{source_type}' != target type '{target_type}'. "
            f"Use 'init: random' to initialize randomly."
        )

    return ExprPlan(
        mappings={
            target_mlp_path / proj / "weight": Ref(key=source_mlp_path / proj / "weight")
            for proj in ["gate_proj", "up_proj", "down_proj"]
        }
    )


def _plan_random_mlp(
    target_layer_idx: int,
    target_mlp: dict,
    hidden_size: int,
) -> ExprPlan:
    """Random initialization for MLP."""
    target_mlp_path = W("model", "decoder", "blocks", target_layer_idx, "mlp")
    intermediate_size = target_mlp["intermediate_size"]
    return ExprPlan(
        mappings={
            target_mlp_path
            / "gate_proj"
            / "weight": Init(shape=(intermediate_size, hidden_size), init_type="kaiming"),
            target_mlp_path / "up_proj" / "weight": Init(shape=(intermediate_size, hidden_size), init_type="kaiming"),
            target_mlp_path
            / "down_proj"
            / "weight": Init(shape=(hidden_size, intermediate_size), init_type="kaiming"),
        }
    )


def _plan_norms(
    target_layer_idx: int,
    source_layer_idx: int,
    source_block: dict,
    target_block: dict,
    hidden_size: int,
) -> ExprPlan:
    """Plan normalization layer weights."""
    target_norm = target_block.get("normalization", {})
    if target_norm.get("init") == "random":
        return _plan_random_norms(target_layer_idx, hidden_size)
    return _plan_norms_transfer(target_layer_idx, source_layer_idx, source_block, target_block, hidden_size)


def _plan_norms_transfer(
    target_layer_idx: int,
    source_layer_idx: int,
    source_block: dict,
    target_block: dict,
    hidden_size: int,
) -> ExprPlan:
    """Passthrough for normalization layer weights."""
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

    return ExprPlan(
        mappings={
            target_layer / norm_name / "weight": Ref(key=source_layer / norm_name / "weight")
            for norm_name in ["input_layernorm", "post_attention_layernorm"]
        }
    )


def _plan_random_norms(
    target_layer_idx: int,
    hidden_size: int,
) -> ExprPlan:
    """Random initialization for normalization layers."""
    target_layer = W("model", "decoder", "blocks", target_layer_idx)
    return ExprPlan(
        mappings={
            target_layer / norm_name / "weight": Init(shape=(hidden_size,), init_type="ones")
            for norm_name in ["input_layernorm", "post_attention_layernorm"]
        }
    )
