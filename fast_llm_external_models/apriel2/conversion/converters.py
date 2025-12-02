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
    - `init: transfer` → use converters (MIL, DIL, passthrough)
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
    """MIL: Q→C, K→B, V→x, O→out_proj, z/conv/dt/A_log/D→random."""
    # in_proj layout: [z, x, B, C] sizes [d_inner, d_xb, d_xb, d_inner]
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

    result = {
        target_prefix / "in_proj" / "weight": in_proj_expr,
        target_prefix / "out_proj" / "weight": Ref(key=source_prefix / "o_proj" / "weight"),
        target_prefix / "dt_in_proj" / "weight": Init(shape=(dt_rank, hidden_size), init_type="kaiming"),
        target_prefix / "dt_proj" / "weight": Init(shape=(d_inner, dt_rank), init_type="kaiming"),
        target_prefix / "conv1d" / "weight": Init(shape=(conv_channels, 1, d_conv), init_type="kaiming"),
        target_prefix / "A_log": Init(shape=(d_inner, d_state), init_type="s4d"),
        target_prefix / "D": Init(shape=(d_inner,), init_type="ones"),
    }

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
    """DIL: Q/K/V→in_proj_qkvz (tiled for GQA), O→out_proj, Z/ba/conv/A_log/dt_bias/norm→init."""
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim
    v_heads_per_group = num_v_heads // num_k_heads
    conv_dim = 2 * key_dim + value_dim

    q_ref = Ref(key=source_prefix / "q_proj" / "weight")
    k_ref = Ref(key=source_prefix / "k_proj" / "weight")
    v_ref = Ref(key=source_prefix / "v_proj" / "weight")

    # Build per-group [Q_g, K_g, V_group_g, Z_group_g] for in_proj_qkvz
    group_exprs: list[Expr] = []
    for g in range(num_k_heads):
        # Q_g from teacher Q head (g mod source_num_q_heads)
        q_head_idx = g % source_num_q_heads
        q_row_start = q_head_idx * source_head_dim
        q_rows = Slice(
            expr=q_ref,
            slices=((q_row_start, q_row_start + head_k_dim, None), (None, None, None)),
        )

        # K_g from teacher KV head (g mod source_num_kv_heads)
        k_head_idx = g % source_num_kv_heads
        k_row_start = k_head_idx * source_head_dim
        k_rows = Slice(
            expr=k_ref,
            slices=((k_row_start, k_row_start + head_k_dim, None), (None, None, None)),
        )

        # V_group_g: tile v_heads_per_group from source KV heads
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

        z_group = Init(shape=(v_heads_per_group * head_v_dim, hidden_size), init_type="zeros")
        group_block = Concat(exprs=(q_rows, k_rows, v_group, z_group), dim=0)
        group_exprs.append(group_block)

    in_proj_qkvz_expr: Expr = Concat(exprs=tuple(group_exprs), dim=0)
    in_proj_ba_expr = Init(shape=(2 * num_v_heads, hidden_size), init_type="zeros")  # b=a=0 → β=0.5
    out_proj_expr = Ref(key=source_prefix / "o_proj" / "weight")
    conv_weight_expr = Init(shape=(conv_dim, 1, conv_kernel_size), init_type="scaled_identity_conv")
    A_log_expr = Init(shape=(num_v_heads,), init_type="slow_decay")
    dt_bias_expr = Init(shape=(num_v_heads,), init_type="zeros")
    norm_weight_expr = Init(shape=(head_v_dim,), init_type="ones")

    # Apriel2GatedDeltaNet wraps actual GDN in self.gdn; Qwen3NextGatedDeltaNet has bias=False
    gdn = target_prefix / "gdn"
    return ExprPlan(
        mappings={
            gdn / "in_proj_qkvz" / "weight": in_proj_qkvz_expr,
            gdn / "in_proj_ba" / "weight": in_proj_ba_expr,
            gdn / "out_proj" / "weight": out_proj_expr,
            gdn / "conv1d" / "weight": conv_weight_expr,
            # gdn / "conv1d" / "bias": Init(shape=(conv_dim,), init_type="zeros"),  # GDN conv1d has no bias
            gdn / "A_log": A_log_expr,
            gdn / "dt_bias": dt_bias_expr,
            gdn / "norm" / "weight": norm_weight_expr,
        }
    )


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


def plan_surgery(
    source_config: dict,
    target_config: dict,
) -> ExprPlan:
    """Build plan for Apriel2→Apriel2 surgery (MIL, DIL, stochastic mixers, etc.)."""
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
            target_layer_idx, source_layer_idx,
            source_block.get("mixer", {}), target_block.get("mixer", {}),
            hidden_size,
        )
        plan += _plan_mlp(
            target_layer_idx, source_layer_idx,
            source_block.get("mlp", {}), target_block.get("mlp", {}),
            hidden_size,
        )
        plan += _plan_norms(
            target_layer_idx, source_layer_idx,
            source_block, target_block,
            hidden_size,
        )

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
                    matched_source_type, sub_type,
                    matched_source, sub_config,
                    source_prefix, target_prefix, hidden_size,
                )

        # Passthrough source sub-mixers not in target spec
        if source_type == "stochastic":
            for sub_name, sub_config in source_mixers.items():
                if sub_name not in target_mixers_spec:
                    sub_type = sub_config.get("type", "attention")
                    source_prefix = source_layer / "mixer" / "mixers" / sub_name
                    target_prefix = target_layer / "mixer" / "mixers" / sub_name
                    plan += _plan_mixer_transfer(
                        sub_type, sub_type, sub_config, sub_config,
                        source_prefix, target_prefix, hidden_size,
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
            main_source_type, target_type,
            main_source, target_mixer,
            source_prefix, target_prefix, hidden_size,
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
    """Transfer weights. Raises ValueError if no converter for this type pair."""
    # Attention → Attention
    if source_type in ("attention", "sliding_window") and target_type in ("attention", "sliding_window"):
        return ExprPlan(
            mappings={
                target_prefix / proj / "weight": Ref(key=source_prefix / proj / "weight")
                for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]
            }
        )

    # Attention → Mamba (MIL)
    if source_type in ("attention", "sliding_window") and target_type == "mamba":
        d_inner = target_config.get("d_inner", 2 * hidden_size)
        dt_rank = target_config.get("dt_rank", hidden_size // 16)
        d_xb = target_config.get("d_xb", hidden_size // 4)
        d_state = target_config["d_state"]
        d_conv = target_config["d_conv"]
        repeat_kv_before_conv = target_config["repeat_kv_before_conv"]
        conv_bias = target_config["conv_bias"]
        dt_bias = target_config["dt_proj_bias"]
        dt_min = target_config["dt_min"]
        dt_max = target_config["dt_max"]
        dt_init_floor = target_config["dt_init_floor"]

        return plan_mil_attention_to_mamba(
            layer_idx=0,
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

    # Mamba → Mamba
    if source_type == "mamba" and target_type == "mamba":
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

    # Attention → GatedDeltaNet (DIL)
    if source_type in ("attention", "sliding_window") and target_type == "gated_delta_net":
        source_heads = source_config["heads"]
        source_kv_heads = source_config["head_groups"]
        source_head_size = source_config["head_size"]
        num_v_heads = target_config.get("num_value_heads", source_heads)
        num_k_heads = target_config.get("num_key_heads", source_kv_heads)
        head_k_dim = target_config.get("key_head_dim", source_head_size)
        head_v_dim = target_config.get("value_head_dim", source_head_size)
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

    # GatedDeltaNet → GatedDeltaNet
    if source_type == "gated_delta_net" and target_type == "gated_delta_net":
        return ExprPlan(
            mappings={
                target_prefix / name: Ref(key=source_prefix / name)
                for name in [
                    "gdn.in_proj_qkvz.weight",
                    "gdn.in_proj_ba.weight",
                    "gdn.out_proj.weight",
                    "gdn.conv1d.weight",
                    # "gdn.conv1d.bias",  # GDN conv1d has no bias (Qwen3NextGatedDeltaNet uses bias=False)
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
    mappings: dict[W, Expr] = {}

    if mixer_type in ("attention", "sliding_window"):
        heads = config["heads"]
        head_groups = config["head_groups"]
        head_size = config["head_size"]
        q_size = heads * head_size
        kv_size = head_groups * head_size

        mappings[prefix / "q_proj" / "weight"] = Init(shape=(q_size, hidden_size), init_type="kaiming")
        mappings[prefix / "k_proj" / "weight"] = Init(shape=(kv_size, hidden_size), init_type="kaiming")
        mappings[prefix / "v_proj" / "weight"] = Init(shape=(kv_size, hidden_size), init_type="kaiming")
        mappings[prefix / "o_proj" / "weight"] = Init(shape=(hidden_size, q_size), init_type="kaiming")

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

        conv_channels = d_inner if repeat_kv_before_conv else d_xb
        mappings[prefix / "in_proj" / "weight"] = Init(
            shape=(2 * d_inner + 2 * d_xb, hidden_size), init_type="kaiming"
        )
        mappings[prefix / "out_proj" / "weight"] = Init(shape=(hidden_size, d_inner), init_type="kaiming")
        mappings[prefix / "dt_in_proj" / "weight"] = Init(shape=(dt_rank, hidden_size), init_type="kaiming")
        mappings[prefix / "dt_proj" / "weight"] = Init(shape=(d_inner, dt_rank), init_type="kaiming")
        mappings[prefix / "conv1d" / "weight"] = Init(shape=(conv_channels, 1, d_conv), init_type="kaiming")
        if conv_bias:
            mappings[prefix / "conv1d" / "bias"] = Init(shape=(conv_channels,), init_type="zeros")
        if dt_bias:
            mappings[prefix / "dt_proj" / "bias"] = Init(
                shape=(d_inner,),
                init_type="dt_bias",
                init_params={"dt_min": dt_min, "dt_max": dt_max, "dt_init_floor": dt_init_floor},
            )
        mappings[prefix / "A_log"] = Init(shape=(d_inner, d_state), init_type="s4d")
        mappings[prefix / "D"] = Init(shape=(d_inner,), init_type="ones")

    elif mixer_type == "gated_delta_net":
        num_v_heads = config["num_value_heads"]
        num_k_heads = config["num_key_heads"]
        head_k_dim = config["key_head_dim"]
        head_v_dim = config["value_head_dim"]
        conv_kernel_size = config.get("conv_kernel_size", 4)
        key_dim = head_k_dim * num_k_heads
        value_dim = head_v_dim * num_v_heads
        q_dim = head_k_dim * num_v_heads
        conv_dim = key_dim * 2 + value_dim
        gdn = prefix / "gdn"
        qkvz_size = q_dim + key_dim + value_dim * 2
        mappings[gdn / "in_proj_qkvz" / "weight"] = Init(shape=(qkvz_size, hidden_size), init_type="kaiming")
        mappings[gdn / "in_proj_ba" / "weight"] = Init(shape=(key_dim * 2, hidden_size), init_type="zeros")
        mappings[gdn / "out_proj" / "weight"] = Init(shape=(hidden_size, value_dim), init_type="kaiming")
        mappings[gdn / "conv1d" / "weight"] = Init(
            shape=(conv_dim, 1, conv_kernel_size), init_type="scaled_identity_conv"
        )
        mappings[gdn / "A_log"] = Init(shape=(num_v_heads,), init_type="slow_decay")
        mappings[gdn / "dt_bias"] = Init(shape=(num_v_heads,), init_type="zeros")
        mappings[gdn / "norm" / "weight"] = Init(shape=(value_dim,), init_type="ones")

    return ExprPlan(mappings=mappings)


def _plan_mlp(
    target_layer_idx: int,
    source_layer_idx: int,
    source_mlp: dict,
    target_mlp: dict,
    hidden_size: int,
) -> ExprPlan:
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
    target_mlp_path = W("model", "decoder", "blocks", target_layer_idx, "mlp")
    intermediate_size = target_mlp["intermediate_size"]
    return ExprPlan(mappings={
        target_mlp_path / "gate_proj" / "weight": Init(shape=(intermediate_size, hidden_size), init_type="kaiming"),
        target_mlp_path / "up_proj" / "weight": Init(shape=(intermediate_size, hidden_size), init_type="kaiming"),
        target_mlp_path / "down_proj" / "weight": Init(shape=(hidden_size, intermediate_size), init_type="kaiming"),
    })


def _plan_norms(
    target_layer_idx: int,
    source_layer_idx: int,
    source_block: dict,
    target_block: dict,
    hidden_size: int,
) -> ExprPlan:
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
    target_layer = W("model", "decoder", "blocks", target_layer_idx)
    return ExprPlan(mappings={
        target_layer / norm_name / "weight": Init(shape=(hidden_size,), init_type="ones")
        for norm_name in ["input_layernorm", "post_attention_layernorm"]
    })
