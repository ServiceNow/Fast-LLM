"""Qwen2/Qwen2.5 to Apriel2 weight conversion plan."""

from fast_llm_external_models.apriel2.conversion.expr import Expr, ExprPlan, Ref, W


def plan_qwen2_to_apriel2(qwen2_config: dict) -> ExprPlan:
    """Build an expression plan for Qwen2/Qwen2.5 to Apriel2 conversion.

    This is a pure mapping (all Ref expressions) since Qwen2→Apriel2
    is just renaming keys. The weight tensors are identical.

    Key mapping (source keys have "model." prefix in safetensors):
        Qwen2 (safetensor key)                      Apriel2
        ----------------------                      -------
        model.embed_tokens.weight                -> model.embed_tokens.weight
        model.norm.weight                        -> model.norm.weight
        model.layers.{i}.input_layernorm.weight  -> model.decoder.blocks.{i}.input_layernorm.weight
        model.layers.{i}.post_attention_layernorm.weight -> model.decoder.blocks.{i}.post_attention_layernorm.weight
        model.layers.{i}.self_attn.q_proj.weight -> model.decoder.blocks.{i}.mixer.q_proj.weight
        model.layers.{i}.self_attn.q_proj.bias   -> model.decoder.blocks.{i}.mixer.q_proj.bias
        model.layers.{i}.self_attn.k_proj.weight -> model.decoder.blocks.{i}.mixer.k_proj.weight
        model.layers.{i}.self_attn.k_proj.bias   -> model.decoder.blocks.{i}.mixer.k_proj.bias
        model.layers.{i}.self_attn.v_proj.weight -> model.decoder.blocks.{i}.mixer.v_proj.weight
        model.layers.{i}.self_attn.v_proj.bias   -> model.decoder.blocks.{i}.mixer.v_proj.bias
        model.layers.{i}.self_attn.o_proj.weight -> model.decoder.blocks.{i}.mixer.o_proj.weight
        model.layers.{i}.mlp.gate_proj.weight    -> model.decoder.blocks.{i}.mlp.gate_proj.weight
        model.layers.{i}.mlp.up_proj.weight      -> model.decoder.blocks.{i}.mlp.up_proj.weight
        model.layers.{i}.mlp.down_proj.weight    -> model.decoder.blocks.{i}.mlp.down_proj.weight

    Note: Qwen2 has QKV biases but no O bias. The Apriel2 config uses per-layer
    bias settings (query_layer.bias.enabled=True, dense_layer.bias.enabled=False)
    to match this exactly - no workarounds needed.

    Args:
        qwen2_config: HuggingFace Qwen2Config as dict

    Returns:
        ExprPlan with Ref mappings
    """
    mappings: dict[str, Expr] = {}

    num_layers = qwen2_config["num_hidden_layers"]

    # Static mappings (embeddings and final norm)
    # Note: Qwen2 safetensor keys have "model." prefix
    static_mappings = [
        (W("model", "embed_tokens", "weight"), W("model", "embed_tokens", "weight")),
        (W("model", "norm", "weight"), W("model", "norm", "weight")),
    ]

    # lm_head - only if not tied
    if not qwen2_config.get("tie_word_embeddings", False):
        static_mappings.append((W("lm_head", "weight"), W("lm_head", "weight")))

    for src, tgt in static_mappings:
        mappings[tgt] = Ref(key=src)

    # Layer mappings
    for layer in range(num_layers):
        # Source has "model.layers.{i}" prefix
        qwen_layer = W("model", "layers", layer)
        apriel_layer = W("model", "decoder", "blocks", layer)

        # Attention projection weights
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            src = qwen_layer / "self_attn" / proj / "weight"
            tgt = apriel_layer / "mixer" / proj / "weight"
            mappings[tgt] = Ref(key=src)

        # QKV biases (Qwen2 has these, but not O bias)
        for proj in ["q_proj", "k_proj", "v_proj"]:
            src = qwen_layer / "self_attn" / proj / "bias"
            tgt = apriel_layer / "mixer" / proj / "bias"
            mappings[tgt] = Ref(key=src)

        # Note: o_proj has no bias in Qwen2, and Apriel2 config has dense_layer.bias.enabled=False

        # MLP projections
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            src = qwen_layer / "mlp" / proj / "weight"
            tgt = apriel_layer / "mlp" / proj / "weight"
            mappings[tgt] = Ref(key=src)

        # Layer norms
        mappings[apriel_layer / "input_layernorm" / "weight"] = Ref(key=qwen_layer / "input_layernorm" / "weight")
        mappings[apriel_layer / "post_attention_layernorm" / "weight"] = Ref(
            key=qwen_layer / "post_attention_layernorm" / "weight"
        )

    return ExprPlan(
        mappings=mappings,
        source_format="qwen2",
        target_format="apriel2",
        metadata={
            "num_layers": num_layers,
            "hidden_size": qwen2_config["hidden_size"],
            "num_attention_heads": qwen2_config["num_attention_heads"],
            "num_key_value_heads": qwen2_config.get("num_key_value_heads", qwen2_config["num_attention_heads"]),
        },
    )
