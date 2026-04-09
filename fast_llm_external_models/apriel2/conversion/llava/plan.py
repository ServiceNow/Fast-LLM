"""Llava to Apriel2 weight conversion plan."""

from fast_llm_external_models.apriel2.conversion.expr import Expr, ExprPlan, Ref, W
from fast_llm_external_models.apriel2.modeling_apriel2 import _TRANSFORMERS_V4


def plan_llava_to_apriel2(llava_config: dict) -> ExprPlan:
    """Build an expression plan for Llava to Apriel2 conversion.

    This is a pure mapping (all Ref expressions) since Llava→Apriel2
    is just renaming keys.
    """
    mappings: dict[str, Expr] = {}

    num_text_layers = llava_config.get("text_config", {}).get("num_hidden_layers", 0)
    num_vision_layers = llava_config.get("vision_config", {}).get("num_hidden_layers", 0)

    # In transformers 5.x, LlavaForConditionalGeneration adds a "model." prefix to most submodules.
    # 4.x: language_model.model.*, vision_tower.*, multi_modal_projector.*
    # 5.x: model.language_model.model.*, model.vision_tower.*, model.multi_modal_projector.*
    # lm_head stays as language_model.lm_head.weight in both versions.
    if _TRANSFORMERS_V4:
        text_model_prefix = ("language_model", "model")
        vision_tower_prefix = ("vision_tower",)
        projector_prefix = ("multi_modal_projector",)
    else:
        text_model_prefix = ("model", "language_model", "model")
        vision_tower_prefix = ("model", "vision_tower")
        projector_prefix = ("model", "multi_modal_projector")

    # Static mappings
    static_mappings = [
        (W(*text_model_prefix, "embed_tokens", "weight"), W("model", "embed_tokens", "weight")),
        (W("language_model", "lm_head", "weight"), W("lm_head", "weight")),
        (W(*text_model_prefix, "norm", "weight"), W("model", "norm", "weight")),
        (
            W(*vision_tower_prefix, "patch_conv", "weight"),
            W("model", "vision_encoder", "embeddings", "patch_embeddings", "weight"),
        ),
        (
            W(*vision_tower_prefix, "ln_pre", "weight"),
            W("model", "vision_encoder", "embeddings", "normalization", "weight"),
        ),
        (
            W(*projector_prefix, "linear_1", "weight"),
            W("model", "vision_encoder", "adapter", "linear_1", "weight"),
        ),
        (W(*projector_prefix, "linear_1", "bias"), W("model", "vision_encoder", "adapter", "linear_1", "bias")),
        (
            W(*projector_prefix, "linear_2", "weight"),
            W("model", "vision_encoder", "adapter", "linear_2", "weight"),
        ),
        (W(*projector_prefix, "linear_2", "bias"), W("model", "vision_encoder", "adapter", "linear_2", "bias")),
    ]

    for src, tgt in static_mappings:
        mappings[tgt] = Ref(key=src)

    # Text decoder layers
    for layer in range(num_text_layers):
        llava_layer = W(*text_model_prefix, "layers", layer)
        apriel_layer = W("model", "decoder", "blocks", layer)

        # Attention projections
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            src = llava_layer / "self_attn" / proj / "weight"
            tgt = apriel_layer / "mixer" / proj / "weight"
            mappings[tgt] = Ref(key=src)

        # MLP projections
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            src = llava_layer / "mlp" / proj / "weight"
            tgt = apriel_layer / "mlp" / proj / "weight"
            mappings[tgt] = Ref(key=src)

        # Layer norms
        mappings[apriel_layer / "input_layernorm" / "weight"] = Ref(key=llava_layer / "input_layernorm" / "weight")
        mappings[apriel_layer / "post_attention_layernorm" / "weight"] = Ref(
            key=llava_layer / "post_attention_layernorm" / "weight"
        )

    # Vision encoder layers
    for layer in range(num_vision_layers):
        llava_layer = W(*vision_tower_prefix, "transformer", "layers", layer)
        apriel_layer = W("model", "vision_encoder", "encoder", "blocks", layer)

        # Attention projections
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            src = llava_layer / "attention" / proj / "weight"
            tgt = apriel_layer / "mixer" / proj / "weight"
            mappings[tgt] = Ref(key=src)

        # MLP projections (llava uses feed_forward, apriel uses mlp)
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            src = llava_layer / "feed_forward" / proj / "weight"
            tgt = apriel_layer / "mlp" / proj / "weight"
            mappings[tgt] = Ref(key=src)

        # Layer norms (different naming)
        mappings[apriel_layer / "input_layernorm" / "weight"] = Ref(key=llava_layer / "attention_norm" / "weight")
        mappings[apriel_layer / "post_attention_layernorm" / "weight"] = Ref(key=llava_layer / "ffn_norm" / "weight")

    return ExprPlan(
        mappings=mappings,
        source_format="llava",
        target_format="apriel2",
        metadata={
            "num_text_layers": num_text_layers,
            "num_vision_layers": num_vision_layers,
        },
    )
