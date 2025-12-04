"""Llava to Apriel2 config conversion."""


def convert_config(llava_config: dict) -> dict:
    """Convert Llava config to Apriel2 format.

    This is a pure 1-to-1 mapping - no architecture modifications.
    The resulting config has attention-only decoder matching the source structure.

    Args:
        llava_config: Source Llava/Pixtral config dict.

    Returns:
        Apriel2 config dict with equivalent architecture.
    """
    text_config = llava_config["text_config"]

    # Get token IDs - prefer top-level, fall back to text_config
    bos_token_id = llava_config.get("bos_token_id") or text_config.get("bos_token_id")
    eos_token_id = llava_config.get("eos_token_id") or text_config.get("eos_token_id")
    pad_token_id = llava_config.get("pad_token_id") or text_config.get("pad_token_id")

    # Build decoder config (attention-only, matching source)
    hidden_size = text_config["hidden_size"]
    num_heads = text_config["num_attention_heads"]
    num_kv_heads = text_config["num_key_value_heads"]
    rope_theta = text_config["rope_theta"]
    # Use explicit head_dim if available (some models have head_dim != hidden_size // num_heads)
    # Note: MistralConfig.head_dim is None by default, so we must check for None explicitly
    head_dim = text_config.get("head_dim")
    if head_dim is None:
        head_dim = hidden_size // num_heads

    decoder_config = {
        "type": "fixed",
        "num_blocks": text_config["num_hidden_layers"],
        "block": {
            "mixer": {
                "type": "attention",
                "heads": num_heads,
                "head_groups": num_kv_heads,
                "head_size": head_dim,
                "add_linear_biases": False,
                "rotary": {"type": "mistral_1d", "theta": rope_theta},
            },
            "mlp": {
                "type": "mlp",
                "intermediate_size": text_config["intermediate_size"],
                "activation": text_config["hidden_act"],
                "gated": True,
                "add_linear_biases": False,
            },
            "normalization": {
                "type": "rms_norm",
                "epsilon": text_config["rms_norm_eps"],
            },
        },
    }

    apriel2_config = {
        "architectures": ["Apriel2ForConditionalGeneration"],
        "model_type": "apriel2",
        "auto_map": {
            "AutoConfig": "configuration_apriel2.Apriel2Config",
            "AutoModel": "modeling_apriel2.Apriel2Model",
            "AutoModelForCausalLM": "modeling_apriel2.Apriel2ForConditionalGeneration",
        },
        "hidden_size": hidden_size,
        "vocab_size": text_config["vocab_size"],
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
        "tie_word_embeddings": text_config["tie_word_embeddings"],
        "use_cache": text_config.get("use_cache", True),
        "image_token_index": llava_config["image_token_index"],
        "decoder": decoder_config,
        "embeddings": {
            "max_position_embeddings": text_config["max_position_embeddings"],
        },
        "head": {
            "normalization": {
                "type": "rms_norm",
                "epsilon": text_config["rms_norm_eps"],
            },
        },
        "vision_encoder": _convert_vision_config(llava_config),
    }

    return apriel2_config


def _convert_vision_config(llava_config: dict) -> dict:
    """Convert Llava vision_config to Apriel2 vision_encoder format."""
    vision_config = llava_config["vision_config"]
    text_config = llava_config["text_config"]

    hidden_size = vision_config["hidden_size"]
    num_heads = vision_config["num_attention_heads"]
    num_layers = vision_config["num_hidden_layers"]
    intermediate_size = vision_config["intermediate_size"]
    rope_theta = vision_config["rope_theta"]
    patch_size = vision_config["patch_size"]
    num_channels = vision_config["num_channels"]
    # Use explicit head_dim if available
    # Note: head_dim may be None in HF configs, so check explicitly
    head_dim = vision_config.get("head_dim")
    if head_dim is None:
        head_dim = hidden_size // num_heads

    return {
        "hidden_size": hidden_size,
        "embeddings": {
            "patch_height": patch_size,
            "patch_width": patch_size,
            "input_channels": num_channels,
            "normalization": {"type": "rms_norm", "epsilon": 1e-5},
        },
        "encoder": {
            "type": "fixed",
            "num_blocks": num_layers,
            "block": {
                "mixer": {
                    "type": "attention",
                    "heads": num_heads,
                    "head_groups": num_heads,
                    "head_size": head_dim,
                    "add_linear_biases": False,
                    "causal": False,
                    "cross_document_attention": False,
                    "rotary": {
                        "type": "pixtral_2d",
                        "theta": rope_theta,
                        "patch_size": patch_size,
                        # max_image_size determines the max 2D position table size
                        # Pixtral default is 1024, but we use a larger value to be safe
                        "max_image_size": vision_config.get("image_size", 4096),
                    },
                },
                "mlp": {
                    "type": "mlp",
                    "intermediate_size": intermediate_size,
                    "activation": vision_config["hidden_act"],
                    "gated": True,
                    "add_linear_biases": False,
                },
                "normalization": {"type": "rms_norm", "epsilon": 1e-5},
            },
        },
        "adapter": {
            "type": "mlp",
            "intermediate_size": text_config["hidden_size"],
            "activation": llava_config["projector_hidden_act"],
            "add_linear_biases": True,
            "gated": False,
        },
    }
