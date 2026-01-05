"""Qwen2/Qwen2.5 to Apriel2 config conversion."""


def convert_config(qwen2_config: dict) -> dict:
    """Convert Qwen2/Qwen2.5 config to Apriel2TextConfig format.

    Qwen2.5 architecture:
    - Standard transformer with GQA (grouped query attention)
    - QKV bias enabled, O bias disabled
    - MLP bias disabled
    - Gated SwiGLU MLP
    - RMSNorm
    - RoPE embeddings

    Args:
        qwen2_config: HuggingFace Qwen2Config as dict

    Returns:
        Apriel2TextConfig-compatible dict
    """
    hidden_size = qwen2_config["hidden_size"]
    num_attention_heads = qwen2_config["num_attention_heads"]
    num_key_value_heads = qwen2_config.get("num_key_value_heads", num_attention_heads)
    head_dim = hidden_size // num_attention_heads

    # Qwen2 uses QKV bias but not O bias - mirror Fast-LLM's per-layer config
    return {
        "model_type": "apriel2_text",
        "architectures": ["Apriel2ForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_apriel2.Apriel2TextConfig",
            "AutoModel": "modeling_apriel2.Apriel2TextModel",
            "AutoModelForCausalLM": "modeling_apriel2.Apriel2ForCausalLM",
        },
        "hidden_size": hidden_size,
        "vocab_size": qwen2_config["vocab_size"],
        "tie_word_embeddings": qwen2_config.get("tie_word_embeddings", False),
        "decoder": {
            "type": "fixed",
            "num_blocks": qwen2_config["num_hidden_layers"],
            "block": {
                "mixer": {
                    "type": "attention",
                    "heads": num_attention_heads,
                    "head_groups": num_key_value_heads,
                    "head_size": head_dim,
                    # Per-layer bias config matching Fast-LLM structure
                    "query_layer": {"bias": {"enabled": True}},
                    "key_layer": {"bias": {"enabled": True}},
                    "value_layer": {"bias": {"enabled": True}},
                    "dense_layer": {"bias": {"enabled": False}},
                    "rotary": {
                        "type": "mistral_1d",
                        "theta": qwen2_config.get("rope_theta", 1000000.0),
                    },
                },
                "mlp": {
                    "type": "mlp",
                    "intermediate_size": qwen2_config["intermediate_size"],
                    "activation": qwen2_config.get("hidden_act", "silu"),
                    "gated": True,
                    "add_linear_biases": False,
                },
                "normalization": {
                    "type": "rms_norm",
                    "epsilon": qwen2_config.get("rms_norm_eps", 1e-6),
                },
            },
        },
        "head": {
            "normalization": {
                "type": "rms_norm",
                "epsilon": qwen2_config.get("rms_norm_eps", 1e-6),
            }
        },
        "embeddings": {
            "max_position_embeddings": qwen2_config.get("max_position_embeddings", 32768),
        },
    }
