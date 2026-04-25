import pytest
import torch

from fast_llm.models.gpt.conversion.auto import AutoGPTHuggingfaceCheckpointHandler
from fast_llm.models.gpt.conversion.config import Gemma4CheckpointFormat
from fast_llm.models.gpt.conversion.gemma4 import (
    Gemma4BaseModelConverter,
    Gemma4ExpertLayer1Converter,
    Gemma4ExpertLayer2Converter,
    Gemma4HuggingfaceCheckpointHandler,
    Gemma4LayerScalarConverter,
)


def _gemma4_config() -> dict:
    return {
        "model_type": "gemma4_text",
        "architectures": ["Gemma4ForCausalLM"],
        "enable_moe_block": True,
        "attention_k_eq_v": True,
        "use_bidirectional_attention": "vision",
        "hidden_size_per_layer_input": 0,
        "num_kv_shared_layers": 0,
        "use_double_wide_mlp": False,
        "hidden_size": 32,
        "intermediate_size": 24,
        "moe_intermediate_size": 8,
        "num_experts": 4,
        "top_k_experts": 2,
        "num_hidden_layers": 6,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_global_key_value_heads": 1,
        "head_dim": 8,
        "global_head_dim": 16,
        "sliding_window": 16,
        "final_logit_softcapping": 30.0,
        "tie_word_embeddings": True,
        "vocab_size": 128,
        "rms_norm_eps": 1e-6,
        "hidden_activation": "gelu_pytorch_tanh",
        "attention_dropout": 0.0,
        "rope_parameters": {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
            "full_attention": {
                "rope_type": "proportional",
                "rope_theta": 1_000_000.0,
                "partial_rotary_factor": 0.25,
            },
        },
        "layer_types": ["sliding_attention"] * 5 + ["full_attention"],
    }


def test_gemma4_config_import_builds_pattern_decoder():
    model_config = Gemma4HuggingfaceCheckpointHandler._import_config(_gemma4_config())
    base = model_config.base_model

    assert base.embeddings.scale_by_sqrt_hidden_size
    assert base.head.final_logit_softcap == 30.0
    assert base.decoder.expanded_pattern == ["sliding"] * 5 + ["full"]
    assert not base.decoder.blocks["sliding"].mixer.attention_k_eq_v
    assert base.decoder.blocks["sliding"].mixer.window_size == 16
    assert base.decoder.blocks["sliding"].mixer.softmax_scale_power == 0.0
    assert base.decoder.blocks["sliding"].mixer.rotary.theta == 10_000.0
    assert base.decoder.blocks["full"].mixer.attention_k_eq_v
    assert base.decoder.blocks["full"].mixer.head_size == 16
    assert base.decoder.blocks["full"].mixer.softmax_scale_power == 0.0
    assert base.decoder.blocks["full"].mixer.rotary.theta == 1_000_000.0
    assert base.decoder.blocks["full"].mixer.rotary.partial_rotary_factor == 0.25
    assert base.decoder.blocks["full"].mlp.moe_intermediate_size == 8


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("hidden_size_per_layer_input", 1, "per-layer-input"),
        ("num_kv_shared_layers", 1, "KV sharing"),
        ("use_double_wide_mlp", True, "use_double_wide_mlp"),
        ("use_bidirectional_attention", "all", "non-causal"),
        ("attention_bias", True, "attention biases"),
        ("attention_k_eq_v", False, "shared K=V"),
    ],
)
def test_gemma4_config_rejects_unsupported_branches(field, value, message):
    config = _gemma4_config()
    config[field] = value

    with pytest.raises(NotImplementedError, match=message):
        Gemma4BaseModelConverter.import_config(config)


def test_gemma4_expert_weight_converters_roundtrip():
    mlp_config = Gemma4HuggingfaceCheckpointHandler._import_config(_gemma4_config()).base_model.decoder.blocks[
        "full"
    ].mlp

    layer_1_converter = Gemma4ExpertLayer1Converter("fast", "hf", mlp_config)
    hf_gate_up = torch.arange(4 * 16 * 32, dtype=torch.float32).reshape(4, 16, 32)
    fast_gate_up, = layer_1_converter.import_weight((hf_gate_up,))
    assert fast_gate_up.shape == (64, 32)
    roundtrip_gate_up, = layer_1_converter.export_weight((fast_gate_up,))
    torch.testing.assert_close(roundtrip_gate_up, hf_gate_up)

    layer_2_converter = Gemma4ExpertLayer2Converter("fast", "hf", mlp_config)
    hf_down = torch.arange(4 * 32 * 8, dtype=torch.float32).reshape(4, 32, 8)
    fast_down, = layer_2_converter.import_weight((hf_down,))
    assert fast_down.shape == (32, 32)
    roundtrip_down, = layer_2_converter.export_weight((fast_down,))
    torch.testing.assert_close(roundtrip_down, hf_down)


def test_gemma4_weight_converter_keys_follow_sliding_and_full_attention():
    base_config = Gemma4HuggingfaceCheckpointHandler._import_config(_gemma4_config()).base_model
    exported_config = Gemma4BaseModelConverter.export_config(base_config)
    converters = Gemma4BaseModelConverter.get_converters(base_config, exported_config)
    export_names = {name for converter in converters for name in converter.export_name}

    assert "model.layers.4.self_attn.k_proj.weight" in export_names
    assert "model.layers.4.self_attn.v_proj.weight" in export_names
    assert "model.layers.5.self_attn.k_proj.weight" in export_names
    assert "model.layers.5.self_attn.v_proj.weight" not in export_names
    assert "model.layers.5.router.scale" in export_names
    assert "model.layers.5.router.per_expert_scale" in export_names
    assert "model.layers.5.experts.gate_up_proj" in export_names
    assert "model.layers.5.experts.down_proj" in export_names
    assert "model.layers.5.layer_scalar" in export_names
    assert exported_config["rope_parameters"] == _gemma4_config()["rope_parameters"]


def test_gemma4_layer_scalar_import_requires_one():
    converter = Gemma4LayerScalarConverter((), "model.layers.0.layer_scalar")

    assert converter.import_weight((torch.ones(1),)) == ()
    with pytest.raises(AssertionError):
        converter.import_weight((torch.zeros(1),))


def test_gemma4_registered_with_gpt_auto_handler():
    assert AutoGPTHuggingfaceCheckpointHandler.get_handler_class(Gemma4CheckpointFormat.name) is (
        Gemma4HuggingfaceCheckpointHandler
    )
    assert AutoGPTHuggingfaceCheckpointHandler.get_handler_class("gemma4_text") is Gemma4HuggingfaceCheckpointHandler
