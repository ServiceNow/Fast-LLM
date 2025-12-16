"""Test fixtures for Apriel2 model tests."""

from pathlib import Path
from typing import Generator

import pytest
import torch
from transformers import LlavaConfig, LlavaForConditionalGeneration, MistralConfig


# Skip marker for tests that require CUDA for Mamba forward pass
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="SSM mixers (Mamba) require CUDA for forward pass"
)


@pytest.fixture(autouse=True)
def set_default_device():
    """Set default device to CUDA for all tests (Mamba requires CUDA)."""
    if torch.cuda.is_available():
        old_device = torch.get_default_device()
        torch.set_default_device("cuda")
        yield
        torch.set_default_device(old_device)
    else:
        yield


@pytest.fixture(autouse=True)
def set_default_dtype():
    """Set default dtype to float32 for numerical comparison tests."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    yield
    torch.set_default_dtype(old_dtype)


# =============================================================================
# Llava Source Model Fixtures (Pixtral-based, matching Apriel 1.5 structure)
# =============================================================================


def create_llava_pixtral_model(
    hidden_size: int = 256,
    num_heads: int = 8,
    num_kv_heads: int = 4,
    num_layers: int = 5,
    intermediate_size: int = 512,
    vocab_size: int = 1000,
    vision_hidden_size: int = 128,
    vision_num_heads: int = 4,
    vision_num_layers: int = 3,
) -> LlavaForConditionalGeneration:
    """Create a small LlavaForConditionalGeneration with Pixtral vision encoder.

    This produces the same weight format as Apriel 1.5 when saved with save_pretrained().
    """
    text_config = MistralConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        num_hidden_layers=num_layers,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        hidden_act="silu",
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        max_position_embeddings=4096,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=None,
    )

    vision_config = {
        "model_type": "pixtral",
        "hidden_size": vision_hidden_size,
        "num_attention_heads": vision_num_heads,
        "num_hidden_layers": vision_num_layers,
        "intermediate_size": vision_hidden_size * 4,
        "patch_size": 16,
        "num_channels": 3,
        "rope_theta": 10000.0,
        "hidden_act": "silu",
    }

    config = LlavaConfig(
        text_config=text_config,
        vision_config=vision_config,
        image_token_index=10,
        projector_hidden_act="gelu",
        # Use "full" to include all patches - Pixtral doesn't have CLS token
        # so "default" (which removes first token) would drop a real patch
        vision_feature_select_strategy="full",
        # Use final layer output (-1) to match Apriel2's vision encoder behavior
        # Llava default is -2 (second-to-last), but Apriel2 returns final output
        vision_feature_layer=-1,
    )

    return LlavaForConditionalGeneration(config)


@pytest.fixture
def small_pixtral_model() -> LlavaForConditionalGeneration:
    """Create a small Pixtral model for equivalence testing.

    Uses smaller dimensions than create_llava_pixtral_model() defaults
    for faster testing while still exercising all code paths.
    """
    model = create_llava_pixtral_model(
        hidden_size=256,
        num_heads=4,
        num_kv_heads=2,
        num_layers=2,
        intermediate_size=512,
        vocab_size=1000,
        vision_hidden_size=128,
        vision_num_heads=2,
        vision_num_layers=2,
    )
    model.eval()
    return model


@pytest.fixture(params=["identity", "converted"])
def model_pair(request, small_pixtral_model, tmp_path):
    """Parameterized fixture providing source and target models for comparison.

    Parameters:
        identity: Target is identical copy of source (validates test infrastructure)
        converted: Target is Apriel2 model converted from source (tests conversion)

    Returns:
        tuple: (source_model, target_model, expected_atol, variant_name)
    """
    import json
    from safetensors import safe_open

    from fast_llm_external_models.apriel2.configuration_apriel2 import Apriel2Config
    from fast_llm_external_models.apriel2.conversion import (
        convert_llava_config,
        execute,
        plan_llava_to_apriel2,
    )
    from fast_llm_external_models.apriel2.modeling_apriel2 import Apriel2ForConditionalGeneration

    source = small_pixtral_model

    if request.param == "identity":
        # Target is identical copy of source (sanity check)
        target = create_llava_pixtral_model(
            hidden_size=256,
            num_heads=4,
            num_kv_heads=2,
            num_layers=2,
            intermediate_size=512,
            vocab_size=1000,
            vision_hidden_size=128,
            vision_num_heads=2,
            vision_num_layers=2,
        )
        target.load_state_dict(source.state_dict())
        target.eval()
        expected_atol = 1e-6  # Should be essentially identical
    else:
        # Target is converted Apriel2 model
        # Save source to checkpoint (save_pretrained applies key transformations)
        source.save_pretrained(tmp_path)

        # Load config and fix missing fields
        with open(tmp_path / "config.json") as f:
            llava_config = json.load(f)

        llava_config["text_config"]["bos_token_id"] = 1
        llava_config["text_config"]["eos_token_id"] = 2
        llava_config["text_config"]["pad_token_id"] = None
        llava_config["text_config"]["tie_word_embeddings"] = False

        # Load weights from checkpoint
        with safe_open(tmp_path / "model.safetensors", framework="pt") as f:
            source_weights = {key: f.get_tensor(key) for key in f.keys()}

        # Convert
        apriel2_config_dict = convert_llava_config(llava_config)
        plan = plan_llava_to_apriel2(llava_config)
        apriel2_weights = execute(plan, source_weights, seed=0)

        # Create and load Apriel2 model
        apriel2_config = Apriel2Config(**apriel2_config_dict)
        target = Apriel2ForConditionalGeneration(apriel2_config)
        target.load_state_dict(apriel2_weights, strict=False)
        target.eval()
        # Strict tolerance for isolation tests: Each component receives identical
        # inputs, so should produce identical outputs. Integration tests use
        # looser tolerance to account for FP accumulation.
        expected_atol = 1e-6

    return source, target, expected_atol, request.param


@pytest.fixture
def llava_pixtral_config() -> dict:
    """Small Llava config (Pixtral-based) for testing.

    Note: HF's to_dict() omits some config fields that have default values.
    We manually add the missing fields to match the real Apriel 1.5 config format.
    """
    model = create_llava_pixtral_model()
    config = model.config.to_dict()

    # Add missing fields to text_config (matching Apriel 1.5 format)
    config["text_config"]["bos_token_id"] = 1
    config["text_config"]["eos_token_id"] = 2
    config["text_config"]["pad_token_id"] = None
    config["text_config"]["tie_word_embeddings"] = False

    return config


@pytest.fixture
def llava_pixtral_checkpoint(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary Llava checkpoint for converter testing.

    Creates a small random-initialized Llava model using HF's save_pretrained(),
    which produces the same weight format as Apriel 1.5.

    Note: HF's save_pretrained() omits some config fields that have default values.
    We manually add the missing fields to match the real Apriel 1.5 config format.
    """
    import json

    model = create_llava_pixtral_model()
    model.save_pretrained(tmp_path)

    # HF doesn't serialize these fields when they're defaults - add them explicitly
    config_path = tmp_path / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    # Add missing fields to text_config (matching Apriel 1.5 format)
    config["text_config"]["bos_token_id"] = 1
    config["text_config"]["eos_token_id"] = 2
    config["text_config"]["pad_token_id"] = None
    config["text_config"]["tie_word_embeddings"] = False

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    yield tmp_path


# =============================================================================
# Apriel2 Config Fixtures
# =============================================================================


@pytest.fixture
def apriel2_config_tiny():
    """Tiny Apriel2 config for fast testing."""
    from fast_llm_external_models.apriel2.configuration_apriel2 import Apriel2Config

    return Apriel2Config(
        vocab_size=100,
        hidden_size=64,
        decoder={
            "type": "fixed",
            "num_blocks": 2,
            "block": {
                "mixer": {
                    "type": "attention",
                    "heads": 4,
                    "head_groups": 2,
                    "head_size": 16,
                    "rotary": {"type": "mistral_1d", "theta": 10000.0},
                },
                "mlp": {"type": "mlp", "intermediate_size": 256, "gated": True},
                "normalization": {"type": "rms_norm", "epsilon": 1e-5},
            },
        },
    )


@pytest.fixture
def apriel2_config_stochastic():
    """Apriel2 config with stochastic mixer for testing routing."""
    from fast_llm_external_models.apriel2.configuration_apriel2 import Apriel2Config

    return Apriel2Config(
        vocab_size=100,
        hidden_size=64,
        decoder={
            "type": "pattern",
            "num_blocks": 2,
            "pattern": ["attn", "stoch"],
            "blocks": {
                "attn": {
                    "mixer": {
                        "type": "attention",
                        "heads": 4,
                        "head_groups": 2,
                        "head_size": 16,
                        "rotary": {"type": "mistral_1d", "theta": 10000.0},
                    },
                    "mlp": {"type": "mlp", "intermediate_size": 256, "gated": True},
                    "normalization": {"type": "rms_norm", "epsilon": 1e-5},
                },
                "stoch": {
                    "mixer": {
                        "type": "stochastic",
                        "main_mixer_name": "attention",
                        "mixers": {
                            "attention": {
                                "type": "attention",
                                "heads": 4,
                                "head_groups": 2,
                                "head_size": 16,
                                "window_size": 4096,
                                "rotary": {"type": "mistral_1d", "theta": 250000.0},
                            },
                            "mamba": {
                                "type": "mamba",
                                "d_inner": 128,
                                "d_state": 16,
                                "dt_rank": 4,
                                "d_xb": 32,
                                "d_conv": 4,
                                "repeat_kv_before_conv": True,
                                "conv_bias": True,
                                "dt_proj_bias": True,
                                "dt_min": 0.001,
                                "dt_max": 0.1,
                                "dt_init_floor": 1e-4,
                            },
                        },
                    },
                    "mlp": {"type": "mlp", "intermediate_size": 256, "gated": True},
                    "normalization": {"type": "rms_norm", "epsilon": 1e-5},
                },
            },
        },
    )


@pytest.fixture
def apriel2_config_multi_mixer():
    """Apriel2 config with multiple mixers of same type."""
    from fast_llm_external_models.apriel2.configuration_apriel2 import Apriel2Config

    return Apriel2Config(
        vocab_size=100,
        hidden_size=64,
        decoder={
            "type": "pattern",
            "num_blocks": 1,
            "pattern": ["multi"],
            "blocks": {
                "multi": {
                    "mixer": {
                        "type": "stochastic",
                        "main_mixer_name": "attn_small",
                        "mixers": {
                            "attn_small": {
                                "type": "attention",
                                "heads": 4,
                                "head_groups": 2,
                                "head_size": 16,
                                "window_size": 2048,
                                "rotary": {"type": "mistral_1d", "theta": 10000.0},
                            },
                            "attn_large": {
                                "type": "attention",
                                "heads": 4,
                                "head_groups": 2,
                                "head_size": 16,
                                "window_size": 8192,
                                "rotary": {"type": "mistral_1d", "theta": 500000.0},
                            },
                            "mamba_v1": {
                                "type": "mamba",
                                "d_inner": 128,
                                "d_state": 16,
                                "dt_rank": 4,
                                "d_xb": 32,
                                "d_conv": 4,
                                "repeat_kv_before_conv": True,
                                "conv_bias": True,
                                "dt_proj_bias": True,
                                "dt_min": 0.001,
                                "dt_max": 0.1,
                                "dt_init_floor": 1e-4,
                            },
                            "mamba_v2": {
                                "type": "mamba",
                                "d_inner": 128,
                                "d_state": 16,
                                "dt_rank": 4,
                                "d_xb": 32,
                                "d_conv": 4,
                                "repeat_kv_before_conv": True,
                                "conv_bias": True,
                                "dt_proj_bias": True,
                                "dt_min": 0.001,
                                "dt_max": 0.1,
                                "dt_init_floor": 1e-4,
                            },
                        },
                    },
                    "mlp": {"type": "mlp", "intermediate_size": 256, "gated": True},
                    "normalization": {"type": "rms_norm", "epsilon": 1e-5},
                },
            },
        },
    )


@pytest.fixture
def apriel2_config_all_mixers():
    """Apriel2 config with all 4 mixer types in one stochastic mixer.

    This config is critical for testing:
    - All mixer types work (attention, swa, mamba, gated_delta_net)
    - Cache correctly isolates different mixer types
    - Switching between mixers preserves independent state
    """
    from fast_llm_external_models.apriel2.configuration_apriel2 import Apriel2Config

    return Apriel2Config(
        vocab_size=100,
        hidden_size=64,
        decoder={
            "type": "pattern",
            "num_blocks": 2,
            "pattern": ["attn", "all_mixers"],
            "blocks": {
                "attn": {
                    "mixer": {
                        "type": "attention",
                        "heads": 4,
                        "head_groups": 2,
                        "head_size": 16,
                        "rotary": {"type": "mistral_1d", "theta": 10000.0},
                    },
                    "mlp": {"type": "mlp", "intermediate_size": 256, "gated": True},
                    "normalization": {"type": "rms_norm", "epsilon": 1e-5},
                },
                "all_mixers": {
                    "mixer": {
                        "type": "stochastic",
                        "main_mixer_name": "attention",
                        "mixers": {
                            "attention": {
                                "type": "attention",
                                "heads": 4,
                                "head_groups": 2,
                                "head_size": 16,
                                "rotary": {"type": "mistral_1d", "theta": 10000.0},
                            },
                            "swa": {
                                "type": "attention",
                                "heads": 4,
                                "head_groups": 2,
                                "head_size": 16,
                                "window_size": 2048,
                                "rotary": {"type": "mistral_1d", "theta": 1000000.0},
                            },
                            "mamba": {
                                "type": "mamba",
                                "d_inner": 128,
                                "d_state": 16,
                                "dt_rank": 4,
                                "d_xb": 32,
                                "d_conv": 4,
                                "repeat_kv_before_conv": True,
                                "conv_bias": True,
                                "dt_proj_bias": True,
                                "dt_min": 0.001,
                                "dt_max": 0.1,
                                "dt_init_floor": 1e-4,
                            },
                            "gdn": {
                                "type": "gdn",
                                "value_heads": 4,
                                "key_heads": 2,
                                "key_head_dim": 16,
                                "value_head_dim": 16,
                                "convolution_layer": {"kernel_size": 4},
                            },
                        },
                    },
                    "mlp": {"type": "mlp", "intermediate_size": 256, "gated": True},
                    "normalization": {"type": "rms_norm", "epsilon": 1e-5},
                },
            },
        },
    )


@pytest.fixture
def apriel2_config_kda():
    """Apriel2 config with pure KDA (Kimi Delta Attention) layers.

    Tests KDA-specific cache behavior:
    - Tuple conv states (q, k, v) instead of single tensor
    - Recurrent state handling
    """
    from fast_llm_external_models.apriel2.configuration_apriel2 import Apriel2Config

    return Apriel2Config(
        vocab_size=100,
        hidden_size=64,
        decoder={
            "type": "fixed",
            "num_blocks": 2,
            "block": {
                "mixer": {
                    "type": "kda",
                    "heads": 4,
                    "head_dim": 16,
                    "convolution_layer": {"kernel_size": 4},
                    "normalization": {"epsilon": 1e-5},
                },
                "mlp": {"type": "mlp", "intermediate_size": 256, "gated": True},
                "normalization": {"type": "rms_norm", "epsilon": 1e-5},
            },
        },
    )


@pytest.fixture
def apriel2_config_all_mixers_with_kda():
    """Apriel2 config with all 5 mixer types including KDA.

    This config exercises:
    - All mixer types (attention, swa, mamba, gdn, kda)
    - KDA's tuple conv state handling in stochastic context
    - Cache isolation between all mixer types
    """
    from fast_llm_external_models.apriel2.configuration_apriel2 import Apriel2Config

    return Apriel2Config(
        vocab_size=100,
        hidden_size=64,
        decoder={
            "type": "pattern",
            "num_blocks": 2,
            "pattern": ["attn", "all_mixers"],
            "blocks": {
                "attn": {
                    "mixer": {
                        "type": "attention",
                        "heads": 4,
                        "head_groups": 2,
                        "head_size": 16,
                        "rotary": {"type": "mistral_1d", "theta": 10000.0},
                    },
                    "mlp": {"type": "mlp", "intermediate_size": 256, "gated": True},
                    "normalization": {"type": "rms_norm", "epsilon": 1e-5},
                },
                "all_mixers": {
                    "mixer": {
                        "type": "stochastic",
                        "main_mixer_name": "attention",
                        "mixers": {
                            "attention": {
                                "type": "attention",
                                "heads": 4,
                                "head_groups": 2,
                                "head_size": 16,
                                "rotary": {"type": "mistral_1d", "theta": 10000.0},
                            },
                            "swa": {
                                "type": "attention",
                                "heads": 4,
                                "head_groups": 2,
                                "head_size": 16,
                                "window_size": 2048,
                                "rotary": {"type": "mistral_1d", "theta": 1000000.0},
                            },
                            "mamba": {
                                "type": "mamba",
                                "d_inner": 128,
                                "d_state": 16,
                                "dt_rank": 4,
                                "d_xb": 32,
                                "d_conv": 4,
                                "repeat_kv_before_conv": True,
                                "conv_bias": True,
                                "dt_proj_bias": True,
                                "dt_min": 0.001,
                                "dt_max": 0.1,
                                "dt_init_floor": 1e-4,
                            },
                            "gdn": {
                                "type": "gdn",
                                "value_heads": 4,
                                "key_heads": 2,
                                "key_head_dim": 16,
                                "value_head_dim": 16,
                                "convolution_layer": {"kernel_size": 4},
                            },
                            "kda": {
                                "type": "kda",
                                "heads": 4,
                                "head_dim": 16,
                                "convolution_layer": {"kernel_size": 4},
                                "normalization": {"epsilon": 1e-5},
                            },
                        },
                    },
                    "mlp": {"type": "mlp", "intermediate_size": 256, "gated": True},
                    "normalization": {"type": "rms_norm", "epsilon": 1e-5},
                },
            },
        },
    )


@pytest.fixture
def apriel2_config_comprehensive():
    """Comprehensive Apriel2 config combining all features for thorough testing.

    This config exercises:
    - Pattern decoder with 6 different block types
    - Pure attention (full context)
    - Pure sliding window attention
    - Pure mamba
    - Pure gated delta net
    - Stochastic mixer: attention + mamba
    - Stochastic mixer: swa + gated_delta_net
    """
    from fast_llm_external_models.apriel2.configuration_apriel2 import Apriel2Config

    return Apriel2Config(
        vocab_size=100,
        hidden_size=64,
        decoder={
            "type": "pattern",
            "num_blocks": 6,
            "pattern": [
                "attn",          # 0: pure full attention
                "swa",           # 1: pure sliding window attention
                "mamba",         # 2: pure mamba
                "gdn",           # 3: pure gated delta net
                "stoch_attn_mamba",   # 4: stochastic attention + mamba
                "stoch_swa_gdn",      # 5: stochastic swa + gated delta net
            ],
            "blocks": {
                "attn": {
                    "mixer": {
                        "type": "attention",
                        "heads": 4,
                        "head_groups": 2,
                        "head_size": 16,
                        "rotary": {"type": "mistral_1d", "theta": 10000.0},
                    },
                    "mlp": {"type": "mlp", "intermediate_size": 256, "gated": True},
                    "normalization": {"type": "rms_norm", "epsilon": 1e-5},
                },
                "swa": {
                    "mixer": {
                        "type": "attention",
                        "heads": 4,
                        "head_groups": 2,
                        "head_size": 16,
                        "window_size": 512,
                        "rotary": {"type": "mistral_1d", "theta": 100000.0},
                    },
                    "mlp": {"type": "mlp", "intermediate_size": 256, "gated": True},
                    "normalization": {"type": "rms_norm", "epsilon": 1e-5},
                },
                "mamba": {
                    "mixer": {
                        "type": "mamba",
                        "d_inner": 128,
                        "d_state": 16,
                        "dt_rank": 4,
                        "d_xb": 16,
                        "d_conv": 4,
                        "repeat_kv_before_conv": True,
                        "conv_bias": True,
                        "dt_proj_bias": True,
                        "dt_min": 0.001,
                        "dt_max": 0.1,
                        "dt_init_floor": 1e-4,
                    },
                    "mlp": {"type": "mlp", "intermediate_size": 256, "gated": True},
                    "normalization": {"type": "rms_norm", "epsilon": 1e-5},
                },
                "gdn": {
                    "mixer": {
                        "type": "gdn",
                        "value_heads": 4,
                        "key_heads": 2,
                        "key_head_dim": 16,
                        "value_head_dim": 16,
                        "convolution_layer": {"kernel_size": 4},
                    },
                    "mlp": {"type": "mlp", "intermediate_size": 256, "gated": True},
                    "normalization": {"type": "rms_norm", "epsilon": 1e-5},
                },
                "stoch_attn_mamba": {
                    "mixer": {
                        "type": "stochastic",
                        "main_mixer_name": "attention",
                        "mixers": {
                            "attention": {
                                "type": "attention",
                                "heads": 4,
                                "head_groups": 2,
                                "head_size": 16,
                                "rotary": {"type": "mistral_1d", "theta": 10000.0},
                            },
                            "mamba": {
                                "type": "mamba",
                                "d_inner": 128,
                                "d_state": 16,
                                "dt_rank": 4,
                                "d_xb": 16,
                                "d_conv": 4,
                                "repeat_kv_before_conv": True,
                                "conv_bias": True,
                                "dt_proj_bias": True,
                                "dt_min": 0.001,
                                "dt_max": 0.1,
                                "dt_init_floor": 1e-4,
                            },
                        },
                    },
                    "mlp": {"type": "mlp", "intermediate_size": 256, "gated": True},
                    "normalization": {"type": "rms_norm", "epsilon": 1e-5},
                },
                "stoch_swa_gdn": {
                    "mixer": {
                        "type": "stochastic",
                        "main_mixer_name": "swa",
                        "mixers": {
                            "swa": {
                                "type": "attention",
                                "heads": 4,
                                "head_groups": 2,
                                "head_size": 16,
                                "window_size": 256,
                                "rotary": {"type": "mistral_1d", "theta": 500000.0},
                            },
                            "gdn": {
                                "type": "gdn",
                                "value_heads": 4,
                                "key_heads": 2,
                                "key_head_dim": 16,
                                "value_head_dim": 16,
                                "convolution_layer": {"kernel_size": 4},
                            },
                        },
                    },
                    "mlp": {"type": "mlp", "intermediate_size": 256, "gated": True},
                    "normalization": {"type": "rms_norm", "epsilon": 1e-5},
                },
            },
        },
    )


@pytest.fixture
def apriel2_cache(apriel2_config_tiny):
    """Create empty Apriel2Cache from tiny config."""
    from fast_llm_external_models.apriel2.cache import Apriel2Cache

    return Apriel2Cache(apriel2_config_tiny)


@pytest.fixture
def sample_input_ids():
    """Sample input token IDs for testing."""
    return torch.randint(0, 100, (2, 10))  # batch_size=2, seq_len=10


@pytest.fixture
def sample_attention_states():
    """Sample attention key/value states for cache testing."""
    batch_size, num_heads, seq_len, head_dim = 2, 8, 10, 64
    key = torch.randn(batch_size, num_heads, seq_len, head_dim)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim)
    return key, value


@pytest.fixture
def sample_ssm_states():
    """Sample SSM conv/recurrent states for cache testing."""
    batch_size, d_inner, d_conv = 2, 128, 4
    conv = torch.randn(batch_size, d_inner, d_conv)
    recurrent = torch.randn(batch_size, d_inner, 16)  # d_state=16
    return conv, recurrent


# =============================================================================
# Surgery Chain Fixtures
# =============================================================================


@pytest.fixture
def additive_surgery_chain():
    """Additive-only surgery chain that composes cleanly.

    This chain exercises:
    - Non-stochastic → stochastic transition
    - Adding multiple mixer types (attention, sliding_window, GDN)
    - Weight transfer via init: transfer

    S1: attention → stochastic{attention}
    S2: add sliding_window to stochastic
    S3: add gated_delta_net to stochastic (DIL derivation)
    """
    return [
        # S1: Convert to stochastic with attention
        {
            "decoder": {
                "block": {
                    "mixer": {
                        "type": "stochastic",
                        "main_mixer_name": "attention",
                        "mixers": {
                            "attention": {"init": "transfer"},
                        },
                    },
                    "mlp": {"init": "transfer"},
                    "normalization": {"init": "transfer"},
                },
            },
        },
        # S2: Add sliding_window
        {
            "decoder": {
                "block": {
                    "mixer": {
                        "mixers": {
                            "sliding_window": {
                                "type": "attention",
                                "init": "transfer",
                                "window_size": 512,
                            },
                        },
                    },
                },
            },
        },
        # S3: Add gated_delta_net (DIL)
        {
            "decoder": {
                "block": {
                    "mixer": {
                        "mixers": {
                            "gdn": {
                                "type": "gdn",
                                "init": "transfer",
                                "convolution_layer": {"kernel_size": 4},
                            },
                        },
                    },
                },
            },
        },
    ]


@pytest.fixture
def comprehensive_torture_chain():
    """Comprehensive torture chain exercising ALL conversion paths.

    This is the REAL stress test. It exercises:
    - Fixed → Pattern decoder transitions
    - Per-layer heterogeneity
    - All type conversions: FA ↔ SWA ↔ Mamba ↔ GDN ↔ KDA
    - Stochastic wrapping/unwrapping
    - Both init: transfer and init: random
    - Destructive operations (remove sub-mixers, collapse stochastic)

    The model has 5 layers. Each step changes the architecture significantly.
    """
    # Mamba params - dimensions must be compatible with MIL conversion!
    # Source attention: heads=8, head_groups=4, head_size=32, hidden_size=256
    # - Q has shape [heads*head_size, hidden_size] = [256, 256]
    # - K has shape [head_groups*head_size, hidden_size] = [128, 256]
    # - V has shape [head_groups*head_size, hidden_size] = [128, 256]
    # MIL requires: d_inner <= Q rows (256), d_xb <= K/V rows (128)
    mamba_params = {
        "d_inner": 256,  # Must be <= heads*head_size = 256
        "d_xb": 64,      # Must be <= head_groups*head_size = 128
        "dt_rank": 16,
        "d_state": 16,
        "d_conv": 4,
        "repeat_kv_before_conv": True,
        "conv_bias": True,
        "dt_proj_bias": True,
        "dt_min": 0.001,
        "dt_max": 0.1,
        "dt_init_floor": 1e-4,
    }

    # Rotary config for attention mixers that can't inherit from source
    # (e.g., init: random, or cross-type from mamba/gdn)
    rotary_config = {"type": "mistral_1d", "theta": 10000.0}

    return [
        # =====================================================================
        # STEP 1: Fixed attention → Pattern with FA/SWA alternating
        # Layers: [attn, swa, attn, swa, attn]
        # =====================================================================
        {
            "decoder": {
                "type": "pattern",
                "pattern": ["attn", "swa", "attn", "swa", "attn"],
                "blocks": {
                    "attn": {
                        "mixer": {"type": "attention", "init": "transfer"},
                        "mlp": {"init": "transfer"},
                        "normalization": {"init": "transfer"},
                    },
                    "swa": {
                        "mixer": {
                            "type": "attention",
                            "init": "transfer",
                            "window_size": 512,
                        },
                        "mlp": {"init": "transfer"},
                        "normalization": {"init": "transfer"},
                    },
                },
            },
        },
        # =====================================================================
        # STEP 2: Add stochastic wrappers with MIL/DIL/KIL conversions
        # Layer 0: stochastic{attn, mamba:MIL}
        # Layer 1: swa (unchanged)
        # Layer 2: stochastic{attn, gdn:DIL}
        # Layer 3: swa (unchanged)
        # Layer 4: stochastic{attn, kda:KIL}
        # =====================================================================
        {
            "decoder": {
                "type": "pattern",
                "pattern": ["stoch_am", "swa", "stoch_ag", "swa", "stoch_ak"],
                "blocks": {
                    "stoch_am": {
                        "mixer": {
                            "type": "stochastic",
                            "main_mixer_name": "attention",
                            "mixers": {
                                "attention": {"type": "attention", "init": "transfer"},
                                "mamba": {
                                    "type": "mamba",
                                    "init": "transfer",  # MIL conversion
                                    **mamba_params,
                                },
                            },
                        },
                        "mlp": {"init": "transfer"},
                        "normalization": {"init": "transfer"},
                    },
                    "swa": {
                        "mixer": {
                            "type": "attention",
                            "init": "transfer",
                            "window_size": 512,
                        },
                        "mlp": {"init": "transfer"},
                        "normalization": {"init": "transfer"},
                    },
                    "stoch_ag": {
                        "mixer": {
                            "type": "stochastic",
                            "main_mixer_name": "attention",
                            "mixers": {
                                "attention": {"type": "attention", "init": "transfer"},
                                "gdn": {
                                    "type": "gdn",
                                    "init": "transfer",  # DIL conversion
                                    "convolution_layer": {"kernel_size": 4},
                                },
                            },
                        },
                        "mlp": {"init": "transfer"},
                        "normalization": {"init": "transfer"},
                    },
                    "stoch_ak": {
                        "mixer": {
                            "type": "stochastic",
                            "main_mixer_name": "attention",
                            "mixers": {
                                "attention": {"type": "attention", "init": "transfer"},
                                "kda": {
                                    "type": "kda",
                                    "init": "transfer",  # KIL conversion
                                    "convolution_layer": {"kernel_size": 4},
                                    "normalization": {"epsilon": 1e-5},
                                },
                            },
                        },
                        "mlp": {"init": "transfer"},
                        "normalization": {"init": "transfer"},
                    },
                },
            },
        },
        # =====================================================================
        # STEP 3: Convert pure mixers to different types (MIL/DIL from SWA)
        # Layer 0: stoch{attn, mamba} (unchanged)
        # Layer 1: mamba (MIL from swa!)
        # Layer 2: stoch{attn, gdn} (unchanged)
        # Layer 3: gdn (DIL from swa!)
        # Layer 4: stoch{attn, kda} (unchanged)
        # =====================================================================
        {
            "decoder": {
                "type": "pattern",
                "pattern": ["stoch_am", "mamba", "stoch_ag", "gdn", "stoch_ak"],
                "blocks": {
                    "stoch_am": {
                        "mixer": {
                            "type": "stochastic",
                            "main_mixer_name": "attention",
                            "mixers": {
                                "attention": {"type": "attention", "init": "transfer"},
                                "mamba": {"type": "mamba", "init": "transfer", **mamba_params},
                            },
                        },
                        "mlp": {"init": "transfer"},
                        "normalization": {"init": "transfer"},
                    },
                    "mamba": {
                        "mixer": {
                            "type": "mamba",
                            "init": "transfer",  # MIL from previous swa
                            **mamba_params,
                        },
                        "mlp": {"init": "transfer"},
                        "normalization": {"init": "transfer"},
                    },
                    "stoch_ag": {
                        "mixer": {
                            "type": "stochastic",
                            "main_mixer_name": "attention",
                            "mixers": {
                                "attention": {"type": "attention", "init": "transfer"},
                                "gdn": {
                                    "type": "gdn",
                                    "init": "transfer",
                                    "convolution_layer": {"kernel_size": 4},
                                },
                            },
                        },
                        "mlp": {"init": "transfer"},
                        "normalization": {"init": "transfer"},
                    },
                    "gdn": {
                        "mixer": {
                            "type": "gdn",
                            "init": "transfer",  # DIL from previous swa
                            "convolution_layer": {"kernel_size": 4},
                        },
                        "mlp": {"init": "transfer"},
                        "normalization": {"init": "transfer"},
                    },
                    "stoch_ak": {
                        "mixer": {
                            "type": "stochastic",
                            "main_mixer_name": "attention",
                            "mixers": {
                                "attention": {"type": "attention", "init": "transfer"},
                                "kda": {
                                    "type": "kda",
                                    "init": "transfer",
                                    "convolution_layer": {"kernel_size": 4},
                                    "normalization": {"epsilon": 1e-5},
                                },
                            },
                        },
                        "mlp": {"init": "transfer"},
                        "normalization": {"init": "transfer"},
                    },
                },
            },
        },
        # =====================================================================
        # STEP 4: Add random-init sub-mixers to stochastic blocks
        # Layer 0: stoch{attn, mamba, swa:RANDOM}
        # Layer 1: mamba (unchanged)
        # Layer 2: stoch{attn, gdn, mamba:RANDOM}
        # Layer 3: gdn (unchanged)
        # Layer 4: stoch{attn, kda, swa:RANDOM} (add swa to existing stoch_ak)
        # =====================================================================
        {
            "decoder": {
                "type": "pattern",
                "pattern": ["stoch_ams", "mamba", "stoch_agm", "gdn", "stoch_aks"],
                "blocks": {
                    "stoch_ams": {
                        "mixer": {
                            "type": "stochastic",
                            "main_mixer_name": "attention",
                            "mixers": {
                                "attention": {"type": "attention", "init": "transfer"},
                                "mamba": {"type": "mamba", "init": "transfer", **mamba_params},
                                "swa": {
                                    "type": "attention",
                                    "init": "random",  # Random init!
                                    "heads": 8,
                                    "head_groups": 4,
                                    "head_size": 32,
                                    "window_size": 256,
                                    "rotary": rotary_config,
                                },
                            },
                        },
                        "mlp": {"init": "transfer"},
                        "normalization": {"init": "transfer"},
                    },
                    "mamba": {
                        "mixer": {"type": "mamba", "init": "transfer", **mamba_params},
                        "mlp": {"init": "transfer"},
                        "normalization": {"init": "transfer"},
                    },
                    "stoch_agm": {
                        "mixer": {
                            "type": "stochastic",
                            "main_mixer_name": "attention",
                            "mixers": {
                                "attention": {"type": "attention", "init": "transfer"},
                                "gdn": {
                                    "type": "gdn",
                                    "init": "transfer",
                                    "convolution_layer": {"kernel_size": 4},
                                },
                                "mamba": {
                                    "type": "mamba",
                                    "init": "random",  # Random init!
                                    **mamba_params,
                                },
                            },
                        },
                        "mlp": {"init": "transfer"},
                        "normalization": {"init": "transfer"},
                    },
                    "gdn": {
                        "mixer": {
                            "type": "gdn",
                            "init": "transfer",
                            "convolution_layer": {"kernel_size": 4},
                        },
                        "mlp": {"init": "transfer"},
                        "normalization": {"init": "transfer"},
                    },
                    "stoch_aks": {
                        "mixer": {
                            "type": "stochastic",
                            "main_mixer_name": "attention",
                            "mixers": {
                                "attention": {"type": "attention", "init": "transfer"},
                                "kda": {
                                    "type": "kda",
                                    "init": "transfer",
                                    "convolution_layer": {"kernel_size": 4},
                                    "normalization": {"epsilon": 1e-5},
                                },
                                "swa": {
                                    "type": "attention",
                                    "init": "random",  # Random init!
                                    "heads": 8,
                                    "head_groups": 4,
                                    "head_size": 32,
                                    "window_size": 128,
                                    "rotary": rotary_config,
                                },
                            },
                        },
                        "mlp": {"init": "transfer"},
                        "normalization": {"init": "transfer"},
                    },
                },
            },
        },
        # =====================================================================
        # STEP 5: Destructive - collapse some stochastic, remove sub-mixers
        # Layer 0: stoch{mamba, swa} (REMOVE attention!)
        # Layer 1: attn (random init - type change from mamba!)
        # Layer 2: gdn (collapse stochastic, keep gdn)
        # Layer 3: swa (random init - type change from gdn!)
        # Layer 4: kda (collapse stochastic, keep kda - tests KDA passthrough)
        # =====================================================================
        {
            "decoder": {
                "type": "pattern",
                "pattern": ["stoch_ms", "attn", "gdn", "swa", "kda"],
                "blocks": {
                    "stoch_ms": {
                        "mixer": {
                            "type": "stochastic",
                            "main_mixer_name": "mamba",  # Changed main!
                            "mixers": {
                                # attention REMOVED (null deletion would be explicit)
                                "mamba": {"type": "mamba", "init": "transfer", **mamba_params},
                                "swa": {
                                    "type": "attention",
                                    "init": "transfer",  # Now transfer from previous
                                    "window_size": 256,
                                },
                            },
                        },
                        "mlp": {"init": "transfer"},
                        "normalization": {"init": "transfer"},
                    },
                    "attn": {
                        "mixer": {
                            "type": "attention",
                            "init": "random",  # Can't transfer from mamba!
                            "heads": 8,
                            "head_groups": 4,
                            "head_size": 32,
                            "rotary": rotary_config,
                        },
                        "mlp": {"init": "transfer"},
                        "normalization": {"init": "transfer"},
                    },
                    "gdn": {
                        "mixer": {
                            "type": "gdn",
                            "init": "transfer",  # Transfer from stoch's gdn
                            "convolution_layer": {"kernel_size": 4},
                        },
                        "mlp": {"init": "transfer"},
                        "normalization": {"init": "transfer"},
                    },
                    "swa": {
                        "mixer": {
                            "type": "attention",
                            "init": "random",  # Can't transfer from gdn!
                            "heads": 8,
                            "head_groups": 4,
                            "head_size": 32,
                            "window_size": 512,
                            "rotary": rotary_config,
                        },
                        "mlp": {"init": "transfer"},
                        "normalization": {"init": "transfer"},
                    },
                    "kda": {
                        "mixer": {
                            "type": "kda",
                            "init": "transfer",  # Transfer from stoch's kda
                            "convolution_layer": {"kernel_size": 4},
                            "normalization": {"epsilon": 1e-5},
                        },
                        "mlp": {"init": "transfer"},
                        "normalization": {"init": "transfer"},
                    },
                },
            },
        },
        # =====================================================================
        # STEP 6: Build supernet where possible, preserve incompatible layers
        # After step 5:
        #   Layer 0: stoch{mamba (main), swa}
        #   Layer 1: attention
        #   Layer 2: gdn
        #   Layer 3: swa
        #   Layer 4: kda
        # Layers 1,3 have attention-based sources → can MIL/DIL/KIL to full supernet
        # Layers 0,2,4 have mamba/gdn/kda sources → keep structure, just transfer
        # =====================================================================
        {
            "decoder": {
                "type": "pattern",
                "pattern": ["stoch_ms", "supernet", "gdn", "supernet", "kda"],
                "blocks": {
                    "stoch_ms": {
                        # Layer 0: preserve stoch{mamba, swa}
                        "mixer": {
                            "type": "stochastic",
                            "main_mixer_name": "mamba",
                            "mixers": {
                                "mamba": {"type": "mamba", "init": "transfer", **mamba_params},
                                "swa": {
                                    "type": "attention",
                                    "init": "transfer",
                                    "window_size": 256,
                                },
                            },
                        },
                        "mlp": {"init": "transfer"},
                        "normalization": {"init": "transfer"},
                    },
                    "gdn": {
                        # Layer 2: preserve pure gdn
                        "mixer": {
                            "type": "gdn",
                            "init": "transfer",
                            "convolution_layer": {"kernel_size": 4},
                        },
                        "mlp": {"init": "transfer"},
                        "normalization": {"init": "transfer"},
                    },
                    "supernet": {
                        # Layers 1,3: full supernet via MIL/DIL/KIL from attention
                        # NOTE: Explicit geometry required because this is a NEW block
                        # and the default base (stoch_ms) is mamba-based, so geometry
                        # can't be derived via cross-type composition.
                        "mixer": {
                            "type": "stochastic",
                            "main_mixer_name": "attention",
                            "mixers": {
                                "attention": {
                                    "type": "attention",
                                    "init": "transfer",
                                    "heads": 8,
                                    "head_groups": 4,
                                    "head_size": 32,
                                    "rotary": rotary_config,
                                },
                                "swa": {
                                    "type": "attention",
                                    "init": "transfer",
                                    "heads": 8,
                                    "head_groups": 4,
                                    "head_size": 32,
                                    "window_size": 512,
                                    "rotary": rotary_config,
                                },
                                "mamba": {"type": "mamba", "init": "transfer", **mamba_params},
                                "gdn": {
                                    "type": "gdn",
                                    "init": "transfer",
                                    "value_heads": 8,
                                    "key_heads": 4,
                                    "key_head_dim": 32,
                                    "value_head_dim": 32,
                                    "convolution_layer": {"kernel_size": 4},
                                },
                                "kda": {
                                    "type": "kda",
                                    "init": "transfer",  # KIL conversion
                                    "heads": 8,
                                    "head_dim": 32,
                                    "convolution_layer": {"kernel_size": 4},
                                    "normalization": {"epsilon": 1e-5},
                                },
                            },
                        },
                        "mlp": {"init": "transfer"},
                        "normalization": {"init": "transfer"},
                    },
                    "kda": {
                        # Layer 4: preserve pure kda
                        "mixer": {
                            "type": "kda",
                            "init": "transfer",
                            "convolution_layer": {"kernel_size": 4},
                            "normalization": {"epsilon": 1e-5},
                        },
                        "mlp": {"init": "transfer"},
                        "normalization": {"init": "transfer"},
                    },
                },
            },
        },
    ]


@pytest.fixture
def torture_surgery_chain():
    """Full 11-step torture chain for testing config composition.

    This chain exercises:
    - Non-stochastic → stochastic → non-stochastic → stochastic transitions
    - Accumulating mixers in stochastic wrappers
    - Cross-type derivations (attention → GDN, attention → mamba)
    - Partial rotary config override (theta only)
    - Top-level scalar overrides

    Note: Steps S7-S11 involve "destructive" operations that break
    the compatibility law for config composition.
    """
    return [
        # S1: attention → stochastic{attention}
        {
            "decoder": {
                "block": {
                    "mixer": {
                        "type": "stochastic",
                        "main_mixer_name": "attention",
                        "mixers": {
                            "attention": {"init": "transfer"},
                        },
                    },
                },
            },
        },
        # S2: add sliding_window to stochastic
        {
            "decoder": {
                "block": {
                    "mixer": {
                        "mixers": {
                            "sliding_window": {"init": "transfer", "sliding_window": 2048},
                        },
                    },
                },
            },
        },
        # S3: change rotary theta on sliding_window (tests partial rotary config override)
        {
            "decoder": {
                "block": {
                    "mixer": {
                        "mixers": {
                            "sliding_window": {"rotary": {"theta": 500000.0}},
                        },
                    },
                },
            },
        },
        # S4: add gated_delta_net to stochastic (DIL derivation)
        {
            "decoder": {
                "block": {
                    "mixer": {
                        "mixers": {
                            "gdn": {
                                "type": "gdn",
                                "init": "transfer",
                                "convolution_layer": {"kernel_size": 4},
                            },
                        },
                    },
                },
            },
        },
        # S5: change main_mixer_name + add sampling_strategy
        {
            "decoder": {
                "block": {
                    "mixer": {
                        "main_mixer_name": "sliding_window",
                        "sampling_strategy": "weighted",
                    },
                },
            },
        },
        # S6: add mamba (now 4 mixers!)
        {
            "decoder": {
                "block": {
                    "mixer": {
                        "mixers": {
                            "mamba": {
                                "type": "mamba",
                                "init": "transfer",
                                "d_state": 64,
                                "d_conv": 4,
                            },
                        },
                    },
                },
            },
        },
        # S7: collapse to plain sliding_window (non-stochastic) - DESTRUCTIVE
        {
            "decoder": {
                "block": {
                    "mixer": {
                        "type": "attention",
                        "init": "transfer",
                        "window_size": 4096,
                    },
                },
            },
        },
        # S8: convert to gated_delta_net (DIL derivation from current attention)
        {
            "decoder": {
                "block": {
                    "mixer": {
                        "type": "gdn",
                        "init": "transfer",
                        "convolution_layer": {"kernel_size": 8},
                    },
                },
            },
        },
        # S9: wrap in stochastic{gdn, attention}
        # NOTE: attention uses explicit geometry (init: random) because
        # the current mixer is GDN - can't derive attention from GDN.
        {
            "decoder": {
                "block": {
                    "mixer": {
                        "type": "stochastic",
                        "main_mixer_name": "gdn",
                        "mixers": {
                            "gdn": {"init": "transfer"},
                            "attention": {
                                "type": "attention",
                                "init": "random",
                                "heads": 16,
                                "head_groups": 4,
                                "head_size": 32,
                                "rotary": {"type": "mistral_1d", "theta": 10000.0},
                            },
                        },
                    },
                },
            },
        },
        # S10: override vocab_size (top-level scalar)
        {
            "vocab_size": 50000,
        },
        # S11: add mamba to current stochastic
        {
            "decoder": {
                "block": {
                    "mixer": {
                        "mixers": {
                            "mamba": {
                                "type": "mamba",
                                "init": "transfer",
                                "d_state": 128,
                                "d_conv": 8,
                            },
                        },
                    },
                },
            },
        },
    ]
