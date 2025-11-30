"""Test fixtures for Apriel2 model tests."""

from pathlib import Path
from typing import Generator

import pytest
import torch
from transformers import LlavaConfig, LlavaForConditionalGeneration, MistralConfig

# Apriel 1.5 model ID on HuggingFace
APRIEL_1_5_MODEL_ID = "ServiceNow-AI/Apriel-1.5-15b-Thinker"


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow (requires large model download)"
    )


@pytest.fixture(autouse=True)
def set_default_device():
    """Set default device to CUDA for all tests (Mamba requires CUDA)."""
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
        yield
        torch.set_default_device("cpu")
    else:
        yield


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
    )

    return LlavaForConditionalGeneration(config)


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


@pytest.fixture
def apriel_1_5_config() -> dict:
    """Download and return the Apriel 1.5 config from HuggingFace.

    This is lightweight - only downloads config.json, not the weights.
    """
    import json

    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(APRIEL_1_5_MODEL_ID, "config.json")
    with open(config_path) as f:
        return json.load(f)


@pytest.fixture
def apriel_1_5_checkpoint() -> str:
    """Return the HuggingFace model ID for Apriel 1.5.

    This fixture returns the model ID (not a local path). The converter
    can accept either a local path or an HF model ID.

    Tests using this fixture should be marked with @pytest.mark.slow
    to skip by default (run with: pytest -m slow).
    """
    return APRIEL_1_5_MODEL_ID


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
                },
                "mlp": {"type": "mlp", "intermediate_size": 256},
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
                    },
                    "mlp": {"type": "mlp", "intermediate_size": 256},
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
                                "sliding_window": 4096,
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
                    "mlp": {"type": "mlp", "intermediate_size": 256},
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
                                "sliding_window": 2048,
                            },
                            "attn_large": {
                                "type": "attention",
                                "heads": 4,
                                "head_groups": 2,
                                "head_size": 16,
                                "sliding_window": 8192,
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
                    "mlp": {"type": "mlp", "intermediate_size": 256},
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
                    },
                    "mlp": {"type": "mlp", "intermediate_size": 256},
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
                            },
                            "swa": {
                                "type": "attention",
                                "heads": 4,
                                "head_groups": 2,
                                "head_size": 16,
                                "sliding_window": 2048,
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
                            "gated_delta_net": {
                                "type": "gated_delta_net",
                            },
                        },
                    },
                    "mlp": {"type": "mlp", "intermediate_size": 256},
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
                    },
                    "mlp": {"type": "mlp", "intermediate_size": 256},
                    "normalization": {"type": "rms_norm", "epsilon": 1e-5},
                },
                "swa": {
                    "mixer": {
                        "type": "attention",
                        "heads": 4,
                        "head_groups": 2,
                        "head_size": 16,
                        "sliding_window": 512,
                    },
                    "mlp": {"type": "mlp", "intermediate_size": 256},
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
                    "mlp": {"type": "mlp", "intermediate_size": 256},
                    "normalization": {"type": "rms_norm", "epsilon": 1e-5},
                },
                "gdn": {
                    "mixer": {
                        "type": "gated_delta_net",
                        "num_value_heads": 4,
                        "num_key_heads": 2,
                        "key_head_dim": 16,
                        "value_head_dim": 16,
                        "conv_kernel_size": 4,
                    },
                    "mlp": {"type": "mlp", "intermediate_size": 256},
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
                    "mlp": {"type": "mlp", "intermediate_size": 256},
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
                                "sliding_window": 256,
                            },
                            "gated_delta_net": {
                                "type": "gated_delta_net",
                                "num_value_heads": 4,
                                "num_key_heads": 2,
                                "key_head_dim": 16,
                                "value_head_dim": 16,
                                "conv_kernel_size": 4,
                            },
                        },
                    },
                    "mlp": {"type": "mlp", "intermediate_size": 256},
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
