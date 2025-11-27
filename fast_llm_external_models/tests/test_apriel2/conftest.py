"""Test fixtures for Apriel2 model tests."""

import pytest
import torch


@pytest.fixture
def apriel2_config_tiny():
    """Tiny Apriel2 config for fast testing."""
    from fast_llm_external_models.apriel2.configuration_apriel2 import Apriel2Config

    return Apriel2Config(
        vocab_size=100,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        decoder={
            "type": "fixed",
            "num_blocks": 2,
            "block": {
                "mixer": {"type": "attention"},
                "mlp": {"type": "mlp"},
                "normalization": {"type": "rms_norm"},
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
        num_attention_heads=4,
        num_key_value_heads=2,
        decoder={
            "type": "pattern",
            "num_blocks": 2,
            "pattern": ["attn", "stoch"],
            "blocks": {
                "attn": {"mixer": {"type": "attention"}},
                "stoch": {
                    "mixer": {
                        "type": "stochastic",
                        "main_mixer_name": "attention",
                        "mixers": {
                            "attention": {"type": "attention", "sliding_window": 4096},
                            "mamba": {
                                "type": "mamba",
                                "conv_bias": True,
                                "dt_proj_bias": True
                            }
                        }
                    }
                }
            }
        }
    )


@pytest.fixture
def apriel2_config_multi_mixer():
    """Apriel2 config with multiple mixers of same type."""
    from fast_llm_external_models.apriel2.configuration_apriel2 import Apriel2Config

    return Apriel2Config(
        vocab_size=100,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
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
                            "attn_small": {"type": "attention", "sliding_window": 2048},
                            "attn_large": {"type": "attention", "sliding_window": 8192},
                            "mamba_v1": {
                                "type": "mamba",
                                "conv_bias": True,
                                "dt_proj_bias": True
                            },
                            "mamba_v2": {
                                "type": "mamba",
                                "conv_bias": True,
                                "dt_proj_bias": True
                            }
                        }
                    }
                }
            }
        }
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
        num_attention_heads=4,
        num_key_value_heads=2,
        decoder={
            "type": "pattern",
            "num_blocks": 2,
            "pattern": ["attn", "all_mixers"],
            "blocks": {
                "attn": {"mixer": {"type": "attention"}},
                "all_mixers": {
                    "mixer": {
                        "type": "stochastic",
                        "main_mixer_name": "attention",
                        "mixers": {
                            "attention": {
                                "type": "attention"
                            },
                            "swa": {
                                "type": "attention",
                                "sliding_window": 2048
                            },
                            "mamba": {
                                "type": "mamba",
                                "conv_bias": True,
                                "dt_proj_bias": True
                            },
                            "gated_delta_net": {
                                "type": "gated_delta_net"
                            }
                        }
                    }
                }
            }
        }
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
