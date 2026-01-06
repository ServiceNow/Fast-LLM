"""Tests for compose_configs - config composition laws.

These tests verify the laws that compose_configs must satisfy:
1. IDENTITY:      compose_configs(config, {}) == config
2. ASSOCIATIVITY: compose_configs(compose_configs(A, B), C) == compose_configs(A, compose_configs(B, C))
3. OVERRIDE:      surgery values override source values (overlay wins)
4. INHERITANCE:   config params are inherited based on structure (not `init`)
5. CROSS-TYPE:    attention→gdn derives GDN dims from attention geometry
6. STOCHASTIC:    sub-mixers inherit from base mixer
7. NULL-DELETE:   setting a key to None removes it

Note: `init` is for WEIGHT handling only. Config inheritance is structural.
"""

import json
from functools import reduce
from pathlib import Path

import pytest
import yaml

from fast_llm_external_models.apriel2.configuration_apriel2 import Apriel2Config
from fast_llm_external_models.apriel2.conversion.config import compose_configs


class TestComposeConfigsLaws:
    """Test the fundamental laws of compose_configs."""

    @pytest.fixture
    def source_config(self):
        """A complete Apriel2 config (as would come from Llava conversion)."""
        return {
            "model_type": "apriel2",
            "architectures": ["Apriel2ForConditionalGeneration"],
            "hidden_size": 256,
            "vocab_size": 1000,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "tie_word_embeddings": False,
            "image_token_index": 100,
            "decoder": {
                "type": "fixed",
                "num_blocks": 4,
                "block": {
                    "mixer": {
                        "type": "attention",
                        "heads": 8,
                        "head_groups": 4,
                        "head_size": 32,
                        "rope_theta": 10000.0,
                    },
                    "mlp": {
                        "type": "mlp",
                        "intermediate_size": 512,
                    },
                    "normalization": {
                        "type": "rms_norm",
                        "epsilon": 1e-5,
                    },
                },
            },
            "vision_encoder": {
                "hidden_size": 128,
                "embeddings": {
                    "patch_height": 16,
                    "patch_width": 16,
                    "input_channels": 3,
                },
                "encoder": {
                    "num_blocks": 2,
                },
                "adapter": {
                    "add_linear_biases": True,
                },
            },
        }

    @pytest.mark.parametrize("empty_surgery", [{}, None])
    def test_identity(self, source_config, empty_surgery):
        """Law 1: compose_configs(config, empty) == config for empty in [{}, None]"""
        result = compose_configs(source_config, empty_surgery)
        assert result == source_config

    def test_override_explicit_values(self, source_config):
        """Law 3: Surgery values override source values."""
        surgery = {"hidden_size": 512, "vocab_size": 2000}
        result = compose_configs(source_config, surgery)

        assert result["hidden_size"] == 512
        assert result["vocab_size"] == 2000

    def test_same_type_inheritance(self, source_config):
        """Law 4: Same type inherits unspecified params via deep merge."""
        surgery = {
            "decoder": {
                "block": {
                    "mixer": {
                        "init": "transfer",  # For weight handling
                        "window_size": 512,  # Add this field
                    },
                },
            },
        }
        result = compose_configs(source_config, surgery)

        mixer = result["decoder"]["block"]["mixer"]
        assert mixer["type"] == "attention"  # Inherited
        assert mixer["heads"] == 8  # Inherited
        assert mixer["head_groups"] == 4  # Inherited
        assert mixer["head_size"] == 32  # Inherited
        assert mixer["rope_theta"] == 10000.0  # Inherited
        assert mixer["window_size"] == 512  # Added
        # init is preserved for plan_surgery to see (stripped only at final output)

    def test_cross_type_attention_to_gdn(self, source_config):
        """Law 5: attention→gdn derives GDN dims from attention geometry."""
        surgery = {
            "decoder": {
                "block": {
                    "mixer": {
                        "type": "gdn",
                        "init": "transfer",  # For weight handling
                        "convolution_layer": {"kernel_size": 4},
                    },
                },
            },
        }
        result = compose_configs(source_config, surgery)

        mixer = result["decoder"]["block"]["mixer"]
        assert mixer["type"] == "gdn"
        # Derived from source attention geometry
        assert mixer["value_heads"] == 8  # from heads
        assert mixer["key_heads"] == 4  # from head_groups
        assert mixer["key_head_dim"] == 32  # from head_size
        assert mixer["value_head_dim"] == 32  # from head_size
        assert mixer["convolution_layer"]["kernel_size"] == 4  # from surgery

    def test_cross_type_attention_to_mamba(self, source_config):
        """attention→mamba derives Mamba dims from hidden_size."""
        surgery = {
            "decoder": {
                "block": {
                    "mixer": {
                        "type": "mamba",
                        "init": "transfer",
                        "d_state": 64,
                        "d_conv": 4,
                    },
                },
            },
        }
        result = compose_configs(source_config, surgery)

        mixer = result["decoder"]["block"]["mixer"]
        assert mixer["type"] == "mamba"
        # Derived from hidden_size=256
        assert mixer["d_inner"] == 512  # 2 * hidden_size
        assert mixer["d_xb"] == 64  # hidden_size // 4
        assert mixer["dt_rank"] == 16  # hidden_size // 16
        # From surgery
        assert mixer["d_state"] == 64
        assert mixer["d_conv"] == 4

    def test_cross_type_attention_to_kda(self, source_config):
        """attention→kda derives KDA dims from attention geometry."""
        surgery = {
            "decoder": {
                "block": {
                    "mixer": {
                        "type": "kda",
                        "init": "transfer",
                        "convolution_layer": {"kernel_size": 4},
                        "normalization": {"epsilon": 1e-5},
                    },
                },
            },
        }
        result = compose_configs(source_config, surgery)

        mixer = result["decoder"]["block"]["mixer"]
        assert mixer["type"] == "kda"
        # Derived from source attention geometry
        assert mixer["heads"] == 8  # from heads
        assert mixer["head_dim"] == 32  # from head_size
        # From surgery
        assert mixer["convolution_layer"]["kernel_size"] == 4
        assert mixer["normalization"]["epsilon"] == 1e-5

    def test_stochastic_submixer_inheritance(self, source_config):
        """Law 6: Sub-mixers inherit from base mixer when wrapping in stochastic."""
        surgery = {
            "decoder": {
                "block": {
                    "mixer": {
                        "type": "stochastic",
                        "main_mixer_name": "attention",
                        "mixers": {
                            "attention": {"init": "transfer"},  # Inherits from source attention
                            "sliding_window": {"init": "transfer", "window_size": 512},
                            "gdn": {"type": "gdn", "init": "transfer", "convolution_layer": {"kernel_size": 4}},
                        },
                    },
                },
            },
        }
        result = compose_configs(source_config, surgery)

        mixers = result["decoder"]["block"]["mixer"]["mixers"]

        # Attention sub-mixer inherits from source
        assert mixers["attention"]["type"] == "attention"
        assert mixers["attention"]["heads"] == 8
        assert mixers["attention"]["head_groups"] == 4
        assert mixers["attention"]["head_size"] == 32
        assert mixers["attention"]["rope_theta"] == 10000.0

        # Sliding window inherits geometry, adds window_size
        assert mixers["sliding_window"]["type"] == "attention"
        assert mixers["sliding_window"]["heads"] == 8
        assert mixers["sliding_window"]["window_size"] == 512

        # GDN derives from source attention geometry
        assert mixers["gdn"]["type"] == "gdn"
        assert mixers["gdn"]["value_heads"] == 8
        assert mixers["gdn"]["key_heads"] == 4
        assert mixers["gdn"]["convolution_layer"]["kernel_size"] == 4

    def test_null_deletion(self, source_config):
        """Law 7: Null deletion removes keys."""
        surgery = {
            "vision_encoder": None,
        }
        result = compose_configs(source_config, surgery)

        assert "vision_encoder" not in result

    def test_init_preserved_for_plan_surgery(self, source_config):
        """Verify `init` keys are preserved so plan_surgery can see them.

        The `init` field controls weight initialization (transfer vs random).
        It's preserved through composition and only stripped at final output.
        """
        from fast_llm_external_models.apriel2.conversion.config import strip_init_fields

        surgery = {
            "decoder": {
                "block": {
                    "mixer": {
                        "type": "stochastic",
                        "main_mixer_name": "attention",
                        "mixers": {
                            "attention": {"init": "transfer"},
                            "gdn": {"type": "gdn", "init": "random", "convolution_layer": {"kernel_size": 4}},
                        },
                    },
                },
            },
        }
        result = compose_configs(source_config, surgery)

        # init is preserved in composed config
        mixers = result["decoder"]["block"]["mixer"]["mixers"]
        assert mixers["attention"].get("init") == "transfer"
        assert mixers["gdn"].get("init") == "random"

        # strip_init_fields removes them for final output
        stripped = strip_init_fields(result)
        assert "init" not in stripped["decoder"]["block"]["mixer"]["mixers"]["attention"]
        assert "init" not in stripped["decoder"]["block"]["mixer"]["mixers"]["gdn"]

    def test_init_random_still_inherits_config(self, source_config):
        """init: random is for weights only - config params still inherited."""
        surgery = {
            "decoder": {
                "block": {
                    "mixer": {
                        "init": "random",  # Random weights, but config inherited
                        "window_size": 512,
                    },
                },
            },
        }
        result = compose_configs(source_config, surgery)

        mixer = result["decoder"]["block"]["mixer"]
        # Config params inherited despite init: random
        assert mixer["heads"] == 8
        assert mixer["head_groups"] == 4
        assert mixer["window_size"] == 512

    # =========================================================================
    # Monoid Laws: compose_configs forms a monoid action on configs
    # =========================================================================

    def test_surgery_monoid_associativity(self):
        """MONOID: merge(merge(A, B), C) == merge(A, merge(B, C)) for partial configs."""
        surgery_a = {"decoder": {"block": {"mixer": {"type": "stochastic", "main_mixer_name": "attention"}}}}
        surgery_b = {"decoder": {"block": {"mixer": {"mixers": {"sliding_window": {"window_size": 512}}}}}}
        surgery_c = {"decoder": {"block": {"mixer": {"mixers": {"gdn": {"type": "gdn"}}}}}}

        # Left-associated: (A ∘ B) ∘ C
        ab_c = compose_configs(compose_configs(surgery_a, surgery_b), surgery_c)
        # Right-associated: A ∘ (B ∘ C)
        a_bc = compose_configs(surgery_a, compose_configs(surgery_b, surgery_c))

        assert ab_c == a_bc, "Surgery monoid should be associative"

    @pytest.mark.parametrize("num_surgeries", [2, 3])
    def test_monoid_action_compatibility(self, source_config, num_surgeries):
        """MONOID ACTION: apply(apply(c, A), B) == apply(c, merge(A, B))

        This is the key law: applying surgeries sequentially equals merging first.
        Parameterized to test with 2 and 3 surgeries.
        """
        surgeries = [
            {
                "decoder": {
                    "block": {
                        "mixer": {"type": "stochastic", "main_mixer_name": "attention", "mixers": {"attention": {}}}
                    }
                }
            },
            {"decoder": {"block": {"mixer": {"mixers": {"sliding_window": {"window_size": 512}}}}}},
            {"decoder": {"block": {"mixer": {"mixers": {"gdn": {"type": "gdn"}}}}}},
        ][:num_surgeries]

        # Sequential: ((c ⊳ A) ⊳ B) ⊳ ...
        result_sequential = source_config
        for s in surgeries:
            result_sequential = compose_configs(result_sequential, s)

        # Merged: c ⊳ (A ∘ B ∘ ...)
        merged = surgeries[0]
        for s in surgeries[1:]:
            merged = compose_configs(merged, s)
        result_merged = compose_configs(source_config, merged)

        assert result_sequential == result_merged, f"Monoid action compatibility failed for {num_surgeries} surgeries"


class TestBiasConfigInheritance:
    """Test per-layer bias inheritance through surgery composition.

    These tests verify that the per-layer bias configuration (mirroring Fast-LLM's
    AffineLinearConfig) is correctly inherited through surgery operations:
    - query_layer.bias.enabled, key_layer.bias.enabled, etc. for attention
    - layer_1.bias.enabled, layer_2.bias.enabled for MLP
    """

    @pytest.fixture
    def source_config_with_bias(self):
        """Source config with Qwen-style bias (QKV enabled, O disabled)."""
        return {
            "model_type": "apriel2",
            "architectures": ["Apriel2ForCausalLM"],
            "hidden_size": 256,
            "vocab_size": 1000,
            "decoder": {
                "type": "fixed",
                "num_blocks": 4,
                "block": {
                    "mixer": {
                        "type": "attention",
                        "heads": 8,
                        "head_groups": 4,
                        "head_size": 32,
                        "rotary": {"type": "mistral_1d", "theta": 10000.0},
                        # Qwen-style per-layer bias
                        "query_layer": {"bias": {"enabled": True}},
                        "key_layer": {"bias": {"enabled": True}},
                        "value_layer": {"bias": {"enabled": True}},
                        "dense_layer": {"bias": {"enabled": False}},
                    },
                    "mlp": {
                        "type": "mlp",
                        "intermediate_size": 512,
                        "gated": False,
                        # Per-layer MLP bias
                        "layer_1": {"bias": {"enabled": True}},
                        "layer_2": {"bias": {"enabled": False}},
                    },
                    "normalization": {"type": "rms_norm", "epsilon": 1e-5},
                },
            },
        }

    def test_same_type_inherits_attention_bias(self, source_config_with_bias):
        """Same-type surgery inherits per-layer attention bias settings."""
        surgery = {
            "decoder": {
                "block": {
                    "mixer": {
                        "window_size": 512,  # Add sliding window behavior
                    },
                },
            },
        }
        result = compose_configs(source_config_with_bias, surgery)

        mixer = result["decoder"]["block"]["mixer"]
        assert mixer["query_layer"]["bias"]["enabled"] is True
        assert mixer["key_layer"]["bias"]["enabled"] is True
        assert mixer["value_layer"]["bias"]["enabled"] is True
        assert mixer["dense_layer"]["bias"]["enabled"] is False

    def test_same_type_inherits_mlp_bias(self, source_config_with_bias):
        """Same-type surgery inherits per-layer MLP bias settings."""
        surgery = {
            "decoder": {
                "block": {
                    "mlp": {
                        "intermediate_size": 1024,  # Change size
                    },
                },
            },
        }
        result = compose_configs(source_config_with_bias, surgery)

        mlp = result["decoder"]["block"]["mlp"]
        assert mlp["layer_1"]["bias"]["enabled"] is True
        assert mlp["layer_2"]["bias"]["enabled"] is False
        assert mlp["intermediate_size"] == 1024

    def test_cross_type_attention_to_sliding_window_preserves_bias(self, source_config_with_bias):
        """attention→sliding_window cross-type preserves per-layer bias."""
        surgery = {
            "decoder": {
                "block": {
                    "mixer": {
                        "type": "sliding_window",  # Cross-type derivation
                        "window_size": 512,
                    },
                },
            },
        }
        result = compose_configs(source_config_with_bias, surgery)

        mixer = result["decoder"]["block"]["mixer"]
        assert mixer["type"] == "sliding_window"
        # Bias settings preserved through cross-type
        assert mixer["query_layer"]["bias"]["enabled"] is True
        assert mixer["key_layer"]["bias"]["enabled"] is True
        assert mixer["value_layer"]["bias"]["enabled"] is True
        assert mixer["dense_layer"]["bias"]["enabled"] is False

    def test_stochastic_wrapper_inherits_bias(self, source_config_with_bias):
        """Wrapping in stochastic inherits bias settings to all sub-mixers."""
        surgery = {
            "decoder": {
                "block": {
                    "mixer": {
                        "type": "stochastic",
                        "main_mixer_name": "attention",
                        "mixers": {
                            "attention": {"init": "transfer"},
                            "sliding_window": {
                                "type": "sliding_window",
                                "window_size": 512,
                                "init": "transfer",
                            },
                        },
                    },
                },
            },
        }
        result = compose_configs(source_config_with_bias, surgery)

        mixers = result["decoder"]["block"]["mixer"]["mixers"]

        # Attention sub-mixer inherits bias
        assert mixers["attention"]["query_layer"]["bias"]["enabled"] is True
        assert mixers["attention"]["dense_layer"]["bias"]["enabled"] is False

        # Sliding window sub-mixer also inherits bias
        assert mixers["sliding_window"]["query_layer"]["bias"]["enabled"] is True
        assert mixers["sliding_window"]["dense_layer"]["bias"]["enabled"] is False

    def test_surgery_can_override_bias(self, source_config_with_bias):
        """Surgery can explicitly override inherited bias settings."""
        surgery = {
            "decoder": {
                "block": {
                    "mixer": {
                        "dense_layer": {"bias": {"enabled": True}},  # Override O bias
                    },
                },
            },
        }
        result = compose_configs(source_config_with_bias, surgery)

        mixer = result["decoder"]["block"]["mixer"]
        # Q/K/V unchanged
        assert mixer["query_layer"]["bias"]["enabled"] is True
        # O bias overridden
        assert mixer["dense_layer"]["bias"]["enabled"] is True


class TestComposeConfigsRealYAML:
    """Test compose_configs with real YAML surgery files."""

    def test_stochastic_supernet_yaml(self, llava_pixtral_checkpoint):
        """Test that stochastic_supernet.yaml produces valid config."""
        from fast_llm_external_models.apriel2.conversion.llava import convert_config

        # Load source config and convert to Apriel2
        with open(llava_pixtral_checkpoint / "config.json") as f:
            llava_config = json.load(f)
        intermediate_config = convert_config(llava_config)

        # Load surgery YAML
        yaml_path = Path(__file__).parent.parent.parent / "apriel2" / "examples" / "stochastic_supernet.yaml"
        with open(yaml_path) as f:
            surgery_config = yaml.safe_load(f)

        # Compose
        result = compose_configs(intermediate_config, surgery_config)

        # Verify completeness
        assert "hidden_size" in result
        assert "vocab_size" in result
        assert "vision_encoder" in result
        assert result["decoder"]["num_blocks"] == intermediate_config["decoder"]["num_blocks"]

        # Verify stochastic mixer structure
        mixer = result["decoder"]["block"]["mixer"]
        assert mixer["type"] == "stochastic"
        assert "attention" in mixer["mixers"]
        assert "sliding_window" in mixer["mixers"]
        assert "gdn" in mixer["mixers"]

        # Verify sub-mixer configs are complete (inherited from source)
        attn = mixer["mixers"]["attention"]
        assert "heads" in attn
        assert "head_groups" in attn
        assert "head_size" in attn

        gdn = mixer["mixers"]["gdn"]
        assert "value_heads" in gdn
        assert "key_heads" in gdn
        assert "convolution_layer" in gdn

        # Should be instantiatable
        config = Apriel2Config(**result)
        assert config.hidden_size == intermediate_config["hidden_size"]

    def test_comprehensive_yaml(self, llava_pixtral_checkpoint):
        """Test that comprehensive.yaml produces valid config."""
        from fast_llm_external_models.apriel2.conversion.llava import convert_config

        # Load source config and convert to Apriel2
        with open(llava_pixtral_checkpoint / "config.json") as f:
            llava_config = json.load(f)
        intermediate_config = convert_config(llava_config)

        # Load surgery YAML
        yaml_path = Path(__file__).parent.parent.parent / "apriel2" / "examples" / "comprehensive.yaml"
        with open(yaml_path) as f:
            surgery_config = yaml.safe_load(f)

        # Compose
        result = compose_configs(intermediate_config, surgery_config)

        # Verify pattern decoder
        assert result["decoder"]["type"] == "pattern"
        assert "pattern" in result["decoder"]
        assert "blocks" in result["decoder"]

        # Should be instantiatable
        config = Apriel2Config(**result)
        assert config.decoder["type"] == "pattern"


class TestComposeConfigsEndToEnd:
    """Test the full conversion flow with compose_configs."""

    def test_build_plan_returns_complete_config(self, llava_pixtral_checkpoint):
        """Verify build_plan returns a complete, valid config when using YAML surgery."""
        from fast_llm_external_models.apriel2.convert import build_plan

        # Load source config
        with open(llava_pixtral_checkpoint / "config.json") as f:
            llava_config = json.load(f)

        # Load surgery YAML
        yaml_path = Path(__file__).parent.parent.parent / "apriel2" / "examples" / "stochastic_supernet.yaml"
        with open(yaml_path) as f:
            surgery_config = yaml.safe_load(f)

        # Build plan
        plan, final_config = build_plan(llava_config, [surgery_config])

        # The key test: final_config should be COMPLETE
        assert "hidden_size" in final_config
        assert "vocab_size" in final_config
        assert "vision_encoder" in final_config
        assert "bos_token_id" in final_config
        assert "eos_token_id" in final_config

        # Should be instantiatable
        config = Apriel2Config(**final_config)
        assert config.hidden_size > 0
        assert config.vocab_size > 0

        # Verify stochastic mixer is properly configured
        mixer = config.decoder["block"]["mixer"]
        assert mixer["type"] == "stochastic"

        # Each sub-mixer should have complete config
        # (init is preserved for plan_surgery, stripped only at final output)
        for name, sub_mixer in mixer["mixers"].items():
            assert "type" in sub_mixer


class TestCompositionTortureTest:
    """Comprehensive stress test for config composition.

    Tests the full 10-step surgery chain with proper `init` usage for weights.
    """

    @pytest.fixture
    def complete_config(self):
        """Starting point: complete Apriel2 config with attention mixer."""
        return {
            "model_type": "apriel2",
            "architectures": ["Apriel2ForConditionalGeneration"],
            "hidden_size": 512,
            "vocab_size": 32000,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "tie_word_embeddings": False,
            "image_token_index": 100,
            "decoder": {
                "type": "fixed",
                "num_blocks": 24,
                "block": {
                    "mixer": {
                        "type": "attention",
                        "heads": 16,
                        "head_groups": 4,
                        "head_size": 32,
                        "rope_theta": 10000.0,
                    },
                    "mlp": {"type": "mlp", "intermediate_size": 2048},
                    "normalization": {"type": "rms_norm", "epsilon": 1e-5},
                },
            },
        }

    def test_additive_chain_compatibility(self, complete_config, additive_surgery_chain):
        """Test compatibility law for additive surgery chain.

        apply(apply(c, A), B) == apply(c, merge(A, B))
        """
        # Sequential application
        result_seq = complete_config
        for surgery in additive_surgery_chain:
            result_seq = compose_configs(result_seq, surgery)

        # Merged application
        merged_surgery = reduce(compose_configs, additive_surgery_chain, {})
        result_merged = compose_configs(complete_config, merged_surgery)

        assert result_seq == result_merged, "Additive chain should satisfy compatibility"

    def test_every_prefix_compatibility(self, complete_config, additive_surgery_chain):
        """Test compatibility law for every prefix of the chain."""
        for k in range(1, len(additive_surgery_chain) + 1):
            prefix = additive_surgery_chain[:k]

            # Sequential
            result_seq = complete_config
            for surgery in prefix:
                result_seq = compose_configs(result_seq, surgery)

            # Merged
            merged_surgery = reduce(compose_configs, prefix, {})
            result_merged = compose_configs(complete_config, merged_surgery)

            assert result_seq == result_merged, f"Prefix of length {k} should satisfy compatibility"

    def test_intermediate_configs_are_valid(self, complete_config, additive_surgery_chain):
        """Every intermediate config should be instantiatable as Apriel2Config."""
        result = complete_config
        for i, surgery in enumerate(additive_surgery_chain):
            result = compose_configs(result, surgery)

            try:
                config = Apriel2Config(**result)
                assert config.hidden_size > 0
                assert config.vocab_size > 0
            except Exception as e:
                pytest.fail(f"Step {i+1} produced invalid config: {e}")

    def test_final_config_structure(self, complete_config, additive_surgery_chain):
        """Verify the final config has expected structure."""
        result = complete_config
        for surgery in additive_surgery_chain:
            result = compose_configs(result, surgery)

        # Mixer should be stochastic with 3 sub-mixers
        mixer = result["decoder"]["block"]["mixer"]
        assert mixer["type"] == "stochastic"
        assert mixer["main_mixer_name"] == "attention"
        assert set(mixer["mixers"].keys()) == {"attention", "sliding_window", "gdn"}

        # Sub-mixers should have inherited geometry
        assert mixer["mixers"]["attention"]["heads"] == 16
        assert mixer["mixers"]["sliding_window"]["heads"] == 16
        assert mixer["mixers"]["sliding_window"]["window_size"] == 512
        assert mixer["mixers"]["gdn"]["value_heads"] == 16

    def test_init_keys_preserved_for_planning(self, complete_config, additive_surgery_chain):
        """Verify 'init' keys are preserved for plan_surgery to see.

        The `init` field is metadata for weight initialization. It's preserved
        through composition and only stripped when saving final output.
        """
        from fast_llm_external_models.apriel2.conversion.config import strip_init_fields

        result = complete_config
        for i, surgery in enumerate(additive_surgery_chain):
            result = compose_configs(result, surgery)

        # init should be in the composed config
        mixer = result["decoder"]["block"]["mixer"]
        if "mixers" in mixer:
            has_init = any("init" in m for m in mixer["mixers"].values())
            assert has_init, "init should be preserved in composed config"

        # strip_init_fields removes them
        stripped = strip_init_fields(result)
        mixer = stripped["decoder"]["block"]["mixer"]
        if "mixers" in mixer:
            assert all("init" not in m for m in mixer["mixers"].values())

    def test_full_torture_chain(self, complete_config, torture_surgery_chain):
        """Test the full 10-step torture chain produces valid configs."""
        result = complete_config
        for i, surgery in enumerate(torture_surgery_chain):
            result = compose_configs(result, surgery)

            try:
                config = Apriel2Config(**result)
                assert config.hidden_size > 0
            except Exception as e:
                pytest.fail(f"Step {i+1} produced invalid config: {e}")

        # Verify final state
        assert result["vocab_size"] == 50000  # S9 changed this
        mixer = result["decoder"]["block"]["mixer"]
        assert mixer["type"] == "stochastic"
        assert "mamba" in mixer["mixers"]  # S10 added this
