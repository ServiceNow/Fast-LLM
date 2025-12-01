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
from fast_llm_external_models.apriel2.conversion.config import apply_surgery, compose_configs


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
                "patch_convolution": {
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

    def test_identity_empty_surgery(self, source_config):
        """Law 1: compose_configs(config, {}) == config"""
        result = compose_configs(source_config, {})
        assert result == source_config

    def test_identity_none_surgery(self, source_config):
        """Law 1: compose_configs(config, None) == config"""
        result = compose_configs(source_config, None)
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
                        "sliding_window": 512,  # Add this field
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
        assert mixer["sliding_window"] == 512  # Added
        assert "init" not in mixer  # Stripped by apply_surgery

    def test_cross_type_attention_to_gdn(self, source_config):
        """Law 5: attention→gdn derives GDN dims from attention geometry."""
        surgery = {
            "decoder": {
                "block": {
                    "mixer": {
                        "type": "gated_delta_net",
                        "init": "transfer",  # For weight handling
                        "conv_kernel_size": 4,
                    },
                },
            },
        }
        result = compose_configs(source_config, surgery)

        mixer = result["decoder"]["block"]["mixer"]
        assert mixer["type"] == "gated_delta_net"
        # Derived from source attention geometry
        assert mixer["num_value_heads"] == 8  # from heads
        assert mixer["num_key_heads"] == 4  # from head_groups
        assert mixer["key_head_dim"] == 32  # from head_size
        assert mixer["value_head_dim"] == 32  # from head_size
        assert mixer["conv_kernel_size"] == 4  # from surgery

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
                            "sliding_window": {"init": "transfer", "sliding_window": 512},
                            "gdn": {"type": "gated_delta_net", "init": "transfer", "conv_kernel_size": 4},
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

        # Sliding window inherits geometry, adds sliding_window
        assert mixers["sliding_window"]["type"] == "attention"
        assert mixers["sliding_window"]["heads"] == 8
        assert mixers["sliding_window"]["sliding_window"] == 512

        # GDN derives from source attention geometry
        assert mixers["gdn"]["type"] == "gated_delta_net"
        assert mixers["gdn"]["num_value_heads"] == 8
        assert mixers["gdn"]["num_key_heads"] == 4
        assert mixers["gdn"]["conv_kernel_size"] == 4

    def test_null_deletion(self, source_config):
        """Law 7: Null deletion removes keys."""
        surgery = {
            "vision_encoder": None,
        }
        result = compose_configs(source_config, surgery)

        assert "vision_encoder" not in result

    def test_init_stripped_from_result(self, source_config):
        """Verify `init` keys are stripped from final result."""
        surgery = {
            "decoder": {
                "block": {
                    "mixer": {
                        "type": "stochastic",
                        "main_mixer_name": "attention",
                        "mixers": {
                            "attention": {"init": "transfer"},
                            "gdn": {"type": "gated_delta_net", "init": "random", "conv_kernel_size": 4},
                        },
                    },
                    "mlp": {"init": "transfer"},
                    "normalization": {"init": "transfer"},
                },
            },
        }
        result = compose_configs(source_config, surgery)

        def check_no_init(d, path=""):
            assert "init" not in d, f"Found 'init' key at {path}"
            for k, v in d.items():
                if isinstance(v, dict):
                    check_no_init(v, f"{path}.{k}")

        check_no_init(result)

    def test_init_random_still_inherits_config(self, source_config):
        """init: random is for weights only - config params still inherited."""
        surgery = {
            "decoder": {
                "block": {
                    "mixer": {
                        "init": "random",  # Random weights, but config inherited
                        "sliding_window": 512,
                    },
                },
            },
        }
        result = compose_configs(source_config, surgery)

        mixer = result["decoder"]["block"]["mixer"]
        # Config params inherited despite init: random
        assert mixer["heads"] == 8
        assert mixer["head_groups"] == 4
        assert mixer["sliding_window"] == 512


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
        assert "gated_delta_net" in mixer["mixers"]

        # Verify sub-mixer configs are complete (inherited from source)
        attn = mixer["mixers"]["attention"]
        assert "heads" in attn
        assert "head_groups" in attn
        assert "head_size" in attn

        gdn = mixer["mixers"]["gated_delta_net"]
        assert "num_value_heads" in gdn
        assert "num_key_heads" in gdn
        assert "conv_kernel_size" in gdn

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

        # Each sub-mixer should have complete config (no init keys)
        for name, sub_mixer in mixer["mixers"].items():
            assert "init" not in sub_mixer, f"Sub-mixer {name} still has 'init' key"
            assert "type" in sub_mixer


class TestMonoidLaws:
    """Test the algebraic laws of compose_configs.

    Surgery specs form a MONOID under deep-merge:
    - Identity: {}
    - Operation: deep merge (overlay wins)
    - Associativity: merge(merge(A, B), C) == merge(A, merge(B, C))

    compose_configs is a MONOID ACTION on configs:
    - Identity action: apply(config, {}) == config
    - Compatibility: apply(apply(c, A), B) == apply(c, merge(A, B))
    """

    @pytest.fixture
    def complete_config(self):
        """A complete Apriel2 config."""
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
                    "mlp": {"type": "mlp", "intermediate_size": 512},
                    "normalization": {"type": "rms_norm", "epsilon": 1e-5},
                },
            },
        }

    @pytest.fixture
    def surgery_a(self):
        """First surgery: wrap in stochastic with attention."""
        return {
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
        }

    @pytest.fixture
    def surgery_b(self):
        """Second surgery: add sliding window mixer."""
        return {
            "decoder": {
                "block": {
                    "mixer": {
                        "mixers": {
                            "sliding_window": {"init": "transfer", "sliding_window": 512},
                        },
                    },
                },
            },
        }

    def test_identity_action(self, complete_config):
        """apply(config, {}) == config"""
        result = compose_configs(complete_config, {})
        assert result == complete_config

    def test_surgery_monoid_associativity(self, surgery_a, surgery_b):
        """merge(merge(A, B), C) == merge(A, merge(B, C)) for partial configs."""
        surgery_c = {
            "decoder": {
                "block": {
                    "mixer": {
                        "mixers": {
                            "gdn": {"type": "gated_delta_net", "init": "transfer", "conv_kernel_size": 4},
                        },
                    },
                },
            },
        }

        # Left-associated: (A ∘ B) ∘ C
        ab = compose_configs(surgery_a, surgery_b)
        ab_c = compose_configs(ab, surgery_c)

        # Right-associated: A ∘ (B ∘ C)
        bc = compose_configs(surgery_b, surgery_c)
        a_bc = compose_configs(surgery_a, bc)

        assert ab_c == a_bc, "Surgery monoid should be associative"

    def test_monoid_action_compatibility(self, complete_config, surgery_a, surgery_b):
        """apply(apply(c, A), B) == apply(c, merge(A, B))

        This is the key law: applying surgeries sequentially should equal
        merging the surgeries first, then applying once.
        """
        # Sequential application: (c ⊳ A) ⊳ B
        result_sequential = compose_configs(compose_configs(complete_config, surgery_a), surgery_b)

        # Merged application: c ⊳ (A ∘ B)
        merged_surgery = compose_configs(surgery_a, surgery_b)
        result_merged = compose_configs(complete_config, merged_surgery)

        # These should be equivalent
        assert result_sequential == result_merged, "Monoid action should satisfy compatibility law"

    def test_three_way_compatibility(self, complete_config, surgery_a, surgery_b):
        """Test with three surgeries for stronger confidence."""
        surgery_c = {
            "decoder": {
                "block": {
                    "mixer": {
                        "mixers": {
                            "gdn": {"type": "gated_delta_net", "init": "transfer", "conv_kernel_size": 4},
                        },
                    },
                },
            },
        }

        # Sequential: ((c ⊳ A) ⊳ B) ⊳ C
        seq = compose_configs(
            compose_configs(compose_configs(complete_config, surgery_a), surgery_b),
            surgery_c
        )

        # Merged: c ⊳ ((A ∘ B) ∘ C)
        merged = compose_configs(
            complete_config,
            compose_configs(compose_configs(surgery_a, surgery_b), surgery_c)
        )

        assert seq == merged, "Three-way monoid action should satisfy compatibility"


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
        assert mixer["mixers"]["sliding_window"]["sliding_window"] == 512
        assert mixer["mixers"]["gdn"]["num_value_heads"] == 16

    def test_no_init_keys_in_result(self, complete_config, additive_surgery_chain):
        """Verify no 'init' keys leak through."""

        def check_no_init(d, path=""):
            if isinstance(d, dict):
                assert "init" not in d, f"Found 'init' key at {path}"
                for k, v in d.items():
                    check_no_init(v, f"{path}.{k}")

        result = complete_config
        for i, surgery in enumerate(additive_surgery_chain):
            result = compose_configs(result, surgery)
            check_no_init(result, f"step_{i+1}")

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
