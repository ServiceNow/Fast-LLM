"""Integration tests for Qwen2 -> Apriel2 -> Fast-LLM conversion pipeline.

Tests verify the full conversion chain:
1. Qwen2 -> Apriel2 (external module conversion)
2. Apriel2 + Surgery -> Supernet (stochastic mixer creation)
3. Supernet -> Fast-LLM -> Supernet (roundtrip through training format)

Test Strategy:
- Use real HuggingFace model (Qwen2.5-0.5B) for meaningful validation
- Separate config preservation tests from numerical equivalence tests
- Parameterize both conversion stages AND input variations
- Single test implementation applied across all stages
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from fast_llm_external_models.apriel2.configuration_apriel2 import Apriel2Config
from fast_llm_external_models.apriel2.modeling_apriel2 import Apriel2ForCausalLM
from fast_llm_external_models.apriel2.conversion import (
    compose,
    compose_configs,
    execute,
    plan_surgery,
)
from fast_llm_external_models.apriel2.conversion.expr import W
from fast_llm_external_models.apriel2.conversion.qwen2.config import convert_config as convert_qwen2_config
from fast_llm_external_models.apriel2.conversion.qwen2.plan import plan_qwen2_to_apriel2

from .conftest import requires_fastllm


# =============================================================================
# Test Input Variations
# =============================================================================

TEST_INPUTS = pytest.mark.parametrize(
    "prompts,max_new_tokens",
    [
        pytest.param(["Hello world"], 10, id="single_short"),
        pytest.param(["Hi", "The quick brown fox jumps over the lazy dog"], 20, id="batch_varied"),
        pytest.param(["Once upon a time"], 50, id="long_generation"),
    ],
)


# =============================================================================
# Conversion Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def qwen2_source():
    """Load Qwen2.5-0.5B as the source/reference model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, trust_remote_code=True
    )
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return {
        "model": model,
        "tokenizer": tokenizer,
        "config_dict": config.to_dict(),
        "state_dict": model.state_dict(),
    }


@pytest.fixture(scope="module")
def apriel2_converted(qwen2_source):
    """Stage 1: Qwen2 -> Apriel2."""
    config_dict = convert_qwen2_config(qwen2_source["config_dict"])
    plan = plan_qwen2_to_apriel2(qwen2_source["config_dict"])
    weights = execute(plan, {W(k): v for k, v in qwen2_source["state_dict"].items()}, seed=42)

    config = Apriel2Config(**config_dict)
    model = Apriel2ForCausalLM(config)
    model.load_state_dict({str(k): v for k, v in weights.items()}, strict=False)
    model.eval()

    return {"model": model, "config_dict": config_dict, "plan": plan, "name": "Apriel2"}


@pytest.fixture(scope="module")
def supernet_converted(qwen2_source, apriel2_converted):
    """Stage 2: Apriel2 + Surgery -> Supernet."""
    surgery_spec = {
        "decoder": {
            "block": {
                "mixer": {
                    "type": "stochastic",
                    "main_mixer_name": "attention",
                    "mixers": {
                        "attention": {"type": "attention", "init": "transfer"},
                        "sliding_window": {
                            "type": "attention",
                            "init": "transfer",
                            "window_size": 4096,
                        },
                    },
                },
            },
        },
    }

    apriel_config = apriel2_converted["config_dict"]
    supernet_config = compose_configs(apriel_config, surgery_spec)

    full_plan = compose(
        apriel2_converted["plan"],
        plan_surgery(apriel_config, supernet_config),
    )

    weights = execute(full_plan, {W(k): v for k, v in qwen2_source["state_dict"].items()}, seed=42)

    config = Apriel2Config(**supernet_config)
    model = Apriel2ForCausalLM(config)
    model.load_state_dict({str(k): v for k, v in weights.items()}, strict=False)
    model.eval()

    return {"model": model, "config_dict": supernet_config, "name": "Supernet"}


@pytest.fixture(scope="module")
def roundtrip_converted(supernet_converted, qwen2_source):
    """Stage 3: Supernet -> Fast-LLM -> Supernet."""
    from fast_llm.engine.checkpoint.config import (
        CheckpointLoadConfig,
        CheckpointSaveConfig,
        FastLLMCheckpointFormat,
    )
    from fast_llm.engine.checkpoint.convert import ConvertConfig
    from fast_llm.models.gpt.config import GPTModelConfig
    from fast_llm.models.gpt.conversion.config import Apriel2TextCheckpointFormat

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        supernet_path = tmpdir / "supernet"
        fastllm_path = tmpdir / "fastllm"
        roundtrip_path = tmpdir / "roundtrip"

        supernet_converted["model"].save_pretrained(supernet_path)
        qwen2_source["tokenizer"].save_pretrained(supernet_path)

        ConvertConfig(
            model=GPTModelConfig,
            input=CheckpointLoadConfig(path=supernet_path, format=Apriel2TextCheckpointFormat),
            output=CheckpointSaveConfig(path=fastllm_path, format=FastLLMCheckpointFormat),
        ).run()

        ConvertConfig(
            model=GPTModelConfig,
            input=CheckpointLoadConfig(path=fastllm_path, format=FastLLMCheckpointFormat),
            output=CheckpointSaveConfig(path=roundtrip_path, format=Apriel2TextCheckpointFormat),
        ).run()

        model = Apriel2ForCausalLM.from_pretrained(roundtrip_path)
        model.eval()

        with open(roundtrip_path / "config.json") as f:
            config_dict = json.load(f)

        yield {"model": model, "config_dict": config_dict, "name": "Roundtrip"}


# =============================================================================
# Parameterized Fixture: All Conversion Stages
# =============================================================================


@pytest.fixture(params=["apriel2", "supernet", "roundtrip"])
def converted_model(request, apriel2_converted, supernet_converted, roundtrip_converted):
    """Parameterized fixture providing each conversion stage for testing.

    This allows a single test to run against all stages automatically.
    """
    if request.param == "roundtrip":
        pytest.importorskip("fast_llm")

    return {
        "apriel2": apriel2_converted,
        "supernet": supernet_converted,
        "roundtrip": roundtrip_converted,
    }[request.param]


# =============================================================================
# Config Preservation Tests
# =============================================================================


@pytest.mark.slow
class TestConfigPreservation:
    """Verify configs are correctly preserved through the conversion chain."""

    def test_apriel2_structure(self, qwen2_source, apriel2_converted):
        """Qwen2 -> Apriel2 preserves model dimensions."""
        qwen = qwen2_source["config_dict"]
        apriel = apriel2_converted["config_dict"]

        assert apriel["hidden_size"] == qwen["hidden_size"]
        assert apriel["vocab_size"] == qwen["vocab_size"]
        assert apriel["decoder"]["num_blocks"] == qwen["num_hidden_layers"]

    def test_apriel2_bias_pattern(self, apriel2_converted):
        """Qwen2 -> Apriel2 preserves Qwen-style bias (QKV yes, O no)."""
        mixer = apriel2_converted["config_dict"]["decoder"]["block"]["mixer"]

        assert mixer["query_layer"]["bias"]["enabled"] is True
        assert mixer["key_layer"]["bias"]["enabled"] is True
        assert mixer["value_layer"]["bias"]["enabled"] is True
        assert mixer["dense_layer"]["bias"]["enabled"] is False

    def test_supernet_structure(self, supernet_converted):
        """Surgery creates correct stochastic mixer structure."""
        mixer = supernet_converted["config_dict"]["decoder"]["block"]["mixer"]

        assert mixer["type"] == "stochastic"
        assert mixer["main_mixer_name"] == "attention"
        assert set(mixer["mixers"].keys()) == {"attention", "sliding_window"}

    def test_supernet_bias_inheritance(self, supernet_converted):
        """Submixers inherit bias settings from source."""
        mixer = supernet_converted["config_dict"]["decoder"]["block"]["mixer"]

        for name in ["attention", "sliding_window"]:
            assert mixer["mixers"][name]["query_layer"]["bias"]["enabled"] is True
            assert mixer["mixers"][name]["dense_layer"]["bias"]["enabled"] is False

    @requires_fastllm
    def test_roundtrip_structure(self, roundtrip_converted):
        """Fast-LLM roundtrip preserves stochastic mixer structure."""
        mixer = roundtrip_converted["config_dict"]["decoder"]["block"]["mixer"]

        assert mixer["type"] == "stochastic"
        assert mixer["main_mixer_name"] == "attention"
        assert set(mixer["mixers"].keys()) == {"attention", "sliding_window"}

    @requires_fastllm
    def test_roundtrip_bias_preservation(self, roundtrip_converted):
        """Fast-LLM roundtrip preserves per-layer bias settings."""
        mixer = roundtrip_converted["config_dict"]["decoder"]["block"]["mixer"]

        for name in ["attention", "sliding_window"]:
            assert mixer["mixers"][name]["query_layer"]["bias"]["enabled"] is True
            assert mixer["mixers"][name]["dense_layer"]["bias"]["enabled"] is False


# =============================================================================
# Numerical Equivalence Tests
# =============================================================================


@pytest.mark.slow
class TestNumericalEquivalence:
    """Verify all conversion stages produce numerically identical outputs.

    Uses parameterized fixtures to test all stages with all input variations,
    giving us 3 stages × 3 inputs = 9 test cases from a single test function.
    """

    @TEST_INPUTS
    def test_logits_match(self, qwen2_source, converted_model, prompts, max_new_tokens):
        """Converted model produces identical logits to source."""
        tokenizer = qwen2_source["tokenizer"]
        ref_model = qwen2_source["model"]
        test_model = converted_model["model"]
        stage = converted_model["name"]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        ref_device = next(ref_model.parameters()).device
        test_device = next(test_model.parameters()).device

        with torch.no_grad():
            ref_logits = ref_model(
                input_ids=inputs.input_ids.to(ref_device),
                attention_mask=inputs.attention_mask.to(ref_device),
            ).logits.cpu()

            test_logits = test_model(
                input_ids=inputs.input_ids.to(test_device),
                attention_mask=inputs.attention_mask.to(test_device),
            ).logits.cpu()

        max_diff = (ref_logits - test_logits).abs().max().item()
        assert torch.allclose(ref_logits, test_logits, rtol=1e-4, atol=1e-4), (
            f"{stage} logits mismatch: max diff = {max_diff:.6f}"
        )

    @TEST_INPUTS
    def test_generation_match(self, qwen2_source, converted_model, prompts, max_new_tokens):
        """Converted model produces identical generation to source."""
        tokenizer = qwen2_source["tokenizer"]
        ref_model = qwen2_source["model"]
        test_model = converted_model["model"]
        stage = converted_model["name"]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        ref_device = next(ref_model.parameters()).device
        test_device = next(test_model.parameters()).device

        with torch.no_grad():
            ref_gen = ref_model.generate(
                input_ids=inputs.input_ids.to(ref_device),
                attention_mask=inputs.attention_mask.to(ref_device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            ).cpu()

            test_gen = test_model.generate(
                input_ids=inputs.input_ids.to(test_device),
                attention_mask=inputs.attention_mask.to(test_device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            ).cpu()

        assert torch.equal(ref_gen, test_gen), (
            f"{stage} generation mismatch:\n"
            f"  Reference: {tokenizer.batch_decode(ref_gen, skip_special_tokens=True)}\n"
            f"  Test:      {tokenizer.batch_decode(test_gen, skip_special_tokens=True)}"
        )
