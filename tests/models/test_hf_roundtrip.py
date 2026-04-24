"""Tests for HF → Fast-LLM → HF roundtrip conversion starting from HF format.

Complements test_checkpoint.py (which starts from distributed/Fast-LLM format).
Verifies that HF models survive a roundtrip with identical config and weights.

Model configs are derived from real HuggingFace configs (config-only download, no
weights) with dimensions shrunk for fast testing.
"""

import dataclasses
import pathlib
import typing

import pytest
import torch
from transformers import (
    AutoConfig,
    LlamaConfig,
    LlamaForCausalLM,
    MistralConfig,
    MistralForCausalLM,
    MixtralConfig,
    MixtralForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
    Qwen2Config,
    Qwen2ForCausalLM,
)

from fast_llm.engine.checkpoint.config import (
    CheckpointFormat,
    CheckpointLoadConfig,
    CheckpointSaveConfig,
    FastLLMCheckpointFormat,
)
from fast_llm.engine.checkpoint.convert import ConvertConfig
from fast_llm.models.gpt.config import GPTModelConfig
from fast_llm.models.gpt.conversion.config import (
    Apriel2TextCheckpointFormat,
    LlamaCheckpointFormat,
    MistralCheckpointFormat,
    MixtralCheckpointFormat,
    MTPLlamaCheckpointFormat,
    Qwen2CheckpointFormat,
)
from fast_llm.utils import check_equal_nested
from fast_llm_external_models.apriel2.configuration_apriel2 import Apriel2TextConfig
from fast_llm_external_models.apriel2.conversion import compose_configs
from fast_llm_external_models.apriel2.conversion.qwen2.config import convert_config as convert_qwen2_config
from fast_llm_external_models.apriel2.modeling_apriel2 import Apriel2ForCausalLM
from fast_llm_external_models.mtp_llama.configuration_mtp_llama import MTPLlamaConfig
from fast_llm_external_models.mtp_llama.modeling_mtp_llama import MTPLlamaForCausalLM


@dataclasses.dataclass(frozen=True)
class HFRoundtripCase:
    name: str
    hf_model_name: str
    checkpoint_format: type[CheckpointFormat]
    model_class: type[PreTrainedModel]
    config_class: type[PretrainedConfig]
    dim_overrides: dict[str, typing.Any]
    # max_position_embeddings is not preserved through Fast-LLM internals (LlamaEmbeddingsConverter
    # only stores vocab_size). Must match the HF config class default so source and roundtrip agree.
    max_position_embeddings: int
    # Keys to delete from the config after from_pretrained (e.g. model-version-specific extras
    # that the converter does not preserve and that have no equivalent class default).
    delete_config_keys: tuple[str, ...] = ()

    def make_model(self) -> PreTrainedModel:
        config = self.config_class.from_pretrained(self.hf_model_name)
        for key, value in self.dim_overrides.items():
            setattr(config, key, value)
        config.max_position_embeddings = self.max_position_embeddings
        # head_dim must match hidden_size // num_attention_heads; the converter exports
        # it explicitly for Llama/Mistral, so source must agree.
        if hasattr(config, "head_dim"):
            config.head_dim = config.hidden_size // config.num_attention_heads
        # layer_types length must equal num_hidden_layers (Qwen2 validates this).
        if getattr(config, "layer_types", None) is not None:
            n = config.num_hidden_layers
            lt = config.layer_types
            config.layer_types = (lt * ((n // len(lt)) + 1))[:n]
        for key in self.delete_config_keys:
            if hasattr(config, key):
                delattr(config, key)
        return self.model_class(config)


@dataclasses.dataclass(frozen=True)
class AprielRoundtripCase(HFRoundtripCase):
    surgery_spec: dict | None = None

    def make_model(self) -> PreTrainedModel:
        source_config = AutoConfig.from_pretrained(self.hf_model_name, trust_remote_code=True).to_dict()
        source_config.update(self.dim_overrides)
        source_config["max_position_embeddings"] = self.max_position_embeddings
        converted_config = convert_qwen2_config(source_config)
        if self.surgery_spec is not None:
            converted_config = compose_configs(converted_config, self.surgery_spec)
        return self.model_class(self.config_class(**converted_config))


_TINY_DIMS = {
    "hidden_size": 64,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_hidden_layers": 2,
    "intermediate_size": 128,
    "vocab_size": 256,
}

_HF_ROUNDTRIP_CASES = [
    HFRoundtripCase(
        name="llama",
        hf_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        checkpoint_format=LlamaCheckpointFormat,
        model_class=LlamaForCausalLM,
        config_class=LlamaConfig,
        dim_overrides=_TINY_DIMS,
        max_position_embeddings=2048,  # LlamaConfig default
    ),
    HFRoundtripCase(
        name="mistral",
        hf_model_name="mistralai/Mistral-7B-v0.1",
        checkpoint_format=MistralCheckpointFormat,
        model_class=MistralForCausalLM,
        config_class=MistralConfig,
        dim_overrides=_TINY_DIMS,
        max_position_embeddings=131072,  # MistralConfig default
    ),
    HFRoundtripCase(
        name="qwen2",
        hf_model_name="Qwen/Qwen2.5-0.5B",
        checkpoint_format=Qwen2CheckpointFormat,
        model_class=Qwen2ForCausalLM,
        config_class=Qwen2Config,
        # Reset Qwen2.5-specific fields that the converter does not preserve to their
        # Qwen2Config class defaults, so source and roundtrip configs agree.
        dim_overrides={
            **_TINY_DIMS,
            "bos_token_id": None,
            "eos_token_id": None,
            "max_window_layers": 28,  # Qwen2Config default
        },
        max_position_embeddings=32768,  # Qwen2Config default
        # use_mrope is a Qwen2.5 extension not present in Qwen2Config; deleting it
        # ensures it does not appear in to_dict() of the source config.
        delete_config_keys=("use_mrope",),
    ),
    HFRoundtripCase(
        name="mixtral",
        hf_model_name="mistralai/Mixtral-8x7B-v0.1",
        checkpoint_format=MixtralCheckpointFormat,
        model_class=MixtralForCausalLM,
        config_class=MixtralConfig,
        dim_overrides={
            **_TINY_DIMS,
            "num_local_experts": 2,
            # router_aux_loss_coef is 0.02 in the actual model but the converter does not
            # preserve it; reset to MixtralConfig class default so source and roundtrip agree.
            "router_aux_loss_coef": 0.001,
        },
        max_position_embeddings=131072,  # MixtralConfig default
    ),
    HFRoundtripCase(
        name="mtp_llama",
        hf_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        checkpoint_format=MTPLlamaCheckpointFormat,
        model_class=MTPLlamaForCausalLM,
        config_class=MTPLlamaConfig,
        dim_overrides={
            **_TINY_DIMS,
            # Override model_type so the saved config is recognized as mtp_llama, not llama.
            "model_type": "mtp_llama",
            "prediction_heads": 2,
        },
        max_position_embeddings=2048,  # MTPLlamaConfig default
    ),
    AprielRoundtripCase(
        name="apriel2_attention",
        hf_model_name="Qwen/Qwen2.5-0.5B",
        checkpoint_format=Apriel2TextCheckpointFormat,
        model_class=Apriel2ForCausalLM,
        config_class=Apriel2TextConfig,
        dim_overrides=_TINY_DIMS,
        max_position_embeddings=2048,  # Apriel2TextConfig default
    ),
    AprielRoundtripCase(
        name="apriel2_supernet",
        hf_model_name="Qwen/Qwen2.5-0.5B",
        checkpoint_format=Apriel2TextCheckpointFormat,
        model_class=Apriel2ForCausalLM,
        config_class=Apriel2TextConfig,
        dim_overrides=_TINY_DIMS,
        max_position_embeddings=2048,  # Apriel2TextConfig default
        surgery_spec={
            "decoder": {
                "block": {
                    "mixer": {
                        "type": "stochastic",
                        "main_mixer_name": "attention",
                        "mixers": {
                            "attention": {"type": "attention"},
                            "sliding_window": {
                                "type": "attention",
                                "window_size": 4096,
                            },
                        },
                    },
                },
            },
        },
    ),
]


@pytest.mark.parametrize("case", [pytest.param(c, id=c.name) for c in _HF_ROUNDTRIP_CASES])
def test_hf_roundtrip(case: HFRoundtripCase, result_path: pathlib.Path):
    """HF model survives HF → Fast-LLM → HF with identical config and weights."""
    base = result_path / "hf_roundtrip" / case.name
    source_path = base / "source"
    fastllm_path = base / "fastllm"
    roundtrip_path = base / "roundtrip"

    base.mkdir(parents=True)

    model = case.make_model()
    model.save_pretrained(source_path)
    del model

    use_cpu = not torch.cuda.is_available()

    ConvertConfig(
        model=GPTModelConfig,
        input=CheckpointLoadConfig(path=source_path, format=case.checkpoint_format),
        output=CheckpointSaveConfig(path=fastllm_path, format=FastLLMCheckpointFormat),
        use_cpu=use_cpu,
    ).run()

    ConvertConfig(
        model=GPTModelConfig,
        input=CheckpointLoadConfig(path=fastllm_path, format=FastLLMCheckpointFormat),
        output=CheckpointSaveConfig(path=roundtrip_path, format=case.checkpoint_format),
        use_cpu=use_cpu,
    ).run()

    # Compare config and weights at HF level
    original = case.model_class.from_pretrained(source_path)
    roundtrip = case.model_class.from_pretrained(roundtrip_path)

    original_config = original.config.to_dict()
    roundtrip_config = roundtrip.config.to_dict()
    for key in ("_name_or_path", "auto_map"):
        original_config.pop(key, None)
        roundtrip_config.pop(key, None)
    check_equal_nested(original_config, roundtrip_config)

    original_weights = original.state_dict()
    roundtrip_weights = roundtrip.state_dict()

    assert set(original_weights.keys()) == set(roundtrip_weights.keys()), (
        f"Key mismatch:\n  missing={set(original_weights.keys()) - set(roundtrip_weights.keys())}\n"
        f"  extra={set(roundtrip_weights.keys()) - set(original_weights.keys())}"
    )
    for key in original_weights:
        assert torch.equal(original_weights[key], roundtrip_weights[key]), (
            f"Weight mismatch at '{key}': "
            f"max diff = {(original_weights[key].float() - roundtrip_weights[key].float()).abs().max().item():.6f}"
        )
