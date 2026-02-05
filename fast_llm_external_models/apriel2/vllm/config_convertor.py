"""Config convertor and registration for Apriel2 models.

This module provides:
1. A custom ModelArchConfigConvertor for Apriel2's nested decoder config format
2. A register() function for vLLM's plugin system (entry_points)

Registration is automatic when fast-llm is installed. vLLM discovers the
entry point defined in setup.cfg and calls register() in all processes.
"""

from vllm import ModelRegistry
from vllm.transformers_utils.model_arch_config_convertor import (
    MODEL_ARCH_CONFIG_CONVERTORS,
    ModelArchConfigConvertorBase,
)


class Apriel2TextModelArchConfigConvertor(ModelArchConfigConvertorBase):
    """Config convertor for Apriel2TextConfig with nested decoder structure.

    Apriel2 configs use a nested decoder format instead of standard HuggingFace
    attributes like num_hidden_layers. This convertor extracts the required
    values from the nested structure.
    """

    def _get_first_attention_block(self):
        """Find the first attention block config.

        Handles both regular and stochastic mixer types. For stochastic mixers,
        looks up the main_mixer_name to find the attention config.
        """
        decoder = getattr(self.hf_text_config, "decoder", {})
        decoder_type = decoder.get("type", "fixed")

        if decoder_type == "fixed":
            block = decoder.get("block", {})
            mixer = block.get("mixer", {})
            mixer_type = mixer.get("type", "attention")
            if mixer_type == "stochastic":
                main_mixer_name = mixer.get("main_mixer_name", "attention")
                return mixer.get("mixers", {}).get(main_mixer_name, {})
            elif mixer_type == "attention":
                return mixer
        elif decoder_type == "pattern":
            blocks = decoder.get("blocks", {})
            pattern = decoder.get("pattern", [])
            for block_name in pattern:
                block = blocks.get(block_name, {})
                mixer = block.get("mixer", {})
                if mixer.get("type") == "attention":
                    return mixer
        return {}

    def get_num_hidden_layers(self) -> int:
        decoder = getattr(self.hf_text_config, "decoder", {})
        return decoder.get("num_blocks", 0)

    def get_total_num_attention_heads(self) -> int:
        return self._get_first_attention_block().get("heads", 0)

    def get_total_num_kv_heads(self) -> int:
        return self._get_first_attention_block().get("head_groups", self.get_total_num_attention_heads())

    def get_head_size(self) -> int:
        return self._get_first_attention_block().get("head_size", 0)


def register():
    """Register Apriel2 models and config convertors with vLLM.

    This function is called automatically by vLLM's plugin system via Python's
    entry_points mechanism. The entry point is defined in fast-llm's setup.cfg:

        [options.entry_points]
        vllm.general_plugins =
            apriel2 = fast_llm_external_models.apriel2.vllm.config_convertor:register

    vLLM discovers all entry points in the 'vllm.general_plugins' group using
    importlib.metadata and calls each plugin's register function during startup.
    This happens in every process (parent and subprocesses spawned by AsyncLLM),
    ensuring model registration is available everywhere.

    The VLLM_PLUGINS environment variable can optionally filter which plugins
    are loaded (comma-separated list of plugin names to enable).

    Safe to call multiple times - skips registration if already done.
    """
    # Register config convertors (idempotent).
    #
    # Apriel2 has two HuggingFace config "model_type" variants:
    # - apriel2_text: text-only config
    # - apriel2: multimodal config (text + vision fields), but the text decoder
    #   uses the same nested structure. vLLM still needs the text decoder arch
    #   info even when multimodal fields exist.
    if "apriel2_text" not in MODEL_ARCH_CONFIG_CONVERTORS:
        MODEL_ARCH_CONFIG_CONVERTORS["apriel2_text"] = Apriel2TextModelArchConfigConvertor
    if "apriel2" not in MODEL_ARCH_CONFIG_CONVERTORS:
        MODEL_ARCH_CONFIG_CONVERTORS["apriel2"] = Apriel2TextModelArchConfigConvertor

    # Register HuggingFace config classes for model_type -> config mapping.
    # vLLM loads configs with `trust_remote_code` ignored, so without this
    # Transformers would fall back to a generic PretrainedConfig and `decoder`
    # overrides would replace (not merge) nested dicts.
    try:
        from transformers import AutoConfig

        from fast_llm_external_models.apriel2.configuration_apriel2 import Apriel2Config as HFApriel2Config
        from fast_llm_external_models.apriel2.configuration_apriel2 import Apriel2TextConfig as HFApriel2TextConfig

        AutoConfig.register("apriel2_text", HFApriel2TextConfig, exist_ok=True)
        AutoConfig.register("apriel2", HFApriel2Config, exist_ok=True)

        # Prefer our config classes even when the checkpoint directory contains
        # `configuration_*.py` and vLLM is started with `--trust-remote-code`.
        # vLLM's HFConfigParser checks its own registry first, so this prevents
        # loading an older config implementation from the checkpoint export.
        try:
            from vllm.transformers_utils.config import _CONFIG_REGISTRY

            _CONFIG_REGISTRY["apriel2_text"] = HFApriel2TextConfig
            _CONFIG_REGISTRY["apriel2"] = HFApriel2Config
        except Exception:
            pass
    except Exception:
        # Best-effort only; vLLM can still proceed with the generic config.
        pass

    # Register model class
    # Note: some exported checkpoints may list "Apriel2ForConditionalGeneration"
    # in config.json's "architectures". vLLM's model selection is driven by that
    # field, so we alias it to the same vLLM implementation for text-only usage.
    for arch in ("Apriel2ForCausalLM", "Apriel2ForConditionalGeneration"):
        ModelRegistry.register_model(
            arch,
            "fast_llm_external_models.apriel2.vllm:Apriel2ForCausalLM",
        )
