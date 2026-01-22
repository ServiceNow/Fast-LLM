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
        decoder = getattr(self.hf_text_config, 'decoder', {})
        decoder_type = decoder.get('type', 'fixed')

        if decoder_type == 'fixed':
            block = decoder.get('block', {})
            mixer = block.get('mixer', {})
            mixer_type = mixer.get('type', 'attention')
            if mixer_type == 'stochastic':
                main_mixer_name = mixer.get('main_mixer_name', 'attention')
                return mixer.get('mixers', {}).get(main_mixer_name, {})
            elif mixer_type == 'attention':
                return mixer
        elif decoder_type == 'pattern':
            blocks = decoder.get('blocks', {})
            pattern = decoder.get('pattern', [])
            for block_name in pattern:
                block = blocks.get(block_name, {})
                mixer = block.get('mixer', {})
                if mixer.get('type') == 'attention':
                    return mixer
        return {}

    def get_num_hidden_layers(self) -> int:
        decoder = getattr(self.hf_text_config, 'decoder', {})
        return decoder.get('num_blocks', 0)

    def get_total_num_attention_heads(self) -> int:
        return self._get_first_attention_block().get('heads', 0)

    def get_total_num_kv_heads(self) -> int:
        return self._get_first_attention_block().get(
            'head_groups', self.get_total_num_attention_heads()
        )

    def get_head_size(self) -> int:
        return self._get_first_attention_block().get('head_size', 0)


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
    # Skip if already registered
    if 'apriel2_text' in MODEL_ARCH_CONFIG_CONVERTORS:
        return

    # Register config convertor (only apriel2_text, not apriel2 with vision encoder)
    MODEL_ARCH_CONFIG_CONVERTORS['apriel2_text'] = Apriel2TextModelArchConfigConvertor

    # Register model class
    ModelRegistry.register_model(
        "Apriel2ForCausalLM",
        "fast_llm_external_models.apriel2.vllm:Apriel2ForCausalLM",
    )
