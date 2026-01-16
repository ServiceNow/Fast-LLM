"""Config convertor for Apriel2 models with nested decoder structure.

This module provides a custom ModelArchConfigConvertor that extracts
architecture metadata from Apriel2's nested decoder config format,
allowing vLLM to work with Apriel2 models without requiring standard
HuggingFace config attributes like num_attention_heads.
"""

from vllm.transformers_utils.model_arch_config_convertor import (
    MODEL_ARCH_CONFIG_CONVERTORS,
    ModelArchConfigConvertorBase,
)


class Apriel2TextModelArchConfigConvertor(ModelArchConfigConvertorBase):
    """Config convertor for Apriel2TextConfig with nested decoder structure.

    Apriel2 configs use a nested decoder format:
    {
        "decoder": {
            "type": "pattern",
            "num_blocks": 24,
            "pattern": ["attn_block", "gdn_block"],
            "blocks": {
                "attn_block": {"mixer": {"type": "attention", "heads": 14, ...}},
                "gdn_block": {"mixer": {"type": "gdn", ...}}
            }
        }
    }

    This convertor extracts the required values from this nested structure.
    """

    def _get_first_attention_block(self):
        """Find the first attention block config."""
        decoder = getattr(self.hf_text_config, 'decoder', {})
        decoder_type = decoder.get('type', 'fixed')

        if decoder_type == 'fixed':
            block = decoder.get('block', {})
            mixer = block.get('mixer', {})
            if mixer.get('type') == 'attention':
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
        mixer = self._get_first_attention_block()
        return mixer.get('heads', 0)

    def get_total_num_kv_heads(self) -> int:
        mixer = self._get_first_attention_block()
        return mixer.get('head_groups', self.get_total_num_attention_heads())

    def get_head_size(self) -> int:
        mixer = self._get_first_attention_block()
        return mixer.get('head_size', 0)


def register_config_convertors():
    """Register Apriel2 config convertors with vLLM."""
    MODEL_ARCH_CONFIG_CONVERTORS['apriel2_text'] = Apriel2TextModelArchConfigConvertor
    MODEL_ARCH_CONFIG_CONVERTORS['apriel2'] = Apriel2TextModelArchConfigConvertor
