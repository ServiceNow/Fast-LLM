"""Apriel2 HuggingFace configuration."""

import logging
from typing import Optional

from transformers import PretrainedConfig

logger = logging.getLogger(__name__)

_MERGE_DICT_FIELDS = {"decoder", "embeddings", "head", "vision_encoder"}


def _deep_merge_dict(base: dict, update: dict) -> dict:
    """Recursively merge two dicts without mutating inputs."""
    merged = dict(base)
    for key, value in update.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


class Apriel2TextConfig(PretrainedConfig):
    model_type = "apriel2_text"

    def __setattr__(self, key, value):
        # Transformers' PretrainedConfig.from_dict applies kwargs overrides
        # via setattr() *after* the config is constructed. For nested dict
        # fields like `decoder`, that would normally replace the entire dict.
        # We want to support partial overrides such as:
        #   decoder.block.mixer.main_mixer_name = "gdn"
        if key in _MERGE_DICT_FIELDS and isinstance(value, dict):
            current = self.__dict__.get(key)
            if isinstance(current, dict):
                value = _deep_merge_dict(current, value)
        super().__setattr__(key, value)

    def __init__(
        self,
        hidden_size: int = 4096,
        vocab_size: int = 32000,
        decoder: Optional[dict] = None,
        embeddings: Optional[dict] = None,
        head: Optional[dict] = None,
        tie_word_embeddings: bool = False,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: Optional[int] = None,
        use_cache: bool = True,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.use_cache = use_cache

        self.decoder = decoder or self._default_decoder_config()
        self.embeddings = embeddings or self._default_embeddings_config()
        self.head = head or self._default_head_config()

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _default_decoder_config(self) -> dict:
        return {
            "type": "fixed",
            "num_blocks": 32,
            "block": {
                "mixer": {
                    "type": "attention",
                    "heads": 32,
                    "head_groups": 32,
                    "head_size": self.hidden_size // 32,
                    "rotary": {"type": "default", "theta": 10000.0},
                    "add_linear_biases": False,
                },
                "mlp": {
                    "type": "mlp",
                    "intermediate_size": self.hidden_size * 4,
                    "activation": "silu",
                    "gated": True,
                    "add_linear_biases": False,
                },
                "normalization": {"type": "rms_norm", "epsilon": 1e-5},
            },
        }

    def _default_embeddings_config(self) -> dict:
        return {
            "max_position_embeddings": 2048,
        }

    def _default_head_config(self) -> dict:
        return {
            "normalization": {"type": "rms_norm", "epsilon": 1e-5},
        }

    def get_text_config(self, decoder: bool = False):
        return self

    def get_block_name(self, layer_idx: int) -> str:
        decoder_type = self.decoder.get("type", "fixed")

        if decoder_type == "fixed":
            return "block"
        elif decoder_type == "pattern":
            pattern = self.decoder.get("pattern", [])
            if not pattern:
                raise ValueError("Pattern decoder requires 'pattern' field")
            return pattern[layer_idx % len(pattern)]
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")

    def get_block_config(self, layer_idx: int) -> dict:
        decoder_type = self.decoder.get("type", "fixed")

        if decoder_type == "fixed":
            return self.decoder.get("block", {})
        elif decoder_type == "pattern":
            blocks = self.decoder.get("blocks", {})
            pattern = self.decoder.get("pattern", [])
            if not blocks or not pattern:
                raise ValueError("Pattern decoder requires 'blocks' and 'pattern' fields")
            block_name = pattern[layer_idx % len(pattern)]
            return blocks[block_name]
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")


class Apriel2Config(Apriel2TextConfig):
    model_type = "apriel2"

    def __init__(
        self,
        hidden_size: int = 4096,
        vocab_size: int = 32000,
        decoder: Optional[dict] = None,
        embeddings: Optional[dict] = None,
        head: Optional[dict] = None,
        vision_encoder: Optional[dict] = None,
        image_token_index: Optional[int] = None,
        tie_word_embeddings: bool = False,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: Optional[int] = None,
        use_cache: bool = True,
        **kwargs,
    ):
        super().__init__(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            decoder=decoder,
            embeddings=embeddings,
            head=head,
            tie_word_embeddings=tie_word_embeddings,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            use_cache=use_cache,
            **kwargs,
        )

        self.vision_encoder = vision_encoder
        self.image_token_index = image_token_index
