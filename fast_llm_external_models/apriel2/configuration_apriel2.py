"""
Apriel2 configuration - HuggingFace format that mirrors Fast-LLM's config structure.

Uses inheritance to mirror Fast-LLM's architecture:
- Apriel2TextConfig: Text-only (mirrors LanguageModelConfig)
- Apriel2Config(Apriel2TextConfig): Multimodal (mirrors VisionMultiModalModelConfig)

Config structure mirrors Fast-LLM exactly for trivial conversion:
- decoder: BlockSequenceConfig dict
- embeddings: LanguageModelEmbeddingsConfig dict
- head: LanguageModelHeadConfig dict
- vision_encoder: VisionEncoderConfig dict (multimodal only)
"""

import logging
from typing import Optional

from transformers import PretrainedConfig

logger = logging.getLogger(__name__)


class Apriel2TextConfig(PretrainedConfig):
    """
    Configuration class for Apriel2 text/language model.
    Mirrors Fast-LLM's LanguageModelConfig structure exactly.

    All model configuration lives in hierarchical dicts:
    - decoder: BlockSequenceConfig (structure of transformer blocks)
    - embeddings: LanguageModelEmbeddingsConfig (word/position embeddings)
    - head: LanguageModelHeadConfig (final norm + output layer)

    Decoder structure:
      type: "fixed" or "pattern"
      num_blocks: int
      block:
        mixer: {type: attention, heads: N, head_groups: N, head_size: D, ...}
        mlp: {type: mlp, intermediate_size: N, activation: silu, ...}
        normalization: {type: rms_norm, epsilon: 1e-5}
      # or for pattern: blocks: {...}, pattern: [...]

    Mixer types: attention, mamba, gated_delta_net, kimi_linear_attention, stochastic
    """

    model_type = "apriel2_text"

    def __init__(
        self,
        # Core dimensions (at root for simplicity)
        hidden_size: int = 4096,
        vocab_size: int = 32000,
        # Main Fast-LLM fields (as dicts) - THE source of truth
        decoder: Optional[dict] = None,
        embeddings: Optional[dict] = None,
        head: Optional[dict] = None,
        # HF-required fields
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

        # Main Fast-LLM fields as dicts - these are THE source of truth
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
        """Default decoder config mirroring Fast-LLM."""
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
        """Default embeddings config mirroring Fast-LLM."""
        return {
            "max_position_embeddings": 2048,
        }

    def _default_head_config(self) -> dict:
        """Default head config mirroring Fast-LLM."""
        return {
            "normalization": {"type": "rms_norm", "epsilon": 1e-5},
        }

    def get_text_config(self, decoder: bool = False):
        """Return self to ensure tie_word_embeddings is accessible."""
        return self

    def get_block_name(self, layer_idx: int) -> str:
        """Get the block name for a specific layer."""
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
        """Get the block configuration for a specific layer."""
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
    """
    Configuration class for Apriel2 multimodal model.
    Mirrors Fast-LLM's VisionMultiModalModelConfig structure via inheritance.

    Inherits all text fields from Apriel2TextConfig (decoder, embeddings, head, hidden_size, etc.)
    and adds vision-specific fields.

    Vision encoder structure (mirrors Fast-LLM VisionEncoderConfig):
      vision_encoder:
        hidden_size: int
        patch_convolution:
          patch_height: int
          patch_width: int
          normalization: {type: rms_norm, epsilon: 1e-5}
        encoder:
          type: fixed
          num_blocks: int
          block:
            mixer: {type: attention, heads: N, ...}
            mlp: {type: mlp, ...}
            normalization: {...}
        adapter:
          intermediate_size: int
          activation: gelu
          add_linear_biases: true
    """

    model_type = "apriel2"

    def __init__(
        self,
        # Core dimensions
        hidden_size: int = 4096,
        vocab_size: int = 32000,
        # Main Fast-LLM fields (as dicts)
        decoder: Optional[dict] = None,
        embeddings: Optional[dict] = None,
        head: Optional[dict] = None,
        # Vision-specific (mirrors Fast-LLM VisionMultiModalModelConfig)
        vision_encoder: Optional[dict] = None,
        image_token_index: Optional[int] = None,
        # HF-required fields
        tie_word_embeddings: bool = False,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: Optional[int] = None,
        use_cache: bool = True,
        **kwargs,
    ):
        # Initialize text part via parent
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

        # Vision fields
        self.vision_encoder = vision_encoder
        self.image_token_index = image_token_index
