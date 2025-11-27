"""
Apriel2 configuration - HuggingFace format that mirrors Fast-LLM's config structure.

Uses inheritance to mirror Fast-LLM's architecture:
- Apriel2TextConfig: Text-only (mirrors LanguageModelConfig)
- Apriel2Config(Apriel2TextConfig): Multimodal (mirrors VisionMultiModalModelConfig)
"""

import logging
from typing import Any, Optional

from transformers import PretrainedConfig

logger = logging.getLogger(__name__)


class Apriel2TextConfig(PretrainedConfig):
    """
    Configuration class for Apriel2 text/language model.
    Mirrors Fast-LLM's LanguageModelConfig structure.

    Main fields (as dicts, mirroring Fast-LLM):
    - decoder: BlockSequenceConfig (structure of transformer blocks)
    - embeddings: LanguageModelEmbeddingsConfig (word/position embeddings)
    - head: LanguageModelHeadConfig (final norm + output layer)

    Decoder structure:
      type: "fixed" or "pattern"
      num_blocks: int
      block: {mixer: {...}, mlp: {...}, normalization: {...}}
      # or for pattern: blocks: {...}, pattern: [...]

    Mixer types: attention, mamba, gated_delta_net, kimi_linear_attention, stochastic
    """

    model_type = "apriel2_text"

    def __init__(
        self,
        # Main Fast-LLM fields (as dicts)
        decoder: Optional[dict] = None,
        embeddings: Optional[dict] = None,
        head: Optional[dict] = None,
        # Core dimensions
        hidden_size: int = 4096,
        vocab_size: int = 32000,
        # Convenience fields for HuggingFace compatibility
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
        num_attention_heads: int = 32,
        num_key_value_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        rms_norm_eps: float = 1e-5,
        tie_word_embeddings: bool = False,
        # Generation config
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: Optional[int] = None,
        use_cache: bool = True,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # Convenience fields
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.rms_norm_eps = rms_norm_eps
        self.tie_word_embeddings = tie_word_embeddings
        self.use_cache = use_cache

        # Main Fast-LLM fields as dicts
        self.decoder = decoder or {
            "type": "fixed",
            "num_blocks": 32,
            "block": {
                "mixer": {"type": "attention"},
                "mlp": {"type": "mlp"},
                "normalization": {"type": "rms_norm"},
            },
        }

        self.embeddings = embeddings or {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
        }

        self.head = head or {
            "type": "language_model_head",
            "normalization": {"type": "rms_norm"},
        }

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

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
            # Fixed decoder: all blocks use the same configuration
            return self.decoder.get("block", self._default_block_config())
        elif decoder_type == "pattern":
            # Pattern decoder: blocks follow a repeating pattern
            blocks = self.decoder.get("blocks", {})
            pattern = self.decoder.get("pattern", [])
            if not blocks or not pattern:
                raise ValueError("Pattern decoder requires 'blocks' and 'pattern' fields")
            block_name = pattern[layer_idx % len(pattern)]
            return blocks[block_name]
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")

    def _default_block_config(self) -> dict:
        """Create default block configuration."""
        return {
            "mixer": {"type": "attention"},
            "mlp": {"type": "mlp"},
            "normalization": {"type": "rms_norm"},
        }


class Apriel2Config(Apriel2TextConfig):
    """
    Configuration class for Apriel2 multimodal model.
    Mirrors Fast-LLM's VisionMultiModalModelConfig structure via inheritance.

    Inherits all text fields from Apriel2TextConfig (decoder, embeddings, head, hidden_size, etc.)
    and adds vision-specific fields.

    Args:
        decoder (`dict`, *optional*):
            Decoder configuration (inherited from Apriel2TextConfig).
        embeddings (`dict`, *optional*):
            Embeddings configuration (inherited from Apriel2TextConfig).
        head (`dict`, *optional*):
            Head configuration (inherited from Apriel2TextConfig).
        vision_encoder (`dict`, *optional*):
            Vision encoder configuration (VisionEncoderConfig as dict).
            Structure: {patch_convolution: {...}, encoder: {...}, adapter: {...}, hidden_size: int}
        image_token_index (`int`, *optional*, defaults to None):
            The image token index. Unused by Fast-LLM, required for HuggingFace conversion.
    """

    model_type = "apriel2"

    def __init__(
        self,
        # Inherited text fields
        decoder: Optional[dict] = None,
        embeddings: Optional[dict] = None,
        head: Optional[dict] = None,
        hidden_size: int = 4096,
        vocab_size: int = 32000,
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
        num_attention_heads: int = 32,
        num_key_value_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        rms_norm_eps: float = 1e-5,
        tie_word_embeddings: bool = False,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: Optional[int] = None,
        use_cache: bool = True,
        # New vision fields (mirroring Fast-LLM's VisionMultiModalModelConfig)
        vision_encoder: Optional[dict] = None,
        image_token_index: Optional[int] = None,
        **kwargs,
    ):
        # Initialize text part via parent
        super().__init__(
            decoder=decoder,
            embeddings=embeddings,
            head=head,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            use_cache=use_cache,
            **kwargs,
        )

        # Add vision fields
        self.vision_encoder = vision_encoder
        self.image_token_index = image_token_index
