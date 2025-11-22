"""
Apriel2 configuration - HuggingFace format that mirrors Fast-LLM's config structure.
"""

from typing import Any, Optional, Union

from transformers import PretrainedConfig


class Apriel2Config(PretrainedConfig):
    """
    Configuration class for Apriel2 models.

    This config mirrors Fast-LLM's hierarchical structure:

    decoder:
      type: "fixed" or "pattern"
      num_blocks: int

      # For fixed decoder:
      block:
        mixer: {type, ...params}
        mlp: {type, ...params}
        normalization: {type}

      # For pattern decoder:
      blocks:
        block_name:
          mixer: {type, ...params}
          mlp: {type, ...params}
          normalization: {type}
      pattern: [block_name, ...]

    Mixer types: attention, mamba, gated_delta_net, kimi_linear_attention, stochastic
    For stochastic mixers, mixer.mixers is a dict of {name: mixer_config}
    """

    model_type = "apriel2"

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        # Decoder configuration
        decoder: Optional[dict] = None,
        # Embedding config
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
        # Attention defaults (can be overridden per-block)
        num_attention_heads: int = 32,
        num_key_value_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        # Head config
        rms_norm_eps: float = 1e-5,
        tie_word_embeddings: bool = False,
        # Generation config
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: Optional[int] = None,
        use_cache: bool = True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.rms_norm_eps = rms_norm_eps
        self.tie_word_embeddings = tie_word_embeddings
        self.use_cache = use_cache

        # Decoder configuration with defaults
        self.decoder = decoder or {
            "type": "fixed",
            "num_blocks": 32,
            "block": {
                "mixer": {"type": "attention"},
                "mlp": {"type": "mlp"},
                "normalization": {"type": "rms_norm"},
            },
        }

        # Convenience accessor for HuggingFace compatibility
        self.num_hidden_layers = self.decoder.get("num_blocks", 32)

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
