import typing

from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.engine.checkpoint.external import SplitWeightConverter, WeightConverter
from fast_llm.layers.attention.config import AttentionConfig
from fast_llm.layers.attention.rotary.config import YarnRotaryConfig
from fast_llm.layers.block.config import BlockSequenceConfig, FixedBlockSequenceConfig, PatternBlockSequenceConfig
from fast_llm.layers.decoder.config import DecoderBlockConfig
from fast_llm.layers.decoder.mlp.config import MoEMLPConfig
from fast_llm.models.gpt.conversion.config import GptOssCheckpointFormat
from fast_llm.models.gpt.conversion.llama import (
    LlamaBaseModelConverter,
    LlamaHeadConverter,
    LlamaMLPConverter,
    MLPLayer2Converter,
    get_weight_and_bias_converters,
)
from fast_llm.models.gpt.conversion.mistral import (
    MistralAttentionConverter,
    MistralBlockConverter,
    MistralDecoderConverter,
    MistralHeadConverter,
    MistralHuggingfaceCheckpointHandler,
)
from fast_llm.utils import Assert, safe_merge_dicts


class GptOssAttentionConverter(MistralAttentionConverter):
    """
    GPT-OSS attention converter.

    Key differences from Mistral:
    - Supports attention_bias=True (Mistral doesn't use biases)
    - Uses YARN RoPE scaling (not default)
    - Has both full attention and sliding window attention variants
    """

    @classmethod
    def import_config(cls, config: dict) -> dict:
        # Start with Mistral's config (handles sliding_window if present)
        out = super().import_config(config)

        # Override attention_bias - GPT-OSS supports it unlike Mistral
        out["add_linear_biases"] = config.get("attention_bias", False)

        # Handle YARN RoPE scaling
        rope_scaling = config.get("rope_scaling", {})
        if rope_scaling:
            rope_type = rope_scaling.get("rope_type", "yarn")
            if rope_type == "yarn":
                out["rotary"] = {
                    "type": "yarn",
                    "theta": config["rope_theta"],
                    "attention_factor": rope_scaling.get("attention_factor", 1.0),
                    "beta_fast": rope_scaling["beta_fast"],
                    "beta_slow": rope_scaling["beta_slow"],
                    "original_context_length": rope_scaling["original_max_position_embeddings"],
                }

        return out

    @classmethod
    def export_config(cls, config: AttentionConfig) -> dict:
        # Start with base Mistral export (handles window_size, etc.)
        out = super().export_config(config)

        # Override to support attention_bias
        out["attention_bias"] = config.add_linear_biases

        # Export YARN rotary config
        if isinstance(config.rotary, YarnRotaryConfig):
            out["rope_scaling"] = {
                "rope_type": "yarn",
                "attention_factor": getattr(config.rotary, "attention_factor", 1.0),
                "beta_fast": config.rotary.beta_fast,
                "beta_slow": config.rotary.beta_slow,
                "original_max_position_embeddings": config.rotary.original_context_length,
            }

        return out

    @classmethod
    def _check_config(cls, config: AttentionConfig) -> None:
        # Unlike Mistral, GPT-OSS supports biases
        Assert.is_(type(config), AttentionConfig)
        Assert.incl(config.query_layer.bias.enabled, (None, config.add_linear_biases))
        Assert.incl(config.key_layer.bias.enabled, (None, config.add_linear_biases))
        Assert.incl(config.value_layer.bias.enabled, (None, config.add_linear_biases))
        Assert.incl(config.dense_layer.bias.enabled, (None, config.add_linear_biases))


class GptOssMLPConverter(LlamaMLPConverter):
    """
    GPT-OSS MoE MLP converter.

    Structure matches Mixtral:
    - 128 experts (120B) or fewer (20B)
    - 4 active experts per token
    - Gated MLP with SiLU activation
    - No biases in MLP layers
    """

    @classmethod
    def import_config(cls, config: dict) -> dict:
        base_config = {
            "intermediate_size": config["intermediate_size"],
            "add_linear_biases": False,  # GPT-OSS doesn't use biases in MLP
            "activation": "silu",
            "gated": True,
        }

        # Add MoE-specific config
        if "num_local_experts" in config:
            base_config.update(
                {
                    "type": "moe",
                    "experts": config["num_local_experts"],
                    "experts_per_token": config.get("num_experts_per_tok", config.get("experts_per_token", 4)),
                }
            )

        return base_config

    @classmethod
    def export_config(cls, config: MoEMLPConfig) -> dict:
        Assert.custom(isinstance, config, MoEMLPConfig)
        assert not config.add_linear_biases

        return {
            "intermediate_size": config.intermediate_size,
            "hidden_act": "silu",
            "num_local_experts": config.experts,
            "num_experts_per_tok": config.experts_per_token,
            "experts_per_token": config.experts_per_token,
        }

    @classmethod
    def get_converters(
        cls,
        config: MoEMLPConfig,
        fast_llm_prefix: str,
        hf_prefix: str,
        drop_on_export: bool = False,
    ) -> list[WeightConverter]:
        """Convert MoE weights between Fast-LLM and HuggingFace formats."""
        return [
            # Router/gate
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.router",
                f"{hf_prefix}.gate",
                False,
                drop_on_export=drop_on_export,
            ),
            # Expert layer 1 (gate + up projections)
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.layer_1",
                tuple(f"{hf_prefix}.experts.{i}.{w}" for i in range(config.experts) for w in ("w1", "w3")),
                False,
                SplitWeightConverter,
                drop_on_export=drop_on_export,
            ),
            # Expert layer 2 (down projection)
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.layer_2",
                tuple(f"{hf_prefix}.experts.{i}.w2" for i in range(config.experts)),
                False,
                MLPLayer2Converter,
                drop_on_export=drop_on_export,
            ),
        ]


class GptOssBlockConverter:
    """
    GPT-OSS block converter supporting both sliding and full attention.

    Uses a layout name system to distinguish between block types:
    - "sliding": Sliding window attention block
    - "full": Full attention block
    """

    layout_names = {
        "sliding_attention": "sliding",
        "full_attention": "full",
    }
    reverse_layout_names = {v: k for k, v in layout_names.items()}

    mixer_converter_class: typing.ClassVar[type[GptOssAttentionConverter]] = GptOssAttentionConverter
    mlp_converter_class: typing.ClassVar[type[GptOssMLPConverter]] = GptOssMLPConverter
    hf_mixer_name: typing.ClassVar[str] = "self_attn"
    hf_mlp_name: typing.ClassVar[str] = "block_sparse_moe"
    hf_norm_1_name: typing.ClassVar[str] = "input_layernorm"
    hf_norm_2_name: typing.ClassVar[str] = "post_attention_layernorm"

    @classmethod
    def import_config(cls, config: dict, layer_type: str = "full_attention") -> dict:
        """Import config for a specific layer type."""
        from fast_llm.layers.common.normalization.config import RMSNormalizationConfig

        # Create attention config
        attention_config = cls.mixer_converter_class.import_config(config)

        # For sliding attention, ensure window_size is set
        if layer_type == "sliding_attention":
            if "window_size" not in attention_config:
                attention_config["window_size"] = config.get("sliding_window", 128)
        else:
            # For full attention, remove window_size if present
            attention_config.pop("window_size", None)

        return {
            "mixer": attention_config,
            "mlp": cls.mlp_converter_class.import_config(config),
            "normalization": {"type": "rms_norm", "epsilon": config["rms_norm_eps"]},
        }

    @classmethod
    def export_config(cls, config: DecoderBlockConfig) -> dict:
        Assert.custom(isinstance, config, DecoderBlockConfig)
        from fast_llm.layers.common.normalization.config import RMSNormalizationConfig

        Assert.custom(isinstance, config.normalization, RMSNormalizationConfig)
        assert not config.normalization.zero_centered

        return safe_merge_dicts(
            cls.mixer_converter_class.export_config(config.mixer),
            cls.mlp_converter_class.export_config(config.mlp),
            {"rms_norm_eps": config.normalization.epsilon},
        )

    @classmethod
    def get_converters(
        cls, config: DecoderBlockConfig, fast_llm_prefix: str, hf_prefix: str, drop_on_export: bool = False
    ) -> list[WeightConverter]:
        """Get weight converters for a block."""
        from fast_llm.models.gpt.conversion.llama import LlamaNormalizationConverter

        return [
            *cls.mixer_converter_class.get_converters(
                config.mixer,
                f"{fast_llm_prefix}.mixer",
                f"{hf_prefix}.{cls.hf_mixer_name}",
                drop_on_export,
            ),
            *cls.mlp_converter_class.get_converters(
                config.mlp,
                f"{fast_llm_prefix}.mlp",
                f"{hf_prefix}.{cls.hf_mlp_name}",
                drop_on_export,
            ),
            *LlamaNormalizationConverter.get_converters(
                config.normalization,
                f"{fast_llm_prefix}.norm_1",
                f"{hf_prefix}.{cls.hf_norm_1_name}",
                drop_on_export,
            ),
            *LlamaNormalizationConverter.get_converters(
                config.normalization,
                f"{fast_llm_prefix}.norm_2",
                f"{hf_prefix}.{cls.hf_norm_2_name}",
                drop_on_export,
            ),
        ]


class GptOssDecoderConverter(MistralDecoderConverter):
    """
    GPT-OSS decoder converter with heterogeneous block pattern support.

    Handles the `layer_types` field that specifies alternating attention patterns.
    """

    block_converter_class: typing.ClassVar[type[GptOssBlockConverter]] = GptOssBlockConverter

    @classmethod
    def import_config(cls, config: dict) -> dict:
        """Import decoder config, handling heterogeneous layer types."""
        layer_types = config.get("layer_types", [])

        if not layer_types:
            # No layer_types specified, assume all full attention
            return {
                "block": cls.block_converter_class.import_config(config, "full_attention"),
                "num_blocks": config["num_hidden_layers"],
            }

        # Determine unique layer types
        unique_types = list(dict.fromkeys(layer_types))  # Preserve order

        if len(unique_types) == 1:
            # All layers are the same type - use FixedBlockSequenceConfig
            return {
                "block": cls.block_converter_class.import_config(config, unique_types[0]),
                "num_blocks": config["num_hidden_layers"],
            }
        else:
            # Multiple layer types - use PatternBlockSequenceConfig
            # Create a block config for each unique type
            blocks = {}
            for layer_type in unique_types:
                layout_name = cls.block_converter_class.layout_names.get(layer_type, layer_type)
                blocks[layout_name] = cls.block_converter_class.import_config(config, layer_type)

            # Create pattern using layout names
            pattern = [cls.block_converter_class.layout_names.get(lt, lt) for lt in layer_types]

            return {
                "type": "pattern",
                "blocks": blocks,
                "pattern": pattern,
                "num_blocks": config["num_hidden_layers"],
            }

    @classmethod
    def export_config(cls, config: BlockSequenceConfig) -> dict:
        """Export decoder config, reconstructing layer_types."""
        if type(config) is FixedBlockSequenceConfig:
            # All blocks are the same
            block_configs = [config.block]
            # Determine layer type from window_size
            has_window = hasattr(config.block.mixer, "window_size") and config.block.mixer.window_size is not None
            layer_type = "sliding_attention" if has_window else "full_attention"
            layer_types = [layer_type] * config.num_blocks
        elif type(config) is PatternBlockSequenceConfig:
            # Multiple block types
            block_configs = list(config.blocks.values())
            # Reconstruct layer_types from pattern
            layer_types = []
            for block_name in config.expanded_pattern:
                block_config = config.blocks[block_name]
                has_window = (
                    hasattr(block_config.mixer, "window_size") and block_config.mixer.window_size is not None
                )
                layer_type = "sliding_attention" if has_window else "full_attention"
                layer_types.append(layer_type)
        else:
            raise NotImplementedError(f"Unsupported block sequence type: {type(config)}")

        # Merge all block configs
        return safe_merge_dicts(
            *[cls.block_converter_class.export_config(block_config) for block_config in block_configs],
            {
                "num_hidden_layers": config.num_blocks,
                "layer_types": layer_types,
            },
        )

    @classmethod
    def get_converters(
        cls,
        config: BlockSequenceConfig,
        fast_llm_prefix: str,
        hf_prefix: str,
        drop_on_export: bool = False,
    ) -> list[WeightConverter]:
        """Get weight converters for all blocks in the decoder."""
        converters = []

        if type(config) is FixedBlockSequenceConfig:
            # All blocks use the same config
            for block_index in range(config.num_blocks):
                converters += cls.block_converter_class.get_converters(
                    config.block,
                    f"{fast_llm_prefix}.{block_index}",
                    f"{hf_prefix}.{block_index}",
                    drop_on_export,
                )
        elif type(config) is PatternBlockSequenceConfig:
            # Blocks follow a pattern
            for block_index in range(config.num_blocks):
                block_name = config.expanded_pattern[block_index]
                block_config = config.blocks[block_name]
                converters += cls.block_converter_class.get_converters(
                    block_config,
                    f"{fast_llm_prefix}.{block_index}",
                    f"{hf_prefix}.{block_index}",
                    drop_on_export,
                )
        else:
            raise NotImplementedError(f"Unsupported block sequence type: {type(config)}")

        return converters


class GptOssHeadConverter(MistralHeadConverter):
    block_converter_class: typing.ClassVar[type[GptOssBlockConverter]] = GptOssBlockConverter


class GptOssBaseModelConverter(LlamaBaseModelConverter):
    """
    GPT-OSS base model converter.

    Handles:
    - Vocab size ~201,088 (o200k_harmony tokenizer)
    - Heterogeneous decoder with alternating attention patterns
    - RMS normalization
    - MoE layers
    """

    decoder_converter_class: typing.ClassVar[type[GptOssDecoderConverter]] = GptOssDecoderConverter
    head_converter_class: typing.ClassVar[type[GptOssHeadConverter]] = GptOssHeadConverter


class GptOssHuggingfaceCheckpointHandler(MistralHuggingfaceCheckpointHandler):
    """
    Checkpoint handler for GPT-OSS models.

    Supports conversion between Fast-LLM and HuggingFace GPT-OSS format.
    Handles both gpt-oss-120b (117B params) and gpt-oss-20b (21B params) variants.

    Key features:
    - Mixture of Experts (128 experts for 120B, 4 active per token)
    - Alternating sliding window and full attention patterns
    - YARN RoPE scaling
    - Grouped multi-query attention (8 KV heads)
    """

    format: typing.ClassVar[type[CheckpointFormat]] = GptOssCheckpointFormat
    architecture: typing.ClassVar[str] = "GptOssForCausalLM"
    base_model_converter_class: typing.ClassVar[type[GptOssBaseModelConverter]] = GptOssBaseModelConverter

    @classmethod
    def get_transformers_configuration_class(cls):
        import transformers

        return transformers.GptOssConfig
