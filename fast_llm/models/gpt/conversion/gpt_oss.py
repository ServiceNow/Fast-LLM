import typing

import torch

from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.engine.checkpoint.external import SplitWeightConverter, WeightConverter
from fast_llm.layers.attention.config import AttentionConfig
from fast_llm.layers.block.config import BlockSequenceConfig, FixedBlockSequenceConfig, PatternBlockSequenceConfig
from fast_llm.layers.common.linear.config import AffineLinearConfig
from fast_llm.layers.decoder.config import DecoderBlockConfig
from fast_llm.layers.decoder.mlp.config import MLPConfig, MoEMLPConfig
from fast_llm.models.gpt.conversion.config import GptOssCheckpointFormat
from fast_llm.models.gpt.conversion.llama import (
    LlamaAttentionConverter,
    LlamaBaseModelConverter,
    LlamaBlockConverter,
    LlamaHeadConverter,
    LlamaMLPConverter,
    MLPLayer2Converter,
    get_parameter_converter,
    get_weight_and_bias_converters,
)
from fast_llm.models.gpt.conversion.mistral import MistralHuggingfaceCheckpointHandler
from fast_llm.models.gpt.conversion.mixtral import MixtralMLPConverter
from fast_llm.tensor import SafeTensorSlice
from fast_llm.utils import Assert, safe_merge_dicts


class GptOssAttentionConverter(LlamaAttentionConverter):
    """
    GPT-OSS attention converter.

    Inherits from Llama (which supports YARN RoPE) and adds:
    - attention_bias support
    - attention sinks support
    """

    @classmethod
    def import_config(cls, config: dict) -> dict:
        out = super().import_config(config)
        # GPT-OSS supports attention_bias unlike Llama
        out["add_linear_biases"] = config.get("attention_bias", False)
        # GPT-OSS always uses attention sinks
        out["sinks"] = {"enabled": True}
        return out

    @classmethod
    def export_config(cls, config: AttentionConfig) -> dict:
        out = super().export_config(config)
        out["attention_bias"] = config.add_linear_biases
        # Don't add sinks to config, it's indicated by presence of sinks parameter
        return out

    @classmethod
    def _check_config(cls, config: AttentionConfig) -> None:
        # Unlike Llama/Mistral, GPT-OSS supports biases
        Assert.is_(type(config), AttentionConfig)
        Assert.incl(config.query_layer.bias.enabled, (None, config.add_linear_biases))
        Assert.incl(config.key_layer.bias.enabled, (None, config.add_linear_biases))
        Assert.incl(config.value_layer.bias.enabled, (None, config.add_linear_biases))
        Assert.incl(config.dense_layer.bias.enabled, (None, config.add_linear_biases))

    @classmethod
    def get_converters(
        cls,
        config: AttentionConfig,
        fast_llm_prefix: str,
        hf_prefix: str,
        drop_on_export: bool = False,
    ) -> list[WeightConverter]:
        # Get base converters from parent class
        converters = super().get_converters(config, fast_llm_prefix, hf_prefix, drop_on_export)

        # Add sinks converter if enabled
        if config.sinks.enabled:
            converters.append(
                get_parameter_converter(
                    f"{fast_llm_prefix}.sinks",
                    f"{hf_prefix}.sinks",
                    drop_on_export=drop_on_export,
                )
            )

        return converters


class GptOssMoEWeightConverter(WeightConverter):
    """
    Converter for GPT-OSS MoE weights.

    HF format: (num_experts, dim1, dim2)
    Fast-LLM format: (num_experts * dim1, dim2)

    Flattens/unflattens the expert dimension during conversion.
    """

    def export_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        (weight_tensor,) = weight
        # Fast-LLM: (num_experts * dim1, dim2) -> HF: (num_experts, dim1, dim2)
        weight_loaded = weight_tensor[:]
        num_experts = self._config.experts
        total_dim, dim2 = weight_loaded.shape
        dim1 = total_dim // num_experts
        weight_reshaped = weight_loaded.reshape(num_experts, dim1, dim2)
        return (weight_reshaped,)

    def import_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        (weight_tensor,) = weight
        # HF: (num_experts, dim1, dim2) -> Fast-LLM: (num_experts * dim1, dim2)
        weight_loaded = weight_tensor[:]
        num_experts, dim1, dim2 = weight_loaded.shape
        weight_reshaped = weight_loaded.reshape(num_experts * dim1, dim2)
        return (weight_reshaped,)


class GptOssMoEBiasConverter(WeightConverter):
    """
    Converter for GPT-OSS MoE biases.

    After dequantization, GPT-OSS stores biases in the same format as Fast-LLM:
    Both formats: (num_experts, out_features_per_expert)

    No transposition needed - just pass through.
    """

    def export_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        # Both Fast-LLM and HF use (num_experts, out_features_per_expert)
        return weight

    def import_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        # Both HF and Fast-LLM use (num_experts, out_features_per_expert)
        return weight


def get_gpt_oss_weight_and_bias_converters(
    fast_llm_prefix: str,
    hf_prefix: str,
    use_bias: bool,
    weight_cls=WeightConverter,
    drop_on_export: bool = False,
) -> list[WeightConverter]:
    """
    Get weight and bias converters for GPT-OSS format.

    GPT-OSS uses "_bias" suffix instead of ".bias" for expert biases.
    After dequantization, biases are in (num_experts, out_features_per_expert) format.
    """
    converters = [
        get_parameter_converter(
            f"{fast_llm_prefix}.weight",
            f"{hf_prefix}.weight",
            weight_cls,
            None,
            drop_on_export,
        )
    ]
    if use_bias:
        # GPT-OSS uses "_bias" suffix for expert biases, not ".bias"
        # and stores them transposed
        converters.append(
            get_parameter_converter(
                f"{fast_llm_prefix}.bias",
                f"{hf_prefix}_bias",  # Note: _bias not .bias
                GptOssMoEBiasConverter,
                None,
                drop_on_export,
            )
        )
    return converters


class GptOssMLPConverter(MixtralMLPConverter):
    """
    GPT-OSS MoE MLP converter.

    Handles the dequantized GPT-OSS checkpoint format which uses:
    - Router at .router (not .gate like Mixtral)
    - Router has bias (unlike Mixtral)
    - Concatenated gate_up_proj and down_proj (not separate w1/w2/w3 like Mixtral)
    - Expert biases use "_bias" suffix (not ".bias")
    """

    @classmethod
    def import_config(cls, config: dict) -> dict:
        out = super().import_config(config)
        out["router"] = {
            "type": "affine_linear",
            "bias": {"enabled": True},
        }
        out["add_linear_biases"] = True
        # Use moe_affine_linear type for MoE expert layers to get per-expert biases
        out["layer_1"] = {
            "type": "moe_affine_linear",
            "bias": {"enabled": True},
        }
        out["layer_2"] = {
            "type": "moe_affine_linear",
            "bias": {"enabled": True},
        }
        return out

    @classmethod
    def export_config(cls, config: MoEMLPConfig) -> dict:
        Assert.custom(isinstance, config, MoEMLPConfig)
        # Unlike Mixtral, GPT-OSS supports biases on expert layers
        return safe_merge_dicts(
            # Skip MixtralMLPConverter.export_config to avoid the bias assertion
            # Call grandparent (LlamaMLPConverter) instead
            LlamaMLPConverter.export_config(config),
            {
                "num_local_experts": config.experts,
                "num_experts_per_tok": config.experts_per_token,
            },
        )

    @classmethod
    def get_converters(
        cls,
        config: MoEMLPConfig,
        fast_llm_prefix: str,
        hf_prefix: str,
        drop_on_export: bool = False,
    ) -> list[WeightConverter]:
        return [
            # Router: GPT-OSS uses .router instead of .gate
            # Router has bias in GPT-OSS (unlike Mixtral which doesn't)
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.router",
                f"{hf_prefix}.router",  # Different from Mixtral which uses .gate
                True,
                drop_on_export=drop_on_export,
            ),
            # Experts use concatenated format like Llama (gate_up_proj, down_proj)
            # not separate w1/w2/w3 like Mixtral
            # GPT-OSS uses "_bias" suffix for expert biases
            # Weights need special MoE converter to handle (num_experts, ...) shape
            *get_gpt_oss_weight_and_bias_converters(
                f"{fast_llm_prefix}.layer_1",
                f"{hf_prefix}.experts.gate_up_proj",
                config.add_linear_biases,
                GptOssMoEWeightConverter,
                drop_on_export=drop_on_export,
            ),
            *get_gpt_oss_weight_and_bias_converters(
                f"{fast_llm_prefix}.layer_2",
                f"{hf_prefix}.experts.down_proj",
                config.add_linear_biases,
                GptOssMoEWeightConverter,
                drop_on_export=drop_on_export,
            ),
        ]


class GptOssBlockConverter(LlamaBlockConverter):
    """
    GPT-OSS block converter.

    Uses dynamic MLP converter selection (Llama vs Mixtral) based on config type.
    """

    # Layout names for heterogeneous block patterns
    layout_names = {
        "sliding_attention": "sliding",
        "full_attention": "full",
    }

    # Dynamic converter selection like Apriel
    _mixer_converter_classes = {
        AttentionConfig: GptOssAttentionConverter,
    }
    _mlp_converter_classes = {
        MLPConfig: LlamaMLPConverter,
        MoEMLPConfig: GptOssMLPConverter,
    }

    mixer_converter_class: typing.ClassVar[type[GptOssAttentionConverter]] = GptOssAttentionConverter
    mlp_converter_class: typing.ClassVar[type] = None  # Will be selected dynamically

    hf_mixer_name: typing.ClassVar[str] = "self_attn"
    hf_mlp_name: typing.ClassVar[str] = "mlp"  # GPT-OSS uses .mlp (after dequantization)
    hf_norm_1_name: typing.ClassVar[str] = "input_layernorm"
    hf_norm_2_name: typing.ClassVar[str] = "post_attention_layernorm"

    @classmethod
    def import_config(cls, config: dict, layer_type: str = "full_attention") -> dict:
        # Create attention config
        attention_config = cls.mixer_converter_class.import_config(config)

        # Handle sliding window for this specific layer type
        if layer_type == "sliding_attention":
            if "window_size" not in attention_config:
                attention_config["window_size"] = config.get("sliding_window", 128)
        else:
            # For full attention, remove window_size if present
            attention_config.pop("window_size", None)

        # Determine MLP converter based on config
        if "num_local_experts" in config:
            mlp_converter = cls._mlp_converter_classes[MoEMLPConfig]
        else:
            mlp_converter = cls._mlp_converter_classes[MLPConfig]

        return {
            "mixer": attention_config,
            "mlp": mlp_converter.import_config(config),
            "normalization": cls.normalization_converter_class.import_config(config),
        }

    @classmethod
    def export_config(cls, config: DecoderBlockConfig) -> dict:
        Assert.custom(isinstance, config, DecoderBlockConfig)

        # Select MLP converter based on config type
        mlp_converter = cls._mlp_converter_classes[type(config.mlp)]

        return safe_merge_dicts(
            cls.mixer_converter_class.export_config(config.mixer),
            mlp_converter.export_config(config.mlp),
            cls.normalization_converter_class.export_config(config.normalization),
        )

    @classmethod
    def get_converters(
        cls, config: DecoderBlockConfig, fast_llm_prefix: str, hf_prefix: str, drop_on_export: bool = False
    ) -> list[WeightConverter]:
        # Select MLP converter based on config type
        mlp_converter = cls._mlp_converter_classes[type(config.mlp)]

        return [
            *cls.mixer_converter_class.get_converters(
                config.mixer,
                f"{fast_llm_prefix}.mixer",
                f"{hf_prefix}.{cls.hf_mixer_name}",
                drop_on_export,
            ),
            *mlp_converter.get_converters(
                config.mlp,
                f"{fast_llm_prefix}.mlp",
                f"{hf_prefix}.{cls.hf_mlp_name}",
                drop_on_export,
            ),
            *cls.normalization_converter_class.get_converters(
                config.normalization,
                f"{fast_llm_prefix}.norm_1",
                f"{hf_prefix}.{cls.hf_norm_1_name}",
                drop_on_export,
            ),
            *cls.normalization_converter_class.get_converters(
                config.normalization,
                f"{fast_llm_prefix}.norm_2",
                f"{hf_prefix}.{cls.hf_norm_2_name}",
                drop_on_export,
            ),
        ]


class GptOssDecoderConverter:
    """
    GPT-OSS decoder converter with heterogeneous block pattern support.

    Handles the `layer_types` field that specifies alternating attention patterns.
    """

    block_converter_class: typing.ClassVar[type[GptOssBlockConverter]] = GptOssBlockConverter

    @classmethod
    def _get_layer_type(cls, config: DecoderBlockConfig) -> str:
        """Determine layer type from block config."""
        match config.mixer:
            case AttentionConfig(window_size=window_size) if window_size is not None:
                return "sliding_attention"
            case _:
                return "full_attention"

    @classmethod
    def _find_minimal_repeating_pattern(cls, layer_types: list[str]) -> list[str]:
        """Find the minimal repeating pattern in layer_types.

        Uses the property that the period must divide the length.
        Tries periods in increasing order to find the smallest one.

        Examples:
        - ["A", "B", "A", "B"] -> ["A", "B"]
        - ["A", "B", "C", "A", "B", "C"] -> ["A", "B", "C"]
        - ["A", "B", "C"] -> ["A", "B", "C"] (no repetition)
        """
        n = len(layer_types)

        # Try each possible period length from 1 to n
        for period_len in range(1, n + 1):
            # Period must divide the total length evenly
            if n % period_len == 0:
                candidate_pattern = layer_types[:period_len]
                # Check if repeating this pattern reconstructs the full sequence
                num_repeats = n // period_len
                if candidate_pattern * num_repeats == layer_types:
                    return candidate_pattern

        # Fallback (should never reach here)
        return layer_types

    @classmethod
    def import_config(cls, config: dict) -> dict:
        """Import decoder config, handling heterogeneous layer types."""
        layer_types = config.get("layer_types", ["full_attention"])

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
            # Find the minimal repeating pattern to enable compact representation
            minimal_pattern = cls._find_minimal_repeating_pattern(layer_types)

            # Create a block config for each unique type in the minimal pattern
            # Use dict.fromkeys to preserve order while removing duplicates
            unique_in_pattern = list(dict.fromkeys(minimal_pattern))
            blocks = {}
            for layer_type in unique_in_pattern:
                layout_name = cls.block_converter_class.layout_names.get(layer_type, layer_type)
                blocks[layout_name] = cls.block_converter_class.import_config(config, layer_type)

            # Create pattern using layout names
            pattern = [cls.block_converter_class.layout_names.get(lt, lt) for lt in minimal_pattern]

            return {
                "type": "pattern",
                "blocks": blocks,
                "pattern": pattern,
                "num_blocks": config["num_hidden_layers"],
            }

    @classmethod
    def export_config(cls, config: BlockSequenceConfig) -> dict:
        """Export decoder config, reconstructing layer_types."""
        match config:
            case FixedBlockSequenceConfig():
                # All blocks are the same
                block_configs = [config.block]
                layer_type = cls._get_layer_type(config.block)
                layer_types = [layer_type] * config.num_blocks
            case PatternBlockSequenceConfig():
                # Multiple block types
                block_configs = list(config.blocks.values())
                # Reconstruct layer_types from expanded pattern
                # HuggingFace requires layer_types length to match num_hidden_layers
                layer_types = []
                for block_name in config.expanded_pattern:
                    block_config = config.blocks[block_name]
                    layer_type = cls._get_layer_type(block_config)
                    layer_types.append(layer_type)
            case _:
                raise NotImplementedError(f"Unsupported block sequence type: {type(config)}")

        # Export each block config and handle sliding_window conflicts
        exported_configs = [cls.block_converter_class.export_config(block_config) for block_config in block_configs]

        # Extract sliding_window values to handle heterogeneous blocks
        sliding_window = None
        for exported_config in exported_configs:
            window = exported_config.pop("sliding_window", None)
            if window is not None:
                sliding_window = window

        # Merge all block configs
        result = safe_merge_dicts(
            *exported_configs,
            {
                "num_hidden_layers": config.num_blocks,
                "layer_types": layer_types,
            },
        )

        # Add sliding_window back if any block had it
        if sliding_window is not None:
            result["sliding_window"] = sliding_window

        return result

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


class GptOssHeadConverter(LlamaHeadConverter):
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
    - Mixture of Experts (32-128 experts, 4 active per token)
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
