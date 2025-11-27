"""
Apriel2 multimodal checkpoint format converter.

Apriel2 multimodal uses inheritance (Apriel2Model inherits from Apriel2TextModel),
mirroring Fast-LLM's VisionMultiModalModel(LanguageModel) structure.

This converter is standalone (no LLaVA inheritance) to ensure weight paths match exactly.

Weight path mapping (Fast-LLM → HuggingFace):
- embeddings.word_embeddings_weight → model.embed_tokens.weight
- decoder.{i}.xxx → model.decoder.blocks.{i}.xxx
- head.final_norm.weight → model.norm.weight
- head.output_weights → lm_head.weight
- vision_encoder.patch_convolution.xxx → model.vision_encoder.patch_convolution.xxx
- vision_encoder.encoder.{i}.xxx → model.vision_encoder.encoder.blocks.{i}.xxx
- vision_encoder.adapter.xxx → model.vision_encoder.adapter.xxx

Config structure:
- Flat config (Apriel2Config inherits from Apriel2TextConfig)
- NOT nested (no text_config like LLaVA)
"""

import typing

from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.engine.checkpoint.external import WeightConverter
from fast_llm.engine.checkpoint.huggingface import HuggingfaceStateDictCheckpointHandler
from fast_llm.engine.multi_stage.config import FastLLMModelConfig
from fast_llm.functional.config import ActivationType
from fast_llm.layers.attention.config import AttentionConfig
from fast_llm.layers.attention.rotary.config import Rotary2DConfig
# Normalization config imports done locally where needed
from fast_llm.layers.decoder.mlp.config import MLPConfig
from fast_llm.layers.vision.config import PatchConvolutionConfig, VisionEncoderConfig
from fast_llm.models.gpt.conversion.apriel2 import (
    Apriel2BaseModelConverter,
    Apriel2DecoderConverter,
    Apriel2HeadConverter,
)
from fast_llm.models.gpt.conversion.llama import (
    KeyValueWeightConverter,
    LlamaEmbeddingsConverter,
    LlamaNormalizationConverter,
    MLPLayer2Converter,
    QueryWeightConverter,
    SplitWeightConverter,
    get_parameter_converter,
    get_weight_and_bias_converters,
)
from fast_llm.models.multimodal.config import MultiModalBaseModelConfig, MultiModalModelConfig
from fast_llm.models.multimodal.conversion.config import Apriel2CheckpointFormat
from fast_llm.models.multimodal.model import MultiModalModel
from fast_llm.utils import Assert, safe_merge_dicts


class Apriel2VisionNormalizationConverter(LlamaNormalizationConverter):
    """
    Vision encoder patch convolution normalization.

    Supports both RMSNorm (Fast-LLM default) and LayerNorm (HF default).
    - RMSNorm: weight only
    - LayerNorm: weight + bias
    """

    @classmethod
    def import_config(cls, config: dict) -> dict:
        # Default to RMSNorm to match Fast-LLM
        return {"type": "rms_norm", "epsilon": 1e-5}

    @classmethod
    def export_config(cls, config) -> dict:
        from fast_llm.layers.common.normalization.config import (
            LayerNormalizationConfig,
            RMSNormalizationConfig,
        )

        if isinstance(config, RMSNormalizationConfig):
            return {"normalization": {"type": "rms_norm", "eps": config.epsilon}}
        elif isinstance(config, LayerNormalizationConfig):
            return {"normalization": {"type": "layer_norm", "eps": config.epsilon}}
        else:
            raise ValueError(f"Unsupported normalization type: {type(config)}")

    @classmethod
    def get_converters(
        cls, config, fast_llm_prefix: str, hf_prefix: str, drop_on_export: bool = False
    ) -> list[WeightConverter]:
        """Get converters for normalization (handles both RMSNorm and LayerNorm)."""
        from fast_llm.layers.common.normalization.config import LayerNormalizationConfig

        converters = [
            get_parameter_converter(
                f"{fast_llm_prefix}.weight",
                f"{hf_prefix}.weight",
                drop_on_export=drop_on_export,
            ),
        ]

        # LayerNorm has bias, RMSNorm does not
        if isinstance(config, LayerNormalizationConfig):
            converters.append(
                get_parameter_converter(
                    f"{fast_llm_prefix}.bias",
                    f"{hf_prefix}.bias",
                    drop_on_export=drop_on_export,
                ),
            )

        return converters


class Apriel2VisionAttentionConverter:
    """Converter for vision encoder attention (non-causal, 2D rotary).

    Config structure mirrors Fast-LLM exactly:
    - heads: number of attention heads
    - head_groups: number of KV heads (equals heads for vision)
    - head_size: dimension per head
    - rotary: {type: default_2d, theta: ...}
    """

    @classmethod
    def import_config(cls, mixer_config: dict) -> dict:
        """Import vision attention config (already in Fast-LLM format)."""
        return {
            "type": "attention",
            "heads": mixer_config.get("heads", 16),
            "head_groups": mixer_config.get("head_groups", mixer_config.get("heads", 16)),
            "head_size": mixer_config.get("head_size", 64),
            "rotary": mixer_config.get("rotary", {"type": "default_2d", "theta": 10000.0}),
            "add_linear_biases": mixer_config.get("add_linear_biases", False),
            "causal": mixer_config.get("causal", False),  # Vision is non-causal by default
        }

    @classmethod
    def export_config(cls, config: AttentionConfig) -> dict:
        """Export vision attention config (to Fast-LLM format)."""
        from fast_llm.layers.attention.rotary.config import DefaultRotaryConfig, Rotary2DConfig

        # Determine rotary type
        if type(config.rotary) is Rotary2DConfig:
            rotary_type = "default_2d"
        elif type(config.rotary) is DefaultRotaryConfig:
            rotary_type = "default"
        else:
            rotary_type = "default_2d"

        return {
            "type": "attention",
            "heads": config.heads,
            "head_groups": config.head_groups,
            "head_size": config.head_size,
            "add_linear_biases": config.add_linear_biases,
            "causal": config.causal,
            "rotary": {
                "type": rotary_type,
                "theta": config.rotary.theta,
            },
        }


class Apriel2VisionBlockConverter:
    """Converter for vision encoder blocks.

    Config structure mirrors Fast-LLM exactly:
    block_config = {
        mixer: {type: attention, heads: N, ...}
        mlp: {type: mlp, intermediate_size: N, ...}
        normalization: {type: rms_norm, epsilon: 1e-5}
    }
    """

    mixer_converter_class: typing.ClassVar[type[Apriel2VisionAttentionConverter]] = Apriel2VisionAttentionConverter
    normalization_converter_class: typing.ClassVar[type[LlamaNormalizationConverter]] = LlamaNormalizationConverter

    @classmethod
    def import_config(cls, vision_config: dict, block_config: dict) -> dict:
        """Import block config (already in Fast-LLM format)."""
        mixer_config = block_config.get("mixer", {})
        mlp_config = block_config.get("mlp", {})
        norm_config = block_config.get("normalization", {"type": "rms_norm", "epsilon": 1e-5})

        return {
            "mixer": cls.mixer_converter_class.import_config(mixer_config),
            "mlp": {
                "type": "mlp",
                "intermediate_size": mlp_config.get("intermediate_size", vision_config.get("hidden_size", 1024) * 4),
                "activation": ActivationType.from_hf_name(mlp_config.get("activation", "silu")),
                "gated": mlp_config.get("gated", True),
                "add_linear_biases": mlp_config.get("add_linear_biases", False),
            },
            "normalization": {
                "type": norm_config.get("type", "rms_norm"),
                "epsilon": norm_config.get("epsilon", 1e-5),
            },
        }

    @classmethod
    def export_config(cls, config) -> dict:
        """Export block config (to Fast-LLM format)."""
        from fast_llm.layers.decoder.config import DecoderBlockConfig
        from fast_llm.layers.common.normalization.config import RMSNormalizationConfig

        Assert.custom(isinstance, config, DecoderBlockConfig)

        # Determine normalization type
        if isinstance(config.normalization, RMSNormalizationConfig):
            norm_type = "rms_norm"
        else:
            norm_type = "layer_norm"

        return {
            "mixer": cls.mixer_converter_class.export_config(config.mixer),
            "mlp": {
                "type": "mlp",
                "intermediate_size": config.mlp.intermediate_size,
                "activation": config.mlp.activation.value,
                "gated": config.mlp.gated,
                "add_linear_biases": config.mlp.add_linear_biases,
            },
            "normalization": {
                "type": norm_type,
                "epsilon": config.normalization.epsilon,
            },
        }

    @classmethod
    def get_converters(
        cls,
        config,
        fast_llm_prefix: str,
        hf_prefix: str,
    ) -> list[WeightConverter]:
        """Get weight converters for vision block."""
        converters = []

        # Attention converters - need QueryWeightConverter and KeyValueWeightConverter
        # for proper head dimension handling
        converters.extend([
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.mixer.query",
                f"{hf_prefix}.mixer.self_attn.q_proj",
                config.mixer.add_linear_biases,
                QueryWeightConverter,
                config.mixer,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.mixer.key_value",
                (f"{hf_prefix}.mixer.self_attn.k_proj", f"{hf_prefix}.mixer.self_attn.v_proj"),
                config.mixer.add_linear_biases,
                KeyValueWeightConverter,
                config.mixer,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.mixer.dense",
                f"{hf_prefix}.mixer.self_attn.o_proj",
                config.mixer.add_linear_biases,
            ),
        ])

        # MLP converters - gated MLP (MistralMLP has gate_proj, up_proj, down_proj)
        converters.extend([
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.mlp.layer_1",
                (f"{hf_prefix}.mlp.gate_proj", f"{hf_prefix}.mlp.up_proj"),
                config.mlp.add_linear_biases,
                SplitWeightConverter,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.mlp.layer_2",
                f"{hf_prefix}.mlp.down_proj",
                config.mlp.add_linear_biases,
                MLPLayer2Converter,
            ),
        ])

        # Normalization converters
        converters.extend([
            *cls.normalization_converter_class.get_converters(
                config.normalization,
                f"{fast_llm_prefix}.norm_1",
                f"{hf_prefix}.input_layernorm",
            ),
            *cls.normalization_converter_class.get_converters(
                config.normalization,
                f"{fast_llm_prefix}.norm_2",
                f"{hf_prefix}.post_attention_layernorm",
            ),
        ])

        return converters


class Apriel2VisionEncoderDecoderConverter:
    """Converter for vision encoder block sequence."""

    block_converter_class: typing.ClassVar[type[Apriel2VisionBlockConverter]] = Apriel2VisionBlockConverter

    @classmethod
    def import_config(cls, config: dict) -> dict:
        """Import encoder config from Apriel2 vision format."""
        encoder_config = config.get("encoder", {})
        num_blocks = encoder_config.get("num_blocks", config.get("num_hidden_layers", 24))

        # Vision encoder uses fixed block type
        block_config = encoder_config.get("block", {})
        imported_block = cls.block_converter_class.import_config(config, block_config)

        return {
            "type": "fixed",
            "num_blocks": num_blocks,
            "block": imported_block,
        }

    @classmethod
    def export_config(cls, config) -> dict:
        """Export encoder config to Apriel2 vision format."""
        from fast_llm.layers.block.config import FixedBlockSequenceConfig

        Assert.custom(isinstance, config, FixedBlockSequenceConfig)
        return {
            "encoder": {
                "type": "fixed",
                "num_blocks": config.num_blocks,
                "block": cls.block_converter_class.export_config(config.block),
            },
            "num_hidden_layers": config.num_blocks,
        }

    @classmethod
    def get_converters(
        cls,
        config,
        fast_llm_prefix: str,
        hf_prefix: str,
    ) -> list[WeightConverter]:
        """Get weight converters for encoder."""
        from fast_llm.layers.block.config import FixedBlockSequenceConfig

        converters = []
        Assert.custom(isinstance, config, FixedBlockSequenceConfig)

        for block_index in range(config.num_blocks):
            converters += cls.block_converter_class.get_converters(
                config.block,
                f"{fast_llm_prefix}.{block_index}",
                f"{hf_prefix}.{block_index}",
            )

        return converters


class Apriel2PatchConvolutionConverter:
    """Converter for vision patch convolution."""

    normalization_converter_class: typing.ClassVar[type[Apriel2VisionNormalizationConverter]] = (
        Apriel2VisionNormalizationConverter
    )

    @classmethod
    def import_config(cls, config: dict) -> dict:
        """Import patch convolution config."""
        patch_conv_config = config.get("patch_convolution", {})
        Assert.eq(patch_conv_config.get("input_channels", 3), 3)
        return {
            "normalization": cls.normalization_converter_class.import_config(config),
            "patch_height": patch_conv_config.get("patch_height", config.get("patch_size", 16)),
            "patch_width": patch_conv_config.get("patch_width", config.get("patch_size", 16)),
        }

    @classmethod
    def export_config(cls, config: PatchConvolutionConfig) -> dict:
        """Export patch convolution config."""
        Assert.custom(isinstance, config, PatchConvolutionConfig)
        Assert.eq(config.patch_height, config.patch_width)
        Assert.incl(config.convolution.bias.enabled, (None, False))

        # Get normalization export (returns {"normalization": {...}})
        norm_export = cls.normalization_converter_class.export_config(config.normalization)

        # Build patch_convolution dict with normalization nested inside
        patch_conv_dict = {
            "patch_height": config.patch_height,
            "patch_width": config.patch_width,
            "input_channels": config.input_channels,
        }
        # Merge normalization into patch_convolution
        if "normalization" in norm_export:
            patch_conv_dict["normalization"] = norm_export["normalization"]

        return {
            "patch_convolution": patch_conv_dict,
            "patch_size": config.patch_height,
            "num_channels": config.input_channels,
        }

    @classmethod
    def get_converters(
        cls, config: PatchConvolutionConfig, fast_llm_prefix: str, hf_prefix: str
    ) -> list[WeightConverter]:
        """Get weight converters for patch convolution."""
        return [
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.convolution",
                f"{hf_prefix}.conv",
                False,
            ),
            *cls.normalization_converter_class.get_converters(
                config.normalization, f"{fast_llm_prefix}.normalization", f"{hf_prefix}.norm"
            ),
        ]


class Apriel2VisionAdapterConverter:
    """Converter for vision adapter/projector."""

    @classmethod
    def import_config(cls, config: dict) -> dict:
        """Import adapter config."""
        adapter_config = config.get("adapter", {})
        return {
            "intermediate_size": adapter_config.get("intermediate_size", config.get("hidden_size")),
            "add_linear_biases": adapter_config.get("add_linear_biases", True),
            "gated": False,
            "activation": ActivationType.from_hf_name(adapter_config.get("activation", "gelu_pytorch_tanh")),
        }

    @classmethod
    def export_config(cls, config: MLPConfig) -> dict:
        """Export adapter config."""
        Assert.custom(isinstance, config, MLPConfig)
        Assert.incl(config.layer_1.bias.enabled, (None, config.add_linear_biases))
        Assert.incl(config.layer_2.bias.enabled, (None, config.add_linear_biases))
        assert not config.gated

        return {
            "adapter": {
                "type": "mlp",
                "intermediate_size": config.intermediate_size,
                "activation": config.activation.hf_name,
                "add_linear_biases": config.add_linear_biases,
            },
        }

    @classmethod
    def get_converters(cls, config: MLPConfig, fast_llm_prefix: str, hf_prefix: str) -> list[WeightConverter]:
        """Get weight converters for adapter."""
        return [
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.layer_1",
                f"{hf_prefix}.linear_1",
                config.add_linear_biases,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.layer_2",
                f"{hf_prefix}.linear_2",
                config.add_linear_biases,
                MLPLayer2Converter,
            ),
        ]


class Apriel2VisionModelConverter:
    """Converter for complete vision encoder (patch conv + encoder + adapter)."""

    patch_convolution_converter_class: typing.ClassVar[type[Apriel2PatchConvolutionConverter]] = (
        Apriel2PatchConvolutionConverter
    )
    encoder_converter_class: typing.ClassVar[type[Apriel2VisionEncoderDecoderConverter]] = (
        Apriel2VisionEncoderDecoderConverter
    )
    adapter_converter_class: typing.ClassVar[type[Apriel2VisionAdapterConverter]] = Apriel2VisionAdapterConverter

    @classmethod
    def import_config(cls, config: dict) -> dict:
        """Import complete vision encoder config."""
        vision_config = config.get("vision_encoder", {})
        return {
            "patch_convolution": cls.patch_convolution_converter_class.import_config(vision_config),
            "encoder": cls.encoder_converter_class.import_config(vision_config),
            "adapter": cls.adapter_converter_class.import_config(vision_config),
            "hidden_size": vision_config.get("hidden_size", 1024),
        }

    @classmethod
    def export_config(cls, config: VisionEncoderConfig) -> dict:
        """Export complete vision encoder config."""
        Assert.custom(isinstance, config, VisionEncoderConfig)

        vision_config = safe_merge_dicts(
            cls.patch_convolution_converter_class.export_config(config.patch_convolution),
            cls.encoder_converter_class.export_config(config.encoder),
            {"hidden_size": config.hidden_size},
        )

        return safe_merge_dicts(
            {"vision_encoder": vision_config},
            cls.adapter_converter_class.export_config(config.adapter),
        )

    @classmethod
    def get_converters(cls, config: VisionEncoderConfig) -> list[WeightConverter]:
        """Get weight converters for complete vision encoder."""
        return [
            *cls.patch_convolution_converter_class.get_converters(
                config.patch_convolution, "vision_encoder.patch_convolution", "model.vision_encoder.patch_convolution"
            ),
            *cls.encoder_converter_class.get_converters(
                config.encoder, "vision_encoder.encoder", "model.vision_encoder.encoder.blocks"
            ),
            *cls.adapter_converter_class.get_converters(
                config.adapter, "vision_encoder.adapter", "model.vision_encoder.adapter"
            ),
        ]


class Apriel2MultimodalHeadConverter(Apriel2HeadConverter):
    """Head converter for Apriel2 multimodal (same paths as text-only)."""

    @classmethod
    def get_converters(
        cls,
        config,
        exported_config: dict,
        fast_llm_prefix: str,
    ) -> list[WeightConverter]:
        """Get weight converters for head."""
        return [
            *cls.normalization_converter_class.get_converters(
                config.normalization,
                f"{fast_llm_prefix}.final_norm",
                "model.norm",  # Same as text-only (inheritance)
            ),
            get_parameter_converter(
                f"{fast_llm_prefix}.output_weights",
                "lm_head.weight",
                drop_on_import=exported_config.get("tie_word_embeddings", False),
                drop_on_export=exported_config.get("tie_word_embeddings", False),
            ),
        ]


class Apriel2MultimodalBaseModelConverter:
    """
    Base model converter for Apriel2 multimodal (standalone, no LLaVA inheritance).

    Weight paths (all under model.):
    - embed_tokens: embeddings (inherited from text)
    - decoder.blocks: decoder blocks (inherited from text)
    - norm: final norm (inherited from text)
    - vision_encoder: vision encoder (added)
    - lm_head: output head

    Config structure:
    - Flat (Apriel2Config inherits from Apriel2TextConfig)
    - NOT nested (no text_config like LLaVA)
    """

    vision_model_converter_class: typing.ClassVar[type[Apriel2VisionModelConverter]] = Apriel2VisionModelConverter
    decoder_converter_class: typing.ClassVar[type[Apriel2DecoderConverter]] = Apriel2DecoderConverter
    embeddings_converter_class: typing.ClassVar[type[LlamaEmbeddingsConverter]] = LlamaEmbeddingsConverter
    head_converter_class: typing.ClassVar[type[Apriel2MultimodalHeadConverter]] = Apriel2MultimodalHeadConverter

    @classmethod
    def import_config(cls, config: dict) -> dict:
        """Import multimodal config from Apriel2 format (flat structure)."""
        # Import text components using text converter
        text_config = Apriel2BaseModelConverter.import_config(config)

        # Import vision encoder
        vision_config = cls.vision_model_converter_class.import_config(config) if config.get("vision_encoder") else None

        return safe_merge_dicts(
            text_config,
            {
                "vision_encoder": vision_config,
                "image_token_index": config.get("image_token_index"),
            },
        )

    @classmethod
    def export_config(cls, config: MultiModalBaseModelConfig) -> dict:
        """Export multimodal config to Apriel2 format (flat structure)."""
        Assert.custom(isinstance, config, MultiModalBaseModelConfig)

        # Export text components using text converter
        exported = Apriel2BaseModelConverter.export_config(config)

        # Export vision encoder if present
        if config.vision_encoder is not None:
            exported = safe_merge_dicts(
                exported,
                cls.vision_model_converter_class.export_config(config.vision_encoder),
            )

        # Add image token index
        if config.image_token_index is not None:
            exported["image_token_index"] = config.image_token_index

        return exported

    @classmethod
    def get_converters(cls, config: MultiModalBaseModelConfig, exported_config: dict) -> list[WeightConverter]:
        """Get weight converters with Apriel2-specific paths."""
        converters = []

        # Vision encoder converters
        if config.vision_encoder is not None:
            converters.extend(cls.vision_model_converter_class.get_converters(config.vision_encoder))

        # Text component converters (same paths as text-only, due to inheritance)
        converters.extend(
            cls.embeddings_converter_class.get_converters(config.embeddings, "embeddings", "model")
        )
        converters.extend(
            cls.decoder_converter_class.get_converters(config.decoder, "decoder", "model.decoder.blocks")
        )
        converters.extend(
            cls.head_converter_class.get_converters(config.head, exported_config, "head")
        )

        return converters


class Apriel2HuggingfaceCheckpointHandler(HuggingfaceStateDictCheckpointHandler):
    """HuggingFace checkpoint handler for Apriel2 multimodal format (standalone)."""

    _model: MultiModalModel
    _model_class: typing.ClassVar[FastLLMModelConfig] = MultiModalModelConfig
    format: typing.ClassVar[type[CheckpointFormat]] = Apriel2CheckpointFormat
    architecture: typing.ClassVar[str] = "Apriel2ForConditionalGeneration"
    base_model_converter_class: typing.ClassVar[type[Apriel2MultimodalBaseModelConverter]] = (
        Apriel2MultimodalBaseModelConverter
    )

    @classmethod
    def get_huggingface_model_type(cls) -> str:
        return "apriel2"

    @classmethod
    def get_transformers_configuration_class(cls):
        from fast_llm_external_models.apriel2.configuration_apriel2 import Apriel2Config

        return Apriel2Config

    @classmethod
    def get_model_files(cls) -> tuple[str, str, str | None]:
        from fast_llm_external_models.apriel2 import (
            configuration_apriel2,
            modeling_apriel2,
        )

        return configuration_apriel2.__file__, modeling_apriel2.__file__, None

    @classmethod
    def _export_config(cls, config: MultiModalModelConfig) -> dict[str, typing.Any]:
        """Export config - flat structure (no super() call to LLaVA)."""
        base_model = config.base_model
        exported = safe_merge_dicts(
            cls.base_model_converter_class.export_config(base_model),
            {
                "architectures": [cls.architecture],
                "model_type": cls.get_huggingface_model_type(),
                "auto_map": {
                    "AutoConfig": "configuration_apriel2.Apriel2Config",
                    "AutoModel": "modeling_apriel2.Apriel2Model",
                    "AutoModelForCausalLM": "modeling_apriel2.Apriel2ForConditionalGeneration",
                },
            },
        )
        return exported

    @classmethod
    def _import_config(cls, config: dict[str, typing.Any]) -> dict[str, typing.Any]:
        """Import config - flat structure (not nested like LLaVA)."""
        return {"base_model": cls.base_model_converter_class.import_config(config)}

    @classmethod
    def _get_weight_converters(cls, config: MultiModalModelConfig, export_config: dict) -> list[WeightConverter]:
        """Get weight converters."""
        return cls.base_model_converter_class.get_converters(config.base_model, export_config)
