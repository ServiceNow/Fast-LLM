"""Apriel2 multimodal checkpoint format converter."""

import typing

from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.engine.checkpoint.external import WeightConverter
from fast_llm.engine.checkpoint.huggingface import HuggingfaceStateDictCheckpointHandler
from fast_llm.engine.multi_stage.config import FastLLMModelConfig
from fast_llm.functional.config import ActivationType
from fast_llm.layers.attention.config import AttentionConfig
from fast_llm.layers.decoder.mlp.config import MLPConfig
from fast_llm.layers.vision.config import PatchConvolutionConfig, VisionEncoderConfig
from fast_llm.models.gpt.conversion.apriel2 import (
    Apriel2BaseModelConverter,
    Apriel2DecoderConverter,
    Apriel2HeadConverter,
)
from fast_llm.models.gpt.conversion.llama import (
    LlamaEmbeddingsConverter,
    LlamaNormalizationConverter,
    get_parameter_converter,
    get_weight_and_bias_converters,
)
from fast_llm.models.multimodal.config import MultiModalBaseModelConfig, MultiModalModelConfig
from fast_llm.models.multimodal.conversion.config import Apriel2CheckpointFormat
from fast_llm.models.multimodal.conversion.llava import (
    LlavaVisionAdapterConverter,
    LlavaVisionModelConverter,
    PixtralAttentionConverter,
    PixtralBlockConverter,
    PixtralEncoderConverter,
)
from fast_llm.models.multimodal.model import MultiModalModel
from fast_llm.utils import Assert, safe_merge_dicts


class Apriel2VisionAttentionConverter(PixtralAttentionConverter):
    @classmethod
    def import_config(cls, config: dict) -> dict:
        out = {
            "rotary": config.get("rotary", {"type": "default_2d", "theta": 10000.0}),
            "heads": config.get("heads", config.get("num_attention_heads", 16)),
            "head_groups": config.get("head_groups", config.get("heads", 16)),
            "head_size": config.get("head_size", 64),
            "add_linear_biases": config.get("add_linear_biases", False),
            "causal": config.get("causal", False),
        }
        if isinstance(out["rotary"], dict) and out["rotary"].get("type") == "default":
            out["rotary"]["type"] = "default_2d"
        return out

    @classmethod
    def export_config(cls, config: AttentionConfig) -> dict:
        from fast_llm.layers.attention.rotary.config import DefaultRotaryConfig, Rotary2DConfig

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


class Apriel2VisionBlockConverter(PixtralBlockConverter):
    mixer_converter_class: typing.ClassVar[type[Apriel2VisionAttentionConverter]] = Apriel2VisionAttentionConverter
    hf_mixer_name: typing.ClassVar[str] = "mixer.self_attn"
    hf_mlp_name: typing.ClassVar[str] = "mlp"
    hf_norm_1_name: typing.ClassVar[str] = "input_layernorm"
    hf_norm_2_name: typing.ClassVar[str] = "post_attention_layernorm"

    @classmethod
    def import_config(cls, config: dict, block_config: dict) -> dict:
        mixer_config = block_config.get("mixer", {})
        mlp_config = block_config.get("mlp", {})
        norm_config = block_config.get("normalization", {"type": "rms_norm", "epsilon": 1e-5})

        return {
            "mixer": cls.mixer_converter_class.import_config(mixer_config),
            "mlp": {
                "type": "mlp",
                "intermediate_size": mlp_config.get("intermediate_size", config.get("hidden_size", 1024) * 4),
                "activation": ActivationType.from_hf_name(mlp_config.get("activation", "silu")),
                "gated": mlp_config.get("gated", True),
                "add_linear_biases": mlp_config.get("add_linear_biases", False),
            },
            "normalization": cls.normalization_converter_class.import_config(norm_config),
        }

    @classmethod
    def export_config(cls, config) -> dict:
        from fast_llm.layers.decoder.config import DecoderBlockConfig

        Assert.custom(isinstance, config, DecoderBlockConfig)
        return {
            "mixer": cls.mixer_converter_class.export_config(config.mixer),
            "mlp": {
                "type": "mlp",
                "intermediate_size": config.mlp.intermediate_size,
                "activation": config.mlp.activation.hf_name,
                "gated": config.mlp.gated,
                "add_linear_biases": config.mlp.add_linear_biases,
            },
            "normalization": {
                "type": "rms_norm",
                "epsilon": config.normalization.epsilon,
            },
        }


class Apriel2VisionEncoderConverter(PixtralEncoderConverter):
    block_converter_class: typing.ClassVar[type[Apriel2VisionBlockConverter]] = Apriel2VisionBlockConverter

    @classmethod
    def import_config(cls, config: dict) -> dict:
        encoder_config = config.get("encoder", {})
        num_blocks = encoder_config.get("num_blocks", config.get("num_hidden_layers", 24))
        block_config = encoder_config.get("block", {})

        return {
            "type": "fixed",
            "num_blocks": num_blocks,
            "block": cls.block_converter_class.import_config(config, block_config),
        }

    @classmethod
    def export_config(cls, config) -> dict:
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


class Apriel2PatchConvolutionConverter:
    normalization_converter_class: typing.ClassVar[type[LlamaNormalizationConverter]] = LlamaNormalizationConverter

    @classmethod
    def import_config(cls, config: dict) -> dict:
        patch_conv_config = config.get("patch_convolution", {})
        Assert.eq(patch_conv_config.get("input_channels", 3), 3)
        return {
            "normalization": {"type": "rms_norm", "epsilon": 1e-5},
            "patch_height": patch_conv_config.get("patch_height", config.get("patch_size", 16)),
            "patch_width": patch_conv_config.get("patch_width", config.get("patch_size", 16)),
        }

    @classmethod
    def export_config(cls, config: PatchConvolutionConfig) -> dict:
        Assert.custom(isinstance, config, PatchConvolutionConfig)
        Assert.eq(config.patch_height, config.patch_width)
        Assert.incl(config.convolution.bias.enabled, (None, False))

        return {
            "patch_convolution": {
                "patch_height": config.patch_height,
                "patch_width": config.patch_width,
                "input_channels": config.input_channels,
                "normalization": {"type": "rms_norm", "epsilon": config.normalization.epsilon},
            },
            "patch_size": config.patch_height,
            "num_channels": config.input_channels,
        }

    @classmethod
    def get_converters(
        cls, config: PatchConvolutionConfig, fast_llm_prefix: str, hf_prefix: str
    ) -> list[WeightConverter]:
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


class Apriel2VisionAdapterConverter(LlavaVisionAdapterConverter):
    @classmethod
    def import_config(cls, config: dict) -> dict:
        adapter_config = config.get("adapter", {})
        return {
            "intermediate_size": adapter_config.get("intermediate_size", config.get("hidden_size")),
            "add_linear_biases": adapter_config.get("add_linear_biases", True),
            "gated": False,
            "activation": ActivationType.from_hf_name(adapter_config.get("activation", "gelu_pytorch_tanh")),
        }

    @classmethod
    def export_config(cls, config: MLPConfig) -> dict:
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


class Apriel2VisionModelConverter(LlavaVisionModelConverter):
    vision_adapter_converter_class: typing.ClassVar[type[Apriel2VisionAdapterConverter]] = (
        Apriel2VisionAdapterConverter
    )
    patch_convolution_converter_class: typing.ClassVar[type[Apriel2PatchConvolutionConverter]] = (
        Apriel2PatchConvolutionConverter
    )
    encoder_converter_class: typing.ClassVar[type[Apriel2VisionEncoderConverter]] = Apriel2VisionEncoderConverter

    # HF path prefixes for Apriel2
    hf_patch_conv_prefix: typing.ClassVar[str] = "model.vision_encoder.patch_convolution"
    hf_encoder_prefix: typing.ClassVar[str] = "model.vision_encoder.encoder.blocks"
    hf_adapter_prefix: typing.ClassVar[str] = "model.vision_encoder.adapter"

    @classmethod
    def import_config(cls, config: dict) -> dict:
        vision_config = config.get("vision_encoder", {})
        return {
            "patch_convolution": cls.patch_convolution_converter_class.import_config(vision_config),
            "encoder": cls.encoder_converter_class.import_config(vision_config),
            "adapter": cls.vision_adapter_converter_class.import_config(vision_config),
            "hidden_size": vision_config.get("hidden_size", 1024),
        }

    @classmethod
    def export_config(cls, config: VisionEncoderConfig) -> dict:
        Assert.custom(isinstance, config, VisionEncoderConfig)

        vision_config = safe_merge_dicts(
            cls.patch_convolution_converter_class.export_config(config.patch_convolution),
            cls.encoder_converter_class.export_config(config.encoder),
            {"hidden_size": config.hidden_size},
        )

        return safe_merge_dicts(
            {"vision_encoder": vision_config},
            cls.vision_adapter_converter_class.export_config(config.adapter),
        )

    @classmethod
    def get_converters(cls, config: VisionEncoderConfig) -> list[WeightConverter]:
        return [
            *cls.patch_convolution_converter_class.get_converters(
                config.patch_convolution, "vision_encoder.patch_convolution", cls.hf_patch_conv_prefix
            ),
            *cls.encoder_converter_class.get_converters(
                config.encoder, "vision_encoder.encoder", cls.hf_encoder_prefix
            ),
            *cls.vision_adapter_converter_class.get_converters(
                config.adapter, "vision_encoder.adapter", cls.hf_adapter_prefix
            ),
        ]


class Apriel2MultimodalHeadConverter(Apriel2HeadConverter):
    @classmethod
    def get_converters(
        cls,
        config,
        exported_config: dict,
        fast_llm_prefix: str,
    ) -> list[WeightConverter]:
        return [
            *cls.normalization_converter_class.get_converters(
                config.normalization,
                f"{fast_llm_prefix}.final_norm",
                "model.norm",
            ),
            get_parameter_converter(
                f"{fast_llm_prefix}.output_weights",
                "lm_head.weight",
                drop_on_import=exported_config.get("tie_word_embeddings", False),
                drop_on_export=exported_config.get("tie_word_embeddings", False),
            ),
        ]


class Apriel2MultimodalBaseModelConverter:
    vision_model_converter_class: typing.ClassVar[type[Apriel2VisionModelConverter]] = Apriel2VisionModelConverter
    decoder_converter_class: typing.ClassVar[type[Apriel2DecoderConverter]] = Apriel2DecoderConverter
    embeddings_converter_class: typing.ClassVar[type[LlamaEmbeddingsConverter]] = LlamaEmbeddingsConverter
    head_converter_class: typing.ClassVar[type[Apriel2MultimodalHeadConverter]] = Apriel2MultimodalHeadConverter

    @classmethod
    def import_config(cls, config: dict) -> dict:
        text_config = Apriel2BaseModelConverter.import_config(config)
        vision_config = (
            cls.vision_model_converter_class.import_config(config) if config.get("vision_encoder") else None
        )

        return safe_merge_dicts(
            text_config,
            {
                "vision_encoder": vision_config,
                "image_token_index": config.get("image_token_index"),
            },
        )

    @classmethod
    def export_config(cls, config: MultiModalBaseModelConfig) -> dict:
        Assert.custom(isinstance, config, MultiModalBaseModelConfig)
        exported = Apriel2BaseModelConverter.export_config(config)
        if config.vision_encoder is not None:
            exported = safe_merge_dicts(
                exported,
                cls.vision_model_converter_class.export_config(config.vision_encoder),
            )

        if config.image_token_index is not None:
            exported["image_token_index"] = config.image_token_index

        return exported

    @classmethod
    def get_converters(cls, config: MultiModalBaseModelConfig, exported_config: dict) -> list[WeightConverter]:
        converters = []
        if config.vision_encoder is not None:
            converters.extend(cls.vision_model_converter_class.get_converters(config.vision_encoder))
        converters.extend(cls.embeddings_converter_class.get_converters(config.embeddings, "embeddings", "model"))
        converters.extend(
            cls.decoder_converter_class.get_converters(config.decoder, "decoder", "model.decoder.blocks")
        )
        converters.extend(cls.head_converter_class.get_converters(config.head, exported_config, "head"))

        return converters


class Apriel2HuggingfaceCheckpointHandler(HuggingfaceStateDictCheckpointHandler):
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
        return {"base_model": cls.base_model_converter_class.import_config(config)}

    @classmethod
    def _get_weight_converters(cls, config: MultiModalModelConfig, export_config: dict) -> list[WeightConverter]:
        return cls.base_model_converter_class.get_converters(config.base_model, export_config)
