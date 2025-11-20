import typing

from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.engine.checkpoint.external import WeightConverter
from fast_llm.engine.checkpoint.huggingface import HuggingFaceBaseModelConverter, HuggingfaceStateDictCheckpointHandler
from fast_llm.engine.multi_stage.config import FastLLMModelConfig
from fast_llm.functional.config import ActivationType
from fast_llm.layers.attention.config import AttentionConfig
from fast_llm.layers.attention.rotary.config import Rotary2DConfig
from fast_llm.layers.common.normalization.config import RMSNormalizationConfig
from fast_llm.layers.decoder.mlp.config import MLPConfig
from fast_llm.layers.vision.config import PatchConvolutionConfig, VisionEncoderConfig
from fast_llm.models.gpt.conversion.llama import (
    LlamaAttentionConverter,
    LlamaBlockConverter,
    LlamaDecoderConverter,
    LlamaNormalizationConverter,
    MLPLayer2Converter,
    get_weight_and_bias_converters,
)
from fast_llm.models.gpt.conversion.mistral import MistralBaseModelConverter, MistralMLPConverter
from fast_llm.models.multimodal.config import MultiModalBaseModelConfig, MultiModalModelConfig
from fast_llm.models.multimodal.conversion.config import LlavaCheckpointFormat
from fast_llm.models.multimodal.model import MultiModalModel
from fast_llm.utils import Assert, div, safe_merge_dicts


class PixtralNormalizationConverter(LlamaNormalizationConverter):
    """
    epsilon hard-coded to 1e-5.
    """

    @classmethod
    def import_config(cls, config: dict) -> dict:
        return {"type": "rms_norm", "epsilon": 1e-5}

    @classmethod
    def export_config(cls, config: RMSNormalizationConfig) -> dict:
        Assert.custom(isinstance, config, RMSNormalizationConfig)
        assert not config.zero_centered
        # TODO: Too strict?
        Assert.eq(config.epsilon, 1e-5)
        return {}


class PixtralAttentionConverter(LlamaAttentionConverter):
    @classmethod
    def import_config(cls, config: dict) -> dict:
        config["num_key_value_heads"] = config["num_attention_heads"]
        config["attention_bias"] = False
        out = super().import_config(config)
        out["rotary"]["type"] = "default_2d"
        return out

    @classmethod
    def export_config(cls, config: AttentionConfig) -> dict:
        cls._check_config(config)
        Assert.eq(config.softmax_scale_power, 0.5)
        Assert.is_(type(config.rotary), Rotary2DConfig)
        assert not config.add_linear_biases
        Assert.eq(config.head_groups, config.heads)
        return {
            "num_attention_heads": config.heads,
            "attention_dropout": config.dropout,
            "rope_theta": config.rotary.theta,
            # Not in PixtralConfig, but needed for consistency check in LlavaVisionModelConverter.
            "head_dim": config.head_size,
        }


class PixtralBlockConverter(LlamaBlockConverter):
    mixer_converter_class: typing.ClassVar[type[PixtralAttentionConverter]] = PixtralAttentionConverter
    # TODO: ====== MistralMLPConverter (#391 / #382) ======
    mlp_converter_class: typing.ClassVar[type[MistralMLPConverter]] = MistralMLPConverter
    normalization_converter_class: typing.ClassVar[type[PixtralNormalizationConverter]] = PixtralNormalizationConverter
    hf_mixer_name: typing.ClassVar[str] = "attention"
    hf_mlp_name: typing.ClassVar[str] = "feed_forward"
    hf_norm_1_name: typing.ClassVar[str] = "attention_norm"
    hf_norm_2_name: typing.ClassVar[str] = "ffn_norm"


class PixtralEncoderConverter(LlamaDecoderConverter):
    block_converter_class: typing.ClassVar[type[PixtralBlockConverter]] = PixtralBlockConverter


class PixtralPatchConvolutionConverter:
    normalization_converter_class: typing.ClassVar[type[PixtralNormalizationConverter]] = PixtralNormalizationConverter

    @classmethod
    def import_config(cls, config: dict) -> dict:
        Assert.eq(config["num_channels"], 3)
        return {
            "normalization": cls.normalization_converter_class.import_config(config),
            "patch_height": config["patch_size"],
            "patch_width": config["patch_size"],
        }

    @classmethod
    def export_config(cls, config: PatchConvolutionConfig) -> dict:
        Assert.custom(isinstance, config, PatchConvolutionConfig)
        Assert.eq(config.patch_height, config.patch_width)
        Assert.incl(config.convolution.bias.enabled, (None, False))

        return safe_merge_dicts(
            {
                "patch_size": config.patch_height,
                "num_channels": config.input_channels,
            },
            cls.normalization_converter_class.export_config(config.normalization),
        )

    @classmethod
    def get_converters(
        cls, config: PatchConvolutionConfig, fast_llm_prefix: str, hf_prefix: str
    ) -> list[WeightConverter]:
        return [
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.convolution",
                f"{hf_prefix}.patch_conv",
                False,
                WeightConverter,
            ),
            *cls.normalization_converter_class.get_converters(
                config, f"{fast_llm_prefix}.normalization", f"{hf_prefix}.ln_pre"
            ),
        ]


class LlavaVisionAdapterConverter:
    @classmethod
    def import_config(cls, config: dict) -> dict:
        return {
            "intermediate_size": config["projector_intermediate_size"],
            "add_linear_biases": config["multimodal_projector_bias"],
            "gated": False,
            "activation": ActivationType.from_hf_name(config["projector_hidden_act"]),
        }

    @classmethod
    def export_config(cls, config: MLPConfig) -> dict:
        Assert.custom(isinstance, config, MLPConfig)
        Assert.incl(config.layer_1.bias.enabled, (None, config.add_linear_biases))
        Assert.incl(config.layer_2.bias.enabled, (None, config.add_linear_biases))
        assert not config.gated

        return {
            "projector_hidden_act": config.activation.hf_name,
            "projector_intermediate_size": config.intermediate_size,
            "multimodal_projector_bias": config.add_linear_biases,
        }

    @classmethod
    def get_converters(cls, config: MLPConfig, fast_llm_prefix: str, hf_prefix: str) -> list[WeightConverter]:
        return [
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.layer_1",
                f"{hf_prefix}.linear_1",
                config.add_linear_biases,
                WeightConverter,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.layer_2",
                f"{hf_prefix}.linear_2",
                config.add_linear_biases,
                MLPLayer2Converter,
            ),
        ]


class LlavaVisionModelConverter:
    vision_adapter_converter_class: typing.ClassVar[type[LlavaVisionAdapterConverter]] = LlavaVisionAdapterConverter
    patch_convolution_converter_class: typing.ClassVar[type[PixtralPatchConvolutionConverter]] = (
        PixtralPatchConvolutionConverter
    )
    encoder_converter_class: typing.ClassVar[type[PixtralEncoderConverter]] = PixtralEncoderConverter

    @classmethod
    def import_config(cls, config: dict) -> dict:
        return {
            "patch_convolution": cls.patch_convolution_converter_class.import_config(config["vision_config"]),
            "encoder": cls.encoder_converter_class.import_config(config["vision_config"]),
            "adapter": cls.vision_adapter_converter_class.import_config(config),
            "hidden_size": config["vision_config"]["hidden_size"],
        }

    @classmethod
    def export_config(cls, config: VisionEncoderConfig) -> dict:
        Assert.custom(isinstance, config, VisionEncoderConfig)
        # TODO: ====== image_size? ======
        vision_config = safe_merge_dicts(
            cls.patch_convolution_converter_class.export_config(config.patch_convolution),
            cls.encoder_converter_class.export_config(config.encoder),
            {"hidden_size": config.hidden_size},
        )

        Assert.eq(
            vision_config.pop("head_dim"), div(vision_config["hidden_size"], vision_config["num_attention_heads"])
        )

        return safe_merge_dicts(
            {"vision_config": vision_config},
            cls.vision_adapter_converter_class.export_config(config.adapter),
            # TODO: ====== What about these? ======
            # {
            #     "image_token_index":32000,
            #     "vision_feature_select_strategy":"default",
            #     "vision_feature_layer":-2,
            #     "image_seq_length":576,
            # }
        )

    @classmethod
    def get_converters(
        cls, config: VisionEncoderConfig, fast_llm_prefix: str, hf_prefix: str
    ) -> list[WeightConverter]:
        return [
            *cls.patch_convolution_converter_class.get_converters(
                config.patch_convolution, f"{fast_llm_prefix}.patch_convolution", hf_prefix
            ),
            *cls.encoder_converter_class.get_converters(
                config.encoder, f"{fast_llm_prefix}.encoder", f"{hf_prefix}.transformer"
            ),
            *cls.vision_adapter_converter_class.get_converters(
                config.adapter, f"{fast_llm_prefix}.adapter", f"{hf_prefix}.multi_modal_projector"
            ),
        ]


class LlavaBaseModelConverter(HuggingFaceBaseModelConverter):
    vision_model_converter_class: typing.ClassVar[type[LlavaVisionModelConverter]] = LlavaVisionModelConverter
    # TODO: Make it flexible?
    language_model_converter_class: typing.ClassVar[type[MistralBaseModelConverter]] = MistralBaseModelConverter
    # TODO: ====== Is tie_word_embeddings supported? ======

    @classmethod
    def import_config(cls, config: dict) -> dict:
        return safe_merge_dicts(
            {"vision_encoder": cls.vision_model_converter_class.import_config(config)},
            cls.language_model_converter_class.import_config(config["text_config"]),
        )

    @classmethod
    def export_config(cls, config: MultiModalBaseModelConfig) -> dict:
        Assert.custom(isinstance, config, MultiModalBaseModelConfig)
        return safe_merge_dicts(
            cls.vision_model_converter_class.export_config(config.vision_encoder),
            {"text_config": cls.language_model_converter_class.export_config(config)},
        )

    @classmethod
    def get_converters(cls, config: MultiModalBaseModelConfig, exported_config: dict) -> list[WeightConverter]:
        return [
            *cls.vision_model_converter_class.get_converters(config.vision_encoder, "vision_encoder", "model"),
            *cls.language_model_converter_class.embeddings_converter_class.get_converters(
                config.embeddings, "embeddings", "model.language_model"
            ),
            *cls.language_model_converter_class.decoder_converter_class.get_converters(
                config.decoder, "decoder", "model.language_model.layers"
            ),
            *cls.language_model_converter_class.head_converter_class.get_converters(
                config.head, {"tie_word_embeddings": False}, "head"
            ),
        ]


class LlavaHuggingfaceCheckpointHandler(HuggingfaceStateDictCheckpointHandler):
    _model: MultiModalModel
    _model_class: typing.ClassVar[FastLLMModelConfig] = MultiModalModelConfig
    format: typing.ClassVar[type[CheckpointFormat]] = LlavaCheckpointFormat
    architecture: typing.ClassVar[str] = "LlavaForConditionalGeneration"
    base_model_converter_class: typing.ClassVar[type[LlavaBaseModelConverter]] = LlavaBaseModelConverter

    @classmethod
    def get_transformers_configuration_class(cls):
        import transformers

        return transformers.LlavaConfig
