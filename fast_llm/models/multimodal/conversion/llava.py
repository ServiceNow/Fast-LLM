import typing

import torch

from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.engine.checkpoint.external import WeightConverter
from fast_llm.engine.checkpoint.huggingface import HuggingFaceBaseModelConverter, HuggingfaceStateDictCheckpointHandler
from fast_llm.engine.multi_stage.config import FastLLMModelConfig
from fast_llm.functional.config import ActivationType
from fast_llm.layers.attention.config import AttentionConfig
from fast_llm.layers.attention.rotary.config import Rotary2DConfig
from fast_llm.layers.common.normalization.config import RMSNormalizationConfig
from fast_llm.layers.decoder.mlp.config import MLPConfig
from fast_llm.layers.language_model.config import LanguageModelHeadConfig
from fast_llm.layers.vision.config import PatchEmbeddingsConfig, VisionEncoderConfig
from fast_llm.models.gpt.conversion.llama import (
    LlamaAttentionConverter,
    LlamaBlockConverter,
    LlamaDecoderConverter,
    LlamaNormalizationConverter,
    MLPLayer2Converter,
    get_parameter_converter,
    get_weight_and_bias_converters,
)
from fast_llm.models.gpt.conversion.mistral import MistralBaseModelConverter, MistralHeadConverter, MistralMLPConverter
from fast_llm.models.multimodal.config import MultiModalBaseModelConfig, MultiModalModelConfig
from fast_llm.models.multimodal.conversion.config import LlavaCheckpointFormat
from fast_llm.models.multimodal.model import MultiModalModel
from fast_llm.tensor import SafeTensorSlice
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
        out["causal"] = False
        out["cross_document_attention"] = False
        return out

    @classmethod
    def export_config(cls, config: AttentionConfig) -> dict:
        cls._check_config(config)
        Assert.eq(config.softmax_scale_power, 0.5)
        Assert.is_(type(config.rotary), Rotary2DConfig)
        assert not config.add_linear_biases
        assert not config.causal
        assert not config.cross_document_attention
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


class PatchEmbeddingWeightConverter(WeightConverter):
    _config: PatchEmbeddingsConfig

    def export_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        return tuple(
            weight_[:].view(
                *weight_[:].shape[:-1],
                self._config.input_channels,
                self._config.patch_height,
                self._config.patch_width,
            )
            for weight_ in weight
        )

    def import_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        return tuple(
            weight_[:].view(
                *weight_[:].shape[:-3],
                self._config.input_channels * self._config.patch_height * self._config.patch_width,
            )
            for weight_ in weight
        )


class PixtralEmbeddingsConverter:
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
    def export_config(cls, config: PatchEmbeddingsConfig) -> dict:
        Assert.custom(isinstance, config, PatchEmbeddingsConfig)
        Assert.eq(config.patch_height, config.patch_width)
        Assert.incl(config.patch_embeddings.bias.enabled, (None, False))

        return safe_merge_dicts(
            {
                "patch_size": config.patch_height,
                "num_channels": config.input_channels,
            },
            cls.normalization_converter_class.export_config(config.normalization),
        )

    @classmethod
    def get_converters(
        cls, config: PatchEmbeddingsConfig, fast_llm_prefix: str, hf_prefix: str
    ) -> list[WeightConverter]:
        return [
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.patch_embeddings",
                f"{hf_prefix}.patch_conv",
                False,
                PatchEmbeddingWeightConverter,
                config,
            ),
            *cls.normalization_converter_class.get_converters(
                config, f"{fast_llm_prefix}.normalization", f"{hf_prefix}.ln_pre"
            ),
        ]


class LlavaVisionAdapterConverter:
    @classmethod
    def import_config(cls, config: dict) -> dict:
        return {
            "intermediate_size": config["vision_config"]["hidden_size"],
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
            "multimodal_projector_bias": config.add_linear_biases,
            # Not in LlavaConfig, but needed for consistency check in LlavaBaseModelConverter.
            "projector_intermediate_size": config.intermediate_size,
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
    embeddings_converter_class: typing.ClassVar[type[PixtralEmbeddingsConverter]] = PixtralEmbeddingsConverter
    encoder_converter_class: typing.ClassVar[type[PixtralEncoderConverter]] = PixtralEncoderConverter
    model_type: typing.ClassVar[str] = "pixtral"

    @classmethod
    def import_config(cls, config: dict) -> dict:
        Assert.eq(config["vision_config"]["model_type"], cls.model_type)
        return {
            "embeddings": cls.embeddings_converter_class.import_config(config["vision_config"]),
            "encoder": cls.encoder_converter_class.import_config(config["vision_config"]),
            "adapter": cls.vision_adapter_converter_class.import_config(config),
            "hidden_size": config["vision_config"]["hidden_size"],
        }

    @classmethod
    def export_config(cls, config: VisionEncoderConfig) -> dict:
        Assert.custom(isinstance, config, VisionEncoderConfig)
        # TODO: ====== image_size? ======
        vision_config = safe_merge_dicts(
            cls.embeddings_converter_class.export_config(config.embeddings),
            cls.encoder_converter_class.export_config(config.encoder),
            {"hidden_size": config.hidden_size, "model_type": cls.model_type},
        )

        Assert.eq(
            vision_config.pop("head_dim"), div(vision_config["hidden_size"], vision_config["num_attention_heads"])
        )

        return safe_merge_dicts(
            {"vision_config": vision_config},
            cls.vision_adapter_converter_class.export_config(config.adapter),
        )

    @classmethod
    def get_converters(cls, config: VisionEncoderConfig) -> list[WeightConverter]:
        return [
            *cls.embeddings_converter_class.get_converters(
                config.embeddings, "vision_encoder.embeddings", "model.vision_tower"
            ),
            *cls.encoder_converter_class.get_converters(
                config.encoder, "vision_encoder.encoder", "model.vision_tower.transformer.layers"
            ),
            *cls.vision_adapter_converter_class.get_converters(
                config.adapter, "vision_encoder.adapter", "model.multi_modal_projector"
            ),
        ]


class LlavaHeadConverter(MistralHeadConverter):
    @classmethod
    def get_converters(
        cls,
        config: LanguageModelHeadConfig,
        exported_config: dict,
        fast_llm_prefix: str,
    ) -> list[WeightConverter]:
        return [
            *cls.normalization_converter_class.get_converters(
                config.normalization,
                f"{fast_llm_prefix}.final_norm",
                f"model.language_model.norm",
            ),
            get_parameter_converter(
                f"{fast_llm_prefix}.output_weights",
                "lm_head.weight",
                drop_on_import=exported_config["tie_word_embeddings"],
            ),
        ]


class LlavaLanguageModelConverter(MistralBaseModelConverter):
    head_converter_class: typing.ClassVar[type[LlavaHeadConverter]] = LlavaHeadConverter


class LlavaBaseModelConverter(HuggingFaceBaseModelConverter):
    vision_model_converter_class: typing.ClassVar[type[LlavaVisionModelConverter]] = LlavaVisionModelConverter
    # TODO: Make it flexible?
    language_model_converter_class: typing.ClassVar[type[LlavaLanguageModelConverter]] = LlavaLanguageModelConverter
    # TODO: ====== Is tie_word_embeddings supported? ======

    @classmethod
    def import_config(cls, config: dict) -> dict:
        return safe_merge_dicts(
            {
                "vision_encoder": cls.vision_model_converter_class.import_config(config),
                "image_token_index": config["image_token_index"],
            },
            cls.language_model_converter_class.import_config(config["text_config"]),
        )

    @classmethod
    def export_config(cls, config: MultiModalBaseModelConfig) -> dict:
        Assert.custom(isinstance, config, MultiModalBaseModelConfig)
        assert config.image_token_index is not None
        out = safe_merge_dicts(
            cls.vision_model_converter_class.export_config(config.vision_encoder),
            {
                "text_config": cls.language_model_converter_class.export_config(config),
                "image_token_index": config.image_token_index,
                "vision_feature_select_strategy": "full",
                "vision_feature_layer": -1,
            },
        )
        Assert.eq(out.pop("projector_intermediate_size"), out["text_config"]["hidden_size"])
        return out

    @classmethod
    def get_converters(cls, config: MultiModalBaseModelConfig, exported_config: dict) -> list[WeightConverter]:
        return [
            *cls.vision_model_converter_class.get_converters(config.vision_encoder),
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
