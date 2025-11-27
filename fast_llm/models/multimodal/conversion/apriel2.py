"""
Apriel2 multimodal checkpoint format converter.

Combines Apriel2's flexible decoder (with pattern-based blocks, mamba, attention, etc.)
with vision encoder capabilities.
"""

import typing

from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.engine.checkpoint.external import WeightConverter
from fast_llm.engine.checkpoint.huggingface import HuggingfaceStateDictCheckpointHandler
from fast_llm.engine.multi_stage.config import FastLLMModelConfig
from fast_llm.models.gpt.conversion.apriel2 import (
    Apriel2BaseModelConverter,
    Apriel2DecoderConverter,
    Apriel2HeadConverter,
)
from fast_llm.models.gpt.conversion.llama import get_parameter_converter
from fast_llm.models.multimodal.config import MultiModalBaseModelConfig, MultiModalModelConfig
from fast_llm.models.multimodal.conversion.config import Apriel2CheckpointFormat
from fast_llm.models.multimodal.conversion.llava import (
    LlavaBaseModelConverter,
    LlavaHeadConverter,
    LlavaVisionModelConverter,
)
from fast_llm.models.multimodal.model import MultiModalModel
from fast_llm.utils import Assert, safe_merge_dicts


class Apriel2VisionHeadConverter(Apriel2HeadConverter):
    """Head converter for Apriel2 multimodal - uses language_model prefix."""

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
                "model.language_model.norm",
            ),
            get_parameter_converter(
                f"{fast_llm_prefix}.output_weights",
                "lm_head.weight",
                drop_on_import=exported_config.get("tie_word_embeddings", False),
            ),
        ]


class Apriel2LanguageModelConverter(Apriel2BaseModelConverter):
    """Language model converter for Apriel2 multimodal."""

    head_converter_class: typing.ClassVar[type[Apriel2VisionHeadConverter]] = Apriel2VisionHeadConverter


class Apriel2MultimodalBaseModelConverter(LlavaBaseModelConverter):
    """
    Base model converter for Apriel2 multimodal.

    Uses Apriel2's decoder converters for the language model,
    combined with the vision model converter from Llava.
    """

    vision_model_converter_class: typing.ClassVar[type[LlavaVisionModelConverter]] = LlavaVisionModelConverter
    language_model_converter_class: typing.ClassVar[type[Apriel2LanguageModelConverter]] = Apriel2LanguageModelConverter

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
                config.head, exported_config, "head"
            ),
        ]


class Apriel2HuggingfaceCheckpointHandler(HuggingfaceStateDictCheckpointHandler):
    """HuggingFace checkpoint handler for Apriel2 multimodal format."""

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
        return safe_merge_dicts(
            super()._export_config(config),
            {
                "auto_map": {
                    "AutoConfig": "configuration_apriel2.Apriel2Config",
                    "AutoModel": "modeling_apriel2.Apriel2Model",
                    "AutoModelForCausalLM": "modeling_apriel2.Apriel2ForConditionalGeneration",
                },
            },
        )
