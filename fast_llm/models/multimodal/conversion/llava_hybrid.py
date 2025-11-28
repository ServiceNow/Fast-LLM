import typing

from transformers import PretrainedConfig

from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.models.gpt.conversion.apriel import AprielBaseModelConverter
from fast_llm.models.multimodal.config import MultiModalModelConfig
from fast_llm.models.multimodal.conversion.config import LlavaHybridSSMCheckpointFormat
from fast_llm.models.multimodal.conversion.llava import LlavaBaseModelConverter, LlavaHuggingfaceCheckpointHandler
from fast_llm.utils import safe_merge_dicts


class LlavaHybridBaseModelConverter(LlavaBaseModelConverter):
    language_model_converter_class: typing.ClassVar[type[AprielBaseModelConverter]] = AprielBaseModelConverter


class LlavaHybridSSMHuggingfaceCheckpointHandler(LlavaHuggingfaceCheckpointHandler):
    format: typing.ClassVar[type[CheckpointFormat]] = LlavaHybridSSMCheckpointFormat
    architecture: typing.ClassVar[str] = "LlavaHybridForConditionalGeneration"
    base_model_converter_class: typing.ClassVar[type[LlavaHybridBaseModelConverter]] = LlavaHybridBaseModelConverter

    @classmethod
    def get_transformers_configuration_class(cls) -> type[PretrainedConfig]:
        from fast_llm_external_models.llava_hybrid.configuration_llava_hybrid import LlavaHybridConfig

        return LlavaHybridConfig

    @classmethod
    def get_model_files(cls) -> tuple[str, str, str | None]:
        from fast_llm_external_models.llava_hybrid import configuration_llava_hybrid, modeling_llava_hybrid

        return configuration_llava_hybrid.__file__, modeling_llava_hybrid.__file__, None

    @classmethod
    def _export_config(cls, config: MultiModalModelConfig) -> dict[str, typing.Any]:
        return safe_merge_dicts(
            super()._export_config(config),
            {
                "auto_map": {
                    "AutoConfig": "configuration_llava_hybrid.LlavaHybridConfig",
                    "AutoModel": "modeling_llava_hybrid.LlavaHybridModel",
                    "AutoModelForCausalLM": "modeling_llava_hybrid.LlavaHybridForConditionalGeneration",
                    "AutoModelForVision2Seq": "modeling_llava_hybrid.LlavaHybridForConditionalGeneration",
                    "AutoModelForImageTextToText": "modeling_llava_hybrid.LlavaHybridForConditionalGeneration",
                },
            },
        )
