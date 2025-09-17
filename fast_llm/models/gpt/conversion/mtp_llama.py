import typing

from transformers import PretrainedConfig

from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.engine.checkpoint.external import WeightConverter
from fast_llm.layers.decoder.config import DecoderBlockConfig
from fast_llm.layers.language_model.config import LanguageModelHeadConfig
from fast_llm.models.gpt.config import GPTModelConfig
from fast_llm.models.gpt.conversion.config import MTPLlamaCheckpointFormat
from fast_llm.models.gpt.conversion.llama import (
    LlamaBaseModelConverter,
    LlamaHeadConverter,
    LlamaHuggingfaceCheckpointHandler,
    get_parameter_converter,
)
from fast_llm.utils import safe_merge_dicts


class MTPLlamaHeadConverter(LlamaHeadConverter):
    @classmethod
    def import_config(cls, config: dict) -> dict:
        return safe_merge_dicts(
            super().import_config(config),
            {"prediction_heads": config["prediction_heads"]},
        )

    @classmethod
    def export_config(cls, config: LanguageModelHeadConfig) -> dict:
        return safe_merge_dicts(
            super().export_config(config),
            {"prediction_heads": config.prediction_heads},
        )

    @classmethod
    def get_converters(
        cls, config: LanguageModelHeadConfig, block_config: DecoderBlockConfig, fast_llm_prefix: str, start_index: int
    ) -> list[WeightConverter]:
        converters = []
        for prediction_distance in range(config.prediction_heads):
            if prediction_distance > 0:
                converters += cls.block_converter_class.get_converters(
                    block_config,
                    f"{fast_llm_prefix}.{start_index+2*prediction_distance-1}",
                    f"model.mtp_heads.{prediction_distance - 1}",
                )
            converters += cls.normalization_converter_class.get_converters(
                config.normalization,
                f"{fast_llm_prefix}.{start_index+2*prediction_distance}.final_norm",
                f"model.mtp_norms.{prediction_distance}",
            )
        converters.append(
            get_parameter_converter(
                f"{fast_llm_prefix}.{start_index}.output_weights",
                "lm_head.weight",
                drop_on_import=config.tied_weight,
            )
        )

        return converters


class MTPLlamaBaseModelConverter(LlamaBaseModelConverter):
    head_converter_class: typing.ClassVar[type[MTPLlamaHeadConverter]] = MTPLlamaHeadConverter


class MTPLlamaHuggingfaceCheckpointHandler(LlamaHuggingfaceCheckpointHandler):
    format: typing.ClassVar[type[CheckpointFormat]] = MTPLlamaCheckpointFormat
    architecture: typing.ClassVar[str] = "MTPLlamaForCausalLM"
    base_model_converter_class: typing.ClassVar[type[MTPLlamaBaseModelConverter]] = MTPLlamaBaseModelConverter

    @classmethod
    def _export_config(cls, config: GPTModelConfig) -> dict[str, typing.Any]:
        return safe_merge_dicts(
            super()._export_config(config),
            {
                "auto_map": {
                    "AutoConfig": "configuration_mtp_llama.MTPLlamaConfig",
                    "AutoModel": "modeling_mtp_llama.MTPLlamaModel",
                    "AutoModelForCausalLM": "modeling_mtp_llama.MTPLlamaForCausalLM",
                },
            },
        )

    @classmethod
    def get_transformers_configuration_class(cls) -> type[PretrainedConfig]:
        from fast_llm_external_models.mtp_llama.configuration_mtp_llama import MTPLlamaConfig

        return MTPLlamaConfig

    @classmethod
    def get_model_files(cls) -> tuple[str, str, str | None]:
        from fast_llm_external_models.mtp_llama import configuration_mtp_llama, modeling_mtp_llama

        return configuration_mtp_llama.__file__, modeling_mtp_llama.__file__, None
