import typing

from transformers import PretrainedConfig

from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.engine.checkpoint.external import WeightConverter
from fast_llm.layers.block.config import FixedBlockSequenceConfig
from fast_llm.layers.language_model.config import LanguageModelHeadConfig
from fast_llm.models.gpt.config import GPTModelConfig
from fast_llm.models.gpt.conversion.config import MTPLlamaCheckpointFormat
from fast_llm.models.gpt.conversion.llama import (
    LlamaBaseModelConverter,
    LlamaDecoderConverter,
    LlamaHeadConverter,
    LlamaHuggingfaceCheckpointHandler,
)
from fast_llm.utils import Assert, safe_merge_dicts


class MTPLlamaHeadConverter(LlamaHeadConverter):
    @classmethod
    def import_config(cls, config: dict) -> dict:
        return {
            **super().import_config(config),
            "prediction_heads": config["prediction_heads"],
        }

    @classmethod
    def export_config(cls, config: LanguageModelHeadConfig) -> dict:
        return safe_merge_dicts(
            super().export_config(config),
            {"prediction_heads": config.prediction_heads},
        )

    @classmethod
    def get_converters(
        cls,
        config: LanguageModelHeadConfig,
        exported_config: dict,
    ) -> list[WeightConverter]:
        return super().get_converters(config, exported_config) + [
            cls.normalization_converter_class.get_converters(
                config.head.normalization,
                f"multi_token_prediction.heads.{prediction_distance - 1}.final_norm",
                f"model.mtp_norms.{prediction_distance}",
            )
            for prediction_distance in range(1, config.prediction_heads)
        ]


class MTPLlamaDecoderConverter(LlamaDecoderConverter):
    @classmethod
    def import_config(cls, config: dict) -> dict:
        return {
            "block": cls.block_converter_class.import_config(config),
            "num_blocks": config["num_hidden_layers"] - 1,
        }

    @classmethod
    def export_config(cls, config: FixedBlockSequenceConfig) -> dict:
        # TODO: Support PatternBlockSequenceConfig with compatible configs.
        Assert.custom(isinstance, config, FixedBlockSequenceConfig)
        return safe_merge_dicts(
            cls.block_converter_class.export_config(config.block),
            {"num_hidden_layers": config.num_blocks + 1},
        )


class MTPLlamaBaseModelConverter(LlamaBaseModelConverter):
    decoder_converter_class: typing.ClassVar[type[MTPLlamaDecoderConverter]] = MTPLlamaDecoderConverter
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
