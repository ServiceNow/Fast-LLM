import typing

from transformers import PretrainedConfig

from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.engine.checkpoint.external import WeightConverter
from fast_llm.layers.block.config import FixedBlockSequenceConfig
from fast_llm.layers.language_model.config import LanguageModelHeadConfig, MultiTokenPredictionConfig
from fast_llm.models.gpt.config import GPTModelConfig
from fast_llm.models.gpt.conversion.config import MTPLlamaCheckpointFormat
from fast_llm.models.gpt.conversion.llama import (
    LlamaBaseModelConverter,
    LlamaBlockConverter,
    LlamaDecoderConverter,
    LlamaHeadConverter,
    LlamaHuggingfaceCheckpointHandler,
    get_parameter_converter,
)
from fast_llm.utils import Assert, safe_merge_dicts


class MTPLlamaHeadConverter(LlamaHeadConverter):
    @classmethod
    def import_config(cls, config: dict) -> dict:
        return {
            "type": "multi_token_prediction",
            "block": LlamaBlockConverter.import_config(config),
            "head": super().import_config(config),
            "prediction_heads": config["prediction_heads"],
        }

    @classmethod
    def export_config(cls, config: MultiTokenPredictionConfig) -> dict:
        Assert.custom(isinstance, config, MultiTokenPredictionConfig)
        return safe_merge_dicts(
            super().export_config(config.head),
            {"prediction_heads": config.prediction_heads},
        )

    @classmethod
    def get_converters(
        cls,
        config: LanguageModelHeadConfig,
        exported_config: dict,
        fast_llm_prefix: str,
    ) -> list[WeightConverter]:
        converters = []
        for prediction_distance in range(config.prediction_heads):
            converters += cls.block_converter_class.get_converters(
                config.block,
                f"{fast_llm_prefix}.blocks.{prediction_distance}",
                (
                    f"model.layers.{exported_config["num_hidden_layers"]-1}"
                    if prediction_distance == 0
                    else f"model.mtp_heads.{prediction_distance - 1}"
                ),
            )
            converters += cls.normalization_converter_class.get_converters(
                config.head.normalization,
                f"{fast_llm_prefix}.heads.{prediction_distance}.final_norm",
                f"model.mtp_norms.{prediction_distance}",
            )
        converters.append(
            get_parameter_converter(
                f"{fast_llm_prefix}.heads.0.output_weights",
                "lm_head.weight",
                drop_on_import=exported_config["tie_word_embeddings"],
            )
        )

        return converters


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
