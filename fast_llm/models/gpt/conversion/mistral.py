import typing

from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.layers.attention.config import AttentionConfig
from fast_llm.layers.decoder.mlp.config import MLPConfig
from fast_llm.models.gpt.conversion.config import MistralCheckpointFormat
from fast_llm.models.gpt.conversion.llama import (
    LlamaAttentionConverter,
    LlamaBaseModelConverter,
    LlamaBlockConverter,
    LlamaDecoderConverter,
    LlamaHeadConverter,
    LlamaHuggingfaceCheckpointHandler,
    LlamaMLPConverter,
)
from fast_llm.utils import safe_merge_dicts


class MistralAttentionConverter(LlamaAttentionConverter):
    @classmethod
    def import_config(cls, config: dict) -> dict:
        config["attention_bias"] = False
        return safe_merge_dicts(
            super().import_config(config),
            {"window_size": config["sliding_window"]},
        )

    @classmethod
    def export_config(cls, config: AttentionConfig) -> dict:
        out = safe_merge_dicts(
            super().export_config(config),
            {"sliding_window": config.window_size},
        )
        del out["attention_bias"]
        return out

    @classmethod
    def _check_config(cls, config: AttentionConfig) -> None:
        # Mistral doesn't support biases.
        assert not config.add_linear_biases


class MistralMLPConverter(LlamaMLPConverter):
    @classmethod
    def import_config(cls, config: dict) -> dict:
        config["mlp_bias"] = False
        return super().import_config(config)

    @classmethod
    def export_config(cls, config: MLPConfig) -> dict:
        assert not config.add_linear_biases
        out = super().export_config(config)
        del out["mlp_bias"]
        return out


class MistralBlockConverter(LlamaBlockConverter):
    mixer_converter_class: typing.ClassVar[type[MistralAttentionConverter]] = MistralAttentionConverter
    mlp_converter_class: typing.ClassVar[type[MistralMLPConverter]] = MistralMLPConverter


class MistralDecoderConverter(LlamaDecoderConverter):
    block_converter_class: typing.ClassVar[type[MistralBlockConverter]] = MistralBlockConverter


class MistralHeadConverter(LlamaHeadConverter):
    block_converter_class: typing.ClassVar[type[MistralBlockConverter]] = MistralBlockConverter


class MistralBaseModelConverter(LlamaBaseModelConverter):
    decoder_converter_class: typing.ClassVar[type[MistralDecoderConverter]] = MistralDecoderConverter
    head_converter_class: typing.ClassVar[type[MistralHeadConverter]] = MistralHeadConverter


class MistralHuggingfaceCheckpointHandler(LlamaHuggingfaceCheckpointHandler):
    format: typing.ClassVar[type[CheckpointFormat]] = MistralCheckpointFormat
    architecture: typing.ClassVar[str] = "MistralForCausalLM"
    base_model_converter_class: typing.ClassVar[type[MistralBaseModelConverter]] = MistralBaseModelConverter

    @classmethod
    def get_transformers_configuration_class(cls):
        import transformers

        return transformers.MistralConfig
