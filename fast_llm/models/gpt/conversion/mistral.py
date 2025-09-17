import typing

from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.layers.attention.config import AttentionConfig
from fast_llm.models.gpt.conversion.config import MistralCheckpointFormat
from fast_llm.models.gpt.conversion.llama import (
    LlamaAttentionConverter,
    LlamaBaseModelConverter,
    LlamaBlockConverter,
    LlamaDecoderConverter,
    LlamaHeadConverter,
    LlamaHuggingfaceCheckpointHandler,
)
from fast_llm.utils import safe_merge_dicts


class MistralAttentionConverter(LlamaAttentionConverter):
    @classmethod
    def import_config(cls, config: dict, hidden_size: int) -> dict:
        return safe_merge_dicts(super().import_config(config, hidden_size), {"window_size": config["sliding_window"]})

    @classmethod
    def export_config(cls, config: AttentionConfig) -> dict:
        return safe_merge_dicts(
            super().export_config(config),
            {"sliding_window": config.window_size},
        )

    @classmethod
    def _check_config(cls, config: AttentionConfig) -> None:
        # Mistral doesn't support biases.
        assert not config.add_linear_biases


class MistralBlockConverter(LlamaBlockConverter):
    mixer_converter_class: typing.ClassVar[type[MistralAttentionConverter]] = MistralAttentionConverter


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
