import typing

from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.engine.checkpoint.external import ConstantImportConfigConverter, RenameConfigConverter
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


class MistralAttentionConverter(LlamaAttentionConverter):
    @classmethod
    def _create_config_converters(cls) -> dict:
        return {
            **super()._create_config_converters(),
            # Mistral has no `attention_bias` HF field; biases are always disabled.
            "add_linear_biases": ConstantImportConfigConverter(("add_linear_biases",), False),
            "window_size": RenameConfigConverter(("window_size",), ("sliding_window",)),
        }


class MistralMLPConverter(LlamaMLPConverter):
    @classmethod
    def _create_config_converters(cls) -> dict:
        return {
            **super()._create_config_converters(),
            # Mistral has no `mlp_bias` HF field; biases are always disabled.
            "add_linear_biases": ConstantImportConfigConverter(("add_linear_biases",), False),
        }


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
