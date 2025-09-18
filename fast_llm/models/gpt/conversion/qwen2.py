import typing

from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.layers.attention.config import AttentionConfig
from fast_llm.models.gpt.conversion.config import Qwen2CheckpointFormat
from fast_llm.models.gpt.conversion.llama import (
    LlamaAttentionConverter,
    LlamaBaseModelConverter,
    LlamaBlockConverter,
    LlamaDecoderConverter,
    LlamaHeadConverter,
    LlamaHuggingfaceCheckpointHandler,
)
from fast_llm.utils import Assert


class Qwen2AttentionConverter(LlamaAttentionConverter):
    # TODO: Support sliding window with max_window_layers (need 2 kinds of block?)

    @classmethod
    def _check_config(cls, config: AttentionConfig) -> None:
        Assert.is_(type(config), AttentionConfig)
        # There are multiple ways to enable biases on QKV only
        if config.add_linear_biases:
            Assert.incl(config.query_layer.bias.enabled, (None, True))
            Assert.incl(config.key_layer.bias.enabled, (None, True))
            Assert.incl(config.value_layer.bias.enabled, (None, True))
            Assert.is_(config.dense_layer.bias.enabled, False)
        else:
            Assert.is_(config.query_layer.bias.enabled, True)
            Assert.is_(config.key_layer.bias.enabled, True)
            Assert.is_(config.value_layer.bias.enabled, True)
            Assert.incl(config.dense_layer.bias.enabled, (None, False))


class Qwen2BlockConverter(LlamaBlockConverter):
    mixer_converter_class: typing.ClassVar[type[Qwen2AttentionConverter]] = Qwen2AttentionConverter


class Qwen2DecoderConverter(LlamaDecoderConverter):
    block_converter_class: typing.ClassVar[type[Qwen2BlockConverter]] = Qwen2BlockConverter


class Qwen2HeadConverter(LlamaHeadConverter):
    block_converter_class: typing.ClassVar[type[Qwen2BlockConverter]] = Qwen2BlockConverter


class Qwen2BaseModelConverter(LlamaBaseModelConverter):
    decoder_converter_class: typing.ClassVar[type[Qwen2DecoderConverter]] = Qwen2DecoderConverter
    head_converter_class: typing.ClassVar[type[Qwen2HeadConverter]] = Qwen2HeadConverter


class Qwen2HuggingfaceCheckpointHandler(LlamaHuggingfaceCheckpointHandler):
    format: typing.ClassVar[type[CheckpointFormat]] = Qwen2CheckpointFormat
    architecture: typing.ClassVar[str] = "Qwen2ForCausalLM"
    base_model_converter_class: typing.ClassVar[type[Qwen2BaseModelConverter]] = Qwen2BaseModelConverter

    @classmethod
    def get_transformers_configuration_class(cls):
        import transformers

        return transformers.Qwen2Config
