import typing

from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.engine.checkpoint.external import WeightConverter
from fast_llm.layers.attention.config import AttentionConfig
from fast_llm.layers.decoder.mlp.config import MLPConfig
from fast_llm.models.gpt.conversion.config import Qwen2CheckpointFormat
from fast_llm.models.gpt.conversion.llama import (
    KeyValueWeightConverter,
    LlamaAttentionConverter,
    LlamaBaseModelConverter,
    LlamaBlockConverter,
    LlamaDecoderConverter,
    LlamaHeadConverter,
    LlamaHuggingfaceCheckpointHandler,
    LlamaMLPConverter,
    QueryWeightConverter,
    get_weight_and_bias_converters,
)
from fast_llm.utils import Assert


class Qwen2AttentionConverter(LlamaAttentionConverter):
    # TODO: Support sliding window with max_window_layers (need 2 kinds of block?)

    @classmethod
    def import_config(cls, config: dict) -> dict:
        config["attention_bias"] = True
        out = super().import_config(config)
        out["query_layer"] = {"bias": {"enabled": True}}
        out["key_layer"] = {"bias": {"enabled": True}}
        out["value_layer"] = {"bias": {"enabled": True}}
        out["dense_layer"] = {"bias": {"enabled": False}}
        return out

    @classmethod
    def export_config(cls, config: AttentionConfig) -> dict:
        out = super().export_config(config)
        del out["attention_bias"]
        return out

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

    @classmethod
    def get_converters(
        cls,
        config: AttentionConfig,
        fast_llm_prefix: str,
        hf_prefix: str,
        drop_on_export: bool = False,
    ) -> list[WeightConverter]:
        return [
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.query",
                f"{hf_prefix}.q_proj",
                True,
                QueryWeightConverter,
                config,
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.key_value",
                (f"{hf_prefix}.k_proj", f"{hf_prefix}.v_proj"),
                True,
                KeyValueWeightConverter,
                config,
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.dense",
                f"{hf_prefix}.o_proj",
                False,
                drop_on_export=drop_on_export,
            ),
        ]


class Qwen2MLPConverter(LlamaMLPConverter):
    @classmethod
    def import_config(cls, config: dict) -> dict:
        config["mlp_bias"] = False
        return super().import_config(config)

    @classmethod
    def export_config(cls, config: MLPConfig) -> dict:
        out = super().export_config(config)
        del out["mlp_bias"]
        return out


class Qwen2BlockConverter(LlamaBlockConverter):
    mixer_converter_class: typing.ClassVar[type[Qwen2AttentionConverter]] = Qwen2AttentionConverter
    mlp_converter_class: typing.ClassVar[type[Qwen2MLPConverter]] = Qwen2MLPConverter


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
