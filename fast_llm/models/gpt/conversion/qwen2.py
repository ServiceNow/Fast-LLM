import typing

from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.engine.checkpoint.external import (
    ConstantImportConfigConverter,
    CustomConfigConverter,
    WeightConverter,
)
from fast_llm.layers.attention.config import AttentionConfig
from fast_llm.layers.block.config import FixedBlockSequenceConfig
from fast_llm.models.gpt.config import GPTBaseModelConfig
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
from fast_llm.utils import Assert, div


class Qwen2AttentionConverter(LlamaAttentionConverter):
    # TODO: Support sliding window with max_window_layers (need 2 kinds of block?)

    @classmethod
    def _create_config_converters(cls) -> dict:
        out = super()._create_config_converters()
        # Qwen2 has no `attention_bias` HF field; the model always has Q/K/V biases enabled and no dense bias.
        out["add_linear_biases"] = ConstantImportConfigConverter(("add_linear_biases",), False)
        # Qwen2Config does not have `head_dim`; it is always derivable as `hidden_size // num_attention_heads`.
        out["head_size"] = CustomConfigConverter(
            fast_llm_paths=(("head_size",),),
            export_fn=lambda config: {},
            import_fn=lambda hf: {("head_size",): div(hf["hidden_size"], hf["num_attention_heads"])},
        )
        # Override Llama's blanket per-layer bias ignore with Qwen2's hardcoded layer biases.
        # On export the per-layer biases must be compatible with `add_linear_biases`; see ``_validate_export``.
        out["linear_layers"] = CustomConfigConverter(
            fast_llm_paths=(("query_layer",), ("key_layer",), ("value_layer",), ("dense_layer",)),
            export_fn=lambda config: {},
            import_fn=lambda hf: {
                ("query_layer",): {"bias": {"enabled": True}},
                ("key_layer",): {"bias": {"enabled": True}},
                ("value_layer",): {"bias": {"enabled": True}},
                ("dense_layer",): {"bias": {"enabled": False}},
            },
        )
        return out

    @classmethod
    def _validate_export(cls, config: AttentionConfig) -> None:
        Assert.is_(type(config), AttentionConfig)
        # There are multiple ways to enable biases on QKV only.
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
    def _create_config_converters(cls) -> dict:
        return {
            **super()._create_config_converters(),
            # Qwen2 has no `mlp_bias` HF field; biases are always disabled.
            "add_linear_biases": ConstantImportConfigConverter(("add_linear_biases",), False),
        }


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

    @classmethod
    def import_config(cls, hf_dict: dict) -> dict:
        assert hf_dict.get("use_mrope") is not True, "MRoPE (use_mrope=True) is not supported by the Qwen2 converter"
        return super().import_config(hf_dict)

    @classmethod
    def _validate_export(cls, config: GPTBaseModelConfig) -> None:
        super()._validate_export(config)
        block = (
            config.decoder.block
            if isinstance(config.decoder, FixedBlockSequenceConfig)
            else next(iter(config.decoder.blocks.values()))
        )
        if isinstance(block.mixer, AttentionConfig):
            Assert.eq(
                block.mixer.heads * block.mixer.head_size,
                config.hidden_size,
                msg="Qwen2 format omits head_dim; requires heads * head_size == hidden_size",
            )


class Qwen2HuggingfaceCheckpointHandler(LlamaHuggingfaceCheckpointHandler):
    format: typing.ClassVar[type[CheckpointFormat]] = Qwen2CheckpointFormat
    architecture: typing.ClassVar[str] = "Qwen2ForCausalLM"
    base_model_converter_class: typing.ClassVar[type[Qwen2BaseModelConverter]] = Qwen2BaseModelConverter

    @classmethod
    def get_transformers_configuration_class(cls):
        import transformers

        return transformers.Qwen2Config
