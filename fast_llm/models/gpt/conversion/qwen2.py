import functools
import typing

from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.engine.checkpoint.external import (
    ConstantImportConfigConverter,
    IgnoredConfigConverter,
    ImportOnlyConfigConverter,
    KeyValueWeightConverter,
    LinearWeightConverter,
    WeightConverter,
)
from fast_llm.layers.attention.config import AttentionConfig
from fast_llm.layers.block.config import FixedBlockSequenceConfig
from fast_llm.models.gpt.config import GPTBaseModelConfig
from fast_llm.models.gpt.conversion.config import Qwen2CheckpointFormat
from fast_llm.models.gpt.conversion.llama import (
    LlamaAttentionConverter,
    LlamaBaseModelConverter,
    LlamaBlockConverter,
    LlamaHeadConverter,
    LlamaHuggingfaceCheckpointHandler,
    LlamaMLPConverter,
)
from fast_llm.utils import Assert, div


class Qwen2AttentionConverter(LlamaAttentionConverter):
    # TODO: Support sliding window with max_window_layers (need 2 kinds of block?)

    @classmethod
    def _create_config_converters(cls) -> dict:
        return {
            **super()._create_config_converters(),
            # Qwen2 has no `attention_bias` HF field; the model always has Q/K/V biases enabled and no dense bias.
            "add_linear_biases": ConstantImportConfigConverter(("add_linear_biases",), False),
            # Qwen2Config does not have `head_dim`; it is always derivable as `hidden_size // num_attention_heads`.
            "head_size": ImportOnlyConfigConverter(
                fast_llm_paths=(("head_size",),),
                import_fn=lambda hf: {("head_size",): div(hf["hidden_size"], hf["num_attention_heads"])},
            ),
            # Override Llama's blanket per-layer bias ignore with Qwen2's hardcoded layer biases.
            # On export the per-layer biases must be compatible with `add_linear_biases`; see ``_validate_export``.
            "linear_layers": ImportOnlyConfigConverter(
                fast_llm_paths=(("query_layer",), ("key_layer",), ("value_layer",), ("dense_layer",)),
                import_fn=lambda hf: {
                    ("query_layer",): {"bias": {"enabled": True}},
                    ("key_layer",): {"bias": {"enabled": True}},
                    ("value_layer",): {"bias": {"enabled": True}},
                    ("dense_layer",): {"bias": {"enabled": False}},
                },
                fast_llm_recurses=True,
            ),
            # Sliding-window machinery surfaced by Qwen2 HF but not yet supported here (see TODO above).
            "sliding_window_unsupported": IgnoredConfigConverter(
                hf_paths=(("sliding_window",), ("use_sliding_window",), ("max_window_layers",), ("layer_types",)),
            ),
        }

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

    # Qwen2 hardcodes Q/K/V biases on, dense bias off — independent of ``add_linear_biases`` (which is
    # pinned to False on the config side because there's no HF ``attention_bias`` field).
    @classmethod
    @functools.cache
    def _create_weight_converters(cls) -> dict[str, WeightConverter]:
        return {
            "query": LinearWeightConverter("query", "q_proj", bias_fn=lambda c: True),
            "key_value": LinearWeightConverter(
                "key_value", ("k_proj", "v_proj"), transform=KeyValueWeightConverter, bias_fn=lambda c: True
            ),
            "dense": LinearWeightConverter("dense", "o_proj", bias_fn=lambda c: False),
        }


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


class Qwen2HeadConverter(LlamaHeadConverter):
    block_converter_class: typing.ClassVar[type[Qwen2BlockConverter]] = Qwen2BlockConverter


def _qwen2_mrope_guard_import(hf_dict: dict) -> dict:
    if hf_dict.get("use_mrope") is True:
        raise NotImplementedError("MRoPE (use_mrope=True) is not supported by the Qwen2 converter")
    return {}


class Qwen2BaseModelConverter(LlamaBaseModelConverter):
    block_converter_class: typing.ClassVar[type[Qwen2BlockConverter]] = Qwen2BlockConverter
    head_converter_class: typing.ClassVar[type[Qwen2HeadConverter]] = Qwen2HeadConverter

    @classmethod
    def _create_config_converters(cls) -> dict:
        return {
            **super()._create_config_converters(),
            # Refuse MRoPE on import; the export path can't produce ``use_mrope=True`` because Fast-LLM
            # has no rotary type that maps to it.
            "use_mrope_guard": ImportOnlyConfigConverter(
                fast_llm_paths=(),
                hf_paths=(("use_mrope",),),
                import_fn=_qwen2_mrope_guard_import,
            ),
        }

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
